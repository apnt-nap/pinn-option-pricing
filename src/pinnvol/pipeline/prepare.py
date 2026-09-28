r"""Stage 1: build the option panel used by every later stage.

Starting from the cleaned files (``clean_spx.py`` already applied DTE 7-365,
$0.8 \le K/S \le 1.2$, bid > 0, ask > bid, mid >= 0.375 and valid vendor IV), this stage

1. keeps the configured contract family (third-Friday monthlies by default);
2. measures $\tau$ in calendar years to settlement: AM-settled monthlies settle at the
   09:30 open, so $\tau = (\text{DTE} - 6.5/24)/365$ for them (Section 16.8);
3. attaches the maturity-matched risk-free rate $r(t, \tau)$ from FRED;
4. backs out the forward $F_{t,T}$ and dividend yield $q_{t,T}$ from put-call parity;
5. applies the study filters: $7 \le \text{DTE} \le 180$, $0.8 \le K/F \le 1.2$,
   relative spread $\le 20\%$, and static no-arbitrage bounds on the mid;
6. computes the market implied volatility $\sigma^{imp}$ from the mid with the same $r$ and $q$.

The panel covers the whole sample (training, validation and test years) and is saved
as ``panel.pkl`` with a per-filter log ``prepare_log.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data.forwards import implied_forwards
from ..data.loader import load_options
from ..data.rates import load_rate_curve
from ..pricing.implied_vol import implied_vol
from ..vol.horizon import trading_horizon


def prepare_panel(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save the filtered option panel. Returns ``(panel, filter_log)``."""
    p, s = cfg.paths, cfg.sample
    opts = load_options(p.data_dir, cfg.period.start, cfg.period.test_end, s.contract_family)
    log = [("loaded", len(opts))]

    am = opts["is_third_friday"].to_numpy() == 1
    opts["tau"] = (opts["dte"].to_numpy(np.float64) - am * s.am_settlement_hours / 24.0) / 365.0

    curve = load_rate_curve(p.rates_dir, cfg.rates.series, cfg.rates.fallback_rate)
    opts["r"] = curve.rate(opts["quote_date"].to_numpy(), opts["tau"].to_numpy())

    f = cfg.forwards
    fw = implied_forwards(opts, f.n_strikes, f.default_q, tuple(f.q_clip))
    opts = opts.merge(fw, on=["quote_date", "expire_date"], how="left")
    opts["fwd_moneyness"] = opts["K"] / opts["F"]
    opts["k"] = np.log(opts["fwd_moneyness"])
    opts["x"] = np.log(opts["S"] / opts["K"])

    def keep(mask: np.ndarray, name: str) -> None:
        nonlocal opts
        opts = opts[mask].reset_index(drop=True)
        log.append((name, len(opts)))

    keep(opts["dte"].between(s.dte_min, s.dte_max).to_numpy(), "dte_range")
    keep(opts["fwd_moneyness"].between(s.fwd_moneyness_min, s.fwd_moneyness_max).to_numpy(), "fwd_moneyness")
    keep((opts["rel_spread"] <= s.max_rel_spread).to_numpy(), "rel_spread")

    S, K, tau, r, q = (opts[c].to_numpy(np.float64) for c in ("S", "K", "tau", "r", "q"))
    call = opts["is_call"].to_numpy()
    fwd_disc, disc = S * np.exp(-q * tau), K * np.exp(-r * tau)
    lower = np.where(call, np.maximum(fwd_disc - disc, 0), np.maximum(disc - fwd_disc, 0))
    upper = np.where(call, fwd_disc, disc)
    mid = opts["mid"].to_numpy()
    keep((mid > lower) & (mid < upper), "no_arbitrage")

    opts["iv_mkt"] = implied_vol(opts["mid"], opts["S"], opts["K"], opts["tau"], opts["r"], opts["q"], opts["is_call"])
    keep(np.isfinite(opts["iv_mkt"].to_numpy()), "iv_solvable")
    opts["H"] = trading_horizon(opts["dte"].to_numpy())

    panel = opts.drop(columns=["rel_spread"]).assign(rel_spread=(opts["ask"] - opts["bid"]) / opts["mid"])
    log_df = pd.DataFrame(log, columns=["step", "rows_remaining"])
    log_df["removed"] = -log_df["rows_remaining"].diff().fillna(0).astype(int)
    out = cfg.output_dir
    panel.to_pickle(out / "panel.pkl")
    log_df.to_csv(out / "prepare_log.csv", index=False)
    fw.to_csv(out / "implied_forwards.csv.gz", index=False)
    print(f"[prepare] panel: {len(panel):,} options, {panel['quote_date'].nunique()} days "
          f"({panel['quote_date'].min().date()} to {panel['quote_date'].max().date()})")
    print(log_df.to_string(index=False))
    return panel, log_df
