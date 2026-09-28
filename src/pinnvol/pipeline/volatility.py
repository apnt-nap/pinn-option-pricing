r"""Stage 2: volatility forecasts and their attachment to each option.

For every option $i$ quoted on date $t$ with forecast horizon $H_i$ trading days, this
stage adds one column per volatility input (annualised):

| Column | Input | Information used |
|---|---|---|
| ``sigma_garch`` | Treatment A, GARCH(1,1), exact horizon aggregation | returns through $t$ |
| ``sigma_lstm`` | Treatment B, LSTM, total-variance interpolation | returns through $t$ |
| ``sigma_iv_atm`` | Treatment C1, lagged ATM IV term structure | quotes on $t^-$ |
| ``sigma_iv_surface`` | Treatment C2, lagged smile surface | quotes on $t^-$ |
| ``sigma_hv21``, ``sigma_hv63`` | historical volatility baselines | returns through $t$ |
| ``sigma_har`` | HAR-RV baseline | returns through $t$ |
| ``sigma_gjr`` | GJR-GARCH robustness (if listed in treatments) | returns through $t$ |
| ``sigma_iv_atm_same_day`` | reference bound | quotes on $t$ |
| ``sigma_iv_contract_same_day`` | reference bound (the contract's own IV) | the target itself |

It also evaluates the forecasts against forward realised variance on the horizon grid
(``vol_forecast_metrics.csv``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import load_underlying
from ..evaluation.metrics import vol_forecast_metrics
from ..pinn.trainer import resolve_device
from ..vol.garch import garch_horizon_var, garch_walk_forward
from ..vol.har import har_walk_forward
from ..vol.horizon import interp_total_variance
from ..vol.implied import atm_term_structure, atm_vol_at, attach_iv_treatments, smile_fits
from ..vol.lstm import LSTMConfig, lstm_walk_forward
from ..vol.realized import forward_realized_var, historical_vol_table


def _grid_lookup(table: pd.DataFrame, dates: pd.Series, H: np.ndarray) -> np.ndarray:
    grid = np.asarray(table.columns, dtype=np.float64)
    rows = table.reindex(pd.DatetimeIndex(dates)).to_numpy()
    out = np.full(len(H), np.nan)
    ok = np.isfinite(rows).all(axis=1)
    out[ok] = np.sqrt(interp_total_variance(grid, rows[ok], H[ok]))
    return out


def _garch_lookup(state: pd.DataFrame, dates: pd.Series, H: np.ndarray) -> np.ndarray:
    st = state.reindex(pd.DatetimeIndex(dates))
    return np.sqrt(garch_horizon_var(st["h_next"].to_numpy(), st["vbar"].to_numpy(), st["phi"].to_numpy(), H))


def build_volatility(cfg: Config, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit every volatility model walk-forward, attach inputs to the panel, save ``panel_vol.pkl``."""
    out = cfg.output_dir
    vdir = out / "vol"
    vdir.mkdir(exist_ok=True)
    v = cfg.volatility
    if panel is None:
        panel = pd.read_pickle(out / "panel.pkl")
    und = load_underlying(cfg.paths.data_dir, cfg.paths.extra_prices_csv)
    ret = und["log_return"]
    start = v.forecast_start
    horizons = list(v.horizons)
    treatments = set(cfg.treatments)

    print("[vol] GARCH walk-forward")
    g = v.garch
    garch = garch_walk_forward(ret, start, g.kind, g.refit, g.min_obs)
    garch.to_pickle(vdir / "garch_state.pkl")
    gjr = None
    if "gjr" in treatments:
        print("[vol] GJR-GARCH walk-forward")
        gjr = garch_walk_forward(ret, start, "gjr", g.refit, g.min_obs)
        gjr.to_pickle(vdir / "gjr_state.pkl")

    print("[vol] HAR walk-forward")
    har = har_walk_forward(ret, start, horizons, v.har.refit, v.har.min_obs)
    har.to_pickle(vdir / "har_var.pkl")

    lstm_var = None
    if "lstm" in treatments:
        print("[vol] LSTM walk-forward")
        lc = dict(v.lstm)
        lc["seeds"] = tuple(lc["seeds"])
        lstm_var, lstm_disp = lstm_walk_forward(ret, start, horizons, LSTMConfig(**lc), resolve_device(cfg.device))
        lstm_var.to_pickle(vdir / "lstm_var.pkl")
        lstm_disp.to_pickle(vdir / "lstm_seed_dispersion.pkl")

    hv = historical_vol_table(ret, list(v.hv_windows))
    hv.to_pickle(vdir / "hv.pkl")

    # Lagged implied volatility needs the day before the first forecast date as well.
    first = pd.Timestamp(start)
    prior = panel.loc[panel["quote_date"] < first, "quote_date"]
    lo = prior.max() if len(prior) else first
    sub = panel[panel["quote_date"] >= lo].copy()
    print("[vol] implied-volatility term structures and smiles")
    ts = atm_term_structure(sub, v.iv.atm_band)
    fits = smile_fits(sub, v.iv.smile_band, v.iv.min_smile_points)
    ts.to_pickle(vdir / "iv_atm_term_structure.pkl")
    fits.to_pickle(vdir / "iv_smile_fits.pkl")
    sub = attach_iv_treatments(sub, ts, fits)
    sub = sub[sub["quote_date"] >= first].reset_index(drop=True)

    dates, H = sub["quote_date"], sub["H"].to_numpy()
    sub["sigma_garch"] = _garch_lookup(garch, dates, H)
    if gjr is not None:
        sub["sigma_gjr"] = _garch_lookup(gjr, dates, H)
    sub["sigma_har"] = _grid_lookup(har, dates, H)
    if lstm_var is not None:
        sub["sigma_lstm"] = _grid_lookup(lstm_var, dates, H)
    for n in v.hv_windows:
        sub[f"sigma_hv{n}"] = hv[f"hv{n}"].reindex(pd.DatetimeIndex(dates)).to_numpy()
    sub["sigma_iv_contract_same_day"] = sub["iv_mkt"]
    ts_by = {d: grp.sort_values("tau") for d, grp in ts.groupby("quote_date")}
    atm30 = {d: atm_vol_at(ts_by.get(d), np.array([30 / 365]))[0] for d in sub["quote_date"].unique()}
    sub["atm30_same_day"] = sub["quote_date"].map(atm30)

    sub.to_pickle(out / "panel_vol.pkl")
    cols = [c for c in sub.columns if c.startswith("sigma_")]
    print("[vol] coverage (share of options with a forecast):")
    print(sub[cols].notna().mean().round(4).to_string())

    vm = evaluate_vol_forecasts(cfg, ret, garch, har, lstm_var, hv, ts, gjr)
    vm.to_csv(out / "vol_forecast_metrics.csv", index=False)
    return sub


def evaluate_vol_forecasts(cfg: Config, ret: pd.Series, garch: pd.DataFrame, har: pd.DataFrame,
                           lstm_var: pd.DataFrame | None, hv: pd.DataFrame, ts: pd.DataFrame,
                           gjr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast accuracy per model, horizon and split, against forward realised variance."""
    horizons = list(cfg.volatility.horizons)
    dates = garch.index
    ts_by = {d: grp.sort_values("tau") for d, grp in ts.groupby("quote_date")}
    rows = []
    splits = {"validation": (cfg.volatility.forecast_start, cfg.period.valid_end),
              "test": (pd.Timestamp(cfg.period.valid_end) + pd.Timedelta(days=1), cfg.period.test_end)}
    for H in horizons:
        rv2 = forward_realized_var(ret, H).reindex(dates).to_numpy()
        Hs = np.full(len(dates), H)
        models = {
            "garch": garch_horizon_var(garch["h_next"].to_numpy(), garch["vbar"].to_numpy(), garch["phi"].to_numpy(), Hs),
            "har": har[H].reindex(dates).to_numpy(),
            "iv_atm": np.array([atm_vol_at(ts_by.get(d), np.array([H / 252]))[0] for d in dates]) ** 2,
        }
        if gjr is not None:
            models["gjr"] = garch_horizon_var(gjr["h_next"].to_numpy(), gjr["vbar"].to_numpy(), gjr["phi"].to_numpy(), Hs)
        if lstm_var is not None:
            models["lstm"] = lstm_var[H].reindex(dates).to_numpy()
        for c in hv.columns:
            models[c] = hv[c].reindex(dates).to_numpy() ** 2
        for split, (a, b) in splits.items():
            m = np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))
            for name, f in models.items():
                rows.append({"model": name, "horizon": H, "split": split, **vol_forecast_metrics(f[m], rv2[m])})
    return pd.DataFrame(rows)
