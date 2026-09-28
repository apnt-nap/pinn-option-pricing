r"""Stage 5: result tables.

Writes to the run folder:

* ``metrics_market.csv``: every engine x volatility input against the market mid
  (RMSE, MAE, MAPE, $R^2$, bias, in-spread rate, IVRMSE);
* ``metrics_vs_bs.csv``: every engine against analytical Black-Scholes with the same inputs
  (solver error: PINN, Crank-Nicolson, and the data-driven models);
* ``metrics_stratified.csv``: the same metrics by maturity, moneyness ($K/F$), option type
  and market regime;
* ``metrics_mode_h_seeds.csv``: Mode H metrics per seed (seed dispersion);
* ``dm_tests.csv``: pairwise Diebold-Mariano tests between volatility inputs per engine;
* ``report.md``: the headline tables in Markdown.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.metrics import market_extras, pricing_metrics
from ..evaluation.stats import diebold_mariano
from ..evaluation.stratify import add_buckets
from ..pricing.implied_vol import implied_vol

_COL = re.compile(r"^V_(?P<engine>bs|pinn_p|cn|ffnn|pinn_h[0-9.e-]+)_(?P<treatment>.+?)(?P<seed>_s\d+)?$")


def _engine_columns(df: pd.DataFrame) -> list[tuple[str, str, str | None, str]]:
    out = []
    for c in df.columns:
        m = _COL.match(c)
        if m:
            out.append((m["engine"], m["treatment"], m["seed"][2:] if m["seed"] else None, c))
    return out


def evaluate(cfg: Config, pred: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Compute and save all result tables from ``predictions.pkl``."""
    out = cfg.output_dir
    if pred is None:
        pred = pd.read_pickle(out / "predictions.pkl")
    ev = cfg.evaluation
    pred = add_buckets(pred, list(ev.maturity_buckets), list(ev.moneyness_buckets), dict(ev.regimes), ev.calm_atm_iv)
    cols = _engine_columns(pred)
    mid, bid, ask = pred["mid"].to_numpy(), pred["bid"].to_numpy(), pred["ask"].to_numpy()
    args = {c: pred[c].to_numpy() for c in ("S", "K", "tau", "r", "q", "is_call")}

    market, vs_bs, seeds = [], [], []
    for engine, t, seed, c in cols:
        V = pred[c].to_numpy()
        ok = np.isfinite(V)
        if not ok.any():
            continue
        row = {"engine": engine, "treatment": t}
        if seed is not None:
            seeds.append({**row, "seed": int(seed), **pricing_metrics(V[ok], mid[ok], ev.mape_min_price)})
            continue
        iv_hat = implied_vol(V[ok], *(a[ok] for a in args.values()))
        market.append({**row, **pricing_metrics(V[ok], mid[ok], ev.mape_min_price),
                       **market_extras(V[ok], bid[ok], ask[ok], iv_hat, pred["iv_mkt"].to_numpy()[ok])})
        if engine != "bs":
            ref = pred[f"V_bs_{t}"].to_numpy()
            vs_bs.append({**row, **pricing_metrics(V[ok], ref[ok], ev.mape_min_price)})

    tables = {
        "metrics_market": pd.DataFrame(market),
        "metrics_vs_bs": pd.DataFrame(vs_bs),
        "metrics_mode_h_seeds": pd.DataFrame(seeds),
        "metrics_stratified": _stratified(pred, cols, ev.mape_min_price),
        "dm_tests": _dm_tests(pred, cols, ev.dm_loss),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "report.md").write_text(_markdown(cfg, pred, tables))
    print((out / "report.md").read_text())
    return tables


def _stratified(pred: pd.DataFrame, cols: list, mape_min: float) -> pd.DataFrame:
    rows = []
    mid = pred["mid"].to_numpy()
    for dim in ("maturity_bucket", "moneyness_bucket", "option_type", "regime"):
        groups = pred.groupby(dim, observed=True).indices
        for engine, t, seed, c in cols:
            if seed is not None or engine == "cn":
                continue
            V = pred[c].to_numpy()
            for g, idx in groups.items():
                ok = idx[np.isfinite(V[idx])]
                if len(ok):
                    rows.append({"dimension": dim, "bucket": str(g), "engine": engine, "treatment": t,
                                 **pricing_metrics(V[ok], mid[ok], mape_min)})
    return pd.DataFrame(rows)


def _dm_tests(pred: pd.DataFrame, cols: list, loss: str) -> pd.DataFrame:
    rows = []
    mid = pred["mid"].to_numpy()
    dates = pred["quote_date"].to_numpy()
    L = (lambda e: e**2) if loss == "squared" else np.abs
    by_engine: dict[str, dict[str, str]] = {}
    for engine, t, seed, c in cols:
        if seed is None and engine != "cn":
            by_engine.setdefault(engine, {})[t] = c
    for engine, tc in by_engine.items():
        for a, b in itertools.combinations(sorted(tc), 2):
            Va, Vb = pred[tc[a]].to_numpy(), pred[tc[b]].to_numpy()
            ok = np.isfinite(Va) & np.isfinite(Vb)
            res = diebold_mariano(L(Va[ok] - mid[ok]), L(Vb[ok] - mid[ok]), dates[ok])
            rows.append({"engine": engine, "treatment_a": a, "treatment_b": b, "loss": loss, **res})
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_(none)_\n"
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.4g}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n"


def _markdown(cfg: Config, pred: pd.DataFrame, t: dict[str, pd.DataFrame]) -> str:
    m = t["metrics_market"]
    s = [f"# Results: {cfg.output_dir}\n",
         f"Test sample: {len(pred):,} options, {pred['quote_date'].nunique()} quote dates "
         f"({pred['quote_date'].min().date()} to {pred['quote_date'].max().date()}); "
         "every volatility input is evaluated on the same contracts.\n",
         "## Pricing error against the market mid (index points)\n",
         _fmt(m.sort_values(["engine", "rmse"]) if not m.empty else m,
              ["engine", "treatment", "n", "rmse", "mae", "mape", "r2", "in_spread", "ivrmse"]),
         "\n## Solver error against analytical Black-Scholes (same inputs)\n",
         _fmt(t["metrics_vs_bs"], ["engine", "treatment", "n", "rmse", "mae", "mape"]),
         "\n## Diebold-Mariano tests (negative: treatment_a has lower loss)\n",
         _fmt(t["dm_tests"], ["engine", "treatment_a", "treatment_b", "dm", "p_value", "days"])]
    return "\n".join(s)
