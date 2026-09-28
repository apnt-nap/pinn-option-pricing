r"""Direct multi-horizon HAR-RV baseline (Corsi, 2009).

For each grid horizon $H$, regress forward realised variance on daily, weekly and
monthly realised variance:

$$
RV^2_{t,H} = \beta_0 + \beta_d RV^{(1)2}_t + \beta_w RV^{(5)2}_t + \beta_m RV^{(22)2}_t + \varepsilon_{t,H},
$$

with $RV^{(n)2}_t = \frac{252}{n}\sum_{j<n} r^2_{t-j}$. Coefficients are estimated by OLS
on an expanding window at each refit date, using only pairs whose target window
$t+1..t+H$ ended before the refit date (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .garch import refit_dates
from .realized import forward_realized_var


def _features(returns: pd.Series) -> pd.DataFrame:
    r2 = 252.0 * returns**2
    return pd.DataFrame({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()})


def har_walk_forward(returns: pd.Series, start: str, horizons: list[int], refit: str = "M",
                     min_obs: int = 750) -> pd.DataFrame:
    """Annualised variance forecasts, one column per horizon, for each date from ``start``."""
    r = returns.dropna()
    X = _features(r)
    Xa = np.column_stack([np.ones(len(X)), X.to_numpy()])
    targets = {H: forward_realized_var(r, H).to_numpy() for H in horizons}
    pos = np.arange(len(r))
    refits = refit_dates(r.index, start, refit)
    out = []
    for i, d0 in enumerate(refits):
        d1 = refits[i + 1] if i + 1 < len(refits) else r.index[-1] + pd.Timedelta(days=1)
        n0 = int(np.searchsorted(r.index, d0))  # first index at/after d0
        if n0 < min_obs:
            raise ValueError(f"HAR: only {n0} returns before {d0.date()}; need {min_obs}.")
        block_idx = pos[(r.index >= d0) & (r.index < d1)]
        pred = {}
        for H in horizons:
            ok = (pos + H < n0) & np.isfinite(Xa).all(axis=1) & np.isfinite(targets[H])
            beta, *_ = np.linalg.lstsq(Xa[ok], targets[H][ok], rcond=None)
            floor = 0.05 * np.nanmean(targets[H][ok])
            pred[H] = np.maximum(Xa[block_idx] @ beta, floor)
        out.append(pd.DataFrame(pred, index=r.index[block_idx]))
    return pd.concat(out)
