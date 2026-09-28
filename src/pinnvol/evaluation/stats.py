r"""Diebold-Mariano test on daily-aggregated pricing losses (Section 14).

Contracts quoted on the same day are strongly correlated, so the loss differential is
averaged within each day before testing. For treatments $A$, $B$ and the $N_t$ contracts
on day $t$:

$$
d_t = \frac{1}{N_t}\sum_{i\in t}\left[L(e_{A,i}) - L(e_{B,i})\right],\qquad
DM = \frac{\bar d}{\sqrt{\widehat{LRV}(d_t)/T}} \;\overset{a}{\sim}\; \mathcal N(0,1),
$$

with the Newey-West (Bartlett kernel) long-run variance. A negative statistic means $A$
has the lower loss. The Harvey-Leybourne-Newbold small-sample correction is applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_lrv(d: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of a series with Bartlett weights."""
    d = d - d.mean()
    T = len(d)
    lrv = d @ d / T
    for k in range(1, min(lags, T - 1) + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * (d[k:] @ d[:-k]) / T
    return float(lrv)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, dates: np.ndarray,
                    lags: int | None = None) -> dict[str, float]:
    """DM test of equal expected loss on daily-averaged differentials.

    Args:
        loss_a, loss_b: per-contract losses of the two treatments (same contracts).
        dates: quote date of each contract.
        lags: Newey-West lags; default $\\lfloor 4(T/100)^{2/9} \\rfloor$.
    """
    df = pd.DataFrame({"d": np.asarray(loss_a) - np.asarray(loss_b), "t": dates}).dropna()
    d = df.groupby("t")["d"].mean().sort_index().to_numpy()
    T = len(d)
    if T < 10:
        return {"dm": np.nan, "p_value": np.nan, "mean_diff": float(d.mean()) if T else np.nan, "days": T}
    h = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    lrv = newey_west_lrv(d, h)
    dm = d.mean() / np.sqrt(lrv / T) if lrv > 0 else np.nan
    # Harvey, Leybourne and Newbold (1997) correction, compared with Student t(T-1).
    corr = np.sqrt((T + 1 - 2 * (h + 1) + (h + 1) * h / T) / T) if h + 1 < T else 1.0
    dm_hln = dm * corr
    p = 2 * stats.t.sf(abs(dm_hln), df=T - 1)
    return {"dm": float(dm_hln), "p_value": float(p), "mean_diff": float(d.mean()), "days": T, "lags": h}
