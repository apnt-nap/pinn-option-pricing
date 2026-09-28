r"""Historical and realised volatility.

Trailing realised volatility over $n$ trading days (zero-mean convention, Section 10.2):

$$
RV^{(n)}_t = \sqrt{\frac{252}{n}\sum_{j=0}^{n-1} r_{t-j}^2}.
$$

Forward realised variance over the $H$ days after $t$, the target for forecast evaluation:

$$
RV^2_{t,H} = \frac{252}{H}\sum_{h=1}^{H} r_{t+h}^2 .
$$

Squared close-to-close returns are a noisy proxy; a range-based estimator needs daily
OHLC data, which the cleaned files do not carry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_rv(returns: pd.Series, window: int) -> pd.Series:
    """Annualised trailing realised volatility $RV^{(n)}_t$ (uses returns up to and including t)."""
    return np.sqrt(252.0 * (returns**2).rolling(window, min_periods=window).mean())


def forward_realized_var(returns: pd.Series, H: int) -> pd.Series:
    """Annualised realised variance over days $t+1, \\dots, t+H$ (NaN where not yet observed)."""
    r2 = returns.to_numpy() ** 2
    c = np.concatenate([[0.0], np.cumsum(r2)])
    n = len(r2)
    out = np.full(n, np.nan)
    idx = np.arange(n - H)
    out[idx] = (c[idx + H + 1] - c[idx + 1]) * 252.0 / H
    return pd.Series(out, index=returns.index)


def historical_vol_table(returns: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Columns ``hv{n}``: trailing annualised volatility for each window."""
    return pd.DataFrame({f"hv{n}": trailing_rv(returns, n) for n in windows})
