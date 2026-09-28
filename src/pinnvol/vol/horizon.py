r"""Horizon conventions shared by every volatility model.

* Option maturity uses calendar days: $\tau = \text{DTE}/365$.
* Return volatility is annualised with 252 trading days.
* An option with DTE calendar days is matched to a forecast horizon of
  $H = \max(1, \operatorname{round}(252\,\text{DTE}/365))$ trading days.

Models that forecast on a fixed horizon grid (LSTM, HAR) are interpolated linearly in
**total variance** $w(H) = \sigma^2(H)\,H$, with flat volatility outside the grid.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def trading_horizon(dte: ArrayLike) -> NDArray[np.int64]:
    """Calendar days to expiry -> trading-day forecast horizon $H$."""
    return np.maximum(1, np.rint(np.asarray(dte, dtype=np.float64) * 252.0 / 365.0)).astype(np.int64)


def interp_total_variance(grid: ArrayLike, var_grid: NDArray[np.float64], H: ArrayLike) -> NDArray[np.float64]:
    """Interpolate annualised variances given on a horizon grid.

    Args:
        grid: increasing horizons $H_1 < \\dots < H_m$ (trading days or years).
        var_grid: shape ``(n, m)``, annualised variance at each grid horizon, one row per target.
        H: shape ``(n,)``, target horizon per row.

    Returns:
        Annualised variance at ``H`` (shape ``(n,)``).
    """
    g = np.asarray(grid, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    w = var_grid * g[None, :]
    j = np.clip(np.searchsorted(g, H) - 1, 0, len(g) - 2)
    rows = np.arange(n)
    lam = (H - g[j]) / (g[j + 1] - g[j])
    wi = (1 - lam) * w[rows, j] + lam * w[rows, j + 1]
    out = wi / H
    out = np.where(H <= g[0], var_grid[:, 0], out)
    out = np.where(H >= g[-1], var_grid[:, -1], out)
    return out
