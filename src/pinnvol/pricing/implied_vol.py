r"""Vectorised Black-Scholes implied volatility.

Solves $V_{BS}(\sigma) = V$ for $\sigma$ by bisection on $[\sigma_{lo}, \sigma_{hi}]$.
The price is strictly increasing in $\sigma$, so bisection always converges; 60
iterations give an interval narrower than $10^{-16}$, which is simpler and more
robust across millions of quotes than Newton-Raphson (whose step $\Delta V/\mathcal V$
blows up where vega vanishes, e.g. deep in- or out-of-the-money short-dated options).

Prices outside the no-arbitrage band
$\big(\max(Se^{-q\tau} - Ke^{-r\tau}, 0),\ Se^{-q\tau}\big)$ for calls (and the parity
equivalent for puts) have no implied volatility and return ``NaN``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .black_scholes import bs_call_u


def implied_vol(V: ArrayLike, S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike,
                is_call: ArrayLike, sigma_lo: float = 1e-4, sigma_hi: float = 5.0,
                n_iter: int = 60) -> NDArray[np.float64]:
    """Implied volatility of option prices ``V`` (vectorised bisection)."""
    V, S, K, tau, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (V, S, K, tau, r, q)))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), V.shape)
    x = np.log(S / K)
    u = V / K
    fwd_disc = np.exp(x - q * tau)
    disc = np.exp(-r * tau)
    # Work in call space: u_call = u_put + e^{x-q tau} - e^{-r tau}.
    u_call = np.where(is_call, u, u + fwd_disc - disc)
    lower = np.maximum(fwd_disc - disc, 0.0)
    valid = (u_call > lower) & (u_call < fwd_disc) & (tau > 0) & np.isfinite(u_call)

    lo = np.full(V.shape, sigma_lo)
    hi = np.full(V.shape, sigma_hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        above = bs_call_u(x, tau, r, q, mid) > u_call
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    iv = 0.5 * (lo + hi)
    # Solutions pinned at the bracket edge are not reliable.
    edge = (iv <= sigma_lo * 1.001) | (iv >= sigma_hi * 0.999)
    return np.where(valid & ~edge, iv, np.nan)
