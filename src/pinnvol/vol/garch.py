r"""GARCH(1,1) and GJR-GARCH(1,1) by Gaussian (quasi-)maximum likelihood.

Model for daily log returns $r_t$ (Section 10.1):

$$
r_t = \mu + \epsilon_t,\quad \epsilon_t = \sigma_t z_t,\quad
\sigma_t^2 = \omega + (\alpha + \gamma\,\mathbb 1[\epsilon_{t-1} < 0])\,\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2,
$$

with $\gamma = 0$ for plain GARCH. With persistence $\phi = \alpha + \gamma/2 + \beta$ and
long-run variance $\bar\sigma^2 = \omega/(1-\phi)$, the $h$-step forecast is
$\sigma^2_{t+h|t} = \bar\sigma^2 + \phi^{h-1}(\sigma^2_{t+1|t} - \bar\sigma^2)$, and the
maturity-consistent annualised volatility for an $H$-day option is

$$
\hat\sigma_{t,H} = \sqrt{\frac{252}{H}\sum_{h=1}^{H}\sigma^2_{t+h|t}}
= \sqrt{\frac{252}{H}\left[H\bar\sigma^2 + (\sigma^2_{t+1|t} - \bar\sigma^2)\frac{1-\phi^H}{1-\phi}\right]}.
$$

Walk-forward: parameters are re-estimated on an expanding window at each refit date
(monthly by default) using returns strictly before that date, then held fixed while the
variance is filtered forward day by day. The forecast made at the close of day $t$ uses
returns through $t$ only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

_SCALE = 100.0  # estimate on percent returns for numerical conditioning


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def long_run_var(self) -> float:
        return self.omega / (1.0 - self.persistence)


def _filter(eps: NDArray[np.float64], omega: float, alpha: float, gamma: float, beta: float,
            h0: float) -> NDArray[np.float64]:
    """Conditional variances $\\sigma^2_t$ for t = 0..n (the last is the one-step forecast)."""
    n = len(eps)
    h = np.empty(n + 1)
    h[0] = h0
    for t in range(n):
        e = eps[t]
        h[t + 1] = omega + (alpha + gamma * (e < 0.0)) * e * e + beta * h[t]
    return h


def fit_garch(returns: NDArray[np.float64], kind: str = "garch") -> GarchParams:
    """Gaussian QML estimate of GARCH(1,1) (``kind="garch"``) or GJR-GARCH(1,1) (``"gjr"``)."""
    y = np.asarray(returns, dtype=np.float64) * _SCALE
    var0 = y.var()
    gjr = kind == "gjr"

    def nll(p: NDArray[np.float64]) -> float:
        mu, omega, alpha, beta = p[0], p[1], p[2], p[3]
        gamma = p[4] if gjr else 0.0
        eps = y - mu
        h = _filter(eps, omega, alpha, gamma, beta, var0)[:-1]
        if np.any(h <= 0):
            return 1e10
        return 0.5 * float(np.sum(np.log(h) + eps**2 / h))

    x0 = [y.mean(), var0 * 0.05, 0.05, 0.90] + ([0.05] if gjr else [])
    bounds = [(-1.0, 1.0), (1e-6, 10 * var0), (0.0, 0.5), (0.0, 0.999)] + ([(0.0, 0.5)] if gjr else [])
    cons = [{"type": "ineq", "fun": lambda p: 0.9995 - (p[2] + p[3] + (0.5 * p[4] if gjr else 0.0))}]
    res = minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    p = res.x
    return GarchParams(mu=p[0] / _SCALE, omega=p[1] / _SCALE**2, alpha=p[2],
                       gamma=p[4] if gjr else 0.0, beta=p[3])


def refit_dates(dates: pd.DatetimeIndex, start: str, freq: str) -> list[pd.Timestamp]:
    """First trading date of each period (``"M"`` month, ``"Q"`` quarter) on or after ``start``."""
    d = dates[dates >= pd.Timestamp(start)]
    periods = d.to_period(freq)
    first = ~pd.Series(periods).duplicated().to_numpy()
    return list(d[first])


def garch_walk_forward(returns: pd.Series, start: str, kind: str = "garch", refit: str = "M",
                       min_obs: int = 750) -> pd.DataFrame:
    """Daily GARCH state for every date from ``start``.

    Returns a frame indexed by date with columns ``h_next`` (one-step daily variance
    $\\sigma^2_{t+1|t}$), ``vbar`` (daily long-run variance), ``phi`` (persistence) and
    the parameters in force. Use :func:`garch_horizon_vol` to turn it into $\\hat\\sigma_{t,H}$.
    """
    r = returns.dropna()
    dates = r.index
    refits = refit_dates(dates, start, refit)
    out = []
    for i, d0 in enumerate(refits):
        d1 = refits[i + 1] if i + 1 < len(refits) else dates[-1] + pd.Timedelta(days=1)
        est = r[r.index < d0]
        if len(est) < min_obs:
            raise ValueError(f"Only {len(est)} returns before {d0.date()}; need {min_obs}. "
                             "Move volatility.forecast_start later or supply paths.extra_prices_csv.")
        p = fit_garch(est.to_numpy(), kind)
        upto = r[r.index < d1]
        eps = upto.to_numpy() - p.mu
        h = _filter(eps, p.omega, p.alpha, p.gamma, p.beta, float(np.var(est.to_numpy())))
        # h[t+1] is the forecast for day t+1 made at the close of day t.
        block = pd.DataFrame({"h_next": h[1:]}, index=upto.index)
        block = block[block.index >= d0]
        block["vbar"] = p.long_run_var
        block["phi"] = p.persistence
        for k, v in vars(p).items():
            block[k] = v
        out.append(block)
    return pd.concat(out)


def garch_horizon_var(h_next: NDArray, vbar: NDArray, phi: NDArray, H: NDArray) -> NDArray[np.float64]:
    """Annualised average variance $\\frac{252}{H}\\sum_{h=1}^H \\sigma^2_{t+h|t}$."""
    H = np.asarray(H, dtype=np.float64)
    phi = np.minimum(np.asarray(phi, dtype=np.float64), 0.99999)
    total = H * vbar + (h_next - vbar) * (1.0 - phi**H) / (1.0 - phi)
    return 252.0 * total / H
