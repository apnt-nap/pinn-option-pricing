r"""Crank-Nicolson finite-difference benchmark for the Black-Scholes PDE.

Solves the scaled forward equation used by the PINN,

$$
\frac{\partial u}{\partial \tau}
= \tfrac12\sigma^2 u_{xx} + \left(r - q - \tfrac12\sigma^2\right)u_x - r u,
\qquad u(x, 0) = \max(e^x - 1, 0),
$$

on $x \in [x_{min}, x_{max}]$ with Dirichlet far-field conditions
$u(x_{min}, \tau) = 0$ and $u(x_{max}, \tau) = e^{x_{max} - q\tau} - e^{-r\tau}$.
The first two steps are fully implicit (Rannacher start-up) to damp the oscillations
that the payoff kink otherwise causes in Crank-Nicolson. Puts use put-call parity.

This is an independent numerical solution of the same PDE (Section 12 of the
proposal); it is slow compared with the closed form, so it is run on a subsample.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded


def cn_call_u(x: float, tau: float, r: float, q: float, sigma: float,
              nx: int = 400, nt: int = 200, width: float | None = None) -> float:
    """Scaled call price $u(x, \\tau)$ from a Crank-Nicolson solve."""
    if tau <= 0:
        return max(np.exp(x) - 1.0, 0.0)
    # Grid half-width: at least 6 standard deviations, and covers x.
    half = width if width is not None else max(6.0 * sigma * np.sqrt(tau), 0.5)
    half = max(half, abs(x) + 0.1)
    grid = np.linspace(-half, half, nx + 1)
    h = grid[1] - grid[0]
    dt = tau / nt
    a = 0.5 * sigma**2 / h**2
    b = (r - q - 0.5 * sigma**2) / (2 * h)
    # L u_i = lo * u_{i-1} + di * u_i + up * u_{i+1}
    lo, di, up = a - b, -2 * a - r, a + b

    u = np.maximum(np.exp(grid) - 1.0, 0.0)
    n_in = nx - 1

    def step(u_old: NDArray[np.float64], t_new: float, theta: float) -> NDArray[np.float64]:
        left, right = 0.0, np.exp(grid[-1] - q * t_new) - np.exp(-r * t_new)
        ab = np.zeros((3, n_in))
        ab[0, 1:] = -theta * dt * up
        ab[1, :] = 1 - theta * dt * di
        ab[2, :-1] = -theta * dt * lo
        inner = u_old[1:-1]
        rhs = inner + (1 - theta) * dt * (lo * u_old[:-2] + di * inner + up * u_old[2:])
        rhs[0] += theta * dt * lo * left
        rhs[-1] += theta * dt * up * right
        out = np.empty_like(u_old)
        out[1:-1] = solve_banded((1, 1), ab, rhs)
        out[0], out[-1] = left, right
        return out

    t = 0.0
    for n in range(nt):
        t += dt
        u = step(u, t, 1.0 if n < 2 else 0.5)
    return float(np.interp(x, grid, u))


def cn_price(S: NDArray, K: NDArray, tau: NDArray, r: NDArray, q: NDArray, sigma: NDArray,
             is_call: NDArray, nx: int = 400, nt: int = 200) -> NDArray[np.float64]:
    """Crank-Nicolson prices for a batch of options (loops over options)."""
    out = np.empty(len(S))
    for i in range(len(S)):
        x = np.log(S[i] / K[i])
        c = cn_call_u(x, tau[i], r[i], q[i], sigma[i], nx=nx, nt=nt)
        if not is_call[i]:
            c = c - np.exp(x - q[i] * tau[i]) + np.exp(-r[i] * tau[i])
        out[i] = K[i] * c
    return out
