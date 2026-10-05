r"""Closed-form Black-Scholes-Merton prices with a continuous dividend yield.

Two parameterisations are provided:

* original variables, $V(S, K, \tau, r, q, \sigma)$;
* scaled variables used by the PINN, $V = K\,u(x, \tau)$ with $x = \ln(S/K)$:

$$
u_{call}(x,\tau) = e^{x - q\tau}N(d_1) - e^{-r\tau}N(d_2), \qquad
d_{1,2} = \frac{x + (r - q \pm \tfrac12\sigma^2)\tau}{\sigma\sqrt\tau}.
$$

Puts follow from put-call parity, $u_{put} = u_{call} - e^{x - q\tau} + e^{-r\tau}$.
At $\tau = 0$ the functions return the payoff.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

_TINY = 1e-12


def bs_call_u(x: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    """Scaled call price $u = C/K$ as a function of $x = \\ln(S/K)$."""
    x, tau, r, q, sigma = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (x, tau, r, q, sigma)))
    sq = sigma * np.sqrt(np.maximum(tau, 0.0))
    live = sq > _TINY
    safe = np.where(live, sq, 1.0)
    d1 = (x + (r - q + 0.5 * sigma**2) * tau) / safe
    d2 = d1 - safe
    u = np.exp(x - q * tau) * ndtr(d1) - np.exp(-r * tau) * ndtr(d2)
    intrinsic = np.maximum(np.exp(x - q * tau) - np.exp(-r * tau), 0.0)
    return np.where(live, u, intrinsic)


def bs_u(x: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike,
         is_call: ArrayLike) -> NDArray[np.float64]:
    """Scaled price $u = V/K$ for calls (``is_call`` true) or puts."""
    x, tau, r, q = (np.asarray(a, dtype=np.float64) for a in (x, tau, r, q))
    c = bs_call_u(x, tau, r, q, sigma)
    p = c - np.exp(x - q * tau) + np.exp(-r * tau)
    return np.where(np.asarray(is_call, dtype=bool), c, np.maximum(p, 0.0))


def bs_price(S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike,
             sigma: ArrayLike, is_call: ArrayLike) -> NDArray[np.float64]:
    """Black-Scholes-Merton price $V(S, K, \\tau, r, q, \\sigma)$."""
    S, K = np.asarray(S, dtype=np.float64), np.asarray(K, dtype=np.float64)
    return K * bs_u(np.log(S / K), tau, r, q, sigma, is_call)


def bs_vega(S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike,
            sigma: ArrayLike) -> NDArray[np.float64]:
    """Vega $\\partial V/\\partial\\sigma = S e^{-q\\tau}\\varphi(d_1)\\sqrt\\tau$ (same for calls and puts)."""
    S, K, tau, r, q, sigma = (np.asarray(a, dtype=np.float64) for a in (S, K, tau, r, q, sigma))
    sq = np.maximum(sigma * np.sqrt(tau), _TINY)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / sq
    return S * np.exp(-q * tau) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(tau)


def bs_delta(S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike,
             sigma: ArrayLike, is_call: ArrayLike) -> NDArray[np.float64]:
    """Delta $\\partial V/\\partial S$: $e^{-q\\tau}N(d_1)$ for calls, $e^{-q\\tau}(N(d_1)-1)$ for puts."""
    S, K, tau, r, q, sigma = (np.asarray(a, dtype=np.float64) for a in (S, K, tau, r, q, sigma))
    sq = np.maximum(sigma * np.sqrt(tau), _TINY)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / sq
    call = np.exp(-q * tau) * ndtr(d1)
    return np.where(np.asarray(is_call, dtype=bool), call, call - np.exp(-q * tau))


def bs_call_u_torch(x: torch.Tensor, tau: torch.Tensor, r: torch.Tensor, q: torch.Tensor,
                    sigma: torch.Tensor) -> torch.Tensor:
    """Differentiable torch version of :func:`bs_call_u` (used in tests and validation)."""
    sq = sigma * torch.sqrt(torch.clamp(tau, min=0.0))
    safe = torch.clamp(sq, min=_TINY)
    d1 = (x + (r - q + 0.5 * sigma**2) * tau) / safe
    d2 = d1 - safe
    ncdf = torch.special.ndtr
    u = torch.exp(x - q * tau) * ncdf(d1) - torch.exp(-r * tau) * ncdf(d2)
    intrinsic = torch.clamp(torch.exp(x - q * tau) - torch.exp(-r * tau), min=0.0)
    return torch.where(sq > _TINY, u, intrinsic)
