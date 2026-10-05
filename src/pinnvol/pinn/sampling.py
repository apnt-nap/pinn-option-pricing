r"""Training points for the PINN, drawn with a scrambled Sobol sequence.

* Interior collocation points $(x, \tau, r, q, \sigma)$ fill the whole domain; a fraction
  (``center_fraction``) is placed in the pricing region $|x| \le 0.35$, and a fraction
  (``kink_fraction``) is concentrated near the payoff kink, $|x| \le 0.1$ and small
  $\tau$ (drawn as $\tau = 0.1\,U^2$), where the solution has the largest curvature.
* Terminal points sit on $\tau = 0$; half of them lie in $|x| \le 0.2$ around the kink.
* Boundary points sit on $x = x_{min}$ and $x = x_{max}$.
"""
from __future__ import annotations

import torch

from .model import INPUTS


def _sobol(n: int, dim: int, seed: int) -> torch.Tensor:
    return torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(n).double()


def _scale(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * u


def sample_interior(domain: dict, n: int, seed: int, kink_fraction: float = 0.25,
                    dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu",
                    center_fraction: float = 0.0, center_x: float = 0.35) -> dict[str, torch.Tensor]:
    U = _sobol(n, len(INPUTS), seed)
    pts = {k: _scale(U[:, i], *domain[k]) for i, k in enumerate(INPUTS)}
    n_c = int(n * center_fraction)
    if n_c:
        # Points in the pricing region |x| <= center_x, drawn from the tail of the Sobol set.
        pts["x"][-n_c:] = _scale(U[-n_c:, 0], -center_x, center_x)
    n_k = int(n * kink_fraction)
    if n_k:
        g = torch.Generator().manual_seed(seed)
        tau_hi = min(0.1, domain["tau"][1])
        pts["x"][:n_k] = (torch.rand(n_k, generator=g, dtype=torch.float64) * 2 - 1) * 0.1
        pts["tau"][:n_k] = domain["tau"][0] + tau_hi * torch.rand(n_k, generator=g, dtype=torch.float64) ** 2
    return {k: v.to(dtype=dtype, device=device) for k, v in pts.items()}


def sample_terminal(domain: dict, n: int, seed: int, dtype: torch.dtype = torch.float32,
                    device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    pts = sample_interior(domain, n, seed, kink_fraction=0.0, dtype=dtype, device=device)
    pts["tau"] = torch.full_like(pts["tau"], domain["tau"][0])
    # Half of the terminal points resolve the payoff kink at x = 0.
    half = n // 2
    g = torch.Generator().manual_seed(seed + 17)
    pts["x"][:half] = ((torch.rand(half, generator=g, dtype=torch.float64) * 2 - 1) * 0.2).to(pts["x"])
    return pts


def sample_boundary(domain: dict, n: int, seed: int, dtype: torch.dtype = torch.float32,
                    device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    pts = sample_interior(domain, n, seed, kink_fraction=0.0, dtype=dtype, device=device)
    half = n // 2
    pts["x"][:half] = domain["x"][0]
    pts["x"][half:] = domain["x"][1]
    return pts
