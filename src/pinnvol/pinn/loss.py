r"""PINN loss terms in scaled variables $u(x, \tau)$, $x = \ln(S/K)$, $V = K u$.

PDE residual (forward in time-to-maturity $\tau$):

$$
f_\theta = \frac{\partial u_\theta}{\partial \tau}
- \tfrac12\sigma^2 \frac{\partial^2 u_\theta}{\partial x^2}
- \left(r - q - \tfrac12\sigma^2\right)\frac{\partial u_\theta}{\partial x} + r u_\theta,
\qquad \mathcal L_{PDE} = \frac{1}{N_f}\sum_j f_\theta(z_j)^2 .
$$

Terminal condition (call network): $u(x, 0) = \max(e^x - 1, 0)$.

Far-field boundaries: either the linearity condition $V_{SS} = 0$, i.e. $u_{xx} - u_x = 0$ (default),
or the Dirichlet values $u(x_{min}, \tau) \approx 0$ and $u(x_{max}, \tau) = e^{x_{max} - q\tau} - e^{-r\tau}$.

Market data (Mode H), with $u_{market} = V_{market}/K$ and puts priced by parity:
$\mathcal L_{data} = \frac{1}{N_d}\sum_i (u_\theta(z_i) - u_{market,i})^2$.

The PDE, terminal and boundary errors are multiplied by $1/(e^{-r\tau}(1 + e^{x+(r-q)\tau}))$
so that deep in-the-money points (where $u \sim e^x$) do not dominate.

Total: $\mathcal L = w_f\mathcal L_{PDE} + w_T\mathcal L_{terminal} + w_B\mathcal L_{boundary} + w_D\mathcal L_{data}$.
"""
from __future__ import annotations

import torch

from .model import BlackScholesPINN


def scale_weight(p: dict[str, torch.Tensor]) -> torch.Tensor:
    """$1 / (e^{-r\\tau}(1 + e^{x + (r-q)\\tau}))$: puts every error on the O(1) scale of the solution.

    The call price grows like $e^x$ in the money, so unweighted squared errors on a wide
    $x$-domain would be dominated by deep in-the-money points.
    """
    y = p["x"] + (p["r"] - p["q"]) * p["tau"]
    return torch.exp(p["r"] * p["tau"]) / (1.0 + torch.exp(y))


def pde_residual(model: BlackScholesPINN, p: dict[str, torch.Tensor]) -> torch.Tensor:
    """Residual $f_\\theta$ at collocation points ``p``."""
    x = p["x"].detach().requires_grad_(True)
    tau = p["tau"].detach().requires_grad_(True)
    r, q, s = p["r"], p["q"], p["sigma"]
    u = model(x, tau, r, q, s)
    u_x, u_tau = torch.autograd.grad(u.sum(), (x, tau), create_graph=True)
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_tau - 0.5 * s**2 * u_xx - (r - q - 0.5 * s**2) * u_x + r * u


def pde_loss(model: BlackScholesPINN, p: dict[str, torch.Tensor], v0: float | None = None) -> torch.Tensor:
    """Mean squared residual.

    With ``v0`` set, each residual is weighted by $\rho = \min(1, \sigma\sqrt\tau / v_0)$. Near
    expiry the exact solution has $u_{xx} \sim 1/(\sigma\sqrt\tau)$, so unweighted residuals there
    are unbounded and swamp the rest of the domain; the weight keeps the PDE enforced everywhere
    while bounding that contribution.
    """
    f = pde_residual(model, p) * scale_weight(p)
    if v0:
        f = f * torch.clamp(p["sigma"] * torch.sqrt(p["tau"]) / v0, max=1.0)
    return f.pow(2).mean()


def terminal_loss(model: BlackScholesPINN, p: dict[str, torch.Tensor]) -> torch.Tensor:
    u = model(p["x"], p["tau"], p["r"], p["q"], p["sigma"])
    payoff = torch.clamp(torch.exp(p["x"]) - 1.0, min=0.0)
    return ((u - payoff) * scale_weight(p)).pow(2).mean()


def boundary_target(p: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.clamp(torch.exp(p["x"] - p["q"] * p["tau"]) - torch.exp(-p["r"] * p["tau"]), min=0.0)


def boundary_loss(model: BlackScholesPINN, p: dict[str, torch.Tensor], kind: str = "linear") -> torch.Tensor:
    """Far-field loss at $x = x_{min}, x_{max}$.

    ``"dirichlet"``: $u = \max(e^{x - q\tau} - e^{-r\tau}, 0)$, accurate only when the boundary
    is many standard deviations $\sigma\sqrt\tau$ away from the money.

    ``"linear"``: $V_{SS} = 0$, i.e. $u_{xx} - u_x = 0$ in log-moneyness (the price is linear in
    $S$ far from the strike). It holds in both the deep in- and out-of-the-money limits without
    fixing the value, so it stays accurate on a narrower domain (Windcliff, Forsyth and Vetzal, 2004).
    """
    if kind == "dirichlet":
        u = model(p["x"], p["tau"], p["r"], p["q"], p["sigma"])
        return ((u - boundary_target(p)) * scale_weight(p)).pow(2).mean()
    x = p["x"].detach().requires_grad_(True)
    u = model(x, p["tau"], p["r"], p["q"], p["sigma"])
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return ((u_xx - u_x) * scale_weight(p)).pow(2).mean()


def data_loss(model: BlackScholesPINN, d: dict[str, torch.Tensor]) -> torch.Tensor:
    """Squared error against market mids in scaled units; ``d`` also holds ``is_call`` and ``u_mkt``."""
    u = model.u(d["x"], d["tau"], d["r"], d["q"], d["sigma"], d["is_call"])
    return (u - d["u_mkt"]).pow(2).mean()
