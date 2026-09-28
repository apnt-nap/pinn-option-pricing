r"""PINN for the scaled Black-Scholes-Merton equation (Section 11, log-moneyness form).

The network approximates the scaled **call** price $u_\theta(x, \tau, r, q, \sigma) \approx C/K$
with $x = \ln(S/K)$ and $\tau = T - t$. Black-Scholes prices are homogeneous of degree one
in $(S, K)$, so this five-input form covers every strike with outputs of order one,
which keeps the loss terms balanced. Puts come from put-call parity,

$$
u_{put} = u_{call} - e^{x - q\tau} + e^{-r\tau},
$$

so a single network prices both option types and parity holds exactly. Inputs are mapped
affinely to $[-1, 1]$ from the training domain inside ``forward`` so that automatic
differentiation returns derivatives with respect to the raw $(x, \tau)$.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

INPUTS = ("x", "tau", "r", "q", "sigma")
_EPS_V = 2e-3  # smooths x / v at expiry

_ACT = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class MLP(nn.Module):
    """Fully connected network with Xavier initialisation."""

    def __init__(self, n_in: int, n_out: int, hidden: int, layers: int, activation: str = "tanh") -> None:
        super().__init__()
        dims = [n_in] + [hidden] * layers
        mods: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods += [lin, _ACT[activation]()]
        out = nn.Linear(dims[-1], n_out)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        mods.append(out)
        self.net = nn.Sequential(*mods)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BlackScholesPINN(nn.Module):
    """$u_\\theta(x, \\tau, r, q, \\sigma)$: scaled call price."""

    def __init__(self, domain: dict[str, tuple[float, float]], hidden: int = 64, layers: int = 5,
                 activation: str = "tanh", features: str = "similarity") -> None:
        super().__init__()
        self.features = features
        self.domain = {k: tuple(map(float, domain[k])) for k in INPUTS}
        lo = torch.tensor([self.domain[k][0] for k in INPUTS])
        hi = torch.tensor([self.domain[k][1] for k in INPUTS])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        n_in = {"raw": len(INPUTS), "similarity": len(INPUTS) + 2, "canonical": 3}[features]
        self.mlp = MLP(n_in, 1, hidden, layers, activation)
        self.arch = {"hidden": hidden, "layers": layers, "activation": activation, "features": features}

    def forward(self, x: torch.Tensor, tau: torch.Tensor, r: torch.Tensor, q: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        """Scaled call price, shape ``(n,)``; all inputs shape ``(n,)``."""
        v = sigma * torch.sqrt(torch.clamp(tau, min=0.0) + 1e-10)
        v_max = self.hi[4] * torch.sqrt(self.hi[1])
        if self.features == "canonical":
            # u = e^{-r tau} w(y, v) with forward log-moneyness y = x + (r - q) tau and total
            # standard deviation v = sigma sqrt(tau): the change of variables that turns the
            # Black-Scholes-Merton PDE into a two-variable diffusion. The PDE residual is still
            # evaluated in the original (x, tau) by automatic differentiation.
            y = x + (r - q) * tau
            z = torch.stack([y / self.hi[0], 2.0 * v / v_max - 1.0,
                             torch.tanh(y / (2.0 * torch.sqrt(v**2 + _EPS_V**2)))], dim=-1)
            # Output scaled by e^{-r tau}(1 + e^y), which bounds the network output to (0, 1)
            # for every y, so a wide y-domain (accurate far-field conditions) keeps O(1) outputs.
            return torch.exp(-r * tau) * (1.0 + torch.exp(y)) * self.mlp(z).squeeze(-1)
        X = torch.stack([x, tau, r, q, sigma], dim=-1)
        z = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        if self.features == "similarity":
            # Total standard deviation v = sigma sqrt(tau) and the standardised forward
            # log-moneyness (x + (r - q) tau) / v: the scales on which the solution varies. Near
            # expiry this resolves the payoff kink, which a smooth network on raw inputs cannot.
            xi = (x + (r - q) * tau) / torch.sqrt(v**2 + _EPS_V**2)
            z = torch.cat([z, (2.0 * v / v_max - 1.0).unsqueeze(-1), torch.tanh(xi / 2.0).unsqueeze(-1)], dim=-1)
        return self.mlp(z).squeeze(-1)

    def u(self, x: torch.Tensor, tau: torch.Tensor, r: torch.Tensor, q: torch.Tensor, sigma: torch.Tensor,
          is_call: torch.Tensor) -> torch.Tensor:
        """Scaled price for calls or puts (puts via parity)."""
        c = self.forward(x, tau, r, q, sigma)
        p = c - torch.exp(x - q * tau) + torch.exp(-r * tau)
        return torch.where(is_call, c, p)

    @torch.no_grad()
    def price(self, S: ArrayLike, K: ArrayLike, tau: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike,
              is_call: ArrayLike, batch_size: int = 262144) -> NDArray[np.float64]:
        """Option prices $V = K\\,u_\\theta$ for numpy inputs, evaluated in batches."""
        self.eval()
        dev = self.lo.device
        dt = self.lo.dtype
        S, K = np.asarray(S, dtype=np.float64), np.asarray(K, dtype=np.float64)
        cols = [np.log(S / K), tau, r, q, sigma]
        cols = [np.asarray(c, dtype=np.float64) for c in cols]
        ic = np.asarray(is_call, dtype=bool)
        out = np.empty(len(S))
        for b in range(0, len(S), batch_size):
            sl = slice(b, b + batch_size)
            t = [torch.as_tensor(c[sl], dtype=dt, device=dev) for c in cols]
            u = self.u(*t, torch.as_tensor(ic[sl].copy(), device=dev))
            out[sl] = u.double().cpu().numpy()
        return K * out

    def clip_inputs(self, tau: NDArray, r: NDArray, q: NDArray, sigma: NDArray) -> tuple[NDArray, ...]:
        """Clip (tau, r, q, sigma) to the training domain (x is left alone)."""
        d = self.domain
        return (np.clip(tau, *d["tau"]), np.clip(r, *d["r"]), np.clip(q, *d["q"]), np.clip(sigma, *d["sigma"]))

    def checkpoint(self) -> dict:
        return {"state_dict": self.state_dict(), "domain": self.domain, "arch": self.arch}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: torch.device | str = "cpu") -> "BlackScholesPINN":
        m = cls(ckpt["domain"], **ckpt["arch"])
        m.load_state_dict(ckpt["state_dict"])
        return m.to(device)
