import numpy as np
import torch

from pinnvol.config import load_config
from pinnvol.pinn import BlackScholesPINN, train_mode_p
from pinnvol.pinn.loss import pde_residual
from pinnvol.pricing import bs_call_u
from pinnvol.pricing.black_scholes import bs_call_u_torch


class ExactBS(BlackScholesPINN):
    """The closed form plugged into the PINN interface: its PDE residual must vanish."""

    def forward(self, x, tau, r, q, sigma):
        return bs_call_u_torch(x, tau, r, q, sigma)


def _domain():
    return {k: tuple(v) for k, v in load_config().raw["pinn"]["domain"].items()}


def test_residual_of_exact_solution_is_zero():
    m = ExactBS(_domain()).double()
    g = torch.Generator().manual_seed(0)
    n = 500
    p = {"x": torch.rand(n, generator=g, dtype=torch.float64) * 0.6 - 0.3,
         "tau": 0.05 + 0.4 * torch.rand(n, generator=g, dtype=torch.float64),
         "r": 0.05 * torch.rand(n, generator=g, dtype=torch.float64),
         "q": 0.03 * torch.rand(n, generator=g, dtype=torch.float64),
         "sigma": 0.1 + 0.5 * torch.rand(n, generator=g, dtype=torch.float64)}
    assert pde_residual(m, p).abs().max().item() < 1e-8


def test_put_call_parity_is_exact():
    m = BlackScholesPINN(_domain())
    n = 64
    rng = np.random.default_rng(0)
    a = dict(S=np.full(n, 4000.0), K=rng.uniform(3300, 4700, n), tau=rng.uniform(0.02, 0.5, n),
             r=rng.uniform(0, 0.05, n), q=rng.uniform(0, 0.03, n), sigma=rng.uniform(0.1, 0.6, n))
    c = m.price(**a, is_call=np.ones(n, bool))
    p = m.price(**a, is_call=np.zeros(n, bool))
    rhs = a["S"] * np.exp(-a["q"] * a["tau"]) - a["K"] * np.exp(-a["r"] * a["tau"])
    np.testing.assert_allclose(c - p, rhs, atol=1e-3)


def test_short_training_reduces_error():
    """A brief Mode P run must already beat an untrained network by a wide margin."""
    torch.set_num_threads(2)
    cfg = load_config().raw["pinn"]
    cfg["mode_p"].update(adam_steps=600, lbfgs_steps=50, n_collocation=1024, n_terminal=256,
                         n_boundary=128, n_validation=2000)
    model, hist, val = train_mode_p(cfg, verbose=False)
    untrained = BlackScholesPINN(_domain())
    rng = np.random.default_rng(1)
    x, tau = rng.uniform(-0.15, 0.15, 2000), rng.uniform(0.05, 0.5, 2000)
    r, q, s = rng.uniform(0, 0.05, 2000), rng.uniform(0, 0.03, 2000), rng.uniform(0.1, 0.5, 2000)
    true = bs_call_u(x, tau, r, q, s)
    err0 = np.abs(untrained.price(np.exp(x), np.ones(2000), tau, r, q, s, np.ones(2000, bool)) - true).mean()
    assert val["mae_u"] < 0.2 * err0
    assert val["mae_u"] < 0.02
    assert hist.loss[-1] < hist.loss[0]
