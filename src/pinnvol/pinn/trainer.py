r"""Training loops: Mode P (physics only), Mode H (hybrid) and the data-only FFNN control.

**Mode P** ($w_D = 0$, Section 11.5): trained once on synthetic collocation points over
the whole input domain, so the same network prices every volatility treatment. Adam with
cosine learning-rate decay and periodic resampling, followed by L-BFGS. Loss weights are
either fixed or set by learning-rate annealing (Wang, Teng and Perdikaris, 2021):

$$
\hat\lambda_i = \frac{\max_\theta |\nabla_\theta \mathcal L_{PDE}|}{\overline{|\nabla_\theta \mathcal L_i|}},
\qquad \lambda_i \leftarrow \alpha\lambda_i + (1-\alpha)\hat\lambda_i .
$$

**Mode H** ($w_D > 0$): starts from the Mode P weights and fine-tunes on market quotes
with $\lambda = w_D / w_f$ fixed; the PDE, terminal and boundary weights are frozen at
their final Mode P values, so every treatment gets the same weighting policy.

**FFNN control**: the same MLP trained on market quotes only (no PDE residual).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..pricing.black_scholes import bs_call_u
from .loss import boundary_loss, data_loss, pde_loss, terminal_loss
from .model import MLP, BlackScholesPINN
from .sampling import sample_boundary, sample_interior, sample_terminal


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainHistory:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    terminal: list[float] = field(default_factory=list)
    boundary: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)
    w_terminal: list[float] = field(default_factory=list)
    w_boundary: list[float] = field(default_factory=list)

    def log(self, step: int, parts: dict[str, float], weights: dict[str, float]) -> None:
        self.step.append(step)
        self.loss.append(parts["total"])
        for k in ("pde", "terminal", "boundary", "data"):
            getattr(self, k).append(parts.get(k, float("nan")))
        self.w_terminal.append(weights["terminal"])
        self.w_boundary.append(weights["boundary"])


def _grad_stats(loss: torch.Tensor, params: list[nn.Parameter]) -> tuple[float, float]:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([gi.abs().flatten() for gi in g if gi is not None])
    return flat.max().item(), flat.mean().item()


def _points(domain: dict, n_f: int, n_t: int, n_b: int, seed: int, kink: float, dtype: torch.dtype,
            device: torch.device, center: float = 0.3) -> tuple[dict, dict, dict]:
    return (sample_interior(domain, n_f, seed, kink, dtype, device, center_fraction=center),
            sample_terminal(domain, n_t, seed + 1, dtype, device),
            sample_boundary(domain, n_b, seed + 2, dtype, device))


def validate_against_bs(model: BlackScholesPINN, n: int = 20000, seed: int = 12345,
                        x_range: tuple[float, float] = (-0.1823, 0.2231),
                        tau_range: tuple[float, float] = (7 / 365, 180 / 365)) -> dict[str, float]:
    """Compare the call network with closed-form Black-Scholes on a held-out random set.

    The default ranges match the evaluation sample: $0.8 \\le K/S \\le 1.2$ and 7-180 DTE.
    Errors are in scaled units ($u = V/K$) and in index points for $K = 4000$.
    """
    rng = np.random.default_rng(seed)
    d = model.domain
    x = rng.uniform(*x_range, n)
    tau = rng.uniform(*tau_range, n)
    r, q, s = (rng.uniform(*d[k], n) for k in ("r", "q", "sigma"))
    u_true = bs_call_u(x, tau, r, q, s)
    u_hat = model.price(np.exp(x), np.ones(n), tau, r, q, s, np.ones(n, bool))
    err = u_hat - u_true
    return {"mae_u": float(np.mean(np.abs(err))), "rmse_u": float(np.sqrt(np.mean(err**2))),
            "max_u": float(np.max(np.abs(err))), "mae_points_K4000": float(4000 * np.mean(np.abs(err))),
            "rmse_points_K4000": float(4000 * np.sqrt(np.mean(err**2)))}


def train_mode_p(cfg: dict, seed: int = 0, device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32, verbose: bool = True) -> tuple[BlackScholesPINN, TrainHistory, dict]:
    """Train the physics-only PINN. ``cfg`` is the ``pinn`` section of the config."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    mp = cfg["mode_p"]
    domain = {k: tuple(v) for k, v in cfg["domain"].items()}
    model = BlackScholesPINN(domain, cfg["hidden"], cfg["layers"], cfg["activation"], cfg.get("features", "similarity")).to(device=device, dtype=dtype)
    params = [p for p in model.parameters()]
    w = {"pde": float(mp["weights"]["pde"]), "terminal": float(mp["weights"]["terminal"]),
         "boundary": float(mp["weights"]["boundary"])}
    v0 = mp.get("residual_v0")
    bkind = mp.get("boundary", "linear")
    opt = torch.optim.Adam(params, lr=mp["adam_lr"])
    steps = int(mp["adam_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))))
    hist = TrainHistory()
    t0 = time.time()
    P = None
    for step in range(steps):
        if step % mp["resample_every"] == 0:
            P = _points(domain, mp["n_collocation"], mp["n_terminal"], mp["n_boundary"], seed * 100003 + step,
                        mp["kink_fraction"], dtype, device)
        pf, pt, pb = P
        lf, lt, lb = pde_loss(model, pf, v0), terminal_loss(model, pt), boundary_loss(model, pb, bkind)
        if mp["weighting"] == "lra" and step % mp["lra_every"] == 0 and step > 0:
            gmax, _ = _grad_stats(lf, params)
            a = mp["lra_alpha"]
            for name, li in (("terminal", lt), ("boundary", lb)):
                _, gmean = _grad_stats(li, params)
                w[name] = a * w[name] + (1 - a) * gmax / max(gmean, 1e-12)
        loss = w["pde"] * lf + w["terminal"] * lt + w["boundary"] * lb
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            hist.log(step, {"total": loss.item(), "pde": lf.item(), "terminal": lt.item(), "boundary": lb.item()}, w)
        if verbose and (step % 1000 == 0 or step == steps - 1):
            print(f"[mode P] adam {step:6d}  loss {loss.item():.3e}  pde {lf.item():.2e}  term {lt.item():.2e}"
                  f"  bnd {lb.item():.2e}  wT {w['terminal']:.1f} wB {w['boundary']:.1f}  {time.time() - t0:.0f}s")

    if mp["lbfgs_steps"] > 0:
        pf, pt, pb = _points(domain, mp["n_collocation"], mp["n_terminal"], mp["n_boundary"], seed * 7 + 99991,
                             mp["kink_fraction"], dtype, device)
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=int(mp["lbfgs_steps"]), history_size=50,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            l = w["pde"] * pde_loss(model, pf, v0) + w["terminal"] * terminal_loss(model, pt) + \
                w["boundary"] * boundary_loss(model, pb, bkind)
            l.backward()
            return l

        lbfgs.step(closure)
        lf, lt, lb = pde_loss(model, pf, v0), terminal_loss(model, pt), boundary_loss(model, pb, bkind)
        loss = w["pde"] * lf + w["terminal"] * lt + w["boundary"] * lb
        hist.log(steps + int(mp["lbfgs_steps"]), {"total": loss.item(), "pde": lf.item(), "terminal": lt.item(),
                                                  "boundary": lb.item()}, w)
        if verbose:
            print(f"[mode P] lbfgs      loss {loss.item():.3e}  pde {lf.item():.2e}  term {lt.item():.2e}"
                  f"  bnd {lb.item():.2e}  {time.time() - t0:.0f}s")

    val = validate_against_bs(model, n=int(mp["n_validation"]), tau_range=(7 / 365, min(180 / 365, domain["tau"][1])))
    val["final_weights"] = dict(w)
    val["train_seconds"] = time.time() - t0
    if verbose:
        print(f"[mode P] validation vs closed form: MAE(u) {val['mae_u']:.2e}, max {val['max_u']:.2e}, "
              f"MAE {val['mae_points_K4000']:.3f} pts at K=4000")
    return model, hist, val


def _data_tensors(d: dict[str, np.ndarray], dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    out = {k: torch.as_tensor(d[k], dtype=dtype, device=device) for k in ("x", "tau", "r", "q", "sigma", "u_mkt")}
    out["is_call"] = torch.as_tensor(d["is_call"], dtype=torch.bool, device=device)
    return out


def finetune_mode_h(base: BlackScholesPINN, data: dict[str, np.ndarray], lam: float, weights: dict[str, float],
                    cfg: dict, seed: int, device: torch.device | None = None) -> BlackScholesPINN:
    """Hybrid PINN: Mode P weights fine-tuned with data loss weight ``lam`` (relative to the PDE)."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = BlackScholesPINN.from_checkpoint(base.checkpoint(), device)
    dtype = next(model.parameters()).dtype
    mh = cfg["mode_h"]
    bkind = cfg["mode_p"].get("boundary", "linear")
    domain = model.domain
    D = _data_tensors(data, dtype, device)
    opt = torch.optim.Adam(model.parameters(), lr=mh["lr"])
    P = None
    for step in range(int(mh["steps"])):
        if step % 250 == 0:
            P = _points(domain, mh["n_collocation"], mh["n_collocation"] // 4, mh["n_collocation"] // 8,
                        seed * 7919 + step, cfg["mode_p"]["kink_fraction"], dtype, device)
        pf, pt, pb = P
        loss = weights["pde"] * pde_loss(model, pf, cfg["mode_p"].get("residual_v0")) + weights["terminal"] * terminal_loss(model, pt) + \
            weights["boundary"] * boundary_loss(model, pb, bkind) + lam * weights["pde"] * data_loss(model, D)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


class FFNNPricer(nn.Module):
    """Data-driven control: $(x, \\tau, r, q, \\sigma, \\mathbb 1_{call}) \\mapsto V/K$, no physics."""

    def __init__(self, domain: dict, hidden: int = 64, layers: int = 5, activation: str = "tanh") -> None:
        super().__init__()
        from .model import INPUTS
        self.register_buffer("lo", torch.tensor([float(domain[k][0]) for k in INPUTS]))
        self.register_buffer("hi", torch.tensor([float(domain[k][1]) for k in INPUTS]))
        self.mlp = MLP(len(INPUTS) + 1, 1, hidden, layers, activation)

    def forward(self, x, tau, r, q, sigma, is_call) -> torch.Tensor:  # type: ignore[no-untyped-def]
        X = torch.stack([x, tau, r, q, sigma], dim=-1)
        z = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        z = torch.cat([z, (2.0 * is_call.to(z.dtype) - 1.0).unsqueeze(-1)], dim=-1)
        return nn.functional.softplus(self.mlp(z).squeeze(-1)) * 0.1

    @torch.no_grad()
    def price(self, S, K, tau, r, q, sigma, is_call) -> np.ndarray:  # type: ignore[no-untyped-def]
        self.eval()
        dt, dev = self.lo.dtype, self.lo.device
        t = [torch.as_tensor(np.asarray(c, dtype=np.float64), dtype=dt, device=dev)
             for c in (np.log(np.asarray(S) / np.asarray(K)), tau, r, q, sigma)]
        u = self.forward(*t, torch.as_tensor(np.asarray(is_call, bool), device=dev))
        return np.asarray(K) * u.double().cpu().numpy()


def train_ffnn(data: dict[str, np.ndarray], domain: dict, hidden: int, layers: int, steps: int, lr: float,
               seed: int, device: torch.device | None = None, batch_size: int = 4096) -> FFNNPricer:
    """Train the FFNN control on market quotes (mean squared error in scaled units)."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = FFNNPricer(domain, hidden, layers).to(device)
    D = _data_tensors(data, torch.float32, device)
    n = len(D["x"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        j = torch.randint(0, n, (min(batch_size, n),), generator=g).to(device)
        pred = model(D["x"][j], D["tau"][j], D["r"][j], D["q"][j], D["sigma"][j], D["is_call"][j])
        loss = (pred - D["u_mkt"][j]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model
