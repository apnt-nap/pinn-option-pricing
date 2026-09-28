r"""LSTM direct multi-horizon volatility forecaster (Section 10.2).

Input at date $t$: the sequence of the last $L$ daily feature vectors
$(r_s, |r_s|, r_s^2, RV^{(21)}_s)$, $s = t-L+1, \dots, t$, standardised with statistics
from the training window only.

Output: log annualised variance for each grid horizon
$H \in \{5, 10, 21, 42, 63, 126\}$, i.e. $\hat\ell_{t,H} = \ln\hat\sigma^2_{t,H}$.

Loss: QLIKE on variance, which is robust to noise in the realised-variance proxy
(Patton, 2011) and is minimised by the conditional expectation of $RV^2$:

$$
\mathcal L_{QLIKE} = \frac{1}{|\Omega|}\sum_{(t,H)\in\Omega}
\left[\frac{RV^2_{t,H}}{\hat\sigma^2_{t,H}} + \ln\hat\sigma^2_{t,H}\right].
$$

Walk-forward: at each refit date $D$ (quarterly by default), a model is trained on all
pairs $(t, H)$ whose target window ended before $D$ ($\Omega$ masks the rest, so no
future return enters training). It then forecasts every date in $[D, D_{next})$. Each refit
trains several seeds; the forecast is the seed average of $\hat\sigma^2$ and the
seed dispersion of $\hat\sigma$ is kept for reporting (Section 16.7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .garch import refit_dates
from .realized import forward_realized_var


@dataclass
class LSTMConfig:
    seq_len: int = 60
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.0
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    val_frac: float = 0.15
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    refit: str = "Q"
    min_obs: int = 750


class VolLSTM(nn.Module):
    """LSTM encoder followed by a linear head giving one log variance per horizon."""

    def __init__(self, n_features: int, n_horizons: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _feature_matrix(returns: pd.Series) -> np.ndarray:
    rv21 = np.sqrt(252.0 * (returns**2).rolling(21, min_periods=1).mean())
    f = np.column_stack([returns, returns.abs(), returns**2, rv21])
    return f.astype(np.float64)


def _qlike(logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    t = torch.where(mask, target, torch.ones_like(target))
    loss = t * torch.exp(-logvar) + logvar
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _windows(F: np.ndarray, ends: np.ndarray, L: int) -> np.ndarray:
    idx = ends[:, None] + np.arange(-L + 1, 1)[None, :]
    return F[idx]


def _train_one(Xtr: np.ndarray, Ytr: np.ndarray, Mtr: np.ndarray, Xva: np.ndarray, Yva: np.ndarray,
               Mva: np.ndarray, cfg: LSTMConfig, seed: int, bias0: np.ndarray, device: torch.device) -> VolLSTM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VolLSTM(Xtr.shape[2], Ytr.shape[1], cfg.hidden, cfg.layers, cfg.dropout).to(device)
    with torch.no_grad():
        model.head.bias.copy_(torch.as_tensor(bias0, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    Xv, Yv, Mv = tt(Xva), tt(Yva), tt(Mva, torch.bool)
    best, best_state, bad = np.inf, None, 0
    for _ in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), cfg.batch_size):
            j = perm[b:b + cfg.batch_size]
            loss = _qlike(model(tt(Xtr[j])), tt(Ytr[j]), tt(Mtr[j], torch.bool))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = _qlike(model(Xv), Yv, Mv).item() if len(Xva) else loss.item()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def lstm_walk_forward(returns: pd.Series, start: str, horizons: list[int], cfg: LSTMConfig,
                      device: torch.device | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward LSTM forecasts.

    Returns:
        ``(var, disp)``: annualised variance forecasts (seed average) and the
        cross-seed standard deviation of the volatility forecast; one column per horizon.
    """
    device = device or torch.device("cpu")
    r = returns.dropna()
    F = _feature_matrix(r)
    Y = np.column_stack([forward_realized_var(r, H).to_numpy() for H in horizons])
    L = cfg.seq_len
    n = len(r)
    pos = np.arange(n)
    refits = refit_dates(r.index, start, cfg.refit)
    var_blocks, disp_blocks = [], []
    for i, d0 in enumerate(refits):
        d1 = refits[i + 1] if i + 1 < len(refits) else r.index[-1] + pd.Timedelta(days=1)
        n0 = int(np.searchsorted(r.index, d0))
        if n0 < max(cfg.min_obs, L + 30):
            raise ValueError(f"LSTM: only {n0} returns before {d0.date()}; need {cfg.min_obs}.")
        # Standardise with the training window only.
        mu, sd = F[:n0].mean(0), F[:n0].std(0) + 1e-12
        Fz = (F - mu) / sd
        ends = pos[L - 1:n0 - 1]  # sample dates inside the training window
        M = (ends[:, None] + np.asarray(horizons)[None, :] < n0) & np.isfinite(Y[ends])
        keep = M.any(axis=1)
        ends, M = ends[keep], M[keep]
        X = _windows(Fz, ends, L)
        Yt = np.nan_to_num(Y[ends], nan=1.0)
        n_val = int(len(ends) * cfg.val_frac)
        split = len(ends) - n_val
        bias0 = np.log(np.nanmean(np.where(M, Y[ends], np.nan), axis=0))

        target_idx = pos[(r.index >= d0) & (r.index < d1)]
        Xp = torch.as_tensor(_windows(Fz, target_idx, L), dtype=torch.float32, device=device)
        preds = []
        for seed in cfg.seeds:
            model = _train_one(X[:split], Yt[:split], M[:split], X[split:], Yt[split:], M[split:],
                               cfg, seed + 1000 * i, bias0, device)
            with torch.no_grad():
                preds.append(torch.exp(model(Xp)).cpu().numpy())
        P = np.stack(preds)  # (seeds, dates, horizons)
        var_blocks.append(pd.DataFrame(P.mean(0), index=r.index[target_idx], columns=horizons))
        disp_blocks.append(pd.DataFrame(np.sqrt(P).std(0), index=r.index[target_idx], columns=horizons))
        if verbose:
            print(f"[lstm] refit {d0.date()}: {len(ends)} samples, forecast {len(target_idx)} days")
    return pd.concat(var_blocks), pd.concat(disp_blocks)
