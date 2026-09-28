r"""Lagged implied-volatility treatments (Section 10.3).

Same-contract, same-day implied volatility is the target price in disguise,
$\sigma^{imp}_{t,i} = BS^{-1}(V_{market,t,i})$, so the IV treatments use the surface of the
**previous available quote date** $t^- < t$:

* **C1, primary** (``iv_atm``): lagged ATM term structure, one volatility per maturity,
  like GARCH and LSTM: $\hat\sigma_{t,i} = IV_{t^-}(k = 0, \tau_i)$.
* **C2, reference** (``iv_surface``): lagged smile surface,
  $\hat\sigma_{t,i} = IV_{t^-}(k_i, \tau_i)$ with $k_i = \ln(K_i/F_{t,T_i})$.

Construction on each date and expiry, from out-of-the-money quotes (puts for $k<0$,
calls for $k \ge 0$), which are the liquid side of the smile:

* ATM level: IV interpolated linearly in $k$ at $k = 0$ from the nearest quotes on
  each side (within ``atm_band``);
* smile: weighted quadratic $IV(k) = a + bk + ck^2$ fitted on $|k| \le$ ``smile_band``,
  evaluated with flat extrapolation beyond the fitted strikes.

Across maturities both are interpolated linearly in total implied variance
$w = \sigma^2\tau$, with flat volatility beyond the shortest and longest expiry.

The same-day versions (``iv_atm_same_day``, and the contract's own IV) are reported
only as reference bounds, never as treatments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _otm(panel: pd.DataFrame) -> pd.DataFrame:
    k = panel["k"].to_numpy()
    otm = (panel["is_call"].to_numpy() & (k >= 0)) | (~panel["is_call"].to_numpy() & (k < 0))
    return panel.loc[otm & np.isfinite(panel["iv_mkt"].to_numpy()), ["quote_date", "expire_date", "tau", "k", "iv_mkt"]]


def atm_term_structure(panel: pd.DataFrame, atm_band: float = 0.10) -> pd.DataFrame:
    """ATM implied volatility per (quote_date, expire_date).

    ``panel`` needs ``quote_date, expire_date, tau, k, is_call, iv_mkt``.
    """
    q = _otm(panel)
    q = q[q["k"].abs() <= atm_band].sort_values(["quote_date", "expire_date", "k"])
    rows = []
    for (d, e), g in q.groupby(["quote_date", "expire_date"], sort=False):
        k, iv = g["k"].to_numpy(), g["iv_mkt"].to_numpy()
        if (k < 0).any() and (k >= 0).any():
            atm = np.interp(0.0, k, iv)
        elif np.abs(k).min() <= 0.02:
            atm = iv[np.argmin(np.abs(k))]
        else:
            continue
        rows.append((d, e, g["tau"].iat[0], atm))
    return pd.DataFrame(rows, columns=["quote_date", "expire_date", "tau", "atm_iv"])


def smile_fits(panel: pd.DataFrame, band: float = 0.25, min_points: int = 5) -> pd.DataFrame:
    """Quadratic smile $IV(k) = a + bk + ck^2$ per (quote_date, expire_date)."""
    q = _otm(panel)
    q = q[q["k"].abs() <= band]
    rows = []
    for (d, e), g in q.groupby(["quote_date", "expire_date"], sort=False):
        if len(g) < min_points:
            continue
        k, iv = g["k"].to_numpy(), g["iv_mkt"].to_numpy()
        w = np.exp(-0.5 * (k / 0.1) ** 2) + 0.1  # emphasise the centre of the smile
        c, b, a = np.polyfit(k, iv, 2, w=np.sqrt(w))
        rows.append((d, e, g["tau"].iat[0], a, b, c, k.min(), k.max()))
    return pd.DataFrame(rows, columns=["quote_date", "expire_date", "tau", "a", "b", "c", "kmin", "kmax"])


def _interp_total_var(taus: np.ndarray, ivs: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Interpolate volatility across expiries in total variance, flat vol at the ends.

    ``taus`` sorted (m,), ``ivs`` shape (n, m) (per-target IV at each expiry), ``tau`` (n,).
    """
    if len(taus) == 1:
        return ivs[:, 0]
    w = ivs**2 * taus[None, :]
    j = np.clip(np.searchsorted(taus, tau) - 1, 0, len(taus) - 2)
    n = np.arange(len(tau))
    lam = (tau - taus[j]) / (taus[j + 1] - taus[j])
    wi = (1 - lam) * w[n, j] + lam * w[n, j + 1]
    out = np.sqrt(np.maximum(wi, 1e-12) / np.maximum(tau, 1e-12))
    out = np.where(tau <= taus[0], ivs[:, 0], out)
    return np.where(tau >= taus[-1], ivs[:, -1], out)


def atm_vol_at(g: pd.DataFrame | None, tau: np.ndarray) -> np.ndarray:
    """ATM vol at maturities ``tau`` from one date's term structure ``g`` (sorted by tau)."""
    if g is None or g.empty:
        return np.full(len(tau), np.nan)
    taus, iv = g["tau"].to_numpy(), g["atm_iv"].to_numpy()
    return _interp_total_var(taus, np.broadcast_to(iv, (len(tau), len(taus))), np.asarray(tau))


def surface_vol_at(g: pd.DataFrame | None, k: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Smile-surface vol at (k, tau) from one date's smile fits ``g`` (sorted by tau)."""
    if g is None or g.empty:
        return np.full(len(tau), np.nan)
    kk = np.clip(np.asarray(k)[:, None], g["kmin"].to_numpy()[None, :], g["kmax"].to_numpy()[None, :])
    iv = g["a"].to_numpy()[None, :] + g["b"].to_numpy()[None, :] * kk + g["c"].to_numpy()[None, :] * kk**2
    iv = np.maximum(iv, 0.01)
    return _interp_total_var(g["tau"].to_numpy(), iv, np.asarray(tau))


def attach_iv_treatments(panel: pd.DataFrame, ts: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Add ``sigma_iv_atm``, ``sigma_iv_surface`` (lagged) and ``sigma_iv_atm_same_day``."""
    ts_by = {d: g.sort_values("tau") for d, g in ts.groupby("quote_date")}
    fit_by = {d: g.sort_values("tau") for d, g in fits.groupby("quote_date")}
    dates = np.sort(panel["quote_date"].unique())
    prev = dict(zip(dates[1:], dates[:-1]))
    out = {c: np.full(len(panel), np.nan) for c in ("sigma_iv_atm", "sigma_iv_surface", "sigma_iv_atm_same_day")}
    for d, idx in panel.groupby("quote_date").indices.items():
        tau = panel["tau"].to_numpy()[idx]
        k = panel["k"].to_numpy()[idx]
        out["sigma_iv_atm_same_day"][idx] = atm_vol_at(ts_by.get(d), tau)
        if d in prev:
            out["sigma_iv_atm"][idx] = atm_vol_at(ts_by.get(prev[d]), tau)
            out["sigma_iv_surface"][idx] = surface_vol_at(fit_by.get(prev[d]), k, tau)
    for c, v in out.items():
        panel[c] = v
    return panel
