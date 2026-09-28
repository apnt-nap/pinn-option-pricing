r"""Error metrics (Section 13).

Pricing, with $e_i = \hat V_i - V_i$ against a reference $V_i$ (the market mid, or the
analytical Black-Scholes price for solver error):

$$
RMSE = \sqrt{\tfrac1N\textstyle\sum_i e_i^2},\quad
MAE = \tfrac1N\textstyle\sum_i |e_i|,\quad
MAPE = \tfrac{100}{N}\textstyle\sum_i \left|\frac{e_i}{V_i}\right|,\quad
R^2 = 1 - \frac{\sum_i e_i^2}{\sum_i (V_i - \bar V)^2}.
$$

MAPE is computed only for $V_i \ge$ ``mape_min_price`` (0.50 index points by default)
because tiny denominators explode it. Against the market we also report IVRMSE (model
price inverted to implied volatility vs. market IV) and the in-spread rate, the share of
$Bid \le \hat V \le Ask$.

Volatility forecasts, against realised variance $RV^2$ (robust losses, Patton 2011):

$$
MSE_{\sigma^2} = \tfrac1N\textstyle\sum_i(\hat\sigma^2_i - RV^2_i)^2,\qquad
QLIKE = \tfrac1N\textstyle\sum_i\left(\frac{RV^2_i}{\hat\sigma^2_i} - \ln\frac{RV^2_i}{\hat\sigma^2_i} - 1\right),
$$

plus the Mincer-Zarnowitz regression $RV^2_i = a + b\,\hat\sigma^2_i + u_i$.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pricing_metrics(pred: ArrayLike, ref: ArrayLike, mape_min_price: float = 0.5) -> dict[str, float]:
    """RMSE, MAE, MAPE (on ``ref >= mape_min_price``), R^2 and mean error (bias)."""
    pred, ref = np.asarray(pred, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[ok], ref[ok]
    e = pred - ref
    m = ref >= mape_min_price
    sst = np.sum((ref - ref.mean()) ** 2)
    return {
        "n": int(len(e)),
        "rmse": float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
        "mae": float(np.mean(np.abs(e))) if len(e) else np.nan,
        "mape": float(100 * np.mean(np.abs(e[m] / ref[m]))) if m.any() else np.nan,
        "r2": float(1 - np.sum(e**2) / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(e)) if len(e) else np.nan,
    }


def market_extras(pred: ArrayLike, bid: ArrayLike, ask: ArrayLike, iv_pred: ArrayLike | None = None,
                  iv_mkt: ArrayLike | None = None) -> dict[str, float]:
    """In-spread rate and IVRMSE (implied-volatility RMSE, in vol points x 100)."""
    pred, bid, ask = (np.asarray(a, dtype=np.float64) for a in (pred, bid, ask))
    ok = np.isfinite(pred)
    out = {"in_spread": float(np.mean((pred[ok] >= bid[ok]) & (pred[ok] <= ask[ok])))}
    if iv_pred is not None and iv_mkt is not None:
        a, b = np.asarray(iv_pred), np.asarray(iv_mkt)
        m = np.isfinite(a) & np.isfinite(b)
        out["ivrmse"] = float(100 * np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan
        out["iv_invertible"] = float(m.mean())
    return out


def vol_forecast_metrics(var_hat: ArrayLike, rv2: ArrayLike) -> dict[str, float]:
    """MSE on variance, QLIKE, RMSE on volatility, and Mincer-Zarnowitz (a, b, R^2)."""
    f, y = np.asarray(var_hat, dtype=np.float64), np.asarray(rv2, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)
    f, y = f[ok], y[ok]
    if len(f) < 3:
        return {"n": int(len(f))}
    ratio = y / f
    X = np.column_stack([np.ones_like(f), f])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - (a + b * f)
    return {
        "n": int(len(f)),
        "mse_var": float(np.mean((f - y) ** 2)),
        "qlike": float(np.mean(ratio - np.log(ratio) - 1)),
        "rmse_vol": float(np.sqrt(np.mean((np.sqrt(f) - np.sqrt(y)) ** 2))),
        "mean_forecast_vol": float(np.mean(np.sqrt(f))),
        "mean_realized_vol": float(np.mean(np.sqrt(y))),
        "mz_a": float(a),
        "mz_b": float(b),
        "mz_r2": float(1 - resid.var() / y.var()) if y.var() > 0 else np.nan,
    }
