r"""Synthetic data in the exact schema of the cleaned OptionsDX files.

Used to test the whole pipeline without the licensed data. The generator

1. simulates SPX log returns from a GJR-GARCH(1,1) with Student-t shocks;
2. builds a Treasury curve and writes it as FRED-format CSVs;
3. quotes European options on third-Friday expiries (and optionally weekly Fridays)
   with an implied-volatility smile around a risk-neutral ATM level
   $\sigma_{ATM} = c_{VRP}\sqrt{\mathbb E_t[\bar\sigma^2_{t,T}]}$, so implied volatility
   carries a variance risk premium over the physical GARCH forecast;
4. prices them with Black-Scholes-Merton, adds mid-price noise and a bid-ask spread,
   and applies the same filters as ``clean_spx.py``.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ..pricing.black_scholes import bs_delta, bs_price
from .loader import OPTION_COLUMNS
from .rates import RateCurve

FRED_SERIES = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0}

# (date, short rate in %) knots, roughly the 2012-2023 US path
_RATE_KNOTS = [("2012-01-01", 0.05), ("2015-12-01", 0.2), ("2016-12-31", 0.5), ("2018-12-31", 2.4),
               ("2019-12-31", 1.55), ("2020-03-31", 0.05), ("2021-12-31", 0.05), ("2022-12-31", 4.3),
               ("2023-12-31", 5.4)]


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def simulate_returns(dates: pd.DatetimeIndex, rng: np.random.Generator,
                     omega: float = 2e-6, alpha: float = 0.03, gamma: float = 0.15, beta: float = 0.86,
                     nu: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    """GJR-GARCH(1,1)-t daily returns and their conditional variances."""
    n = len(dates)
    persistence = alpha + gamma / 2 + beta
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1 - persistence)
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu)
    for t in range(n):
        if t > 0:
            e = r[t - 1]
            h[t] = omega + (alpha + gamma * (e < 0)) * e**2 + beta * h[t - 1]
        r[t] = 0.0003 + np.sqrt(h[t]) * z[t]
    return r, h


def _garch_avg_var(h_next: float, H: int, omega: float, persistence: float) -> float:
    """Average GJR-GARCH variance over the next H days (daily units)."""
    vbar = omega / (1 - persistence)
    return vbar + (h_next - vbar) * (1 - persistence**H) / ((1 - persistence) * H)


def make_synthetic_dataset(out_dir: str | Path, rates_dir: str | Path, start: str = "2016-01-01",
                           end: str = "2021-12-31", strike_step: float = 0.025, weeklies: bool = False,
                           seed: int = 0, vrp: float = 1.12, noise: float = 0.01) -> None:
    """Write ``options/``, ``spx_underlying_daily.csv`` and FRED rate CSVs."""
    rng = np.random.default_rng(seed)
    out_dir, rates_dir = Path(out_dir), Path(rates_dir)
    (out_dir / "options").mkdir(parents=True, exist_ok=True)
    rates_dir.mkdir(parents=True, exist_ok=True)

    # Two years of burn-in history before `start` so GARCH/LSTM have data.
    dates = pd.bdate_range(pd.Timestamp(start), end)
    omega, alpha, gamma, beta = 2e-6, 0.03, 0.15, 0.86
    r_ret, h = simulate_returns(dates, rng, omega, alpha, gamma, beta)
    S = 2000.0 * np.exp(np.cumsum(r_ret))
    und = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "S": np.round(S, 2)})
    und["log_return"] = np.log(und["S"]).diff()
    und.to_csv(out_dir / "spx_underlying_daily.csv", index=False, float_format="%.8f")

    # Treasury curve: short rate path plus a mild term premium, FRED format (percent).
    cal = pd.date_range(dates[0] - pd.Timedelta(days=30), dates[-1], freq="D")
    knots = pd.Series({pd.Timestamp(d): v for d, v in _RATE_KNOTS})
    short = np.interp(cal.asi8, knots.index.asi8, knots.to_numpy())
    for name, mat in FRED_SERIES.items():
        y = np.maximum(short + 0.25 * mat + rng.normal(0, 0.01, len(cal)), 0.0)
        pd.DataFrame({"observation_date": cal.strftime("%Y-%m-%d"), name: np.round(y, 2)}).to_csv(
            rates_dir / f"{name}.csv", index=False)
    curve = RateCurve.from_fred(rates_dir, FRED_SERIES)

    persistence = alpha + gamma / 2 + beta
    day_curve = curve.curve.reindex(dates, method="ffill").to_numpy()
    rows: dict[str, list[pd.DataFrame]] = {}
    for i, d in enumerate(dates):
        s = S[i]
        h_next = omega + (alpha + gamma * (r_ret[i] < 0)) * r_ret[i] ** 2 + beta * h[i]
        expiries = []
        for k in range(14):
            m = (d.month - 1 + k) % 12 + 1
            y = d.year + (d.month - 1 + k) // 12
            expiries.append(third_friday(y, m))
        if weeklies:
            expiries += list(pd.date_range(d + pd.Timedelta(days=7), periods=8, freq="W-FRI"))
        expiries = sorted(set(e for e in expiries if 7 <= (e - d).days <= 365))
        for e in expiries:
            dte = (e - d).days
            tf = e.day >= 15 and e.day <= 21 and e.weekday() == 4
            tau = (dte - (6.5 / 24 if tf else 0.0)) / 365.0
            H = max(1, round(dte * 252 / 365))
            atm = vrp * np.sqrt(252 * _garch_avg_var(h_next, H, omega, persistence))
            n_k = int(0.2 / strike_step)
            K = np.round(s * (1 + strike_step * np.arange(-n_k, n_k + 1)) / 5) * 5
            r = float(np.interp(tau, curve.maturities, day_curve[i]))
            q = 0.016 + 0.002 * np.sin(i / 200)
            F = s * np.exp((r - q) * tau)
            k = np.log(K / F)
            scale = 1 / np.sqrt(max(tau, 1 / 12))
            iv = np.maximum(atm + (-0.12 * k + 0.35 * k**2) * scale, 0.05)
            for is_call in (True, False):
                mid = bs_price(s, K, tau, r, q, iv, is_call) * (1 + rng.normal(0, noise, len(K)))
                spread = np.minimum(0.02 + 0.5 * np.exp(-mid / 3), 0.6) * mid
                bid, ask = np.round(mid - spread / 2, 2), np.round(mid + spread / 2, 2)
                mid = 0.5 * (bid + ask)
                df = pd.DataFrame({
                    "quote_date": d.strftime("%Y-%m-%d"), "expire_date": e.strftime("%Y-%m-%d"),
                    "option_type": "C" if is_call else "P", "S": round(s, 2), "K": K, "dte": dte,
                    "tau": dte / 365.0, "moneyness": K / s, "log_moneyness": np.log(K / s),
                    "bid": bid, "ask": ask, "mid": mid, "rel_spread": (ask - bid) / np.where(mid > 0, mid, 1),
                    "last": np.round(mid, 2), "volume": rng.poisson(50, len(K)), "iv_vendor": iv,
                    "delta_vendor": bs_delta(s, K, tau, r, q, iv, is_call), "is_third_friday": int(tf),
                })
                keep = (df["bid"] > 0) & (df["ask"] > df["bid"]) & (df["mid"] >= 0.375) & \
                       (df["moneyness"].between(0.8, 1.2))
                rows.setdefault(d.strftime("%Y%m"), []).append(df[keep])

    for month, parts in rows.items():
        df = pd.concat(parts, ignore_index=True)[OPTION_COLUMNS]
        with gzip.open(out_dir / "options" / f"spx_clean_{month}.csv.gz", "wt") as f:
            df.to_csv(f, index=False, float_format="%.6f")
    n = sum(len(p) for parts in rows.values() for p in parts)
    print(f"[synthetic] wrote {len(rows)} monthly files, {n:,} options, {len(dates)} days to {out_dir}")
