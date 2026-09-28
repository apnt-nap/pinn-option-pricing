r"""Risk-free rates from FRED Treasury constant-maturity yields.

The cleaned option files carry no interest rate, so $r$ comes from FRED
(DGS1MO, DGS3MO, DGS6MO, DGS1 by default). CMT yields are quoted as bond-equivalent
(semi-annual) percentages; they are converted to continuous compounding,

$$
r^{cont} = 2\ln\!\left(1 + \frac{y}{200}\right),
$$

forward-filled over missing days, and interpolated linearly in maturity to each
option's $\tau$ (flat outside the quoted range): $r_i = r(t_i, \tau_i)$.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def download_fred(series: list[str], out_dir: str | Path) -> list[Path]:
    """Download FRED series as CSV (no API key needed). Returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in series:
        with urllib.request.urlopen(FRED_URL.format(series=s), timeout=60) as resp:
            text = resp.read().decode()
        p = out / f"{s}.csv"
        p.write_text(text)
        paths.append(p)
        print(f"[rates] wrote {p} ({text.count(chr(10))} lines)")
    return paths


def _read_fred_csv(path: Path, series: str) -> pd.Series:
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    val = pd.to_numeric(df[series], errors="coerce")  # FRED writes "." for missing
    return pd.Series(val.to_numpy(), index=pd.to_datetime(df[date_col]), name=series)


class RateCurve:
    """Daily zero-rate curve, continuous compounding, interpolated in maturity."""

    def __init__(self, curve: pd.DataFrame, maturities: np.ndarray) -> None:
        self.curve = curve  # index: calendar dates, columns: maturities (years)
        self.maturities = maturities

    @classmethod
    def from_fred(cls, rates_dir: str | Path, series: dict[str, float]) -> "RateCurve":
        cols, mats = [], []
        for name, mat in sorted(series.items(), key=lambda kv: kv[1]):
            p = Path(rates_dir) / f"{name}.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            cols.append(_read_fred_csv(p, name))
            mats.append(mat)
        df = pd.concat(cols, axis=1)
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D")).ffill()
        cont = 2.0 * np.log1p(df.to_numpy() / 200.0)
        return cls(pd.DataFrame(cont, index=df.index, columns=mats), np.asarray(mats))

    @classmethod
    def constant(cls, rate: float, start: str = "1990-01-01", end: str = "2030-12-31") -> "RateCurve":
        idx = pd.date_range(start, end, freq="D")
        return cls(pd.DataFrame({1.0: np.full(len(idx), rate)}, index=idx), np.array([1.0]))

    def rate(self, dates: pd.Series | np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Rate for each (date, tau) pair, linear in maturity, flat beyond the ends."""
        dates = pd.to_datetime(np.asarray(dates))
        rows = self.curve.reindex(dates, method="ffill").to_numpy()
        if np.isnan(rows).all(axis=1).any():
            bad = dates[np.isnan(rows).all(axis=1)][:3]
            raise ValueError(f"No rate available for dates such as {list(bad)}")
        m = self.maturities
        tau = np.asarray(tau, dtype=np.float64)
        if len(m) == 1:
            return rows[:, 0]
        j = np.clip(np.searchsorted(m, tau) - 1, 0, len(m) - 2)
        w = np.clip((tau - m[j]) / (m[j + 1] - m[j]), 0.0, 1.0)
        n = np.arange(len(tau))
        return (1 - w) * rows[n, j] + w * rows[n, j + 1]


def load_rate_curve(rates_dir: str | Path, series: dict[str, float], fallback_rate: float | None) -> RateCurve:
    """FRED curve if the CSVs exist, else a constant ``fallback_rate`` (with a warning)."""
    try:
        return RateCurve.from_fred(rates_dir, series)
    except FileNotFoundError as e:
        if fallback_rate is None:
            raise FileNotFoundError(
                f"{e}. Run `pinnvol fetch-rates` first, or set rates.fallback_rate in the config.") from e
        print(f"[rates] warning: {e.filename} missing, using constant r = {fallback_rate}")
        return RateCurve.constant(fallback_rate)
