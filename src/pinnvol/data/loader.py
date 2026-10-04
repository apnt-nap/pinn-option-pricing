"""Readers for the cleaned OptionsDX SPX files written by ``clean_spx.py``.

Expected layout of ``data_dir``::

    data_dir/
        options/spx_clean_YYYYMM.csv.gz   one row per option (call or put)
        spx_underlying_daily.csv          date, S, log_return
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPTION_COLUMNS: list[str] = [
    "quote_date", "expire_date", "option_type", "S", "K", "dte", "tau",
    "moneyness", "log_moneyness", "bid", "ask", "mid", "rel_spread",
    "last", "volume", "iv_vendor", "delta_vendor", "is_third_friday",
]

# Columns the pipeline needs; the rest stay on disk.
_USECOLS = ["quote_date", "expire_date", "option_type", "S", "K", "dte", "bid", "ask", "mid",
            "rel_spread", "volume", "iv_vendor", "is_third_friday"]
_DTYPES = {"option_type": "category", "S": "float64", "K": "float64", "dte": "float32",
           "bid": "float64", "ask": "float64", "mid": "float64", "rel_spread": "float32",
           "volume": "float32", "iv_vendor": "float32", "is_third_friday": "int8"}


def month_files(data_dir: str | Path, start: str, end: str) -> list[Path]:
    """Monthly option files whose month lies in ``[start, end]``."""
    months = pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M")
    folder = Path(data_dir) / "options"
    files = [folder / f"spx_clean_{m.strftime('%Y%m')}.csv.gz" for m in months]
    missing = [f.name for f in files if not f.exists()]
    if len(missing) == len(files):
        raise FileNotFoundError(f"No option files for {start}..{end} in {folder}")
    if missing:
        print(f"[loader] warning: {len(missing)} monthly files missing, e.g. {missing[:3]}")
    return [f for f in files if f.exists()]


def _good_friday(year: int) -> pd.Timestamp:
    """Good Friday (anonymous Gregorian algorithm for Easter Sunday, minus two days)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(year, month, day + 1) - pd.Timedelta(days=2)


def flag_am_monthlies(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive ``is_third_friday`` as "AM-settled standard monthly" and fix ``dte`` to settlement.

    OptionsDX dates the standard monthly by its last trading day, the Thursday before the
    third Friday, until about August 2016, and by the third Friday afterwards. A Thursday
    expiry the day before the third Friday ``F3`` is therefore taken as the monthly when

    * ``F3`` is Good Friday (the exchange moves expiry and AM settlement to Thursday), or
    * no expiry dated ``F3`` is quoted on the same day (the vendor's Thursday convention).
      Settlement is then Friday's open, so ``dte`` gains one day.

    A Thursday expiry quoted alongside an ``F3`` expiry is a PM-settled SPXW daily (2022+)
    and is left alone. ``df`` must hold every expiry of each quote date it contains.
    """
    e = df["expire_date"]
    first = e.dt.to_period("M").dt.start_time
    f3 = first + pd.to_timedelta((4 - first.dt.weekday) % 7 + 14, unit="D")
    years = range(int(e.dt.year.min()), int(e.dt.year.max()) + 1)
    good_friday = f3.isin([_good_friday(y) for y in years])
    pairs = pd.MultiIndex.from_frame(df[["quote_date", "expire_date"]].drop_duplicates())
    f3_listed = pd.MultiIndex.from_arrays([df["quote_date"], f3]).isin(pairs)
    thursday = (e == f3 - pd.Timedelta(days=1)).to_numpy()
    shifted = thursday & ~good_friday.to_numpy() & ~f3_listed
    monthly = (e == f3).to_numpy() | (thursday & (good_friday.to_numpy() | ~f3_listed))
    df["is_third_friday"] = monthly.astype("int8")
    df["dte"] = (df["dte"].to_numpy() + shifted).astype(df["dte"].dtype)
    return df


def load_options(data_dir: str | Path, start: str, end: str, family: str = "all") -> pd.DataFrame:
    """Load cleaned options quoted in ``[start, end]``.

    ``is_third_friday`` is re-derived by :func:`flag_am_monthlies` (the flag in the cleaned
    files misses the Thursday-dated monthlies before September 2016).

    Args:
        data_dir: folder holding ``options/``.
        start, end: ISO dates (inclusive).
        family: ``"monthly"`` keeps AM-settled standard monthlies only; ``"all"`` keeps all.
    """
    frames = []
    for f in month_files(data_dir, start, end):
        df = pd.read_csv(f, usecols=_USECOLS, dtype=_DTYPES, parse_dates=["quote_date", "expire_date"])
        df = flag_am_monthlies(df)
        if family == "monthly":
            df = df[df["is_third_friday"] == 1]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["quote_date"] >= pd.Timestamp(start)) & (out["quote_date"] <= pd.Timestamp(end))]
    out["is_call"] = (out["option_type"] == "C").to_numpy()
    return out.reset_index(drop=True)


def load_underlying(data_dir: str | Path, extra_prices_csv: str | Path | None = None) -> pd.DataFrame:
    """Daily SPX level and log return, indexed by date.

    If ``extra_prices_csv`` (columns ``date``, ``S``) is given, it supplies history
    *before* the first cleaned date, e.g. SPX closes from 2000 for a longer GARCH/LSTM
    estimation sample (Section 7 of the proposal). Overlapping dates use the cleaned file.
    """
    und = pd.read_csv(Path(data_dir) / "spx_underlying_daily.csv", parse_dates=["date"])
    und = und[["date", "S"]]
    if extra_prices_csv:
        extra = pd.read_csv(extra_prices_csv, parse_dates=["date"])[["date", "S"]]
        extra = extra[extra["date"] < und["date"].min()]
        und = pd.concat([extra, und], ignore_index=True)
    und = und.dropna().sort_values("date").drop_duplicates("date").set_index("date")
    und["log_return"] = np.log(und["S"]).diff()
    return und.iloc[1:]
