"""Clean OptionsDX SPX end-of-day option files into a long-format working dataset.

Reads ``dataset/spx-YYYY/spx_eod_YYYYMM.txt`` (read-only) and writes:

* ``data-clean/options/spx_clean_YYYYMM.csv.gz``: one row per option (call or put)
  that survives every filter.
* ``data-clean/spx_underlying_daily.csv``: one SPX level per trading day, for the
  GARCH / LSTM volatility models.
* ``data-clean/cleaning_log.csv``: rows removed by each filter, per month.

Uses only the standard library so it runs without pandas.

Usage (from the folder that holds ``dataset/``, i.e. one level above this repository)::

    python3 code/scripts/clean_spx.py [PROJECT_DIR]

``PROJECT_DIR`` defaults to the current directory.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import gzip
import math
import os
import sys
from collections import Counter
from multiprocessing import Pool
from typing import Iterator

PROJECT_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
RAW_DIR = os.path.join(PROJECT_DIR, "dataset")
OUT_DIR = os.path.join(PROJECT_DIR, "data-clean")

# Filter thresholds.
DTE_MIN, DTE_MAX = 7.0, 365.0            # calendar days
MONEYNESS_MIN, MONEYNESS_MAX = 0.8, 1.2  # K / S
MID_MIN = 0.375                          # minimum mid quote, index points
IV_MIN, IV_MAX = 0.01, 3.0               # vendor implied vol, annualised

# Filter names, in the order they are applied (the log counts first failure).
FILTERS = [
    "closed_day", "missing_quote", "dte_range", "moneyness_range",
    "zero_bid", "crossed_or_locked", "mid_below_min", "iv_invalid",
]

OUT_COLUMNS = [
    "quote_date", "expire_date", "option_type", "S", "K", "dte", "tau",
    "moneyness", "log_moneyness", "bid", "ask", "mid", "rel_spread",
    "last", "volume", "iv_vendor", "delta_vendor", "is_third_friday",
]


# --------------------------------------------------------------------------- #
# NYSE trading calendar
# --------------------------------------------------------------------------- #
def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th ``weekday`` (Mon=0) of a month; n=-1 gives the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = dt.date(year + month // 12, month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: dt.date) -> dt.date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


def nyse_holidays(year: int) -> set[dt.date]:
    """Full-day NYSE closures for ``year`` (valid 2012-2023)."""
    hol = {
        _nth_weekday(year, 1, 0, 3),               # MLK Day
        _nth_weekday(year, 2, 0, 3),               # Presidents' Day
        _easter(year) - dt.timedelta(days=2),      # Good Friday
        _nth_weekday(year, 5, 0, -1),              # Memorial Day
        _observed(dt.date(year, 7, 4)),            # Independence Day
        _nth_weekday(year, 9, 0, 1),               # Labor Day
        _nth_weekday(year, 11, 3, 4),              # Thanksgiving
        _observed(dt.date(year, 12, 25)),          # Christmas
    }
    new_year = dt.date(year, 1, 1)
    if new_year.weekday() != 5:                    # NYSE skips Sat New Year
        hol.add(_observed(new_year))
    if year >= 2022:
        hol.add(_observed(dt.date(year, 6, 19)))   # Juneteenth
    special = {
        dt.date(2012, 10, 29), dt.date(2012, 10, 30),  # Hurricane Sandy
        dt.date(2018, 12, 5),                          # G.H.W. Bush funeral
    }
    return hol | {d for d in special if d.year == year}


def is_trading_day(day: dt.date) -> bool:
    """True if NYSE was open (full or half day) on ``day``."""
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def is_third_friday(day: dt.date) -> bool:
    """Third Friday of the month: standard AM-settled SPX monthly expiry.

    Good-Friday shifts (expiry on the Thursday) are not flagged.
    """
    return day.weekday() == 4 and 15 <= day.day <= 21


# --------------------------------------------------------------------------- #
# Row parsing
# --------------------------------------------------------------------------- #
def _num(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _iter_raw(path: str) -> Iterator[dict[str, str]]:
    """Yield raw rows as dicts keyed by the unbracketed column name."""
    with open(path, newline="") as f:
        header = [h.strip().strip("[]") for h in f.readline().split(",")]
        for line in f:
            parts = line.split(",")
            if len(parts) == len(header):
                yield {h: p.strip() for h, p in zip(header, parts)}


def _clean_side(row: dict[str, str], side: str, S: float, dte: float,
                K: float, quote_ok: bool) -> tuple[str | None, list | None]:
    """Apply the filters to one side (``'C'`` or ``'P'``) of a raw row.

    Returns ``(failed_filter, None)`` or ``(None, output_row)``.
    """
    if not quote_ok:
        return "closed_day", None
    bid, ask = _num(row[f"{side}_BID"]), _num(row[f"{side}_ASK"])
    if bid is None or ask is None:
        return "missing_quote", None
    if not DTE_MIN <= dte <= DTE_MAX:
        return "dte_range", None
    m = K / S
    if not MONEYNESS_MIN <= m <= MONEYNESS_MAX:
        return "moneyness_range", None
    if bid <= 0:
        return "zero_bid", None
    if ask <= bid:
        return "crossed_or_locked", None
    mid = 0.5 * (bid + ask)
    if mid < MID_MIN:
        return "mid_below_min", None
    iv = _num(row[f"{side}_IV"])
    if iv is None or not IV_MIN < iv < IV_MAX:
        return "iv_invalid", None

    expiry = dt.date.fromisoformat(row["EXPIRE_DATE"])
    last, vol, delta = (_num(row[f"{side}_{c}"]) for c in ("LAST", "VOLUME", "DELTA"))
    return None, [
        row["QUOTE_DATE"], row["EXPIRE_DATE"], side, f"{S:.2f}", f"{K:.2f}",
        f"{dte:.0f}", f"{dte / 365.0:.6f}", f"{m:.6f}", f"{math.log(m):.6f}",
        f"{bid:.2f}", f"{ask:.2f}", f"{mid:.3f}", f"{(ask - bid) / mid:.6f}",
        "" if last is None else f"{last:.2f}",
        "" if vol is None else f"{vol:.0f}",
        f"{iv:.6f}", "" if delta is None else f"{delta:.6f}",
        int(is_third_friday(expiry)),
    ]


def clean_file(path: str) -> dict:
    """Clean one monthly file; returns filter counts and daily underlying levels."""
    month = os.path.basename(path)[8:14]
    counts: Counter[str] = Counter()
    underlying: dict[str, float] = {}
    seen: set[tuple] = set()
    out_path = os.path.join(OUT_DIR, "options", f"spx_clean_{month}.csv.gz")
    tmp_path = out_path + ".tmp"
    with gzip.open(tmp_path, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(OUT_COLUMNS)
        for row in _iter_raw(path):
            qd = row["QUOTE_DATE"]
            quote_ok = is_trading_day(dt.date.fromisoformat(qd))
            S, K, dte = _num(row["UNDERLYING_LAST"]), _num(row["STRIKE"]), _num(row["DTE"])
            if quote_ok and S:
                underlying.setdefault(qd, S)
            for side in ("C", "P"):
                counts["raw_options"] += 1
                failed, out = _clean_side(row, side, S, dte, K, quote_ok)
                if failed:
                    counts[failed] += 1
                    continue
                key = (qd, row["EXPIRE_DATE"], side, K)
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(key)
                counts["kept"] += 1
                w.writerow(out)
    os.replace(tmp_path, out_path)
    return {"month": month, "counts": dict(counts), "underlying": underlying}


def main() -> None:
    os.makedirs(os.path.join(OUT_DIR, "options"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "spx-*", "spx_eod_*.txt")))
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        results = pool.map(clean_file, files)

    log_cols = ["month", "raw_options", *FILTERS, "duplicate", "kept"]
    with open(os.path.join(OUT_DIR, "cleaning_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(log_cols)
        for r in results:
            w.writerow([r["month"], *(r["counts"].get(c, 0) for c in log_cols[1:])])

    underlying = {d: s for r in results for d, s in r["underlying"].items()}
    with open(os.path.join(OUT_DIR, "spx_underlying_daily.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "S", "log_return"])
        prev = None
        for d in sorted(underlying):
            s = underlying[d]
            w.writerow([d, f"{s:.2f}", "" if prev is None else f"{math.log(s / prev):.8f}"])
            prev = s
    print(f"cleaned {len(files)} files, {len(underlying)} trading days")


if __name__ == "__main__":
    main()
