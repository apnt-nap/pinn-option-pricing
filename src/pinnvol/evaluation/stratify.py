"""Buckets for the stratified analysis (Section 13.4).

Cut-offs come from the config so they are fixed before looking at test results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_buckets(panel: pd.DataFrame, maturity_edges: list[float], moneyness_edges: list[float],
                regimes: dict[str, list[str]], calm_atm_iv: float | None = None) -> pd.DataFrame:
    """Add ``maturity_bucket``, ``moneyness_bucket`` (on K/F) and ``regime`` columns."""
    panel["maturity_bucket"] = pd.cut(panel["dte"], bins=maturity_edges, include_lowest=True,
                                      labels=[f"{int(a)}-{int(b)}d" for a, b in zip(maturity_edges[:-1], maturity_edges[1:])])
    panel["moneyness_bucket"] = pd.cut(panel["fwd_moneyness"], bins=moneyness_edges, include_lowest=True,
                                       labels=[f"K/F {a:.2f}-{b:.2f}" for a, b in zip(moneyness_edges[:-1], moneyness_edges[1:])])
    regime = np.full(len(panel), "other", dtype=object)
    if calm_atm_iv is not None and "atm30_same_day" in panel:
        regime[(panel["atm30_same_day"] < calm_atm_iv).to_numpy()] = "calm"
    for name, (a, b) in regimes.items():
        m = ((panel["quote_date"] >= pd.Timestamp(a)) & (panel["quote_date"] <= pd.Timestamp(b))).to_numpy()
        regime[m] = name
    panel["regime"] = regime
    return panel
