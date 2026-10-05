r"""Implied forwards and dividend yields from put-call parity.

For each quote date $t$ and expiry $T$, take the strikes $K^*$ where the call and
put mids are closest (the most at-the-money pairs) and compute

$$
F_{t,T} = K^* + e^{r\tau}\big(C(K^*) - P(K^*)\big), \qquad
q_{t,T} = r - \frac{1}{\tau}\ln\frac{F_{t,T}}{S_t}.
$$

The median over the ``n_strikes`` closest pairs is used. This gives $q$ without
buying dividend data and absorbs the timing gap between the 16:00 index level and the
option quotes (Section 6.4 of the proposal, as in the Cboe VIX method). Expiries with no
call/put pair borrow $q$ from the nearest expiry on the same date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_forwards(opts: pd.DataFrame, n_strikes: int = 3, default_q: float = 0.015,
                     q_clip: tuple[float, float] = (-0.02, 0.06)) -> pd.DataFrame:
    """Per (quote_date, expire_date): ``F``, ``q`` and ``q_source``.

    ``opts`` needs ``quote_date, expire_date, is_call, K, S, tau, r, mid``.
    """
    key = ["quote_date", "expire_date"]
    calls = opts.loc[opts["is_call"], key + ["K", "S", "tau", "r", "mid"]]
    puts = opts.loc[~opts["is_call"], key + ["K", "mid"]]
    pairs = calls.merge(puts, on=key + ["K"], suffixes=("_c", "_p"))
    pairs["gap"] = (pairs["mid_c"] - pairs["mid_p"]).abs()
    pairs = pairs.sort_values(key + ["gap"])
    pairs = pairs.groupby(key, sort=False).head(n_strikes)
    pairs["F"] = pairs["K"] + np.exp(pairs["r"] * pairs["tau"]) * (pairs["mid_c"] - pairs["mid_p"])
    fw = pairs.groupby(key).agg(F=("F", "median"), S=("S", "first"), tau=("tau", "first"), r=("r", "first"))
    fw["q"] = fw["r"] - np.log(fw["F"] / fw["S"]) / fw["tau"]
    fw["q_source"] = "parity"

    # Expiries without any call/put pair: nearest-maturity q on the same date.
    allexp = opts.groupby(key).agg(S=("S", "first"), tau=("tau", "first"), r=("r", "first"))
    missing = allexp.index.difference(fw.index)
    if len(missing):
        miss = allexp.loc[missing].reset_index()
        have = fw.reset_index()[["quote_date", "tau", "q"]].sort_values("tau")
        miss = miss.sort_values("tau")
        filled = pd.merge_asof(miss, have.rename(columns={"tau": "tau_ref"}), left_on="tau", right_on="tau_ref",
                               by="quote_date", direction="nearest")
        filled["q"] = filled["q"].fillna(default_q)
        filled["q_source"] = np.where(filled["tau_ref"].isna(), "default", "nearest_expiry")
        filled["F"] = filled["S"] * np.exp((filled["r"] - filled["q"]) * filled["tau"])
        fw = pd.concat([fw, filled.set_index(key)[fw.columns]])

    fw["q"] = fw["q"].clip(*q_clip)
    fw["F"] = fw["S"] * np.exp((fw["r"] - fw["q"]) * fw["tau"])  # consistent with the clipped q
    return fw[["F", "q", "q_source"]].reset_index()
