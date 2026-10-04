import numpy as np
import pandas as pd

from pinnvol.data.forwards import implied_forwards
from pinnvol.data.loader import flag_am_monthlies
from pinnvol.data.rates import RateCurve
from pinnvol.pricing import bs_price


def test_implied_forward_recovers_dividend_yield():
    S, r, q, tau = 4000.0, 0.03, 0.017, 0.25
    K = np.arange(3600, 4401, 25.0)
    rows = []
    for is_call in (True, False):
        mid = bs_price(S, K, tau, r, q, 0.2, is_call)
        rows.append(pd.DataFrame({"quote_date": pd.Timestamp("2021-01-04"), "expire_date": pd.Timestamp("2021-04-05"),
                                  "is_call": is_call, "K": K, "S": S, "tau": tau, "r": r, "mid": mid}))
    fw = implied_forwards(pd.concat(rows))
    assert abs(fw["q"].iat[0] - q) < 1e-6
    assert abs(fw["F"].iat[0] - S * np.exp((r - q) * tau)) < 1e-3


def test_rate_curve_interpolates_in_maturity(tmp_path):
    for name, y in {"DGS1MO": 1.0, "DGS1": 3.0}.items():
        pd.DataFrame({"observation_date": ["2021-01-04", "2021-01-05"], name: [y, "."]}).to_csv(tmp_path / f"{name}.csv", index=False)
    c = RateCurve.from_fred(tmp_path, {"DGS1MO": 1 / 12, "DGS1": 1.0})
    r = c.rate(np.array(["2021-01-05", "2021-01-05"], dtype="datetime64[ns]"), np.array([1 / 12, 0.5]))
    r1, r3 = 2 * np.log1p(0.01 / 2), 2 * np.log1p(0.03 / 2)
    assert abs(r[0] - r1) < 1e-12  # the "." missing value is forward-filled
    w = (0.5 - 1 / 12) / (1 - 1 / 12)
    assert abs(r[1] - ((1 - w) * r1 + w * r3)) < 1e-12


def test_flag_am_monthlies_handles_thursday_dating():
    ts = pd.Timestamp
    rows = [
        ("2013-06-03", "2013-06-20", 17),  # vendor dates the June 2013 monthly on Thursday
        ("2019-04-01", "2019-04-18", 17),  # Good Friday 2019: monthly expires Thursday
        ("2023-06-01", "2023-06-15", 14),  # SPXW daily on the Thursday before a listed F3
        ("2023-06-01", "2023-06-16", 15),  # the June 2023 third Friday itself
        ("2023-06-01", "2023-06-23", 22),  # ordinary weekly Friday
    ]
    df = pd.DataFrame([(ts(q), ts(e), d) for q, e, d in rows], columns=["quote_date", "expire_date", "dte"])
    df["dte"] = df["dte"].astype("float32")
    out = flag_am_monthlies(df)
    assert out["is_third_friday"].tolist() == [1, 1, 0, 1, 0]
    assert out["dte"].tolist() == [18, 17, 14, 15, 22]  # only the vendor-dated Thursday moves to Friday
