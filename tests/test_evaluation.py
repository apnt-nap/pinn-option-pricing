import numpy as np
import pandas as pd

from pinnvol.evaluation import diebold_mariano, pricing_metrics, vol_forecast_metrics


def test_pricing_metrics():
    m = pricing_metrics([1.0, 2.0, 0.2], [1.5, 2.0, 0.1], mape_min_price=0.5)
    assert abs(m["mae"] - 0.2) < 1e-12
    assert abs(m["rmse"] - np.sqrt((0.25 + 0.01) / 3)) < 1e-12
    assert abs(m["mape"] - 100 * (0.5 / 1.5) / 2) < 1e-12  # the 0.1 quote is excluded


def test_dm_detects_better_forecast():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=300), 20)
    e = rng.normal(size=len(dates))
    res = diebold_mariano((0.8 * e) ** 2, e**2 + 0.05, dates)
    assert res["dm"] < -3 and res["p_value"] < 0.01
    same = diebold_mariano(e**2, rng.permutation(e) ** 2, dates)
    assert abs(same["dm"]) < 3


def test_qlike_minimised_at_truth():
    rng = np.random.default_rng(1)
    true_var = rng.uniform(0.01, 0.09, 5000)
    rv2 = true_var * rng.chisquare(5, 5000) / 5
    good = vol_forecast_metrics(true_var, rv2)["qlike"]
    bad = vol_forecast_metrics(true_var * 1.3, rv2)["qlike"]
    assert good < bad
