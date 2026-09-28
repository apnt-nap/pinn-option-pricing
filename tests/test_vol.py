import numpy as np
import pandas as pd

from pinnvol.data.synthetic import simulate_returns
from pinnvol.vol import (fit_garch, forward_realized_var, garch_horizon_var, garch_walk_forward,
                         har_walk_forward, interp_total_variance, trading_horizon)


def _returns(n=3000, seed=0, **kw):
    d = pd.bdate_range("2005-01-03", periods=n)
    r, h = simulate_returns(d, np.random.default_rng(seed), **kw)
    return pd.Series(r, index=d), h


def test_gjr_garch_recovers_parameters():
    r, _ = _returns(6000, seed=3)
    p = fit_garch(r.to_numpy(), "gjr")
    assert abs(p.persistence - (0.03 + 0.15 / 2 + 0.86)) < 0.03
    assert abs(p.beta - 0.86) < 0.05


def test_horizon_aggregation_matches_explicit_sum():
    h_next, vbar, phi = 2e-4, 1e-4, 0.97
    H = 21
    explicit = 252 / H * sum(vbar + phi ** (h - 1) * (h_next - vbar) for h in range(1, H + 1))
    assert abs(garch_horizon_var(np.array([h_next]), vbar, phi, np.array([H]))[0] - explicit) < 1e-12


def test_walk_forward_uses_no_future_returns():
    r, _ = _returns(1500)
    start = r.index[1000]
    base = garch_walk_forward(r, start, refit="M", min_obs=500)
    shocked = r.copy()
    shocked.iloc[1200:] *= 5  # changing the future must not change earlier forecasts
    alt = garch_walk_forward(shocked, start, refit="M", min_obs=500)
    cut = r.index[1199]
    first_refit_after = alt.index[(alt.index > cut) & (alt["omega"] != base["omega"])]
    assert (alt.loc[:cut, "h_next"] == base.loc[:cut, "h_next"]).all()
    assert len(first_refit_after) == 0 or first_refit_after[0] > cut
    har0 = har_walk_forward(r, start, [5, 21], min_obs=500)
    har1 = har_walk_forward(shocked, start, [5, 21], min_obs=500)
    pd.testing.assert_frame_equal(har0.loc[:cut], har1.loc[:cut])


def test_forward_realized_var():
    r = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03])
    rv = forward_realized_var(r, 2)
    assert abs(rv.iat[0] - 252 / 2 * (0.02**2 + 0.01**2)) < 1e-15
    assert np.isnan(rv.iat[3])


def test_total_variance_interpolation():
    grid = np.array([5.0, 21.0])
    v = np.array([[0.04, 0.09]])
    mid = interp_total_variance(grid, v, np.array([13.0]))[0]
    assert abs(mid * 13 - (0.5 * 0.04 * 5 + 0.5 * 0.09 * 21)) < 1e-12
    assert interp_total_variance(grid, v, np.array([2.0]))[0] == 0.04
    assert list(trading_horizon([7, 30, 180])) == [5, 21, 124]
