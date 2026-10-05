import numpy as np

from pinnvol.pricing import bs_price, bs_vega, cn_price, implied_vol


def _grid(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return dict(S=np.full(n, 4000.0), K=rng.uniform(3200, 4800, n), tau=rng.uniform(7 / 365, 0.5, n),
                r=rng.uniform(0, 0.06, n), q=rng.uniform(0, 0.03, n), sigma=rng.uniform(0.08, 0.8, n))


def test_put_call_parity():
    g = _grid()
    c = bs_price(**g, is_call=True)
    p = bs_price(**g, is_call=False)
    rhs = g["S"] * np.exp(-g["q"] * g["tau"]) - g["K"] * np.exp(-g["r"] * g["tau"])
    np.testing.assert_allclose(c - p, rhs, atol=1e-8)


def test_known_value():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5 -> C=4.76, P=0.81
    assert abs(bs_price(42, 40, 0.5, 0.1, 0.0, 0.2, True) - 4.76) < 5e-3
    assert abs(bs_price(42, 40, 0.5, 0.1, 0.0, 0.2, False) - 0.81) < 5e-3


def test_vega_matches_finite_difference():
    g = _grid(50)
    h = 1e-5
    up = bs_price(**{**g, "sigma": g["sigma"] + h}, is_call=True)
    dn = bs_price(**{**g, "sigma": g["sigma"] - h}, is_call=True)
    np.testing.assert_allclose(bs_vega(**g), (up - dn) / (2 * h), rtol=1e-5, atol=1e-6)


def test_implied_vol_roundtrip():
    g = _grid()
    is_call = np.arange(len(g["S"])) % 2 == 0
    V = bs_price(**g, is_call=is_call)
    iv = implied_vol(V, g["S"], g["K"], g["tau"], g["r"], g["q"], is_call)
    ok = np.isfinite(iv)
    assert ok.mean() > 0.97  # a few deep wings have prices below float resolution
    np.testing.assert_allclose(iv[ok], g["sigma"][ok], atol=1e-6)


def test_implied_vol_rejects_arbitrage():
    # A call below intrinsic has no implied volatility.
    assert np.isnan(implied_vol(50.0, 4000, 3800, 0.1, 0.02, 0.01, True))


def test_crank_nicolson_matches_closed_form():
    g = _grid(30, seed=1)
    is_call = np.arange(30) % 2 == 0
    cn = cn_price(g["S"], g["K"], g["tau"], g["r"], g["q"], g["sigma"], is_call, nx=400, nt=200)
    bs = bs_price(**g, is_call=is_call)
    assert np.max(np.abs(cn - bs)) < 0.1  # index points, on prices up to ~1000
