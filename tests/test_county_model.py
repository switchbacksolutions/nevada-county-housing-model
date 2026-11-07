import numpy as np, pandas as pd, pytest
from analysis import county_model as cm

def make_panel(n_c=40, T=24, b=1.5, rho=0.3, seed=1):
    """AR(1) price growth driven by population growth plus county and year effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_c):
        fe, dlp = rng.normal(0, 0.01), 0.0
        pop = rng.uniform(50_000, 500_000)
        for t in range(T):
            dlpop = rng.normal(0.005, 0.01)
            dlp = fe + b * dlpop + rho * dlp + rng.normal(0, 0.02)
            pop *= np.exp(dlpop)
            rows.append(dict(fips=f"{c:05d}", year=2001 + t, dlp=dlp, dlpop=dlpop, dlinc=rng.normal(0, 0.02),
                             permits_pk_lag=rng.uniform(0, 5), population=pop))
    p = pd.DataFrame(rows)
    p["dlp_lag"] = p.groupby("fips")["dlp"].shift()
    p["dlpop_lag"] = p.groupby("fips")["dlpop"].shift()
    return p

def test_fit_panel_recovers_b_and_rho():
    res = cm.fit_panel(make_panel(), "dlpop")
    assert abs(res.params["dlpop"] - 1.5) < 0.25
    assert abs(res.params["dlp_lag"] - 0.3) < 0.08

def test_coef_draws_are_deterministic():
    res = cm.fit_panel(make_panel(), "dlpop")
    a = cm.coef_draws(res, ["dlpop", "dlp_lag"], 500)
    b = cm.coef_draws(res, ["dlpop", "dlp_lag"], 500)
    assert np.array_equal(a, b) and a.shape == (500, 2)

def test_peer_counties_uses_mean_population():
    p = make_panel()
    peers = cm.peer_counties(p, max_pop=200_000)
    means = p.groupby("fips")["population"].mean()
    assert set(peers) == set(means[means < 200_000].index)

def test_spec_curve_has_all_specs():
    p = make_panel()
    p["dlp_lag2"] = p.groupby("fips")["dlp"].shift(2)
    p["dlp_lag3"] = p.groupby("fips")["dlp"].shift(3)
    p["dlpop_pos"] = p["dlpop"].clip(lower=0)
    p["dlpop_neg"] = p["dlpop"].clip(upper=0)
    curve = cm.spec_curve(p, n=200)
    assert list(curve["spec"]) == [name for name, _ in cm.SPECS]
    assert (curve["lr_lo"] <= curve["lr"]).all() and (curve["lr"] <= curve["lr_hi"]).all()

def test_path_effect_zero_growth_is_zero():
    imp = np.random.default_rng(0).normal(size=(50, 10))
    out = cm.path_effect(imp, np.zeros(10))
    assert out.shape == (50, 10) and np.all(out == 0)

def test_path_effect_is_linear_in_log_growth():
    imp = np.random.default_rng(0).normal(size=(50, 10))
    a = cm.path_effect(imp, np.full(10, 0.01))
    b = cm.path_effect(imp, np.full(10, 0.02))
    np.testing.assert_allclose(b, a * np.log1p(0.02) / np.log1p(0.01))

def test_path_effect_matches_manual_convolution():
    imp = np.arange(1, 4, dtype=float).reshape(1, 3)      # [1, 2, 3]
    out = cm.path_effect(imp, np.array([np.e - 1, 0.0, 0.0]))  # log1p = 1 in year 0 only
    np.testing.assert_allclose(out[0], [1, 2, 3])

def test_lp_draws_recover_ar1_impulse():
    p = make_panel(n_c=60, T=30, b=1.5, rho=0.3)
    p["lrp"] = p.groupby("fips")["dlp"].cumsum()
    draws = cm.lp_draws(p, "dlpop", n_boot=100, horizons=range(0, 4), seed=0)
    med = np.median(draws, axis=0)
    expected = 1.5 * np.cumsum(0.3 ** np.arange(4))   # 1.5, 1.95, 2.085, 2.1255
    np.testing.assert_allclose(med, expected, atol=0.35)

def test_bootstrap_slopes_match_dummy_fit():
    p = make_panel()
    p["lrp"] = p.groupby("fips")["dlp"].cumsum()
    res = cm.fit_local_projection(p, "dlpop", 2)
    d = res.sample
    cols = ["dlpop", *cm.LP_CONTROLS]
    years = pd.get_dummies(d["year"].astype(str), drop_first=True, dtype=float).to_numpy()
    sizes = d.groupby("fips", sort=False).size().to_numpy()
    slopes = cm._fe_slopes(d["y"].to_numpy(), np.column_stack([d[cols].to_numpy(), years]), sizes)
    np.testing.assert_allclose(slopes[:len(cols)], res.params[cols], rtol=1e-8)
