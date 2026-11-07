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
