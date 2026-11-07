"""County panel estimation: two-way fixed-effects fit, deterministic coefficient draws, specification curve.

No file I/O. model.py loads data and calls these functions.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 2026

# `counties` is "all", "peer" (mean population under 200,000) or "large" (the rest).
# `drop_years` removes those years before fitting.
SPECS: list[tuple[str, dict]] = [
    ("all/same_year", dict(pop_col="dlpop", counties="all", years=None, extra_cols=())),
    ("all/lagged", dict(pop_col="dlpop_lag", counties="all", years=None, extra_cols=())),
    ("peer/same_year", dict(pop_col="dlpop", counties="peer", years=None, extra_cols=())),
    ("peer/lagged", dict(pop_col="dlpop_lag", counties="peer", years=None, extra_cols=())),
    ("large/same_year", dict(pop_col="dlpop", counties="large", years=None, extra_cols=())),
    ("all/2001-2010", dict(pop_col="dlpop", counties="all", years=(2001, 2010), extra_cols=())),
    ("all/2011-2024", dict(pop_col="dlpop", counties="all", years=(2011, 2024), extra_cols=())),
    ("all/drop_2020_2022", dict(pop_col="dlpop", counties="all", years=None, drop_years=(2020, 2021, 2022), extra_cols=())),
    ("all/three_price_lags", dict(pop_col="dlpop", counties="all", years=None, extra_cols=("dlp_lag2", "dlp_lag3"))),
    ("all/sign_split", dict(pop_col="dlpop_pos", counties="all", years=None, extra_cols=("dlpop_neg",))),
]


def fit_panel(p: pd.DataFrame, pop_col: str, counties: list[str] | None = None, extra_cols: tuple[str, ...] = ()):
    """Two-way fixed-effects OLS with standard errors clustered by county and by year."""
    if counties is not None:
        p = p[p["fips"].isin(counties)]
    x_cols = ["dlp_lag", pop_col, "dlinc", "permits_pk_lag", *extra_cols]
    d = p.dropna(subset=["dlp", *x_cols]).copy()
    d["county_id"] = d["fips"].astype("category").cat.codes
    fe = pd.get_dummies(d[["fips", "year"]].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[x_cols], fe], axis=1))
    # Two-way clustered covariance is not positive definite for the dummy columns. Only the slopes are used.
    # Touching bse here caches it, so the warning fires inside this suppression and not on later reads.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sm.OLS(d["dlp"], X).fit(cov_type="cluster", cov_kwds={"groups": d[["county_id", "year"]].to_numpy()})
        res.bse
    res.sample = d
    return res


def coef_draws(res, cols: list[str], n: int, seed: int = SEED) -> np.ndarray:
    """Fresh generator per call, so every caller sees identical draws."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(res.params[cols], res.cov_params().loc[cols, cols], n)


def long_run(res, pop_col: str, n: int) -> tuple[float, float, float]:
    """Long-run elasticity b_pop / (1 - rho) with a 90% interval from coefficient draws."""
    b = coef_draws(res, [pop_col, "dlp_lag"], n)
    point = res.params[pop_col] / (1 - res.params["dlp_lag"])
    return point, *np.percentile(b[:, 0] / (1 - b[:, 1]), [5, 95])


def peer_counties(p: pd.DataFrame, max_pop: int = 200_000) -> list[str]:
    mean_pop = p.groupby("fips")["population"].mean()
    return sorted(mean_pop[mean_pop < max_pop].index)


def spec_curve(p: pd.DataFrame, n: int) -> pd.DataFrame:
    peers = peer_counties(p)
    rows = []
    for name, s in SPECS:
        d = p
        if s["years"]:
            d = d[d["year"].between(*s["years"])]
        if s.get("drop_years"):
            d = d[~d["year"].isin(s["drop_years"])]
        sel = {"all": None, "peer": peers, "large": [f for f in p["fips"].unique() if f not in peers]}[s["counties"]]
        res = fit_panel(d, s["pop_col"], sel, s["extra_cols"])
        lr, lo, hi = long_run(res, s["pop_col"], n)
        rows.append(dict(
            spec=name, pop_col=s["pop_col"], counties=s["counties"],
            b_pop=res.params[s["pop_col"]], se=res.bse[s["pop_col"]], rho=res.params["dlp_lag"],
            lr=lr, lr_lo=lo, lr_hi=hi, n_obs=int(res.nobs), n_counties=res.sample["fips"].nunique(),
        ))
    return pd.DataFrame(rows)
