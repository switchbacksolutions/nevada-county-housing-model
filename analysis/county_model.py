"""County panel estimation: two-way fixed-effects fit, coefficient draws, specification curve, local projections.

No file I/O. model.py loads data and calls these functions.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 2026
PEER_MAX_POP = 200_000
HORIZONS = range(0, 10)
LP_CONTROLS = ["dlp_lag", "dlinc", "permits_pk_lag"]

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
    # Natural change cannot respond to this year's prices, so b_pop here is the migration coefficient.
    ("all/components", dict(pop_col="r_mig", counties="all", years=None, extra_cols=("r_natural",))),
    ("peer/components", dict(pop_col="r_mig", counties="peer", years=None, extra_cols=("r_natural",))),
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


def peer_counties(p: pd.DataFrame, max_pop: int = PEER_MAX_POP) -> list[str]:
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


def _lp_frame(p: pd.DataFrame, pop_col: str, h: int, counties: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Rows with the cumulative log real price change from t-1 to t+h. Assumes one row per county-year.

    Growth in the next h years enters as controls, so the pop_col slope is exact for convolving a specified path.
    """
    if counties is not None:
        p = p[p["fips"].isin(counties)]
    p = p.sort_values(["fips", "year"])
    g = p.groupby("fips")
    leads = {f"lead_{k}": g[pop_col].shift(-k) for k in range(1, h + 1)}
    x_cols = [pop_col, *leads, *LP_CONTROLS]
    d = p.assign(y=g["lrp"].shift(-h) - g["lrp"].shift(1), **leads)
    return d.dropna(subset=["y", *x_cols]), x_cols


def fit_local_projection(p: pd.DataFrame, pop_col: str, h: int, counties: list[str] | None = None):
    """Price change h years after a one-year population change, later growth held fixed, county and year effects, clustered by county."""
    d, x_cols = _lp_frame(p, pop_col, h, counties)
    fe = pd.get_dummies(d[["fips", "year"]].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[x_cols], fe], axis=1))
    groups = d["fips"].astype("category").cat.codes.to_numpy()
    res = sm.OLS(d["y"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    res.sample = d
    return res


def _fe_slopes(y: np.ndarray, X: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Two-way FE slopes: X carries the year dummies, rows come in contiguous county blocks of `sizes`.

    Demeaning within county blocks is exact for the county effects (Frisch-Waugh), so no county dummies are needed.
    """
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    Z = np.column_stack([y, X])
    Z = Z - np.repeat(np.add.reduceat(Z, starts, axis=0) / sizes[:, None], sizes, axis=0)
    return np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]


def lp_draws(p: pd.DataFrame, pop_col: str, counties: list[str] | None = None, horizons=HORIZONS,
             n_boot: int = 1000, seed: int = SEED) -> np.ndarray:
    """County block bootstrap of the local-projection slopes, shape (n_boot, len(horizons)).

    Every horizon is refit on the same resample, so each row is one jointly consistent impulse response.
    """
    frames = [_lp_frame(p, pop_col, h, counties) for h in horizons]
    fips = sorted(set().union(*(d["fips"] for d, _ in frames)))
    years = sorted(set().union(*(d["year"] for d, _ in frames)))
    blocks = []
    for d, x_cols in frames:
        yd = (d["year"].to_numpy()[:, None] == np.array(years[1:])[None, :]).astype(float)
        X = np.column_stack([d[x_cols].to_numpy(float), yd])
        by = {f: np.flatnonzero(d["fips"].to_numpy() == f) for f in fips}
        blocks.append((d["y"].to_numpy(float), X, by))

    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(frames)))
    for i in range(n_boot):
        pick = rng.choice(fips, size=len(fips), replace=True)
        for j, (y, X, by) in enumerate(blocks):
            # Each sampled copy is its own block, so a county drawn twice gets two county effects.
            rows = [by[f] for f in pick if len(by[f])]
            idx = np.concatenate(rows)
            out[i, j] = _fe_slopes(y[idx], X[idx], np.array([len(r) for r in rows]))[0]
    return out


def path_effect(impulse: np.ndarray, growth_path: np.ndarray) -> np.ndarray:
    """Cumulative log price response to a growth path: column t sums impulse[:, t-s] * log1p(growth_path[s]) over s <= t."""
    x = np.log1p(np.asarray(growth_path, dtype=float))
    T, H = len(x), impulse.shape[1]
    # Past the last estimated horizon the cumulative response is held at its last value.
    if T > H:
        impulse = np.hstack([impulse, np.repeat(impulse[:, -1:], T - H, axis=1)])
    k, t = np.indices((T, T))
    L = np.where(k <= t, x[(t - k).clip(0)], 0.0)
    return impulse[:, :T] @ L


def long_difference(p: pd.DataFrame, pop_col: str, h: int, counties: list[str] | None = None) -> dict:
    """Price change on population change over non-overlapping h-year windows, with window-end year effects."""
    if counties is not None:
        p = p[p["fips"].isin(counties)]
    p = p.sort_values(["fips", "year"])
    g = p.groupby("fips")
    d = p.assign(y=p["lrp"] - g["lrp"].shift(h), x=g[pop_col].transform(lambda s: s.rolling(h).sum()))
    d = d.dropna(subset=["y", "x"])
    last = d["year"].max()
    d = d[(last - d["year"]) % h == 0]
    fe = pd.get_dummies(d["year"].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["x"]], fe], axis=1))
    res = sm.OLS(d["y"], X).fit(cov_type="cluster", cov_kwds={"groups": d["fips"].astype("category").cat.codes.to_numpy()})
    return dict(b=float(res.params["x"]), se=float(res.bse["x"]), n=int(res.nobs))


def _rolling_predict(p: pd.DataFrame, x_cols: list[str], T: int, h: int) -> pd.Series:
    """Predicted cumulative growth over T+1..T+h by county, from a two-way FE fit on years <= T."""
    fit = p[p["year"] <= T].dropna(subset=["dlp", *x_cols]).sort_values(["fips", "year"])
    years = sorted(fit["year"].unique())
    yd = (fit["year"].to_numpy()[:, None] == np.array(years[1:])[None, :]).astype(float)
    X = np.column_stack([fit[x_cols].to_numpy(float), yd])
    sizes = fit.groupby("fips", sort=False).size().to_numpy()
    coef = _fe_slopes(fit["dlp"].to_numpy(float), X, sizes)
    resid = fit["dlp"].to_numpy(float) - X @ coef
    alpha = pd.Series(resid).groupby(fit["fips"].to_numpy(), sort=False).mean()

    wide = {c: p.pivot(index="fips", columns="year", values=c).reindex(alpha.index) for c in x_cols if c != "dlp_lag"}
    prev = p.pivot(index="fips", columns="year", values="dlp")[T].reindex(alpha.index)
    total = pd.Series(0.0, index=alpha.index)
    for t in range(T + 1, T + h + 1):
        g = alpha.copy()
        for name, b in zip(x_cols, coef):
            g += b * (prev if name == "dlp_lag" else wide[name][t])
        prev = g
        total += g
    return total


def rolling_validation(p: pd.DataFrame, pop_col: str, counties: list[str] | None = None,
                       first_test_year: int = 2015, horizons=(1, 5)) -> pd.DataFrame:
    """Rolling-origin out-of-sample check of the cross-county ranking of cumulative price growth.

    For each origin T the model is fit on years <= T and iterated forward h years using each county's actual
    regressors. Predictions and actuals are demeaned across counties, so this scores the ranking, not the level.
    """
    if counties is not None:
        p = p[p["fips"].isin(counties)]
    lrp = p.pivot(index="fips", columns="year", values="lrp")
    last = int(p["year"].max())
    specs = {"full": ["dlp_lag", pop_col, "dlinc", "permits_pk_lag"], "no_population": ["dlp_lag", "dlinc", "permits_pk_lag"]}
    rows = []
    for h in horizons:
        pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in (*specs, "zero")}
        for T in range(first_test_year - 1, last - h + 1):
            actual = lrp[T + h] - lrp[T]
            preds = {m: _rolling_predict(p, cols, T, h).reindex(actual.index) for m, cols in specs.items()}
            ok = actual.notna()
            for s in preds.values():
                ok &= s.notna()
            if ok.sum() < 3:
                continue
            a = (actual[ok] - actual[ok].mean()).to_numpy()
            for m, s in preds.items():
                s = s[ok]
                pooled[m].append(((s - s.mean()).to_numpy(), a))
            pooled["zero"].append((np.zeros(len(a)), a))
        if not pooled["zero"]:
            raise ValueError("no origin has 3 or more counties with complete horizons")
        for m, pairs in pooled.items():
            pred = np.concatenate([x for x, _ in pairs])
            act = np.concatenate([y for _, y in pairs])
            corr = float(np.corrcoef(pred, act)[0, 1]) if pred.std() > 0 else float("nan")
            rows.append(dict(horizon=h, model=m, rmse=float(np.sqrt(np.mean((pred - act) ** 2))), corr=corr, n=len(act)))
    return pd.DataFrame(rows)
