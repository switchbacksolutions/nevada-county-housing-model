"""Estimate how Nevada County population growth moves home values in Grass Valley and Nevada City.

Stage 1 fits real price growth on population growth across all California counties, with county
and year fixed effects. Stage 2 maps Nevada County price growth to each city. Run fetch_data.py first.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).parent
DATA, OUT, SITE = ROOT / "data", ROOT / "output", ROOT / "site"
NEVADA = "06057"
CITIES = ["Grass Valley", "Nevada City"]
BASE_YEAR = 2025  # last complete year of city prices; projections start the year after
HORIZON = 10
DRAWS = 10_000
RNG = np.random.default_rng(2026)

# Slots 1-4 of the dataviz reference palette (light mode), validated for adjacent lines.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LABELS = {
    "dlp_lag": "Last year's real price growth",
    "dlpop": "Population growth (log change)",
    "dlpop_lag": "Last year's population growth",
    "dlpop_lead": "Next year's population growth (placebo)",
    "dlinc": "Real per-capita income growth",
    "permits_pk_lag": "Last year's permits per 1,000 residents",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    nat = pd.read_csv(DATA / "national_annual.csv", index_col="year")
    deflator = nat["cpi"] / nat.loc[BASE_YEAR, "cpi"]  # real values are in BASE_YEAR dollars

    p = pd.read_csv(DATA / "ca_county_panel.csv", dtype={"fips": str})
    p = p[p["year"] <= BASE_YEAR].sort_values(["fips", "year"])
    p["deflator"] = p["year"].map(deflator)
    g = p.groupby("fips")
    p["dlp"] = g["hpi"].transform(lambda s: np.log(s).diff()) - g["deflator"].transform(lambda s: np.log(s).diff())
    p["dlp_lag"] = p.groupby("fips")["dlp"].shift()
    p["dlpop"] = g["population"].transform(lambda s: np.log(s).diff())
    p["dlpop_lag"] = p.groupby("fips")["dlpop"].shift()
    p["dlpop_lead"] = p.groupby("fips")["dlpop"].shift(-1)
    p["dlinc"] = g["income_pc"].transform(lambda s: np.log(s).diff()) - g["deflator"].transform(lambda s: np.log(s).diff())
    p["permits_pk_lag"] = (1000 * p["permits"] / p["population"]).groupby(p["fips"]).shift()

    z = pd.read_csv(DATA / "city_zhvi_annual.csv", index_col="year").loc[:BASE_YEAR]
    z_real = z.div(deflator.loc[z.index], axis=0)
    return p, z_real


def fit_panel(p: pd.DataFrame, pop_col: str, extra_cols: tuple[str, ...] = ()):
    """Two-way fixed-effects OLS with standard errors clustered by county and by year."""
    x_cols = ["dlp_lag", pop_col, "dlinc", "permits_pk_lag", *extra_cols]
    d = p.dropna(subset=["dlp", *x_cols]).copy()
    d["county_id"] = d["fips"].astype("category").cat.codes
    fe = pd.get_dummies(d[["fips", "year"]].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[x_cols], fe], axis=1))
    # Two-way clustered covariance is not positive definite for the dummy columns. Only the slopes are used.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sm.OLS(d["dlp"], X).fit(cov_type="cluster", cov_kwds={"groups": d[["county_id", "year"]].to_numpy()})
        res.bse
    res.sample = d
    return res


def fit_city(dl_city: pd.Series, dl_county: pd.Series):
    d = pd.concat([dl_city, dl_county.rename("county")], axis=1).dropna()
    return sm.OLS(d.iloc[:, 0], sm.add_constant(d["county"])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})


def draws(res, cols: list[str]) -> np.ndarray:
    return RNG.multivariate_normal(res.params[cols], res.cov_params().loc[cols, cols], DRAWS)


def long_run(res, pop_col: str) -> tuple[float, float, float]:
    """Long-run elasticity b_pop / (1 - rho) with a 90% interval from coefficient draws."""
    b = draws(res, [pop_col, "dlp_lag"])
    point = res.params[pop_col] / (1 - res.params["dlp_lag"])
    return point, *np.percentile(b[:, 0] / (1 - b[:, 1]), [5, 95])


def scenarios(p: pd.DataFrame) -> dict[str, float]:
    pop = p[p["fips"] == NEVADA].set_index("year")["population"]
    trend = (pop.loc[BASE_YEAR] / pop.loc[2000]) ** (1 / (BASE_YEAR - 2000)) - 1
    return {
        "Decline (-0.5%/yr)": -0.005,
        f"2000-{BASE_YEAR} trend ({trend:+.2%}/yr)": trend,
        "Growth (+1.0%/yr)": 0.01,
        "Fast growth (+2.0%/yr)": 0.02,
    }


def population_effect(county, cities, pop_growth: float, pop_col: str = "dlpop") -> dict[str, np.ndarray]:
    """Log difference in real city value vs. a flat-population path, shape (DRAWS, HORIZON).

    The same coefficient draws feed every scenario, so scenario gaps reflect only population.
    """
    rng_state = RNG.bit_generator.state
    b = draws(county, [pop_col, "dlp_lag"])
    d, level = np.zeros(DRAWS), np.zeros((DRAWS, HORIZON))
    for t in range(HORIZON):
        d = b[:, 0] * np.log1p(pop_growth) + b[:, 1] * d
        level[:, t] = level[:, t - 1] + d if t else d
    out = {city: level * draws(res, ["county"])[:, [0]] for city, res in cities.items()}
    RNG.bit_generator.state = rng_state
    return out


def impulse(res, pop_col: str) -> dict[str, list[float]]:
    """Cumulative real price response (%) to a one-time 1% population increase."""
    b = draws(res, [pop_col, "dlp_lag"])
    d, level = b[:, 0] * 0.01, np.zeros((DRAWS, HORIZON + 1))
    for t in range(1, HORIZON + 1):
        level[:, t] = level[:, t - 1] + d
        d = b[:, 1] * d
    pct = 100 * np.expm1(level)
    return {k: np.round(np.percentile(pct, q, axis=0), 3).tolist() for k, q in (("p5", 5), ("median", 50), ("p95", 95))}


def coef_json(res) -> list[dict]:
    ci = res.conf_int(0.10)
    return [
        {"term": LABELS[k], "coef": round(res.params[k], 4), "lo": round(ci.loc[k, 0], 4), "hi": round(ci.loc[k, 1], 4), "p": round(res.pvalues[k], 4)}
        for k in res.params.index if k in LABELS
    ]


def export_site(p, z, county, county_lag, cities):
    """Write site/data.json: history, county comparison, model summary, and a scenario grid for the explorer."""
    names = pd.read_csv(DATA / "ca_county_names.csv", dtype={"fips": str}).set_index("fips")["name"]
    nev = p[p["fips"] == NEVADA].set_index("year")
    real_hpi = (nev["hpi"] / nev["deflator"]).dropna()
    years = list(range(2001, BASE_YEAR + 1))
    history = [
        {
            "year": y,
            "population": round(nev.loc[y, "population"]),
            "county_hpi_real": round(real_hpi.get(y, np.nan), 2),
            **{city: round(z.loc[y, city]) for city in CITIES},
        }
        for y in years
    ]

    start, end = 2001, BASE_YEAR - 1
    counties = []
    for fips, g in p.set_index("year").groupby("fips"):
        hpi = g["hpi"] / g["deflator"]
        if pd.isna(hpi.get(start)) or pd.isna(hpi.get(end)):
            continue
        counties.append({
            "fips": fips,
            "name": names[fips],
            "population": round(g.loc[end, "population"]),
            "pop_change": round(100 * (g.loc[end, "population"] / g.loc[start, "population"] - 1), 2),
            "price_change": round(100 * (hpi[end] / hpi[start] - 1), 2),
        })

    grid = []
    for g in np.round(np.arange(-1.0, 3.01, 0.1), 1):
        for model_name, res, col in (("same_year", county, "dlpop"), ("lagged", county_lag, "dlpop_lag")):
            eff = population_effect(res, cities, g / 100, col)
            for city in CITIES:
                v = z.loc[BASE_YEAR, city] * np.exp(np.hstack([np.zeros((DRAWS, 1)), eff[city]]))
                lo, med, hi = np.percentile(v, [5, 50, 95], axis=0).round(-2)
                grid.append({"growth": float(g), "model": model_name, "city": city, "median": med.tolist(), "p5": lo.tolist(), "p95": hi.tolist()})

    data = {
        "base_year": BASE_YEAR,
        "horizon": HORIZON,
        "cities": CITIES,
        "base_values": {city: round(z.loc[BASE_YEAR, city]) for city in CITIES},
        "trend_growth": round(list(scenarios(p).values())[1], 5),
        "elasticity": {
            "same_year": dict(zip(("point", "lo", "hi"), np.round(long_run(county, "dlpop"), 3))),
            "lagged": dict(zip(("point", "lo", "hi"), np.round(long_run(county_lag, "dlpop_lag"), 3))),
        },
        "passthrough": {
            city: {"coef": round(res.params["county"], 3), "lo": round(res.conf_int(0.10).loc["county", 0], 3),
                   "hi": round(res.conf_int(0.10).loc["county", 1], 3), "r2": round(res.rsquared, 3)}
            for city, res in cities.items()
        },
        "coefficients": {"same_year": coef_json(county), "lagged": coef_json(county_lag)},
        "panel": {"counties": county.sample["fips"].nunique(), "first_year": int(county.sample["year"].min()),
                  "last_year": int(county.sample["year"].max()), "n": int(county.nobs)},
        "impulse": {"same_year": impulse(county, "dlpop"), "lagged": impulse(county_lag, "dlpop_lag")},
        "history": history,
        "counties": {"start": start, "end": end, "rows": counties},
        "scenario_grid": grid,
    }
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":"), default=float))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_history(p, z):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    years = z.dropna().index
    base = years[0]
    pop = p[p["fips"] == NEVADA].set_index("year")["population"]
    series = {
        "Nevada County population": pop,
        "Grass Valley real home value": z["Grass Valley"],
        "Nevada City real home value": z["Nevada City"],
    }
    for (name, s), color in zip(series.items(), COLORS):
        idx = 100 * s.loc[years] / s.loc[base]
        ax.plot(years, idx, color=color, linewidth=2, label=name)
        ax.annotate(name, (years[-1], idx.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    style(ax)
    ax.set_title(f"Nevada County population vs. real home values, indexed {base} = 100", loc="left", color=INK, fontsize=12)
    ax.set_ylabel("Index", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    ax.set_ylim(bottom=60)
    ax.set_xlim(years[0], years[-1] + 7)
    fig.tight_layout()
    fig.savefig(OUT / "history.png", dpi=150)
    plt.close(fig)


def plot_scenarios(z, effects):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    years = np.arange(BASE_YEAR, BASE_YEAR + HORIZON + 1)
    for ax, city in zip(axes, CITIES):
        base = z.loc[BASE_YEAR, city]
        ax.axhline(base / 1e3, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate("Flat population", (years[-1], base / 1e3), xytext=(0, -12), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
        for (name, eff), color in zip(effects.items(), COLORS):
            v = base * np.exp(np.hstack([np.zeros((DRAWS, 1)), eff[city]])) / 1e3
            med = np.median(v, axis=0)
            if name.startswith("Growth"):
                lo, hi = np.percentile(v, [5, 95], axis=0)
                ax.fill_between(years, lo, hi, color=color, alpha=0.15, linewidth=0, label="+1.0%/yr 90% range")
            ax.plot(years, med, color=color, linewidth=2, label=name)
            ax.annotate(f"${med[-1]:,.0f}k", (years[-1], med[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        style(ax)
        ax.set_title(city, loc="left", color=INK, fontsize=12)
        ax.set_xlim(years[0], years[-1] + 1.6)
    axes[0].set_ylabel(f"Typical home value, {BASE_YEAR} $ (thousands)", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.suptitle(
        f"Home value by county population path, all other drivers held at {BASE_YEAR} levels",
        x=0.01, ha="left", color=INK, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "scenarios.png", dpi=150)
    plt.close(fig)


def coef_table(res) -> str:
    ci = res.conf_int(0.10)
    lines = ["| Term | Coefficient | 90% CI | p-value |", "|---|---:|---|---:|"]
    for k in [k for k in res.params.index if k in LABELS]:
        lines.append(f"| {LABELS[k]} | {res.params[k]:.3f} | {ci.loc[k, 0]:.3f} to {ci.loc[k, 1]:.3f} | {res.pvalues[k]:.3f} |")
    s = res.sample
    lines.append(
        f"\n{s['fips'].nunique()} counties, years {s['year'].min()}-{s['year'].max()}, n = {int(res.nobs)}. "
        "County and year fixed effects. Standard errors clustered by county and year."
    )
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    p, z = load()

    county = fit_panel(p, "dlpop")
    county_lag = fit_panel(p, "dlpop_lag")
    placebo = fit_panel(p, "dlpop_lead")
    joint = fit_panel(p, "dlpop", ("dlpop_lag", "dlpop_lead"))
    nevada_dlp = p[p["fips"] == NEVADA].set_index("year")["dlp"]
    cities = {city: fit_city(np.log(z[city]).diff(), nevada_dlp) for city in CITIES}

    lr, lr_lo, lr_hi = long_run(county, "dlpop")
    lr2, lr2_lo, lr2_hi = long_run(county_lag, "dlpop_lag")

    effects = {name: population_effect(county, cities, g) for name, g in scenarios(p).items()}
    effects_lag = {name: population_effect(county_lag, cities, g, "dlpop_lag") for name, g in scenarios(p).items()}
    plot_history(p, z)
    plot_scenarios(z, effects)
    export_site(p, z, county, county_lag, cities)

    rows = [
        f"| Scenario | 10-yr population change | City | {BASE_YEAR + HORIZON} value | Effect vs. flat population | 90% range | Lagged-model effect |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for (name, eff), eff_lag, g in zip(effects.items(), effects_lag.values(), scenarios(p).values()):
        for city in CITIES:
            base = z.loc[BASE_YEAR, city]
            dollars = base * (np.exp(eff[city][:, -1]) - 1)
            lo, med, hi = np.percentile(dollars, [5, 50, 95])
            med_lag = base * (np.exp(np.median(eff_lag[city][:, -1])) - 1)
            rows.append(
                f"| {name} | {(1 + g) ** HORIZON - 1:+.1%} | {city} | ${base + med:,.0f} | "
                f"{med / base:+.1%} (${med:+,.0f}) | ${lo:+,.0f} to ${hi:+,.0f} | {med_lag / base:+.1%} (${med_lag:+,.0f}) |"
            )

    city_rows = ["| City | Pass-through of county price growth | 90% CI | R² | Years |", "|---|---:|---|---:|---|"]
    for city, res in cities.items():
        ci = res.conf_int(0.10).loc["county"]
        years = res.model.data.row_labels
        city_rows.append(f"| {city} | {res.params['county']:.2f} | {ci[0]:.2f} to {ci[1]:.2f} | {res.rsquared:.2f} | {years.min()}-{years.max()} |")

    city_table, scenario_table = "\n".join(city_rows), "\n".join(rows)
    report = f"""# Nevada County population and home values: model results

All prices are real {BASE_YEAR} dollars (CPI-U deflated). Generated by `model.py`.

## Stage 1: population and real home-price growth, California counties

Dependent variable: annual log change in each county's real FHFA house price index.
Year fixed effects absorb statewide and national drivers such as mortgage rates.

{coef_table(county)}

**Long-run elasticity of price to population: {lr:.2f}** (90% interval {lr_lo:.2f} to {lr_hi:.2f}).
A county population that is 1% larger, sustained, goes with real home prices about {lr:.1f}% higher
once price momentum plays out.

The same model with last year's population growth in place of this year's, a second timing assumption:

{coef_table(county_lag)}

Long-run elasticity with lagged population: {lr2:.2f} (90% interval {lr2_lo:.2f} to {lr2_hi:.2f}).

## Placebo check

Next year's population growth, which this year's prices cannot respond to, is included as a
regressor. If it predicts this year's price growth, the population coefficient reflects shared
local shocks (jobs, amenities, remote work) rather than a timing story.

Next year's population growth on its own:

{coef_table(placebo)}

Lagged, same-year and next-year growth together:

{coef_table(joint)}

Neither timing choice is a bound on a causal effect. Both are associations.

## Stage 2: Nevada County to city pass-through

Annual log change in each city's real Zillow Home Value Index on the Nevada County change.

{city_table}

## Population scenarios, {BASE_YEAR + 1}-{BASE_YEAR + HORIZON}

Each scenario holds Nevada County population growth at a constant rate. Every other driver stays
where it was in {BASE_YEAR}, so the {BASE_YEAR + HORIZON} value is the {BASE_YEAR} value plus the population effect only.
It is not a forecast of the market. Ranges come from {DRAWS:,} draws of the coefficients.
The last column uses the lagged-population model, the same association under the lagged timing assumption.

{scenario_table}

Charts: `history.png`, `scenarios.png`.
"""
    (OUT / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
