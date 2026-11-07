"""Estimate how Nevada County population growth moves home values in Grass Valley and Nevada City.

Stage 1 fits real price growth on population growth with county and year fixed effects. The headline
uses the California counties under 200,000 people, the set that includes Nevada County. The specification
curve also fits all counties. Local projections give the price response at each horizon, and the scenarios
convolve them with a growth path. Stage 2 maps Nevada County price growth to each city. Run fetch_data.py first.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.county_model import (
    PEER_MAX_POP, SEED, coef_draws, fit_local_projection, fit_panel, long_difference, long_run, lp_draws, path_effect, peer_counties, rolling_validation, spec_curve,
)

ROOT = Path(__file__).parent
DATA, OUT, SITE = ROOT / "data", ROOT / "output", ROOT / "site"
NEVADA = "06057"
CITIES = ["Grass Valley", "Nevada City"]
BASE_YEAR = 2025  # last complete year of city prices; projections start the year after
HORIZON = 10
DRAWS = 10_000
FIRST_TEST_YEAR = 2015
N_BOOT = 1000  # county bootstrap resamples for the local projections
UNIT_PATHS = 200  # draws exported for the browser's live scenarios

# Labels for register rows shown on the history chart. Dates, not causes.
SITE_EVENTS: dict[str, str] = {
    "FIRE_2017_LOBO": "Lobo fire",
    "FIRE_2017_MCCOURTNEY": "McCourtney fire",
    "FIRE_2018_CAMP": "Camp fire",
    "FIRE_2020_JONES": "Jones fire",
    "FIRE_2021_RIVER": "River fire",
    "MOR_2020_08_18": "Nonrenewal moratorium (Jones fire)",
    "INS_2022_Q4_ALLSTATE_PAUSE": "Allstate pauses new policies",
    "INS_2023_05_27_STATE_FARM_NEW_BUSINESS": "State Farm stops new policies",
    "INS_2023_Q3_FARMERS_LIMIT": "Farmers limits new policies",
    "INS_2024_03_20_STATE_FARM_NONRENEWAL": "State Farm nonrenewals",
}
MARKET_EVENTS = [
    {"year": 2006, "date": "2006", "label": "Price peak", "kind": "market"},
    {"year": 2012, "date": "2012", "label": "Price trough", "kind": "market"},
]

# Slots 1-4 of the dataviz reference palette (light mode), validated for adjacent lines.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LABELS = {
    "dlp_lag": "Last year's real price growth",
    "dlpop": "Population growth (log change)",
    "dlpop_lag": "Last year's population growth",
    "dlpop_lead": "Next year's population growth (placebo)",
    "r_mig": "Net migration rate",
    "r_natural": "Natural change rate (births minus deaths)",
    "dlinc": "Real per-capita income growth",
    "permits_pk_lag": "Last year's permits per 1,000 residents",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nat = pd.read_csv(DATA / "national_annual.csv", index_col="year")
    deflator = nat["cpi"] / nat.loc[BASE_YEAR, "cpi"]  # real values are in BASE_YEAR dollars

    p = pd.read_csv(DATA / "ca_county_panel.csv", dtype={"fips": str})
    p = p[p["year"] <= BASE_YEAR].sort_values(["fips", "year"])
    p["deflator"] = p["year"].map(deflator)
    g = p.groupby("fips")
    p["dlp"] = g["hpi"].transform(lambda s: np.log(s).diff()) - g["deflator"].transform(lambda s: np.log(s).diff())
    p["dlp_lag"] = p.groupby("fips")["dlp"].shift()
    p["lrp"] = np.log(p["hpi"]) - np.log(p["deflator"])
    p["dlpop"] = g["population"].transform(lambda s: np.log(s).diff())
    p["dlpop_lag"] = p.groupby("fips")["dlpop"].shift()
    p["dlpop_pos"] = p["dlpop"].clip(lower=0)
    p["dlpop_neg"] = p["dlpop"].clip(upper=0)
    p["dlp_lag2"] = p.groupby("fips")["dlp"].shift(2)
    p["dlp_lag3"] = p.groupby("fips")["dlp"].shift(3)
    p["dlpop_lead"] = p.groupby("fips")["dlpop"].shift(-1)
    lag_pop = g["population"].shift()
    p["r_natural"] = p["natural_change"] / lag_pop
    p["r_mig"] = p["net_migration"] / lag_pop
    p["dlinc"] = g["income_pc"].transform(lambda s: np.log(s).diff()) - g["deflator"].transform(lambda s: np.log(s).diff())
    p["permits_pk_lag"] = (1000 * p["permits"] / p["population"]).groupby(p["fips"]).shift()

    z = pd.read_csv(DATA / "city_zhvi_annual.csv", index_col="year").loc[:BASE_YEAR]
    z_real = z.div(deflator.loc[z.index], axis=0)
    return p, z_real, z


def fit_city(dl_city: pd.Series, dl_county: pd.Series):
    d = pd.concat([dl_city, dl_county.rename("county")], axis=1).dropna()
    return sm.OLS(d.iloc[:, 0], sm.add_constant(d["county"])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})


def scenarios(p: pd.DataFrame) -> list[dict]:
    """Constant-growth presets; growth is a fraction per year."""
    pop = p[p["fips"] == NEVADA].set_index("year")["population"]

    def cagr(start: int) -> float:
        return (pop.loc[BASE_YEAR] / pop.loc[start]) ** (1 / (BASE_YEAR - start)) - 1

    return [
        {"key": "decline", "label": "Decline", "growth": -0.005},
        {"key": "flat", "label": "Flat", "growth": 0.0},
        {"key": "recent", "label": f"Recent 2020-{BASE_YEAR}", "growth": cagr(2020)},
        {"key": "longrun", "label": f"Long-run 2000-{BASE_YEAR}", "growth": cagr(2000)},
        {"key": "growth", "label": "Growth", "growth": 0.01},
    ]


def rate_text(g: float) -> str:
    """Growth fraction as a signed %/yr label: one decimal unless the second one matters."""
    v = round(100 * g, 2)
    text = f"{v:+.1f}" if round(v, 1) == v else f"{v:+.2f}"
    return text.replace("-", "−") + "%/yr"


def support(p: pd.DataFrame) -> dict:
    """Range of 10-year average county growth (percent per year); the slider shades values outside it."""
    win = p.set_index("year").groupby("fips")["dlpop"].rolling(10).mean().rename("avg").reset_index()
    # A window ending in year e starts in e - 9; start years 2001-2015 keep every window complete.
    win = win[win["year"].between(2010, 2024)].dropna(subset=["avg"])
    lo, hi = np.percentile(100 * win["avg"], [5, 95])
    return {"lo": round(lo, 2), "hi": round(hi, 2), "rule": "5th to 95th percentile of 10-year average county growth, 2001-2024"}


def dof_path(dof: pd.DataFrame) -> np.ndarray:
    """Annual Nevada County growth rates from BASE_YEAR to BASE_YEAR + HORIZON in the DOF P-2A projection."""
    pop = dof[dof["fips"] == NEVADA].set_index("year")["population"].loc[BASE_YEAR : BASE_YEAR + HORIZON]
    return (pop.iloc[1:].values / pop.iloc[:-1].values) - 1


def city_draws(cities) -> dict[str, np.ndarray]:
    """Pass-through draws, one per bootstrap resample. A different seed keeps them independent of the county draws."""
    return {city: coef_draws(res, ["county"], N_BOOT, seed=SEED + 1)[:, 0] for city, res in cities.items()}


def population_effect(impulse: np.ndarray, cities_draws: dict[str, np.ndarray], growth_path: np.ndarray) -> dict[str, np.ndarray]:
    """Log difference in real city value vs. a flat-population path, shape (N_BOOT, len(growth_path)).

    The same impulse draws feed every scenario, so scenario gaps reflect only population.
    """
    county = path_effect(impulse, growth_path)
    return {city: county * d[:, None] for city, d in cities_draws.items()}


def band(pct: np.ndarray) -> dict[str, list[float]]:
    return {k: np.round(np.percentile(pct, q, axis=0), 3).tolist() for k, q in (("p5", 5), ("median", 50), ("p95", 95))}


def impulse_lp(imp: np.ndarray) -> dict[str, list[float]]:
    """Cumulative real price response (%) to a one-time 1% population increase; index 0 is the year before, index 1 the same year."""
    return band(100 * np.expm1(0.01 * np.hstack([np.zeros((len(imp), 1)), imp])))


def impulse_ar1(res, pop_col: str) -> dict[str, list[float]]:
    """The momentum model's cumulative real price response (%) to a one-time 1% population increase."""
    b = coef_draws(res, [pop_col, "dlp_lag"], DRAWS)
    d, level = b[:, 0] * 0.01, np.zeros((DRAWS, HORIZON + 1))
    for t in range(1, HORIZON + 1):
        level[:, t] = level[:, t - 1] + d
        d = b[:, 1] * d
    return band(100 * np.expm1(level))


def coef_json(res) -> list[dict]:
    ci = res.conf_int(0.10)
    return [
        {"term": LABELS[k], "coef": round(res.params[k], 4), "lo": round(ci.loc[k, 0], 4), "hi": round(ci.loc[k, 1], 4), "p": round(res.pvalues[k], 4)}
        for k in res.params.index if k in LABELS
    ]


def value_band(z, city: str, eff: np.ndarray) -> dict[str, list[float]]:
    """Median and 90% range of the city value path, with the base-year value first, rounded to $100."""
    v = z.loc[BASE_YEAR, city] * np.exp(np.hstack([np.zeros((len(eff), 1)), eff]))
    lo, med, hi = np.percentile(v, [5, 50, 95], axis=0).round(-2)
    return {"median": med.tolist(), "p5": lo.tolist(), "p95": hi.tolist()}


def site_events() -> list[dict]:
    """Chart events: two fixed market turns plus the labelled register rows."""
    reg = pd.read_csv(DATA / "event_register.csv").set_index("event_id")
    rows = []
    for event_id, label in SITE_EVENTS.items():
        date = reg.loc[event_id, "date_start"]
        kind = "fire" if event_id.startswith("FIRE_") else "insurance"
        rows.append({"year": int(date[:4]), "date": date, "label": label, "kind": kind})
    return sorted(MARKET_EVENTS + rows, key=lambda e: (e["year"], e["date"]))


def explained_export(p, z, imp, cd) -> dict[str, dict[str, float]]:
    """The model's arithmetic applied to Nevada County's observed 2002..BASE_YEAR growth.

    Years past the estimated horizon hold the year-HORIZON response (see path_effect).
    """
    dlpop = p[p["fips"] == NEVADA].set_index("year")["dlpop"].loc[2002:BASE_YEAR]
    final = {c: e[:, -1] for c, e in population_effect(imp, cd, np.expm1(dlpop.values)).items()}
    out = {}
    for city, draws in final.items():
        pop_lo, pop, pop_hi = (100 * np.expm1(np.percentile(draws, q)) for q in (5, 50, 95))
        actual, pop = round(100 * (z.loc[BASE_YEAR, city] / z.loc[2001, city] - 1), 2), round(pop, 2)
        # The parts compound, so the remainder is a log-point difference; built from the rounded values the site shows.
        other = 100 * np.expm1(np.log1p(actual / 100) - np.log1p(pop / 100))
        out[city] = {"start_year": 2001, "end_year": BASE_YEAR, "actual_pct": actual, "population_pct": pop,
                     "population_lo": round(pop_lo, 2), "population_hi": round(pop_hi, 2), "other_pct": round(other, 2)}
    return out


def export_site(p, z, z_nominal, county, county_lag, all_fits, cities, curve, comps, imps, cd, dof, validation):
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
            **{f"{city}_nominal": round(z_nominal.loc[y, city]) for city in CITIES},
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
    for g in np.round(np.arange(-1.0, 2.01, 0.1), 1):
        for model_name, imp in imps.items():
            eff = population_effect(imp, cd, np.full(HORIZON, g / 100))
            for city in CITIES:
                grid.append({"growth": float(g), "model": model_name, "city": city, **value_band(z, city, eff[city])})

    # Effects are linear in log1p(growth), so the browser rescales these 1%/yr draws to any constant rate.
    unit_paths = {
        model_name: {city: np.round(e[:UNIT_PATHS], 5).tolist() for city, e in population_effect(imp, cd, np.full(HORIZON, 0.01)).items()}
        for model_name, imp in imps.items()
    }

    path = dof_path(dof)
    dof_export = {
        "label": "State projection (DOF P-2A)",
        "vintage": json.loads((DATA / "dof_p2a_county.meta.json").read_text())["vintage"],
        "growth": np.round(path, 5).tolist(),
        **{
            model_name: {city: value_band(z, city, eff[city]) for city in CITIES}
            for model_name, eff in ((m, population_effect(imps[m], cd, path)) for m in ("peer", "all"))
        },
    }

    # Two decimals, so the site matches results.md.
    elasticity = {
        f"{set_name}_{key}": dict(zip(("point", "lo", "hi"), np.round(long_run(res, col, DRAWS), 2).tolist()))
        for set_name, fits in (("peer", (county, county_lag)), ("all", all_fits))
        for key, res, col in zip(("same_year", "lagged"), fits, ("dlpop", "dlpop_lag"))
    }
    lp10 = {**dict(zip(("point", "lo", "hi"), np.round(np.percentile(imps["peer"][:, HORIZON - 1], [50, 5, 95]), 2).tolist())),
            "estimate": "bootstrap median"}
    spec_rows = curve.round({"lr": 2, "lr_lo": 2, "lr_hi": 2}).round(3).to_dict("records")

    zips = pd.read_csv(DATA / "panel_zip_year.csv")
    core = zips[(zips["geo_tier"] == "core") & (zips["year"] == 2024)]
    permits = nev["permits"].dropna()

    data = {
        "base_year": BASE_YEAR,
        "horizon": HORIZON,
        "cities": CITIES,
        "events": site_events(),
        # ACS 2020-2024 window; the people translator divides residents by this.
        "persons_per_household": round(core["acs5_population"].sum() / core["acs5_households"].sum(), 2),
        "permits_latest": {"year": int(permits.index[-1]), "count": int(permits.iloc[-1])},
        "population_latest": {"year": BASE_YEAR, "count": round(nev.loc[BASE_YEAR, "population"])},
        "base_values": {city: round(z.loc[BASE_YEAR, city]) for city in CITIES},
        "support": support(p),
        "presets": [{"key": sc["key"], "label": sc["label"], "growth": round(100 * sc["growth"], 2)} for sc in scenarios(p)],
        "scenario_paths": {"dof": dof_export},
        "headline": {"set": "peer", "max_pop": PEER_MAX_POP, "counties": county.sample["fips"].nunique(), **elasticity["peer_same_year"], "lp10": lp10},
        "elasticity": elasticity,
        "passthrough": {
            city: {"coef": round(res.params["county"], 3), "lo": round(res.conf_int(0.10).loc["county", 0], 3),
                   "hi": round(res.conf_int(0.10).loc["county", 1], 3), "r2": round(res.rsquared, 3)}
            for city, res in cities.items()
        },
        "coefficients": {"same_year": coef_json(county), "lagged": coef_json(county_lag)},
        "panel": {"counties": county.sample["fips"].nunique(), "first_year": int(county.sample["year"].min()),
                  "last_year": int(county.sample["year"].max()), "n": int(county.nobs)},
        "impulse": {"lp": impulse_lp(imps["peer"]), "ar1": impulse_ar1(county, "dlpop")},
        "history": history,
        "explained": explained_export(p, z, imps["peer"], cd),
        "counties": {"start": start, "end": end, "rows": counties},
        "spec_curve": spec_rows,
        "components": {
            k: {"b_mig": round(r.params["r_mig"], 2), "b_natural": round(r.params["r_natural"], 2),
                "lo_mig": round(r.conf_int(0.10).loc["r_mig", 0], 2), "hi_mig": round(r.conf_int(0.10).loc["r_mig", 1], 2)}
            for k, r in comps.items()
        },
        "validation": {k: [{**r, "corr": None if np.isnan(r["corr"]) else round(r["corr"], 3), "rmse": round(r["rmse"], 4)} for r in v.to_dict("records")] for k, v in validation.items()},
        "validation_first_origin": FIRST_TEST_YEAR - 1,
        "scenario_grid": grid,
        "unit_paths": unit_paths,
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
            v = base * np.exp(np.hstack([np.zeros((len(eff[city]), 1)), eff[city]])) / 1e3
            med = np.median(v, axis=0)
            if name.startswith("Growth"):
                lo, hi = np.percentile(v, [5, 95], axis=0)
                ax.fill_between(years, lo, hi, color=color, alpha=0.15, linewidth=0, label=f"{name} 90% range")
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
    p, z, z_nominal = load()

    peers = peer_counties(p, PEER_MAX_POP)
    county = fit_panel(p, "dlpop", peers)
    county_lag = fit_panel(p, "dlpop_lag", peers)
    all_fits = (fit_panel(p, "dlpop"), fit_panel(p, "dlpop_lag"))
    curve = spec_curve(p, DRAWS)
    sign = fit_panel(p, "dlpop_pos", extra_cols=("dlpop_neg",))
    comp_peer = fit_panel(p, "r_mig", peers, ("r_natural",))
    comp_all = fit_panel(p, "r_mig", None, ("r_natural",))
    placebo = fit_panel(p, "dlpop_lead")
    joint = fit_panel(p, "dlpop", extra_cols=("dlpop_lag", "dlpop_lead"))
    nevada_dlp = p[p["fips"] == NEVADA].set_index("year")["dlp"]
    cities = {city: fit_city(np.log(z[city]).diff(), nevada_dlp) for city in CITIES}

    lr, lr_lo, lr_hi = long_run(county, "dlpop", DRAWS)
    lr2, lr2_lo, lr2_hi = long_run(county_lag, "dlpop_lag", DRAWS)
    lr_all = curve.set_index("spec").loc["all/same_year", "lr"]

    hs = range(HORIZON)
    imps = {"peer": lp_draws(p, "dlpop", peers, hs, N_BOOT), "all": lp_draws(p, "dlpop", None, hs, N_BOOT)}
    cd = city_draws(cities)
    dof = pd.read_csv(DATA / "dof_p2a_county.csv", dtype={"fips": str})
    paths = {sc["label"]: np.full(HORIZON, sc["growth"]) for sc in scenarios(p)}
    paths["State projection (DOF P-2A)"] = dof_path(dof)
    effects = {name: population_effect(imps["peer"], cd, g) for name, g in paths.items()}
    effects_all = {name: population_effect(imps["all"], cd, g) for name, g in paths.items()}
    plot_history(p, z)
    plot_scenarios(z, {
        name if name.startswith("State") else f"{name} ({rate_text(paths[name][0])})": eff
        for name, eff in effects.items() if name.startswith(("Decline", "Recent", "Growth", "State"))
    })
    validation = {k: rolling_validation(p, "dlpop", sel, FIRST_TEST_YEAR) for k, sel in (("peer", peers), ("all", None))}
    export_site(p, z, z_nominal, county, county_lag, all_fits, cities, curve, {"peer": comp_peer, "all": comp_all}, imps, cd, dof, validation)

    rows = [
        f"| Scenario | 10-yr population change | City | {BASE_YEAR + HORIZON} value | Change vs. flat population | 90% range | All-counties change |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for (name, eff), eff_all, g in zip(effects.items(), effects_all.values(), paths.values()):
        label = name if name.startswith("State") else f"{name} ({g[0]:+.2%}/yr)"
        for city in CITIES:
            base = z.loc[BASE_YEAR, city]
            dollars = base * (np.exp(eff[city][:, -1]) - 1)
            lo, med, hi = np.percentile(dollars, [5, 50, 95])
            med_all = base * (np.exp(np.median(eff_all[city][:, -1])) - 1)
            rows.append(
                f"| {label} | {np.prod(1 + g) - 1:+.1%} | {city} | ${base + med:,.0f} | "
                f"{med / base:+.1%} (${med:+,.0f}) | ${lo:+,.0f} to ${hi:+,.0f} | {med_all / base:+.1%} (${med_all:+,.0f}) |"
            )

    lp_rows = ["| Years after the change | Peer counties | 90% interval | All counties | 90% interval |", "|---:|---:|---|---:|---|"]
    for j in range(HORIZON):
        cells = []
        for imp in imps.values():
            lo, med, hi = np.percentile(imp[:, j], [5, 50, 95])
            cells.append(f"{med:.2f} | {lo:.2f} to {hi:.2f}")
        lp_rows.append(f"| {j + 1} | {' | '.join(cells)} |")
    lp10_med, lp10_lo, lp10_hi = np.percentile(imps["peer"][:, HORIZON - 1], [50, 5, 95])
    lp_first, lp_last = (fit_local_projection(p, "dlpop", h, peers).sample for h in (0, HORIZON - 1))
    lp_window = {k: f"{d['year'].min()}-{d['year'].max()}" for k, d in (("first", lp_first), ("last", lp_last))}
    lp_counties = lp_first["fips"].nunique()
    # 2001-2015 lies inside every horizon's window, so all horizons share one set of origins.
    fixed, unfixed = ([fit_local_projection(p, "dlpop", h - 1, peers, o).params["dlpop"] for h in (1, 5, HORIZON)] for o in ((2001, 2015), None))
    # A sustained path raises log population by HORIZON * g and log price by g * sum(impulse), so this is its implied elasticity.
    sustained = {k: np.median(imp.sum(axis=1)) / HORIZON for k, imp in imps.items()}
    ld = {h: {k: long_difference(p, "dlpop", h, sel) for k, sel in (("peer", peers), ("all", None))} for h in (1, 5, HORIZON)}
    ld_rows = ["| Window (years) | Peer counties (s.e.) | Windows | All counties (s.e.) | Windows |", "|---:|---:|---:|---:|---:|"]
    for h, rs in ld.items():
        ld_rows.append(f"| {h} | " + " | ".join(f"{r['b']:.2f} ({r['se']:.2f}) | {r['n']:,}" for r in rs.values()) + " |")
    ld10_peer, lp_sum_peer = ld[HORIZON]["peer"]["b"], sustained["peer"]
    if ld10_peer < 0.5 * lp_sum_peer:
        ld_sentence = (f"The ten-year long difference ({ld10_peer:.2f}) is far below the ten-year path response ({lp_sum_peer:.2f}); "
                       "the two checks disagree, and the ten-year scenario values should be read with that in mind.")
    else:
        ld_sentence = f"The ten-year long difference ({ld10_peer:.2f}) is within a factor of two of the ten-year path response ({lp_sum_peer:.2f})."

    def val(k, h, m, col):
        return validation[k].set_index(["horizon", "model"]).loc[(h, m), col]

    zero_best = all(val(k, h, "zero", "rmse") < min(val(k, h, m, "rmse") for m in ("full", "no_population")) for k in validation for h in (1, 5))
    corr_full_5, corr_nopop_5 = val("peer", 5, "full", "corr"), val("peer", 5, "no_population", "corr")
    rank = f"population growth helps rank counties (peer correlation {corr_full_5:.2f} vs {corr_nopop_5:.2f} at five years)"
    val_sentence = (f"Predicting no deviation beats every model on RMSE at both horizons; {rank} but not predict the size of their price changes."
                    if zero_best else f"Some model beats predicting no deviation on RMSE; {rank}.")

    val_rows = ["| Years ahead | Model | Peer RMSE | Peer correlation | Peer pairs | All-county RMSE | All-county correlation | All-county pairs |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rp, ra in zip(validation["peer"].itertuples(), validation["all"].itertuples()):
        fc = lambda r: "n/a" if np.isnan(r.corr) else f"{r.corr:.2f}"
        val_rows.append(f"| {rp.horizon} | {rp.model} | {rp.rmse:.3f} | {fc(rp)} | {rp.n:,} | {ra.rmse:.3f} | {fc(ra)} | {ra.n:,} |")

    city_rows = ["| City | Pass-through of county price growth | 90% CI | R² | Years |", "|---|---:|---|---:|---|"]
    for city, res in cities.items():
        ci = res.conf_int(0.10).loc["county"]
        years = res.model.data.row_labels
        city_rows.append(f"| {city} | {res.params['county']:.2f} | {ci[0]:.2f} to {ci[1]:.2f} | {res.rsquared:.2f} | {years.min()}-{years.max()} |")

    curve_rows = ["| Specification | Counties | Population coefficient (s.e.) | Long-run elasticity | 90% interval | Observations |", "|---|---|---:|---:|---|---:|"]
    for r in curve.itertuples():
        curve_rows.append(
            f"| {r.spec} | {r.n_counties} | {r.b_pop:.3f} ({r.se:.3f}) | {r.lr:.2f} | {r.lr_lo:.2f} to {r.lr_hi:.2f} | {r.n_obs:,} |"
        )
    b_pos, b_neg = sign.params["dlpop_pos"], sign.params["dlpop_neg"]
    diff_p = float(sign.t_test("dlpop_pos = dlpop_neg").pvalue)
    sign_head = f"The sign split gives {b_pos:.2f} for growth and {b_neg:.2f} for decline"
    if diff_p >= 0.10:
        sign_sentence = f"{sign_head}, with wide intervals; the data do not reliably separate the two directions (difference p = {diff_p:.2f})."
    else:
        bigger = "declines" if b_neg > b_pos else "growth"
        sign_sentence = f"{sign_head}; {bigger} go{'' if bigger == 'declines' else 'es'} with larger price moves (difference p = {diff_p:.2f})."
    nev_pop = round(p[(p["fips"] == NEVADA) & (p["year"] == BASE_YEAR)]["population"].iloc[0], -3)

    city_table, scenario_table, curve_table = "\n".join(city_rows), "\n".join(rows), "\n".join(curve_rows)
    lp_table, ld_table, val_table = "\n".join(lp_rows), "\n".join(ld_rows), "\n".join(val_rows)
    report = f"""# Nevada County population and home values: model results

All prices are real {BASE_YEAR} dollars (CPI-U deflated). Generated by `model.py`.

## Stage 1: population and real home-price growth, peer counties

Dependent variable: annual log change in each county's real FHFA house price index.
Year fixed effects absorb statewide and national drivers such as mortgage rates.

Headline set: the {len(peers)} California counties with mean population under {PEER_MAX_POP:,}, the set that includes Nevada County ({nev_pop:,.0f}).
{county.sample['fips'].nunique()} of them have price data.
Large counties show a much larger association (see specification curve).

{coef_table(county)}

**Long-run elasticity of price to population, momentum model: {lr:.2f}** (90% interval {lr_lo:.2f} to {lr_hi:.2f}).
A county population that is 1% larger, sustained, goes with real home prices about {lr:.1f}% higher
once price momentum plays out.

The same model with last year's population growth in place of this year's, a second timing assumption:

{coef_table(county_lag)}

Long-run elasticity with lagged population: {lr2:.2f} (90% interval {lr2_lo:.2f} to {lr2_hi:.2f}).

## Local projections

Each row is a separate fit: the log real price change from the year before a population change through
the end of year n, where year 1 is the year of the change itself, on that year's population growth, with
the same controls and county and year fixed effects. Each fit also holds population growth in the later
years of the window fixed, so the value is the cumulative price change in percent for a one-year 1%
population change with later growth unchanged. Intervals come from {N_BOOT:,} county block bootstrap
resamples; every horizon is refit on the same resample. The table shows bootstrap medians.
Each horizon uses every origin year with data through the end of year n, so year 1 uses {lp_window["first"]} and year {HORIZON} uses {lp_window["last"]}.
The peer bootstrap resamples {lp_counties} counties, so its intervals may be too narrow.

{lp_table}

With origins fixed to 2001-2015 for every horizon, the peer responses are {fixed[0]:.2f}, {fixed[1]:.2f}, {fixed[2]:.2f} at years 1, 5 and {HORIZON}.
These are point estimates; with every origin year the point estimates are {unfixed[0]:.2f}, {unfixed[1]:.2f}, {unfixed[2]:.2f}.

**Price change after {HORIZON} years for 1% more people, peer counties, bootstrap median: {lp10_med:.2f}** (90% interval {lp10_lo:.2f} to {lp10_hi:.2f}).

## Long-difference check

Log real price change on log population change over non-overlapping windows, with window-end year
effects and standard errors clustered by county.
The {HORIZON}-year long-difference coefficient measures the same horizon without the momentum, income and permit controls, over two non-overlapping windows per county.

{ld_table}

Applied to a sustained {HORIZON}-year path, the sum of the year 1 to {HORIZON} responses implies {sustained["peer"]:.2f}
for peer counties and {sustained["all"]:.2f} for all counties, against {HORIZON}-year long-difference coefficients of
{ld[HORIZON]["peer"]["b"]:.2f} and {ld[HORIZON]["all"]["b"]:.2f}.

{ld_sentence}

## Out-of-sample check

For each origin year from {FIRST_TEST_YEAR - 1} on, the model is fit on earlier years only. It then predicts each county's
cumulative real price change over the next 1 and 5 years, using the county's actual later population, income and permit
data. Predictions and actual changes are both measured as deviations from the mean of the counties in that sample for that origin, because
year effects are unknown ahead of time. RMSE is in log points. The `zero` model predicts no deviation for every county.
Pairs are the pooled county-origin pairs.

{val_table}

{val_sentence}

This tests whether population growth helps rank counties' price growth ahead of time. It is not evidence of causation.

## Specification curve

The same model under different samples and timing choices. The all-county same-year elasticity is {lr_all:.2f}.
The peer-set same-year elasticity is {lr:.2f}.

{curve_table}

{sign_sentence}

## Migration and natural change

Population growth is split into net migration (domestic plus international) and natural change (births minus deaths),
each as a share of last year's population. Both enter one fit. The first table is the peer set, the second is all counties.

Peer counties:

{coef_table(comp_peer)}

All counties:

{coef_table(comp_all)}

Natural change (births minus deaths) does not respond to this year's prices. The migration coefficient is the association that matters for Nevada County, where natural change is negative.

The natural-change coefficient is large and negative. With county fixed effects it comes from years when a county's births minus deaths moved relative to that county's own average. The largest such moves are the 2020-2021 death surge, which was deepest in older counties while their prices rose. The coefficient reflects age structure and timing, not a price response. Do not read it as a lever.

## Placebo check

These fits use all counties.

Next year's population growth, which this year's prices cannot respond to, is included as a
regressor. If it predicts this year's price growth, the population coefficient may reflect shared
local shocks (jobs, amenities, remote work), anticipated arrivals, or migration that follows prices;
the data cannot tell these apart.

Next year's population growth on its own:

{coef_table(placebo)}

Lagged, same-year and next-year growth together:

{coef_table(joint)}

Neither timing choice is a bound on a causal effect. Both are associations.

## Stage 2: Nevada County to city pass-through

Annual log change in each city's real Zillow Home Value Index on the Nevada County change.

{city_table}

## Population scenarios, {BASE_YEAR + 1}-{BASE_YEAR + HORIZON}

Each scenario holds Nevada County population growth at a constant rate. Each year's growth is
combined with the local-projection response at every later horizon. Every other driver stays
where it was in {BASE_YEAR}, so the {BASE_YEAR + HORIZON} value is the {BASE_YEAR} value plus the population association only.
It is not a forecast of the market. Ranges combine {N_BOOT:,} county bootstrap resamples of the peer-county local projections with draws of the city pass-through coefficient.
The last column uses the local projections fit on all counties.

{scenario_table}

Charts: `history.png`, `scenarios.png`.
"""
    (OUT / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
