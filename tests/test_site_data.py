import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = re.compile(
    r"cannot have caused|caused|effect of|lower bound|conservative|cautious|between the two"
    r"|changes? (typical )?home values|\bcause\b|population effect|after people arrive",
    re.I,
)


def test_no_bound_language():
    for rel in ("output/results.md", "site/index.html", "README.md", "site/app.js"):
        text = (ROOT / rel).read_text()
        assert not BANNED.search(text), f"{rel}: {BANNED.search(text).group(0)!r}"


def test_results_has_placebo_section():
    text = (ROOT / "output/results.md").read_text()
    assert "## Placebo check" in text
    assert "Next year's population growth" in text


def test_results_and_data_json_agree_on_headline():
    import json

    data = json.loads((ROOT / "site/data.json").read_text())
    text = (ROOT / "output/results.md").read_text()
    e = data["headline"]
    assert f"{e['point']:.2f}" in text and f"{e['hi']:.2f}" in text
    assert len(data["spec_curve"]) == 12


def _data():
    import json

    return json.loads((ROOT / "site/data.json").read_text())


def test_support_and_presets_are_exported():
    data = _data()
    sup = data["support"]
    assert sup["lo"] < 0 < sup["hi"] < 2
    assert [p["key"] for p in data["presets"]] == ["decline", "flat", "recent", "longrun", "growth"]
    assert "scenario_grid" not in data and set(data["unit_quantiles"]) == {"peer", "all"}
    assert all(p["growth"] <= 2 for p in data["presets"])


def test_dof_path_is_exported():
    dof = _data()["scenario_paths"]["dof"]
    assert len(dof["growth"]) == 10 and sum(dof["growth"]) < 0
    for model in ("peer", "all"):
        for city in ("Grass Valley", "Nevada City"):
            row = dof[model][city]
            assert len(row["median"]) == 11 and row["p5"][-1] <= row["median"][-1] <= row["p95"][-1]


def test_translator_fields_present():
    data = json.loads((ROOT / "site/data.json").read_text())
    assert 2.0 < data["persons_per_household"] < 3.0
    assert data["permits_latest"]["count"] > 0 and data["population_latest"]["count"] > 90_000


def test_events_and_nominal_exported():
    data = json.loads((ROOT / "site/data.json").read_text())
    kinds = {e["kind"] for e in data["events"]}
    assert {"market", "fire", "insurance"} <= kinds
    assert all(e["year"] >= 2001 for e in data["events"])
    assert data["history"][0]["Grass Valley_nominal"] < data["history"][0]["Grass Valley"]


def test_validation_exported_and_stated_without_causal_claim():
    data = json.loads((ROOT / "site/data.json").read_text())
    assert len(data["validation"]["peer"]) == len(data["validation"]["all"]) == 6
    sentence = "This tests whether population growth helps rank counties' price growth ahead of time. It is not evidence of causation."
    assert sentence in (ROOT / "output/results.md").read_text()
    assert sentence in (ROOT / "site/index.html").read_text()


def test_explained_population_part_within_actual():
    data = _data()
    for city in ("Grass Valley", "Nevada City"):
        e = data["explained"][city]
        assert abs(e["population_pct"]) < abs(e["actual_pct"])
        assert e["population_lo"] <= e["population_pct"] <= e["population_hi"]
        import math

        rest = 100 * math.expm1(math.log1p(e["actual_pct"] / 100) - math.log1p(e["population_pct"] / 100))
        assert abs(e["other_pct"] - rest) < 0.01


def test_unit_quantiles_match_path_effect():
    import numpy as np

    data = _data()
    for model in ("peer", "all"):
        for city in data["cities"]:
            q = {k: np.array(v) for k, v in data["unit_quantiles"][model][city].items()}
            assert all(v.shape == (data["horizon"],) for v in q.values())
            assert (q["p5"] <= q["p50"]).all() and (q["p50"] <= q["p95"]).all()
    # The site's +1%/yr readout and the results.md Growth row come from the same 1,000 draws.
    text = (ROOT / "output/results.md").read_text()
    row = re.search(r"^\| Growth \(\+1\.00%/yr\) \| [^|]+ \| Grass Valley \| \$([\d,]+) \|", text, re.M)
    city = "Grass Valley"
    site = data["base_values"][city] * np.exp(data["unit_quantiles"]["peer"][city]["p50"][-1])
    assert abs(float(row.group(1).replace(",", "")) - site) < 200
    dof = data["scenario_paths"]["dof"]
    assert all(len(dof[m][c][k]) == data["horizon"] + 1 for m in ("peer", "all") for c in data["cities"] for k in ("p5", "median", "p95"))


def test_unit_paths_shape():
    import math

    data = _data()
    for model in ("peer", "all"):
        for city in data["cities"]:
            rows = data["unit_paths"][model][city]
            assert len(rows) == 200 and all(len(r) == data["horizon"] for r in rows)
            assert all(math.isfinite(v) for r in rows for v in r)


def test_dof_growth_path_alignment():
    import csv

    data = _data()
    with (ROOT / "data/dof_p2a_county.csv").open() as f:
        pop = {int(r["year"]): float(r["population"]) for r in csv.DictReader(f) if r["fips"] == "06057"}
    dof = data["scenario_paths"]["dof"]
    assert abs(dof["growth"][0] - (pop[data["base_year"] + 1] / pop[data["base_year"]] - 1)) < 1e-6
    assert len(dof["growth"]) == data["horizon"]
