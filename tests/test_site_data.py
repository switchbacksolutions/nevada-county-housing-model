import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = re.compile(r"cannot have caused|caused|effect of|lower bound|conservative|cautious|between the two", re.I)


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
    grid_growth = {r["growth"] for r in data["scenario_grid"]}
    assert max(grid_growth) == 2.0 and min(grid_growth) == -1.0
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


def test_explained_is_small_share():
    data = _data()
    for city in ("Grass Valley", "Nevada City"):
        e = data["explained"][city]
        assert abs(e["population_pct"]) < abs(e["actual_pct"])
        assert e["population_lo"] <= e["population_pct"] <= e["population_hi"]
        assert abs(e["other_pct"] - (e["actual_pct"] - e["population_pct"])) < 0.01
