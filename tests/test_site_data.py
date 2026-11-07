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
    assert len(data["spec_curve"]) == 10


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
