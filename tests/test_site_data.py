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
