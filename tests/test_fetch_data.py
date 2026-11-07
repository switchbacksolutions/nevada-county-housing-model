from pathlib import Path

import fetch_data

FIX = Path(__file__).parent / "fixtures" / "dof" / "P2A_excerpt.xlsx"


def test_dof_projection_parses_nevada():
    df = fetch_data.parse_dof_projection(FIX)
    nev = df[(df.fips == "06057")].set_index("year")["population"]
    assert nev.loc[2020] == 102142 and nev.loc[2035] == 97387
    assert set(df.columns) == {"fips", "year", "population"}


def test_dof_vintage_reads_about_sheet():
    assert "Vintage 2026 Release" in fetch_data.dof_vintage(FIX)
