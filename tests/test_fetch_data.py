from pathlib import Path

import fetch_data

FIX = Path(__file__).parent / "fixtures" / "dof" / "P2A_excerpt.xlsx"
CENSUS_FIX = Path(__file__).parent / "fixtures" / "census"


def test_dof_projection_parses_nevada():
    df = fetch_data.parse_dof_projection(FIX)
    nev = df[(df.fips == "06057")].set_index("year")["population"]
    assert nev.loc[2020] == 102142 and nev.loc[2035] == 97387
    assert set(df.columns) == {"fips", "year", "population"}


def test_dof_vintage_reads_about_sheet():
    assert "Vintage 2026 Release" in fetch_data.dof_vintage(FIX)


def test_components_2025_nevada_2021():
    df = fetch_data.parse_components(CENSUS_FIX / "co-est2025-alldata-excerpt.csv", 2020, 2025)
    assert list(df.columns) == ["fips", "year", "natural_change", "net_migration"]
    row = df[(df.fips == "06057") & (df.year == 2021)].iloc[0]
    assert (row.natural_change, row.net_migration) == (-422, 1730 + 2)
    assert len(df) == 12


def test_components_older_vintages_use_naturalinc():
    old = fetch_data.parse_components(CENSUS_FIX / "co-est2009-alldata-excerpt.csv", 2000, 2009)
    mid = fetch_data.parse_components(CENSUS_FIX / "co-est2020-alldata-excerpt.csv", 2010, 2020)
    assert set(old.year) == set(range(2000, 2010)) and set(mid.year) == set(range(2010, 2021))
    assert set(old.fips) == {"06057", "06061"}
    nev = old[(old.fips == "06057") & (old.year == 2009)].iloc[0]
    assert (nev.natural_change, nev.net_migration) == (-58, 496)
