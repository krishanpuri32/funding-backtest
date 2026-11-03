from fb import exporter


def test_resolve_columns_maps_aliases_and_reports_unmapped():
    raw = ["APPLICATION_ID", "ACTIVITY", "IC_NAME", "ADMINISTERING_IC", "CORE_PROJECT_NUM",
           "FY", "TOTAL_COST", "BUDGET_START", "BUDGET_END", "WEIRD_COL"]
    mapping, unmapped = exporter.resolve_columns(raw)
    assert mapping["ADMINISTERING_IC"] == "ADMINISTERING_IC"
    assert mapping["CORE_PROJECT_NUM"] == "CORE_PROJECT_NUM"
    assert "IC_NAME" in unmapped and "WEIRD_COL" in unmapped
    # an old spelling resolves to the canonical name
    mapping2, _ = exporter.resolve_columns(["Fiscal_Year", "core_project_number", "Activity_Code"])
    assert set(mapping2.values()) == {"FY", "CORE_PROJECT_NUM", "ACTIVITY"}


def test_parse_exporter_date_handles_both_formats():
    import pandas as pd
    s = pd.Series(["09/22/2000", "2006-06-20", "", None, "junk", " 2015-10-01 "])
    out = exporter.parse_exporter_date(s)
    assert out.iloc[0] == pd.Timestamp("2000-09-22")
    assert out.iloc[1] == pd.Timestamp("2006-06-20")
    assert out.iloc[5] == pd.Timestamp("2015-10-01")
    assert out.iloc[2:5].isna().all()
