import pandas as pd

from fb import cpi as cpimod
from fb.dollars import trial_dollars

CPI = {2010: 218.056, 2011: 224.939, 2020: 258.856}


def _projects():
    return pd.DataFrame([
        # parent rows for P1 (overlap 2010, 2011), one sub-project row in 2010 that must be ignored
        {"CORE_PROJECT_NUM": "P1", "SUBPROJECT_ID": None, "TOTAL_COST": "100", "FY_FILE": 2010, "budget_start": "2010-01-01", "budget_end": "2010-12-31"},
        {"CORE_PROJECT_NUM": "P1", "SUBPROJECT_ID": "5001", "TOTAL_COST": "999999", "FY_FILE": 2010, "budget_start": "2010-01-01", "budget_end": "2010-12-31"},
        {"CORE_PROJECT_NUM": "P1", "SUBPROJECT_ID": None, "TOTAL_COST": "200", "FY_FILE": 2011, "budget_start": "2011-01-01", "budget_end": "2011-12-31"},
        # P1 parent row outside the trial window
        {"CORE_PROJECT_NUM": "P1", "SUBPROJECT_ID": None, "TOTAL_COST": "5000", "FY_FILE": 2014, "budget_start": "2014-01-01", "budget_end": "2014-12-31"},
        # P2 parent row overlapping, cost null
        {"CORE_PROJECT_NUM": "P2", "SUBPROJECT_ID": None, "TOTAL_COST": None, "FY_FILE": 2011, "budget_start": "2011-06-01", "budget_end": "2012-05-31"},
        # P9 not linked to any trial
        {"CORE_PROJECT_NUM": "P9", "SUBPROJECT_ID": None, "TOTAL_COST": "7", "FY_FILE": 2011, "budget_start": "2011-01-01", "budget_end": "2011-12-31"},
    ])


def _trials():
    return pd.DataFrame({"nct_id": ["NCTA", "NCTB"], "trial_start": ["2010-03-01", "2010-03-01"], "trial_end": ["2012-02-28", "2012-02-28"]})


def test_parent_only_sum_with_subproject_row_present():
    pairs = pd.DataFrame({"nct_id": ["NCTA", "NCTA", "NCTB"], "core_project_num": ["P1", "P2", "P1"]})
    for c in ("budget_start", "budget_end"):
        pass
    out = trial_dollars(pairs, _projects(), _trials(), cpi=None).set_index("nct_id")
    assert out.loc["NCTA", "total_cost_nominal"] == 300          # 100 + 200; sub-project 999999 and 2014 row excluded; null -> 0
    assert out.loc["NCTA", "n_fy_overlap"] == 2                  # FY2010, FY2011 (P2's 2011 row shares the year)
    assert out.loc["NCTA", "n_fy_rows_overlap"] == 3
    assert out.loc["NCTA", "n_fy_rows_cost_null"] == 1
    assert out.loc["NCTA", "n_projects_overlap"] == 2
    assert out.loc["NCTB", "total_cost_nominal"] == 300
    assert out.loc["NCTB", "n_projects_overlap"] == 1
    assert pd.isna(out.loc["NCTA", "total_cost_base"])            # nominal only when no CPI


def test_cpi_adjustment_to_2020_dollars():
    pairs = pd.DataFrame({"nct_id": ["NCTA"], "core_project_num": ["P1"]})
    out = trial_dollars(pairs, _projects(), _trials(), cpi=CPI).set_index("nct_id")
    expected = 100 * 258.856 / 218.056 + 200 * 258.856 / 224.939
    assert abs(out.loc["NCTA", "total_cost_base"] - expected) < 1e-9
    assert out.loc["NCTA", "total_cost_nominal"] == 300


def test_trial_with_no_overlapping_parent_row_gets_zero():
    pairs = pd.DataFrame({"nct_id": ["NCTC"], "core_project_num": ["P1"]})
    trials = pd.DataFrame({"nct_id": ["NCTC"], "trial_start": ["2016-01-01"], "trial_end": ["2017-01-01"]})
    out = trial_dollars(pairs, _projects(), trials, cpi=CPI).set_index("nct_id")
    assert out.loc["NCTC", "total_cost_base"] == 0 and out.loc["NCTC", "n_fy_overlap"] == 0


def test_cpi_parse_and_deflator(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("observation_date,CPIAUCSL\n2019-01-01,255.653\n2020-01-01,258.856\n2026-01-01,\n")
    d = cpimod.parse_cpi_csv(p)
    assert d == {2019: 255.653, 2020: 258.856}
    f = cpimod.deflator(d, pd.Series([2019, 2020, 1999]), base_year=2020)
    assert abs(f.iloc[0] - 258.856 / 255.653) < 1e-12 and f.iloc[1] == 1.0 and pd.isna(f.iloc[2])
