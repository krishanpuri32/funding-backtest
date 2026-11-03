"""F and F₂ features on a hand-built 3-trial / 6-row ExPORTER table.

Projects (parent rows, CPI flat at 100 so 2020 dollars = nominal unless stated):
  P1 r1 FY2010 type 1 SY1 budget 2009-12-01..2010-11-30 cost 100 ind 30 PE 2013-11-30
     r2 FY2011 type 5 SY2 budget 2010-12-01..2011-11-30 cost  80 ind 20 PE 2013-11-30
     r3 FY2013 type 5 SY4 budget 2012-12-01..2013-11-30 cost  60 ind 15 PE 2014-11-30  <- project end moved: NCE
     r4 FY2013 type 3 S1  budget 2012-12-01..2013-11-30 cost  20 ind --  PE 2013-11-30  <- supplement
  P2 r5 FY2011 type 1 SY1 budget 2011-01-01..2011-12-31 cost  50 ind 10 PE 2014-12-31
     r6 FY2012 type 5 SY2 budget 2012-01-01..2012-12-31 cost  50 ind 10 PE 2014-12-31
Trials:
  T1 P1      start 2010-01-01 end 2015-06-30, completed, outcome 2015-06-30, planned enrollment 13
  T2 P1+P2   start 2010-01-01 end 2014-12-31, terminated, outcome 2012-06-15 (dies mid-window)
  T3 P3      start 2011-03-01 end 2013-12-31, P3 has no ExPORTER row -> all missing
Expected values are worked out by hand in each test.
"""
import math

import numpy as np
import pandas as pd
import pytest

from fb import features_f as ff

CPI = {y: 100.0 for y in range(2000, 2026)}


def _projects():
    rows = [
        # core, FY, type, SY, suffix, budget_start, budget_end, cost, indirect, project_end
        ("P1", 2010, "1", 1, None, "2009-12-01", "2010-11-30", 100, 30, "2013-11-30"),
        ("P1", 2011, "5", 2, None, "2010-12-01", "2011-11-30", 80, 20, "2013-11-30"),
        ("P1", 2013, "5", 4, None, "2012-12-01", "2013-11-30", 60, 15, "2014-11-30"),
        ("P1", 2013, "3", 4, "S1", "2012-12-01", "2013-11-30", 20, None, "2013-11-30"),
        ("P2", 2011, "1", 1, None, "2011-01-01", "2011-12-31", 50, 10, "2014-12-31"),
        ("P2", 2012, "5", 2, None, "2012-01-01", "2012-12-31", 50, 10, "2014-12-31"),
    ]
    df = pd.DataFrame(rows, columns=["CORE_PROJECT_NUM", "FY_FILE", "APPLICATION_TYPE", "SUPPORT_YEAR", "SUFFIX",
                                     "BUDGET_START", "BUDGET_END", "TOTAL_COST", "INDIRECT_COST_AMT", "PROJECT_END"])
    df["DIRECT_COST_AMT"] = [70, 60, 45, 20, 40, 40]  # direct + indirect = total on every row with both
    df["PROJECT_START"] = np.where(df["CORE_PROJECT_NUM"].eq("P1"), "2009-12-01", "2011-01-01")
    df["SUBPROJECT_ID"] = None
    for c in ["TOTAL_COST", "INDIRECT_COST_AMT", "DIRECT_COST_AMT", "SUPPORT_YEAR"]:
        df[c] = df[c].astype("object").where(df[c].notna(), None).map(lambda v: None if v is None else str(int(v)))
    return df


PAIRS = pd.DataFrame({"nct_id": ["T1", "T2", "T2", "T3"], "core_project_num": ["P1", "P1", "P2", "P3"]})
TRIALS = pd.DataFrame({
    "nct_id": ["T1", "T2", "T3"],
    "trial_start": pd.to_datetime(["2010-01-01", "2010-01-01", "2011-03-01"]),
    "trial_end": pd.to_datetime(["2015-06-30", "2014-12-31", "2013-12-31"]),
    "outcome_date": pd.to_datetime(["2015-06-30", "2012-06-15", "2013-12-31"]),
})
PLANNED = pd.Series({"T1": 13.0, "T2": np.nan, "T3": 40.0})


@pytest.fixture(scope="module")
def feats():
    rows = ff.prepare_rows(_projects(), CPI)
    return ff.compute_features(PAIRS, rows, TRIALS, PLANNED, "F").set_index("nct_id")


@pytest.fixture(scope="module")
def feats2():
    rows = ff.prepare_rows(_projects(), CPI)
    return ff.compute_features(PAIRS, rows, TRIALS, PLANNED, "F2").set_index("nct_id")


def test_t1_full_window(feats):
    t = feats.loc["T1"]
    # annual: 2010 100, 2011 80, 2012 0, 2013 60+20 = 80; total 260
    assert t["F_total"] == pytest.approx(math.log1p(260))
    assert t["F_years"] == 3
    # OLS slope of [100, 80, 0, 80] on [0..3]: Sxy = -70, Sxx = 5 -> -14; / 260
    assert t["F_shape"] == pytest.approx(-14 / 260)
    assert t["F_yoy_min"] == pytest.approx(1.0)        # 80 -> 0 in FY2012
    # gap window FY2010..FY(2015-06-30)-2 = FY2013; FY2012 uncovered
    assert t["F_gap"] == 1 and t["F_gap_count"] == 1 and t["f_gap_window_n"] == 4
    assert t["F_supp"] == 1                              # r4
    assert t["F_nce"] == 1                               # r3 project end 2014-11-30 > 2013-11-30
    assert t["F_indirect_ratio"] == pytest.approx((0.30 + 0.25 + 0.25) / 3)  # FY2013 uses r3 only (r4 ind null)
    assert t["F_per_enrollee"] == pytest.approx(math.log1p(260 / 13))
    assert t["F_noncompeting_share"] == pytest.approx(2 / 4)
    assert t["f_rows_reason"] == "ok" and t["f_n_rows"] == 4


def test_t2_dies_mid_window_sees_no_post_death_money(feats):
    t = feats.loc["T2"]
    # kept: r1 (end 2010-11-30), r2 (2011-11-30), r5 (2011-12-31); r3, r4, r6 end after 2012-06-15
    assert t["f_n_rows"] == 3 and t["f_n_rows_overlap"] == 6
    assert t["F_total"] == pytest.approx(math.log1p(100 + 80 + 50))
    assert t["F_years"] == 2
    assert t["F_shape"] == pytest.approx(30 / 230)       # [100, 130]
    assert t["F_yoy_min"] == 0                           # no cut
    assert t["F_gap_count"] == 0 and t["f_gap_window_n"] == 1  # window FY2010..FY2012-2
    assert t["F_supp"] == 0 and t["F_nce"] == 0          # the r3 extension and r4 supplement are post-death
    assert t["F_indirect_ratio"] == pytest.approx((0.30 + 30 / 130) / 2)
    assert np.isnan(t["F_per_enrollee"])                 # no planned enrollment
    assert t["F_noncompeting_share"] == pytest.approx(1 / 3)


def test_t3_no_rows_is_missing_not_zero(feats):
    t = feats.loc["T3"]
    assert t["f_rows_reason"] == "no_overlap"
    assert all(np.isnan(t[c]) for c in ff.F_COLUMNS)


def test_f2_uses_first_two_fys(feats2):
    t = feats2.loc["T1"]  # f0 = 2010 -> r1, r2 only
    assert "F_nce" not in feats2.columns
    assert t["f_n_rows"] == 2
    assert t["F_total"] == pytest.approx(math.log1p(180))
    assert t["F_years"] == 2
    assert t["F_shape"] == pytest.approx(-20 / 180)
    assert t["F_yoy_min"] == pytest.approx(0.2)
    assert t["F_gap_count"] == 0 and t["f_gap_window_n"] == 2
    assert t["F_supp"] == 0
    assert t["F_indirect_ratio"] == pytest.approx((0.30 + 0.25) / 2)
    assert t["F_noncompeting_share"] == pytest.approx(0.5)


def test_f2_five_fy_trial_sees_fys_1_2_only():
    proj = pd.DataFrame({
        "CORE_PROJECT_NUM": "Q", "SUBPROJECT_ID": None, "FY_FILE": [2010, 2011, 2012, 2013, 2014],
        "APPLICATION_TYPE": ["1", "5", "5", "5", "5"], "SUFFIX": None, "SUPPORT_YEAR": ["1", "2", "3", "4", "5"],
        "PROJECT_START": "2009-10-01", "PROJECT_END": "2014-09-30",
        "BUDGET_START": [f"{y - 1}-10-01" for y in range(2010, 2015)], "BUDGET_END": [f"{y}-09-30" for y in range(2010, 2015)],
        "TOTAL_COST": ["10", "20", "30", "40", "50"], "INDIRECT_COST_AMT": None,
    })
    rows = ff.prepare_rows(proj, CPI)
    tr = pd.DataFrame({"nct_id": ["X"], "trial_start": pd.to_datetime(["2009-10-01"]),
                       "trial_end": pd.to_datetime(["2015-06-30"]), "outcome_date": pd.to_datetime(["2015-06-30"])})
    pairs = pd.DataFrame({"nct_id": ["X"], "core_project_num": ["Q"]})
    f2 = ff.compute_features(pairs, rows, tr, None, "F2").iloc[0]
    f = ff.compute_features(pairs, rows, tr, None, "F").iloc[0]
    assert f2["F_total"] == pytest.approx(math.log1p(30)) and f2["F_years"] == 2
    assert f["F_total"] == pytest.approx(math.log1p(150)) and f["F_years"] == 5


def test_cpi_deflation_uses_row_fy():
    proj = _projects()[lambda d: d["CORE_PROJECT_NUM"].eq("P2")]
    cpi = dict(CPI)
    cpi[2011] = 50.0  # FY2011 dollars are worth double in 2020 terms
    rows = ff.prepare_rows(proj, cpi)
    assert rows.loc[rows["FY_FILE"].eq(2011), "cost_base"].iloc[0] == pytest.approx(100.0)
    assert rows.loc[rows["FY_FILE"].eq(2012), "cost_base"].iloc[0] == pytest.approx(50.0)


def test_null_cost_row_counts_as_award_action_but_not_dollars():
    proj = _projects()[lambda d: d["CORE_PROJECT_NUM"].eq("P2")].copy()
    proj["TOTAL_COST"] = None
    rows = ff.prepare_rows(proj, CPI)
    tr = pd.DataFrame({"nct_id": ["V"], "trial_start": pd.to_datetime(["2011-01-01"]),
                       "trial_end": pd.to_datetime(["2016-01-01"]), "outcome_date": pd.to_datetime(["2016-01-01"])})
    f = ff.compute_features(pd.DataFrame({"nct_id": ["V"], "core_project_num": ["P2"]}), rows, tr, None, "F").iloc[0]
    assert f["f_rows_reason"] == "cost_unknown" and f["f_n_rows_cost_null"] == 2
    assert np.isnan(f["F_total"]) and f["F_years"] == 2 and f["F_noncompeting_share"] == 0.5


def test_nce_zero_obligation_beyond_term():
    g = pd.DataFrame({"app_type": ["5"], "project_end": pd.to_datetime(["2013-11-30"]),
                      "seg_orig_end": pd.to_datetime(["2013-11-30"]), "support_year": [6.0],
                      "seg_term_years": [5.0], "cost": [0.0]})
    assert ff.f_nce(g) == 1
    g["cost"] = [np.nan]  # unknown amount is not evidence of a zero obligation
    assert ff.f_nce(g) == 0


def test_indirect_ratio_uses_direct_plus_indirect_not_total_cost():
    # ExPORTER quirk: TOTAL_COST 25,001 on a budget of 173,707 direct + 84,248 indirect
    g = pd.DataFrame({"FY_FILE": [2006, 2007, 2005], "indirect": [84248.0, 30.0, 10.0],
                      "direct": [173707.0, 70.0, 90.0], "cost": [25001.0, 100.0, 100.0]})
    assert ff.f_indirect_ratio(g) == pytest.approx((84248 / 257955 + 0.30) / 2)  # FY2005 excluded
    assert np.isnan(ff.f_indirect_ratio(g[g["FY_FILE"] < 2006]))


def test_multi_year_budget_covers_later_fy():
    g = pd.DataFrame({"FY_FILE": [2010], "budget_end": pd.to_datetime(["2012-09-30"])})
    assert ff.f_gap(g, [2010, 2011, 2012]) == (0.0, 0.0)
    assert ff.f_gap(g, [2010, 2013]) == (1.0, 1.0)
    assert all(np.isnan(v) for v in ff.f_gap(g, []))
