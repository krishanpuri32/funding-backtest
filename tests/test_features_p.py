"""P features, the planned-value rule and the missing-value rule, on hand-built fixtures."""
import numpy as np
import pandas as pd

from fb import features_p as fp
from fb import missing, planned


def _v0(**over):
    base = dict(nct_id="NCT1", v0_overall_status="NOT_YET_RECRUITING", v0_enrollment="120",
                v0_enrollment_type="ESTIMATED", v0_start="2010-01", v0_primary_completion="2012-01",
                start_date="2010-02-15")
    base.update(over)
    return base


def test_planned_ok_late_actual_blank():
    df = pd.DataFrame([
        _v0(nct_id="OK"),
        _v0(nct_id="LATE", v0_overall_status="COMPLETED", v0_enrollment_type="ACTUAL"),
        _v0(nct_id="ACT", v0_enrollment_type="ACTUAL"),
        _v0(nct_id="BLANK", v0_enrollment=None, v0_enrollment_type=None),
        _v0(nct_id="SUSP", v0_overall_status="SUSPENDED"),
    ])
    p = planned.planned_values(df).set_index("nct_id")
    assert p.loc["OK", "planned_enrollment"] == 120 and p.loc["OK", "planned_enrollment_reason"] == "ok"
    assert p.loc["LATE", "planned_enrollment_reason"] == "late"  # late takes precedence over actual
    assert p.loc["ACT", "planned_enrollment_reason"] == "actual" and np.isnan(p.loc["ACT", "planned_enrollment"])
    assert p.loc["BLANK", "planned_enrollment_reason"] == "blank"
    assert p.loc["SUSP", "registered_late"] and not p.loc["SUSP", "o34_eligible"]
    assert p.loc["OK", "o34_eligible"] and not p.loc["ACT", "o34_eligible"]
    # late / actual also blank the planned duration
    assert np.isnan(p.loc["LATE", "planned_duration_months"]) and p.loc["ACT", "planned_duration_reason"] == "actual"


def test_planned_duration_months_and_current_start_fallback():
    df = pd.DataFrame([_v0(nct_id="A"), _v0(nct_id="B", v0_start=None),
                       _v0(nct_id="C", v0_primary_completion=None),
                       _v0(nct_id="D", v0_primary_completion="2009-06")])
    p = planned.planned_values(df).set_index("nct_id")
    assert abs(p.loc["A", "planned_duration_months"] - 730 / 30.4375) < 1e-9  # 2010-01-01 -> 2012-01-01
    assert p.loc["A", "planned_start_source"] == "v0"
    assert p.loc["B", "planned_start_source"] == "current"
    assert abs(p.loc["B", "planned_duration_months"] - (pd.Timestamp("2012-01-01") - pd.Timestamp("2010-02-15")).days / 30.4375) < 1e-9
    assert p.loc["C", "planned_duration_reason"] == "blank"
    assert p.loc["D", "planned_duration_reason"] == "nonpositive" and np.isnan(p.loc["D", "planned_duration_months"])


def test_registry_date_parser():
    s = planned.parse_registry_date(pd.Series(["2010-03", "2010-03-17", "", None, "March 2010"]))
    assert s.tolist()[:2] == [pd.Timestamp("2010-03-01"), pd.Timestamp("2010-03-17")]
    assert s.iloc[2:].isna().all()


def test_intervention_collapse():
    iv = pd.DataFrame({"nct_id": ["A", "A", "B", "B", "C", "D", "E"],
                       "intervention_type": ["DRUG", "OTHER", "DRUG", "BEHAVIORAL", "OTHER", "RADIATION", "DIETARY_SUPPLEMENT"]})
    got = fp.collapse_intervention(iv, ["A", "B", "C", "D", "E", "F"]).tolist()
    assert got[:5] == ["drug", "multiple", "other", "other", "dietary"]
    assert pd.isna(got[5])


def test_allocation_and_masking():
    assert fp.collapse_allocation(pd.Series(["RANDOMIZED", "NON_RANDOMIZED", "NA", None])).tolist()[:3] == ["randomized", "non-randomized", "N/A"]
    m = fp.collapse_masking(pd.Series(["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", None])).tolist()
    assert m[:5] == ["none", "single", "double+", "double+", "double+"] and pd.isna(m[5])


def test_facility_and_country_counts():
    fac = pd.DataFrame({"nct_id": ["A", "A", "A", "B"]})
    assert fp.count_facilities(fac, ["A", "B", "C"]).tolist() == [3, 1, 0]
    co = pd.DataFrame({"nct_id": ["A", "A", "A", "B"], "name": ["US", "US", "Canada", "France"], "removed": ["f", "f", "t", "f"]})
    assert fp.count_countries(co, ["A", "B", "C"]).tolist() == [1, 1, 0]


def test_missing_rule_indicator_plus_median_and_level():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0], "c": ["a", None, "", "b"]})
    out, med = missing.fill_numeric(df, ["x"])
    assert med["x"] == 3.0 and out["x"].tolist() == [1.0, 3.0, 3.0, 5.0] and out["x_missing"].tolist() == [0, 1, 0, 0]
    out2, _ = missing.fill_numeric(df, ["x"], {"x": 10.0})
    assert out2["x"].tolist()[1] == 10.0
    assert missing.fill_categorical(df, ["c"])["c"].tolist() == ["a", "missing", "missing", "b"]
    cov = missing.coverage_table(df, ["x"], ["c"]).set_index("feature")
    assert cov.loc["x", "n_non_missing"] == 3 and cov.loc["c", "n_non_missing"] == 2
