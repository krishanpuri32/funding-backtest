"""Design matrix: one-hot over training levels with the missing level, training medians,
log planned duration (A4c), F indicators only on request, no label column, and the
population rules (landmark A4b, A2)."""
import math

import numpy as np
import pandas as pd
import pytest

from fb import design, features_f, features_p, models


def _p_raw():
    return pd.DataFrame({
        "nct_id": ["T1", "T2", "T3"],
        "phase_group": ["2", None, "3"],
        "condition_group": ["Neoplasms", "Neoplasms", "Infections"],
        "intervention_type": ["drug", "behavioral", "drug"],
        "allocation": ["randomized", "randomized", None],
        "masking": ["none", "double+", "none"],
        "sponsor_class": ["OTHER", "NIH", "OTHER"],
        "activity_class": ["R", "U", "R"],
        "ic_group": ["CA", "other", "CA"],
        "log_n_facilities": [0.0, 1.0, 2.0],
        "n_countries": [1.0, 1.0, 2.0],
        "log_planned_enrollment": [4.0, np.nan, 5.0],
        "planned_duration_months": [11.0, 23.0, np.nan],
        "start_year": [2010.0, 2011.0, 2012.0],
        "n_attributable_projects": [1.0, 2.0, 1.0],
        "o1": [1, 0, 0],  # a label column on the input must never reach X
    })


P_MED = {"log_n_facilities": 1.0, "n_countries": 1.0, "log_planned_enrollment": 4.7, "planned_duration_months": 31.0,
         "start_year": 2012.0, "n_attributable_projects": 1.0}


def _f_raw():
    f = pd.DataFrame({"nct_id": ["T1", "T2", "T3"]})
    for c in features_f.F_COLUMNS:
        f[c] = [1.0, np.nan, 3.0]
    f["f_rows_reason"] = ["ok", "all_cut", "ok"]
    f["f_n_rows"] = [3, 0, 2]
    return f


F_MED = {c: 10.0 for c in features_f.F_COLUMNS}


def test_p_only_matrix():
    p = _p_raw()
    lv = design.p_levels(p)
    assert lv["phase_group"] == ["2", "3", "missing"]
    X = design.design_matrix(["T1", "T2", "T3"], p, P_MED, lv)
    assert "o1" not in X.columns and not (design.FORBIDDEN & set(X.columns))
    assert X["log_planned_duration"].tolist() == pytest.approx([math.log1p(11), math.log1p(23), math.log1p(31.0)])
    assert X["planned_duration_months_missing"].tolist() == [0, 0, 1]
    assert X["log_planned_enrollment"].tolist() == [4.0, 4.7, 5.0]     # training median, not recomputed
    assert X["phase_group=missing"].tolist() == [0, 1, 0]
    assert X["allocation=missing"].tolist() == [0, 0, 1]
    assert not any(c.startswith("F_") for c in X.columns)
    # every row has exactly one level per categorical
    for c in features_p.P_CATEGORICAL:
        assert (X[[k for k in X.columns if k.startswith(c + "=")]].sum(axis=1) == 1).all()


def test_unseen_level_is_all_zero_and_order_follows_ids():
    lv = design.p_levels(_p_raw().iloc[:2])            # training levels without 'Infections'
    X = design.design_matrix(["T3", "T1"], _p_raw(), P_MED, lv)
    cg = [k for k in X.columns if k.startswith("condition_group=")]
    assert cg == ["condition_group=Neoplasms"]
    assert X[cg[0]].tolist() == [0, 1]
    assert X["start_year"].tolist() == [2012.0, 2010.0]


def test_f_block_with_and_without_indicators():
    p, f = _p_raw(), _f_raw()
    lv = design.p_levels(p)
    X = design.design_matrix(["T1", "T2", "T3"], p, P_MED, lv, f_raw=f, f_medians=F_MED, variant="F")
    assert X["F_total"].tolist() == [1.0, 10.0, 3.0]
    assert not any(c.endswith("_missing") and c.startswith("F_") for c in X.columns)
    assert "f_n_rows" not in X.columns and "f_rows_reason" not in X.columns
    X2 = design.design_matrix(["T1", "T2", "T3"], p, P_MED, lv, f_raw=f, f_medians=F_MED, variant="F2", f_missing=True)
    assert "F_nce" not in X2.columns and X2["F_total_missing"].tolist() == [0, 1, 0]
    assert design.f_block(X2) == [c for c in X2.columns if c.startswith("F_")]


def test_forbidden_guard():
    assert "o1" in design.FORBIDDEN and "f_rows_reason" in design.FORBIDDEN


def test_population_members_landmark_and_a2():
    lab = pd.DataFrame({"nct_id": ["T1", "T2", "T3", "T4"], "withdrawn_zero_enroll": [False, False, True, False]})
    f = pd.DataFrame({"nct_id": ["T1", "T2", "T3"], "f_rows_reason": ["ok", "all_cut", "cost_unknown"]})
    W = {"labels": lab, "F_n20": f}
    assert models.members(W, models.POPULATIONS["primary"]) == ["T1", "T3"]
    assert models.members(W, models.POPULATIONS["S1"]) == ["T1", "T2", "T3"]
    assert models.members(W, models.POPULATIONS["A2"]) == ["T1"]
    assert models.members(W, models.POPULATIONS["P_only"]) == ["T1", "T2", "T3", "T4"]
    assert models.members(W, models.POPULATIONS["P_only_A2"]) == ["T1", "T2", "T4"]
