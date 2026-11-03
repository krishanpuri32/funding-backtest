import pandas as pd
import pytest

from fb import sole_trial


def _fixture():
    # T1, T2 share award SH (2 trials); T3, T4, T5 each hold a sole award; T6 links only to umbrella UM;
    # T7 holds a sole award plus umbrella UM (still one attributable project -> sole); T8 is test window.
    cohort = pd.DataFrame([
        ("T1", "train", 1, 1, True, 1.0e6),
        ("T2", "train", 0, 1, True, 2.0e6),
        ("T3", "train", 1, 1, True, 3.0e6),
        ("T4", "train", 0, 1, True, 4.0e6),
        ("T5", "train", pd.NA, 1, True, 5.0e6),   # censored / unknown: not terminal
        ("T6", "train", 1, 0, False, float("nan")),
        ("T7", "train", 0, 1, True, 7.0e6),
        ("T8", "test", 1, 1, True, 8.0e6),
    ], columns=["nct_id", "window", "o1_label", "n_attributable_n20", "has_attributable_funding_n20",
                "total_cost_base_primary"])
    cohort["o1_label"] = cohort["o1_label"].astype("Int64")
    cp = pd.DataFrame([
        ("T1", "SH", 2, False), ("T2", "SH", 2, False),
        ("T3", "S3", 1, False), ("T4", "S4", 1, False), ("T5", "S5", 1, False),
        ("T6", "UM", 40, True), ("T7", "S7", 1, False), ("T7", "UM", 40, True),
        ("T8", "S8", 1, False),
    ], columns=["nct_id", "core_project_num", "n_nct_linked", "is_umbrella_n20"])
    return cohort, cp


def test_population_rule():
    cohort, cp = _fixture()
    assert sole_trial.sole_trial_ids(cohort, cp) == {"T3", "T4", "T5", "T7", "T8"}


def test_sums_and_rates():
    cohort, cp = _fixture()
    t = sole_trial.build(cohort, cp).set_index(["window", "population"])
    s = t.loc[("train", "sole_trial")]
    assert s["n_terminal"] == 3 and s["o1_events"] == 1            # T3 event; T4, T7 control; T5 not terminal
    assert s["event_rate"] == pytest.approx(1 / 3)
    assert s["dollars_total_base"] == pytest.approx(14.0e6)        # 3 + 4 + 7
    assert s["dollars_noncomplete_base"] == pytest.approx(3.0e6)
    assert s["noncomplete_share"] == pytest.approx(3 / 14)
    assert s["dollars_median_base"] == pytest.approx(4.0e6)
    a = t.loc[("train", "attributable_n20")]
    assert a["n_terminal"] == 5 and a["o1_events"] == 2            # T1, T2, T3, T4, T7
    te = t.loc[("test", "sole_trial")]
    assert te["n_terminal"] == 1 and te["dollars_noncomplete_base"] == pytest.approx(8.0e6)
