import pandas as pd

from fb import base_rates, config


def _tr(n_a, k_a, n_b, k_b):
    rows = []
    for i in range(n_a):
        rows.append({"phase_group": "2", "condition_group": "Neoplasms", "status_group": "event" if i < k_a else "control"})
    for i in range(n_b):
        rows.append({"phase_group": "3", "condition_group": "Infections", "status_group": "event" if i < k_b else "control"})
    rows.append({"phase_group": "2", "condition_group": "Neoplasms", "status_group": "unknown"})
    df = pd.DataFrame(rows)
    df["dollars"] = 1.0e6
    df["enrollment"] = 50
    df["duration_months"] = 24.0
    df["activity_class"] = "R"
    df["ic"] = "CA"
    df["start_year"] = 2010
    return df


def test_phase_x_condition_floor_and_pooling():
    tr = _tr(n_a=config.STRATUM_MIN_TRIALS, k_a=config.STRATUM_MIN_EVENTS, n_b=50, k_b=10)
    f = base_rates.phase_x_condition(tr)
    kept = f[f["phase"] != "other"]
    assert len(kept) == 1 and kept.iloc[0]["condition_group"] == "Neoplasms" and kept.iloc[0]["meets_floor"]
    pooled = f[f["phase"] == "other"].iloc[0]
    assert pooled["n_terminal"] == 50 and pooled["o1_events"] == 10
    assert kept.iloc[0]["n_unknown"] == 1


def test_stratum_row_rate_and_ci():
    r = base_rates.stratum_row(_tr(100, 10, 0, 0))
    assert r["n_terminal"] == 100 and r["o1_events"] == 10 and abs(r["event_rate"] - 0.10) < 1e-12
    assert abs(r["wilson_lo"] - 0.0552) < 2e-4 and abs(r["wilson_hi"] - 0.1744) < 2e-4
    assert r["n_unknown"] == 1 and r["dollars_median"] == 1.0e6


def test_table_for_has_overall_and_each_stratum_var():
    t = base_rates.table_for(_tr(120, 40, 60, 5))
    assert t.iloc[0]["stratum_var"] == "overall" and t.iloc[0]["n_terminal"] == 180
    assert set(t["stratum_var"]) == {"overall", "phase", "condition_group", "activity_class", "ic", "start_year"}


def test_history_summary_blank_and_filled(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("nct_id,url,n_versions,v1_date,v1_enrollment,v1_enrollment_type,v1_primary_completion,notes\nNCT1,u,,,,,,\n")
    assert base_rates.history_summary(p)["status"] == "blank"
    p.write_text("nct_id,url,n_versions,v1_date,v1_enrollment,v1_enrollment_type,v1_primary_completion,notes\n"
                 "NCT1,u,3,2010-01-01,100,ESTIMATED,2012-06-30,\nNCT2,u,,,,,,\nNCT3,u,2,,40,,,\n")
    cohort = pd.DataFrame({"nct_id": ["NCT1", "NCT3"], "enrollment": [80.0, 40.0], "trial_end": ["2013-01-01", "2011-01-01"],
                           "trial_end_source": ["primary_completion", "completion"]})
    s = base_rates.history_summary(p, cohort)
    assert s["n_filled"] == 2 and s["n_with_v1_enrollment"] == 2 and s["n_with_v1_primary_completion"] == 1
    assert s["enrollment_differs_from_current"] == 1 and s["primary_completion_differs_from_current"] == 1
