"""fb.leakage on hand-built fixtures. The worked case: a trial funded FY2010–FY2014,
terminated 2012-06 -> only rows ending before 2012-06 enter; the FY2013–14 rows must not."""
import pandas as pd

from fb import leakage


def _studies(**over):
    base = dict(nct_id="NCT1", status_group="event",
                primary_completion_date="2012-06-15", primary_completion_date_type="ACTUAL",
                completion_date="2012-06-30", completion_date_type="ACTUAL",
                last_update_submitted_date="2013-01-10")
    base.update(over)
    return pd.DataFrame([base])


def test_nih_fy_october_rollover():
    fy = leakage.nih_fy(["2011-09-30", "2011-10-01", "2012-06-15"])
    assert fy.tolist() == [2011, 2012, 2012]


def test_event_uses_earlier_actual_date():
    o = leakage.outcome_dates(_studies())
    assert o.loc[0, "outcome_date"] == pd.Timestamp("2012-06-15")
    assert o.loc[0, "outcome_date_source"] == "primary_completion_actual" and not o.loc[0, "outcome_date_flag"]
    o = leakage.outcome_dates(_studies(completion_date="2012-05-01"))
    assert o.loc[0, "outcome_date"] == pd.Timestamp("2012-05-01") and o.loc[0, "outcome_date_source"] == "completion_actual"


def test_event_without_actual_dates_takes_earliest_and_flags():
    o = leakage.outcome_dates(_studies(primary_completion_date_type="ESTIMATED", completion_date_type=None,
                                       primary_completion_date="2015-01-01", completion_date=None,
                                       last_update_submitted_date="2013-03-01"))
    assert o.loc[0, "outcome_date"] == pd.Timestamp("2013-03-01")
    assert o.loc[0, "outcome_date_source"] == "earliest_non_actual" and o.loc[0, "outcome_date_flag"]


def test_completed_uses_primary_completion_then_fallbacks():
    o = leakage.outcome_dates(_studies(status_group="control"))
    assert o.loc[0, "outcome_date"] == pd.Timestamp("2012-06-15") and o.loc[0, "outcome_date_source"] == "primary_completion"
    o = leakage.outcome_dates(_studies(status_group="control", primary_completion_date=None))
    assert o.loc[0, "outcome_date_source"] == "completion" and o.loc[0, "outcome_date_flag"]


def test_censored_uses_snapshot():
    o = leakage.outcome_dates(_studies(status_group="censored"), snapshot="2026-10-01")
    assert o.loc[0, "outcome_date"] == pd.Timestamp("2026-10-01") and o.loc[0, "outcome_date_source"] == "snapshot"


def test_trial_dying_mid_window_sees_no_post_death_money():
    # FY2010..FY2014 budget periods, July–June; trial terminated 2012-06-15
    rows = pd.DataFrame({
        "FY_FILE": [2010, 2011, 2012, 2013, 2014],
        "budget_end": ["2011-06-30", "2012-05-31", "2013-06-30", "2014-06-30", "2015-06-30"],
        "outcome_date": pd.Timestamp("2012-06-15"),
    })
    kept = leakage.cut_rows(rows)
    assert kept["FY_FILE"].tolist() == [2010, 2011]
    assert not kept["FY_FILE"].isin([2013, 2014]).any()


def test_cut_is_strict_and_drops_null_budget_end():
    rows = pd.DataFrame({"budget_end": ["2012-06-15", "2012-06-14", None], "outcome_date": pd.Timestamp("2012-06-15")})
    assert leakage.cut_rows(rows)["budget_end"].tolist() == ["2012-06-14"]


def test_gap_window_stops_two_fys_before_the_cut():
    # start FY2010 (2010-03), cut 2012-06-15 is FY2012 -> only FY2010 is fully observable
    assert leakage.gap_window_fys(pd.Timestamp("2010-03-01"), pd.Timestamp("2012-06-15")) == [2010]
    assert leakage.gap_window_fys(pd.Timestamp("2010-03-01"), pd.Timestamp("2011-01-01")) == []
    assert leakage.gap_window_fys(None, pd.Timestamp("2011-01-01")) == []
