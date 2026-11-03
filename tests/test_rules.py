import pandas as pd
import pytest

from fb import rules


def test_in_start_window_bounds_and_nulls():
    s = pd.Series(["2007-12-31", "2008-01-01", "2015-06-30", "2020-12-31", "2021-01-01", None, "garbage"])
    out = rules.in_start_window(s)
    assert out.tolist() == [False, True, True, True, False, False, False]


def test_window_of_train_test_split():
    s = pd.Series(["2008-01-01", "2015-12-31", "2016-01-01", "2020-12-31", "2007-06-01", "2021-03-01", None])
    out = rules.window_of(s)
    assert out.tolist()[:4] == ["train", "train", "test", "test"]
    assert all(pd.isna(v) for v in out.tolist()[4:])


def test_status_group_and_o1_label():
    s = pd.Series([
        "COMPLETED", "TERMINATED", "SUSPENDED", "WITHDRAWN", "UNKNOWN",
        "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING",
        "AVAILABLE", None, "Active, not recruiting", "Completed",
    ])
    g = rules.status_group(s).tolist()
    assert g == [
        "control", "event", "event", "event", "unknown",
        "censored", "censored", "censored", "censored",
        "other", "other", "censored", "control",
    ]
    lab = rules.o1_label(s).tolist()
    assert lab[0] == 0 and lab[1] == 1 and lab[2] == 1 and lab[3] == 1
    assert all(pd.isna(v) for v in lab[4:12])
    assert lab[12] == 0


@pytest.mark.parametrize(
    "bs,be,ts,te,expected",
    [
        ("2010-07-01", "2011-06-30", "2010-01-01", "2012-12-31", True),   # FY inside trial
        ("2009-07-01", "2010-06-30", "2010-01-01", "2012-12-31", True),   # FY straddles start
        ("2012-07-01", "2013-06-30", "2010-01-01", "2012-12-31", True),   # FY straddles end
        ("2008-07-01", "2009-06-30", "2010-01-01", "2012-12-31", False),  # FY entirely before
        ("2013-07-01", "2014-06-30", "2010-01-01", "2012-12-31", False),  # FY entirely after
        ("2009-01-01", "2009-12-31", "2010-01-01", "2010-06-30", False),  # ends day before start
        ("2009-01-01", "2010-01-01", "2010-01-01", "2010-06-30", True),   # touches start day (closed)
        ("2010-06-30", "2011-06-29", "2010-01-01", "2010-06-30", True),   # touches end day (closed)
        (None, "2011-06-30", "2010-01-01", "2012-12-31", False),          # null budget start
        ("2010-07-01", "2011-06-30", "2010-01-01", None, False),          # null trial end
    ],
)
def test_fy_overlaps_scalar(bs, be, ts, te, expected):
    assert rules.fy_overlaps(bs, be, ts, te) is expected


def test_fy_overlaps_vec_matches_scalar():
    rows = [
        ("2010-07-01", "2011-06-30", "2010-01-01", "2012-12-31"),
        ("2008-07-01", "2009-06-30", "2010-01-01", "2012-12-31"),
        (None, "2011-06-30", "2010-01-01", "2012-12-31"),
        ("2009-01-01", "2010-01-01", "2010-01-01", "2010-06-30"),
    ]
    df = pd.DataFrame(rows, columns=["budget_start", "budget_end", "trial_start", "trial_end"])
    vec = rules.fy_overlaps_vec(df).tolist()
    sc = [rules.fy_overlaps(*r) for r in rows]
    assert vec == sc == [True, False, False, True]


def test_trial_end_date_fallback_order():
    pc = pd.Series(["2012-05-01", None, None, None])
    cd = pd.Series(["2012-09-01", "2013-01-01", None, None])
    lk = pd.Series(["2011-01-01", "2014-01-01", "2015-01-01", None])
    out = rules.trial_end_date(pc, lk, completion=cd)
    assert out["trial_end_source"].tolist() == ["primary_completion", "completion", "last_update_submitted", "none"]
    assert out["trial_end"].iloc[0] == pd.Timestamp("2012-05-01")
    assert out["trial_end"].iloc[1] == pd.Timestamp("2013-01-01")
    assert out["trial_end"].iloc[2] == pd.Timestamp("2015-01-01")
    assert pd.isna(out["trial_end"].iloc[3])
