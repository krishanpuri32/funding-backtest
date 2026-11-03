"""Person-period expansion and cumulative risk for the discrete-time hazard model (O5)."""
import numpy as np
import pandas as pd
import pytest

from fb import config, hazard


def _trials():
    return pd.DataFrame({
        "nct_id": ["T1", "T2", "T3"],
        # T1 starts in FY2011 (Jan 2011), terminates in FY2013 (Mar 2013): 3 rows, event in the last
        # T2 starts in FY2011, completes in FY2012: 2 rows, no event
        # T3 exit before start (bad dates) -> clamped to one row
        "start_date": pd.to_datetime(["2011-01-15", "2010-11-01", "2012-05-01"]),
        "exit_date": pd.to_datetime(["2013-03-10", "2012-08-31", "2012-01-01"]),
        "event": [1, 0, 1],
    })


def test_three_fy_trial_three_rows_event_last():
    pp = hazard.person_periods(_trials())
    t1 = pp[pp["nct_id"] == "T1"]
    assert t1["fy"].tolist() == [2011, 2012, 2013]
    assert t1["period"].tolist() == [0, 1, 2]
    assert t1["y"].tolist() == [0, 0, 1]
    assert t1["at_risk"].all()
    assert t1["key"].tolist() == ["T1|2011", "T1|2012", "T1|2013"]


def test_completed_trial_no_event_and_clamp():
    pp = hazard.person_periods(_trials())
    t2 = pp[pp["nct_id"] == "T2"]
    assert t2["fy"].tolist() == [2011, 2012] and t2["y"].sum() == 0
    t3 = pp[pp["nct_id"] == "T3"]
    assert t3["fy"].tolist() == [2012] and t3["y"].tolist() == [1]


def test_cut_date_is_fy_start_bounded_by_exit():
    pp = hazard.person_periods(_trials()).set_index("key")
    assert pp.loc["T1|2011", "cut_date"] == pd.Timestamp("2010-10-01")
    assert pp.loc["T1|2013", "cut_date"] == pd.Timestamp("2012-10-01")
    assert pp.loc["T3|2012", "cut_date"] == pd.Timestamp("2011-10-01")  # FY2012 starts before T3's exit
    ext = hazard.person_periods(_trials().iloc[[0]], horizon_months=36).set_index("key")
    assert ext.loc["T1|2014", "cut_date"] == pd.Timestamp("2013-03-10")  # FY start after exit -> exit


def test_horizon_extension_not_at_risk():
    pp = hazard.person_periods(_trials().iloc[[1]], horizon_months=36)
    # start 2010-11-01 + 36 months - 1 day = 2013-10-31 -> FY2014
    assert pp["fy"].tolist() == [2011, 2012, 2013, 2014]
    assert pp["at_risk"].tolist() == [True, True, False, False]
    assert pp["y"].sum() == 0
    assert pp["cut_date"].iloc[-1] == pd.Timestamp("2012-08-31")  # information frozen at exit


def test_period_dummies_cap():
    d = hazard.period_dummies([0, 3, 7, 12])
    k = config.HAZARD_MAX_PERIOD
    assert d.shape == (4, k + 1)
    assert d.iloc[0]["period=0"] == 1 and d.iloc[1]["period=3"] == 1
    assert d.iloc[2][f"period={k}+"] == 1 and d.iloc[3][f"period={k}+"] == 1
    assert (d.sum(axis=1) == 1).all()


def test_cumulative_risk_by_horizon():
    pp = pd.DataFrame({"nct_id": ["A", "A", "A"], "fy": [2011, 2012, 2013]})
    start = pd.Series({"A": pd.Timestamp("2011-01-15")})
    h = np.array([0.1, 0.2, 0.5])
    # 12 m: FY2011 (starts 2010-10-01) and FY2012 (2011-10-01 < 2012-01-15) count
    assert hazard.cumulative_risk(pp, h, start, 12)["A"] == pytest.approx(1 - 0.9 * 0.8)
    assert hazard.cumulative_risk(pp, h, start, 24)["A"] == pytest.approx(1 - 0.9 * 0.8 * 0.5)
