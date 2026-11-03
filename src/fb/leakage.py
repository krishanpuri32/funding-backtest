"""Leakage control (plan §5, §10).

Every F / F₂ feature is computed only from parent fiscal-year rows whose BUDGET_END is
strictly earlier than the trial's *outcome date*. A trial that dies stops drawing money;
money observed after death must never enter a feature.

Outcome date (decided here, before any model):
- O1 event (Terminated / Suspended / Withdrawn): the EARLIER of primary_completion_date and
  completion_date among those typed ACTUAL. AACT has no explicit termination date; for a
  stopped trial these are the actual stop dates. Taking the earlier one is the conservative
  cut (less money enters). If neither date is ACTUAL (typical for Withdrawn / Suspended:
  dates still ESTIMATED), the earliest of {primary_completion, completion,
  last_update_submitted} is used and the trial is flagged.
- Completed: primary_completion_date; else completion_date; else last_update_submitted
  (flagged when not the primary-completion date).
- Censored / unknown / other: the AACT snapshot date (config.AACT_SNAPSHOT_TS), flagged.
"""
from __future__ import annotations

import pandas as pd

from . import config

EVENT_SOURCES = ("primary_completion_actual", "completion_actual", "earliest_non_actual")


def nih_fy(d) -> pd.Series:
    """NIH fiscal year of a date: October–September, FY = calendar year + 1 from October."""
    s = pd.to_datetime(pd.Series(d), errors="coerce")
    fy = s.dt.year + (s.dt.month >= 10).astype("Int64")
    return fy.astype("Int64")


def outcome_dates(df: pd.DataFrame, snapshot: str = config.AACT_SNAPSHOT_TS) -> pd.DataFrame:
    """One outcome date per trial.

    df columns: nct_id, status_group, primary_completion_date, primary_completion_date_type,
    completion_date, completion_date_type, last_update_submitted_date.
    Returns nct_id, outcome_date, outcome_date_source, outcome_date_flag (True when the
    date is not an ACTUAL terminal date of the expected kind).
    """
    pcd = pd.to_datetime(df["primary_completion_date"], errors="coerce")
    cd = pd.to_datetime(df["completion_date"], errors="coerce")
    lu = pd.to_datetime(df["last_update_submitted_date"], errors="coerce")
    pcd_act = pcd.where(df["primary_completion_date_type"].fillna("").str.upper().eq("ACTUAL"))
    cd_act = cd.where(df["completion_date_type"].fillna("").str.upper().eq("ACTUAL"))
    g = df["status_group"]

    out = pd.DataFrame({"nct_id": df["nct_id"].values}, index=df.index)
    out["outcome_date"] = pd.NaT
    out["outcome_date_source"] = "none"
    out["outcome_date_flag"] = True

    ev = g.eq("event")
    act_min = pd.concat([pcd_act, cd_act], axis=1).min(axis=1)
    has_act = ev & act_min.notna()
    out.loc[has_act, "outcome_date"] = act_min[has_act]
    use_pcd = has_act & pcd_act.notna() & (pcd_act <= cd_act.fillna(pd.Timestamp.max))
    out.loc[use_pcd, "outcome_date_source"] = "primary_completion_actual"
    out.loc[has_act & ~use_pcd, "outcome_date_source"] = "completion_actual"
    out.loc[has_act, "outcome_date_flag"] = False
    no_act = ev & act_min.isna()
    earliest = pd.concat([pcd, cd, lu], axis=1).min(axis=1)
    out.loc[no_act, "outcome_date"] = earliest[no_act]
    out.loc[no_act & earliest.notna(), "outcome_date_source"] = "earliest_non_actual"

    ctl = g.eq("control")
    pick = pcd.where(pcd.notna(), cd).where(lambda s: s.notna(), lu)
    out.loc[ctl, "outcome_date"] = pick[ctl]
    out.loc[ctl & pcd.notna(), "outcome_date_source"] = "primary_completion"
    out.loc[ctl & pcd.isna() & cd.notna(), "outcome_date_source"] = "completion"
    out.loc[ctl & pcd.isna() & cd.isna() & lu.notna(), "outcome_date_source"] = "last_update_submitted"
    out.loc[ctl & pcd.notna(), "outcome_date_flag"] = False

    other = ~(ev | ctl)
    out.loc[other, "outcome_date"] = pd.Timestamp(snapshot)
    out.loc[other, "outcome_date_source"] = "snapshot"
    out["outcome_date"] = pd.to_datetime(out["outcome_date"])
    return out.reset_index(drop=True)


def cut_rows(rows: pd.DataFrame, outcome_col: str = "outcome_date") -> pd.DataFrame:
    """Keep rows whose budget_end is strictly before the trial's outcome date.

    rows must carry budget_end and the outcome column (one trial or many, already joined).
    Rows with a null budget_end or null outcome date are dropped: no cut can be asserted.
    """
    be = pd.to_datetime(rows["budget_end"], errors="coerce")
    od = pd.to_datetime(rows[outcome_col], errors="coerce")
    return rows[be.notna() & od.notna() & (be < od)]


def gap_window_fys(trial_start, cut_date) -> list[int]:
    """Fiscal years in which a funding gap can be asserted without looking past the cut.

    A FY f row starts its budget inside FY f and (12-month budget) ends by 30 Sep of
    calendar year f + 1, so FY f is fully observable only when cut_date is in FY f + 2 or
    later. Window: FY(trial_start) .. FY(cut_date) - 2 (empty when the trial is shorter).
    """
    if pd.isna(trial_start) or pd.isna(cut_date):
        return []
    lo = int(nih_fy([trial_start]).iloc[0])
    hi = int(nih_fy([cut_date]).iloc[0]) - 2
    return list(range(lo, hi + 1))
