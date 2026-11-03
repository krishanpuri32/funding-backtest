"""Pure cohort rules from the pre-registered plan (§3, §4). No I/O, no data access.

Every function here is unit-tested on hand-built fixtures (tests/test_rules.py)
before it touches the real files (mock-first guardrail).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from . import config


def in_start_window(start: pd.Series, lo: str = config.COHORT_START, hi: str = config.COHORT_END) -> pd.Series:
    """Plan §3 step 3: start date within [lo, hi] inclusive. NaT -> False."""
    s = pd.to_datetime(start, errors="coerce")
    return (s >= pd.Timestamp(lo)) & (s <= pd.Timestamp(hi))


def window_of(start: pd.Series) -> pd.Series:
    """Plan §7: 'train' for start year 2008–2015, 'test' for 2016–2020, else NA."""
    y = pd.to_datetime(start, errors="coerce").dt.year
    out = pd.Series(pd.NA, index=start.index, dtype="object")
    out[(y >= config.TRAIN_YEARS[0]) & (y <= config.TRAIN_YEARS[1])] = "train"
    out[(y >= config.TEST_YEARS[0]) & (y <= config.TEST_YEARS[1])] = "test"
    return out


def status_group(overall_status: pd.Series) -> pd.Series:
    """Plan §4 O1 mapping on AACT `overall_status` (upper-snake enum in the flat files).

    Returns one of: 'event' (Terminated/Suspended/Withdrawn), 'control' (Completed),
    'unknown', 'censored' (Recruiting / Active not recruiting / Enrolling by invitation /
    Not yet recruiting), 'other' (anything else, incl. null).
    """
    s = overall_status.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "_").str.replace(",", "")
    out = pd.Series("other", index=overall_status.index, dtype="object")
    out[s.isin(config.O1_EVENT_STATUSES)] = "event"
    out[s.isin(config.O1_CONTROL_STATUSES)] = "control"
    out[s.isin(config.UNKNOWN_STATUSES)] = "unknown"
    out[s.isin(config.CENSORED_STATUSES)] = "censored"
    return out


def o1_label(overall_status: pd.Series) -> pd.Series:
    """1 for O1 event, 0 for Completed, NA otherwise (excluded from O1)."""
    g = status_group(overall_status)
    out = pd.Series(pd.NA, index=overall_status.index, dtype="Int64")
    out[g == "event"] = 1
    out[g == "control"] = 0
    return out


def fy_overlaps(
    budget_start, budget_end, trial_start, trial_end
) -> bool:
    """Plan §3 step 5: does the fiscal-year budget period overlap the trial window?

    Closed intervals; any null end-point on either side -> False (no overlap can be
    asserted). Scalar version used by the vectorised wrapper below.
    """
    for v in (budget_start, budget_end, trial_start, trial_end):
        if v is None or pd.isna(v):
            return False
    bs, be, ts, te = (pd.Timestamp(v) for v in (budget_start, budget_end, trial_start, trial_end))
    return bs <= te and be >= ts


def fy_overlaps_vec(df: pd.DataFrame) -> pd.Series:
    """Vectorised `fy_overlaps` over columns budget_start, budget_end, trial_start, trial_end."""
    bs = pd.to_datetime(df["budget_start"], errors="coerce")
    be = pd.to_datetime(df["budget_end"], errors="coerce")
    ts = pd.to_datetime(df["trial_start"], errors="coerce")
    te = pd.to_datetime(df["trial_end"], errors="coerce")
    ok = bs.notna() & be.notna() & ts.notna() & te.notna()
    return ok & (bs <= te) & (be >= ts)


def trial_end_date(primary_completion, last_update_submitted_date, completion=None) -> pd.Series:
    """Trial-window end for the overlap rule (plan §3): primary-completion date, else
    `completion_date`, else `last_update_submitted_date`. Returns the chosen date and a
    source label as a two-column frame.
    """
    pc = pd.to_datetime(primary_completion, errors="coerce")
    lk = pd.to_datetime(last_update_submitted_date, errors="coerce")
    cd = pd.to_datetime(completion, errors="coerce") if completion is not None else pd.Series(pd.NaT, index=pc.index)
    end = pc.copy()
    src = pd.Series("primary_completion", index=pc.index, dtype="object")
    use_cd = end.isna() & cd.notna()
    end[use_cd] = cd[use_cd]
    src[use_cd] = "completion"
    use_lk = end.isna() & lk.notna()
    end[use_lk] = lk[use_lk]
    src[use_lk] = "last_update_submitted"
    src[end.isna()] = "none"
    return pd.DataFrame({"trial_end": end, "trial_end_source": src})
