"""Planned enrollment and planned duration from registry version 0 (plan §5, §10; planned-value rule, PLAN.md Note 2026-10-21).

Source: data/derived/history_v0_cohort.csv (the original registration of each trial).
Rule (ratified 2026-10-22): planned values are MISSING when the trial was registered late
(v0 overall status in config.V0_LATE_STATUSES) or its v0 enrollment is typed ACTUAL; those
trials are also excluded from O3/O4. Current-record enrollment is never used as planned.

Reason precedence: late > actual > blank (> nonpositive for duration).
Planned duration = v0 primary completion − v0 start, in months (days / 30.4375); when v0
start is blank the current-record start date is used instead (flagged).
"""
from __future__ import annotations

import pandas as pd

from . import config

HISTORY_V0_PATH = config.HISTORY_V0_CSV_PATH
DAYS_PER_MONTH = 30.4375


def parse_registry_date(s: pd.Series) -> pd.Series:
    """'YYYY-MM' (month precision -> first of month) or 'YYYY-MM-DD'; anything else -> NaT."""
    s = s.fillna("").astype(str).str.strip()
    month_only = s.str.fullmatch(r"\d{4}-\d{2}")
    s = s.where(~month_only, s + "-01")
    return pd.to_datetime(s.where(s.ne("")), format="%Y-%m-%d", errors="coerce")


def load_v0(path=HISTORY_V0_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def planned_values(df: pd.DataFrame) -> pd.DataFrame:
    """df: nct_id, v0_overall_status, v0_enrollment, v0_enrollment_type, v0_start,
    v0_primary_completion, start_date (current record).
    Returns nct_id, registered_late, v0_enrollment_actual, o34_eligible,
    planned_enrollment, planned_enrollment_reason, planned_duration_months,
    planned_duration_reason, planned_start_source.
    """
    st = df["v0_overall_status"].fillna("").str.strip().str.upper()
    late = st.isin(config.V0_LATE_STATUSES)
    actual = df["v0_enrollment_type"].fillna("").str.strip().str.upper().eq("ACTUAL")
    enr = pd.to_numeric(df["v0_enrollment"], errors="coerce")

    out = pd.DataFrame({"nct_id": df["nct_id"].values}, index=df.index)
    out["registered_late"] = late
    out["v0_enrollment_actual"] = actual
    out["o34_eligible"] = ~late & ~actual

    reason = pd.Series("ok", index=df.index, dtype="object")
    reason[enr.isna()] = "blank"
    reason[actual] = "actual"
    reason[late] = "late"
    out["planned_enrollment"] = enr.where(reason.eq("ok"))
    out["planned_enrollment_reason"] = reason

    v0s = parse_registry_date(df["v0_start"])
    cur = pd.to_datetime(df["start_date"], errors="coerce")
    start = v0s.where(v0s.notna(), cur)
    out["planned_start_source"] = pd.Series("v0", index=df.index).where(v0s.notna(), "current")
    out.loc[start.isna(), "planned_start_source"] = "none"
    pc = parse_registry_date(df["v0_primary_completion"])
    dur = (pc - start).dt.days / DAYS_PER_MONTH
    dreason = pd.Series("ok", index=df.index, dtype="object")
    dreason[dur.notna() & (dur <= 0)] = "nonpositive"
    dreason[pc.isna() | start.isna()] = "blank"
    dreason[actual] = "actual"
    dreason[late] = "late"
    out["planned_duration_months"] = dur.where(dreason.eq("ok"))
    out["planned_duration_reason"] = dreason
    return out.reset_index(drop=True)
