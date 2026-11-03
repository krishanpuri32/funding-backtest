"""Coverage table (plan §3 applied step by step) and the §11 cell.

Nothing here builds a feature or looks at an outcome beyond counting statuses.
"""
from __future__ import annotations

import json

import pandas as pd

from . import aact, config, exporter, rules
from .link import LINK_PATH, PROJECTS_PATH

COVERAGE_PATH = config.DATA_DERIVED / "coverage.json"
COHORT_STEP5_PATH = config.DATA_DERIVED / "cohort_step5.parquet"


def _counts(df: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    d = df if mask is None else df[mask]
    return {
        "all": int(len(d)),
        "train": int((d["window"] == "train").sum()),
        "test": int((d["window"] == "test").sum()),
    }


def _months(a: pd.Series, b: pd.Series) -> pd.Series:
    return (b - a).dt.days / 30.4375


def _iqr(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {"n": int(len(s)), "median": float(s.median()), "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75))}


def build(write: bool = True) -> dict:
    out: dict = {"steps": [], "notes": []}
    st = aact.load_table("studies", usecols=[
        "nct_id", "study_type", "overall_status", "last_known_status", "start_date", "start_date_type",
        "primary_completion_date", "primary_completion_date_type", "completion_date", "completion_date_type",
        "why_stopped", "enrollment", "enrollment_type", "phase", "last_update_submitted_date",
    ])
    st["nct_id"] = st["nct_id"].str.strip().str.upper()
    for c in ["start_date", "primary_completion_date", "completion_date", "last_update_submitted_date"]:
        st[c] = pd.to_datetime(st[c], errors="coerce")
    st["window"] = rules.window_of(st["start_date"])
    out["aact_studies_total"] = int(len(st))
    out["study_type_values"] = {str(k): int(v) for k, v in st["study_type"].value_counts(dropna=False).items()}

    # 1 interventional
    m1 = st["study_type"].fillna("").str.upper().eq("INTERVENTIONAL")
    out["steps"].append({"step": 1, "label": "study_type = Interventional", **_counts(st, m1)})
    # 2 with a start date
    m2 = m1 & st["start_date"].notna()
    out["steps"].append({"step": 2, "label": "... with a start date", **_counts(st, m2)})
    # 3 start in 2008-01-01..2020-12-31
    m3 = m2 & rules.in_start_window(st["start_date"])
    out["steps"].append({"step": 3, "label": "... start 2008-01-01..2020-12-31", **_counts(st, m3)})
    # 4 linked to >= 1 core project
    link = pd.read_parquet(LINK_PATH)
    linked_ids = set(link["nct_id"])
    m4 = m3 & st["nct_id"].isin(linked_ids)
    out["steps"].append({"step": 4, "label": "... linked to >= 1 core project (Clinical Studies file)", **_counts(st, m4)})

    # 5 >= 1 FY row whose budget period overlaps the trial window
    c = st[m4].copy()
    end = rules.trial_end_date(c["primary_completion_date"], c["last_update_submitted_date"], completion=c["completion_date"])
    c["trial_end"] = end["trial_end"]
    c["trial_end_source"] = end["trial_end_source"]
    out["trial_end_source_counts"] = {str(k): int(v) for k, v in c["trial_end_source"].value_counts().items()}
    prj = pd.read_parquet(PROJECTS_PATH, columns=["CORE_PROJECT_NUM", "BUDGET_START", "BUDGET_END", "FY_FILE", "TOTAL_COST"])
    prj["budget_start"] = exporter.parse_exporter_date(prj["BUDGET_START"])
    prj["budget_end"] = exporter.parse_exporter_date(prj["BUDGET_END"])
    out["projects_rows_missing_budget_dates"] = int((prj["budget_start"].isna() | prj["budget_end"].isna()).sum())
    pairs = link[link["nct_id"].isin(set(c["nct_id"]))][["nct_id", "core_project_num"]]
    j = pairs.merge(prj[["CORE_PROJECT_NUM", "budget_start", "budget_end", "FY_FILE"]],
                    left_on="core_project_num", right_on="CORE_PROJECT_NUM", how="inner")
    j = j.merge(c[["nct_id", "start_date", "trial_end"]].rename(columns={"start_date": "trial_start"}), on="nct_id", how="left")
    j["overlap"] = rules.fy_overlaps_vec(j)
    ov = j[j["overlap"]].groupby("nct_id").agg(n_overlap_fy_rows=("FY_FILE", "size"),
                                              n_overlap_fy=("FY_FILE", "nunique"),
                                              n_overlap_projects=("core_project_num", "nunique"))
    c = c.merge(ov, left_on="nct_id", right_index=True, how="left")
    c["n_overlap_fy_rows"] = c["n_overlap_fy_rows"].fillna(0).astype(int)
    m5 = c["n_overlap_fy_rows"] > 0
    out["steps"].append({"step": 5, "label": "... with >= 1 FY row overlapping the trial window", **_counts(c, m5)})
    out["step5_no_overlap_by_trial_end_source"] = {str(k): int(v) for k, v in c.loc[~m5, "trial_end_source"].value_counts().items()}
    c5 = c[m5].copy()

    # 6 by overall_status
    c5["status_group"] = rules.status_group(c5["overall_status"])
    by_status = c5.groupby(["overall_status", "window"], dropna=False).size().unstack(fill_value=0)
    by_status["all"] = c5.groupby("overall_status", dropna=False).size()
    out["step6_by_overall_status"] = {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in by_status.to_dict(orient="index").items()}
    out["step6_by_status_group"] = {str(k): v for k, v in c5.groupby("status_group").agg(
        all=("nct_id", "size"), train=("window", lambda s: int((s == "train").sum())), test=("window", lambda s: int((s == "test").sum()))
    ).astype(int).to_dict(orient="index").items()}

    # 7 O1 events vs Completed, §11 cell
    ev = c5["status_group"] == "event"
    ctl = c5["status_group"] == "control"
    o1 = {
        "events": _counts(c5, ev), "completed": _counts(c5, ctl),
        "o1_denominator": _counts(c5, ev | ctl),
        "events_by_status": {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in
                             c5[ev].groupby(["overall_status", "window"]).size().unstack(fill_value=0).to_dict(orient="index").items()},
    }
    test_events = o1["events"]["test"]
    if test_events >= config.GATE_FULL:
        cell = f">= {config.GATE_FULL}: proceed to the full plan"
    elif test_events >= config.GATE_POOLED:
        cell = f"{config.GATE_POOLED}-{config.GATE_FULL}: pooled O1 only, no strata, no O2-O4"
    else:
        cell = f"< {config.GATE_POOLED}: stop"
    o1["section_11_cell"] = cell
    out["step7_o1"] = o1

    # 8 Withdrawn, enrollment 0, no overlapping funded FY — on the step-4 set (c), counted separately
    enr = pd.to_numeric(c["enrollment"], errors="coerce")
    wd = c["overall_status"].fillna("").str.upper().eq("WITHDRAWN")
    out["step8_never_started"] = {
        "withdrawn_at_step4": _counts(c, wd),
        "withdrawn_enrollment0_at_step4": _counts(c, wd & enr.eq(0)),
        "withdrawn_enrollment0_no_overlap_fy": _counts(c, wd & enr.eq(0) & ~m5),
        "withdrawn_enrollment0_with_overlap_fy_(still_in_step5)": _counts(c, wd & enr.eq(0) & m5),
    }

    # 9 why_stopped non-empty among O1 events
    ws = c5["why_stopped"].fillna("").str.strip().ne("")
    out["step9_why_stopped_nonempty_among_events"] = {"events": _counts(c5, ev), "with_why_stopped": _counts(c5, ev & ws)}

    # 10 actual enrollment among terminal-status trials (event + control)
    term = ev | ctl
    act = c5["enrollment_type"].fillna("").str.upper().eq("ACTUAL")
    out["step10_enrollment_actual_among_terminal"] = {
        "terminal": _counts(c5, term), "enrollment_type_actual": _counts(c5, term & act),
        "enrollment_type_values_among_terminal": {str(k): int(v) for k, v in c5.loc[term, "enrollment_type"].value_counts(dropna=False).items()},
    }

    # duration and planned enrollment at step 5
    dur = _months(c5["start_date"], c5["trial_end"])
    out["step5_duration_months"] = {"all": _iqr(dur), "train": _iqr(dur[c5["window"] == "train"]), "test": _iqr(dur[c5["window"] == "test"])}
    enr5 = pd.to_numeric(c5["enrollment"], errors="coerce")
    out["step5_enrollment_current_record"] = {"all": _iqr(enr5), "train": _iqr(enr5[c5["window"] == "train"]), "test": _iqr(enr5[c5["window"] == "test"])}
    out["step5_enrollment_by_type"] = {str(k): int(v) for k, v in c5["enrollment_type"].value_counts(dropna=False).items()}
    out["step5_phase"] = {str(k): int(v) for k, v in c5["phase"].value_counts(dropna=False).items()}
    out["step5_start_year"] = {int(k): int(v) for k, v in c5["start_date"].dt.year.value_counts().sort_index().items()}
    out["step5_overlap_fy_per_trial"] = _iqr(c5["n_overlap_fy"])
    out["step5_overlap_projects_per_trial"] = _iqr(c5["n_overlap_projects"])
    out["notes"].append("enrollment at step 5 is the CURRENT record value (actual or estimated), not originally planned; planned requires record history")
    out["notes"].append("trial window end = primary_completion_date, else completion_date, else last_update_submitted_date (AACT has no last_known_status date column)")

    if write:
        COVERAGE_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
        c5[["nct_id", "start_date", "trial_end", "trial_end_source", "window", "overall_status", "status_group",
            "n_overlap_fy_rows", "n_overlap_fy", "n_overlap_projects"]].to_parquet(COHORT_STEP5_PATH, index=False)
    return out


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, default=str))
