"""Phase 2 cohort table (plan §3; A1, A2): data/derived/cohort.parquet.

One row per step-5 trial (both windows). Carries the O1 label, the A2 flag, the A1
umbrella flags under all three rules, the five base-rate priors, and the parent-row
CPI-adjusted dollar total under the primary (N = 20 attributable) and the all-linked
definitions. Everything *reported* here is computed on the training window only;
test-window rows carry labels but are never summarised.
"""
from __future__ import annotations

import json

import pandas as pd

from . import aact, config, exporter, priors, rules, umbrella
from .coverage import COHORT_STEP5_PATH
from .cpi import CPI_PATH, load_cpi
from .dollars import trial_dollars
from .link import LINK_PATH, PROJECTS_PATH

COHORT_PATH = config.DATA_DERIVED / "cohort.parquet"
COHORT_PROJECTS_PATH = config.DATA_DERIVED / "cohort_projects.parquet"
MESH_MAP_PATH = config.DATA_DERIVED / "mesh_group_map.csv"
STATS_PATH = config.DATA_DERIVED / "cohort_stats.json"


def _iqr(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "median": float(s.median()), "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)), "p90": float(s.quantile(0.90))}


def _split(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "o1_events": int((df["status_group"] == "event").sum()),
            "completed": int((df["status_group"] == "control").sum()),
            "unknown": int((df["status_group"] == "unknown").sum()),
            "censored": int((df["status_group"] == "censored").sum())}


def _modal_by_trial(pairs: pd.DataFrame, flags: pd.DataFrame, mask_col: str | None) -> pd.DataFrame:
    """Modal activity class and modal IC per trial over the projects selected by mask_col (None = all)."""
    j = pairs.merge(flags[["core_project_num", "activity", "ic", "is_umbrella_n20", "is_umbrella_n50", "is_umbrella_activity"]],
                    on="core_project_num", how="left")
    if mask_col is not None:
        j = j[~j[mask_col].fillna(False).astype(bool)]
    j = j.assign(act_class=priors.activity_class(j["activity"]))
    g = j.groupby("nct_id").agg(
        activity_class=("act_class", priors.modal_activity_class),
        ic_modal=("ic", priors.modal_value),
    ).reset_index()
    return g


def build(write: bool = True) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"notes": []}
    c5 = pd.read_parquet(COHORT_STEP5_PATH)
    ids = set(c5["nct_id"])
    st = aact.load_table("studies", usecols=["nct_id", "overall_status", "enrollment", "enrollment_type", "phase"])
    st["nct_id"] = st["nct_id"].str.strip().str.upper()
    st = st[st["nct_id"].isin(ids)]
    c = c5.merge(st[["nct_id", "enrollment", "enrollment_type", "phase"]], on="nct_id", how="left")
    c["start_year"] = pd.to_datetime(c["start_date"]).dt.year.astype(int)
    c["o1_label"] = rules.o1_label(c["overall_status"])
    c["enrollment"] = pd.to_numeric(c["enrollment"], errors="coerce")
    c["duration_months"] = (pd.to_datetime(c["trial_end"]) - pd.to_datetime(c["start_date"])).dt.days / 30.4375
    c["withdrawn_zero_enroll"] = umbrella.withdrawn_zero_enroll(c["overall_status"], c["enrollment"])

    # --- A1 umbrella flags: per project from the WHOLE link file, per trial for the cohort
    link = pd.read_parquet(LINK_PATH)
    pflags = umbrella.project_flags(link[["nct_id", "core_project_num", "activity"]])
    ic_by_proj = link.groupby("core_project_num")["ic"].agg(priors.modal_value)
    pflags = pflags.merge(ic_by_proj.rename("ic"), left_on="core_project_num", right_index=True, how="left")
    pairs = link.loc[link["nct_id"].isin(ids), ["nct_id", "core_project_num"]].drop_duplicates()
    tflags = umbrella.trial_flags(pairs, pflags)
    c = c.merge(tflags, on="nct_id", how="left")
    cohort_projects = pairs.merge(pflags, on="core_project_num", how="left")
    stats["umbrella_projects"] = {
        "projects_in_link_file": int(len(pflags)),
        "is_umbrella_n20": int(pflags["is_umbrella_n20"].sum()),
        "is_umbrella_n50": int(pflags["is_umbrella_n50"].sum()),
        "is_umbrella_activity_P30_U10": int(pflags["is_umbrella_activity"].sum()),
        "cohort_linked_projects": int(pairs["core_project_num"].nunique()),
        "cohort_projects_umbrella_n20": int(cohort_projects.drop_duplicates("core_project_num")["is_umbrella_n20"].sum()),
        "cohort_projects_umbrella_n50": int(cohort_projects.drop_duplicates("core_project_num")["is_umbrella_n50"].sum()),
        "cohort_projects_umbrella_activity": int(cohort_projects.drop_duplicates("core_project_num")["is_umbrella_activity"].sum()),
    }

    # --- priors
    c["phase_group"] = priors.collapse_phase(c["phase"])
    bc = aact.load_table("browse_conditions", usecols=["nct_id", "mesh_term", "mesh_type"])
    bc["nct_id"] = bc["nct_id"].str.strip().str.upper()
    bc = bc[bc["nct_id"].isin(ids)]
    mg = priors.mesh_groups_for_trials(bc, c["nct_id"])
    c = c.merge(mg, on="nct_id", how="left")
    sp = aact.load_table("sponsors", usecols=["nct_id", "agency_class", "lead_or_collaborator"])
    sp["nct_id"] = sp["nct_id"].str.strip().str.upper()
    lead = sp[sp["nct_id"].isin(ids) & sp["lead_or_collaborator"].fillna("").str.lower().eq("lead")]
    lead = lead.drop_duplicates("nct_id")
    c = c.merge(lead[["nct_id", "agency_class"]], on="nct_id", how="left")
    c["sponsor_class"] = priors.sponsor_class(c["agency_class"])
    c = c.drop(columns=["agency_class"])
    prim = _modal_by_trial(pairs, pflags, "is_umbrella_n20").rename(columns={"activity_class": "activity_class_primary", "ic_modal": "ic_primary"})
    allm = _modal_by_trial(pairs, pflags, None).rename(columns={"activity_class": "activity_class_all", "ic_modal": "ic_all"})
    c = c.merge(prim, on="nct_id", how="left").merge(allm, on="nct_id", how="left")
    c["activity_class_primary"] = c["activity_class_primary"].fillna("none")
    c["ic_group_primary"] = priors.collapse_ic(c["ic_primary"])
    c["ic_group_all"] = priors.collapse_ic(c["ic_all"])

    # --- dollars (parent rows, overlapping FYs, CPI to base year)
    cpi = load_cpi() if CPI_PATH.exists() else None
    stats["cpi"] = {"available": cpi is not None, "base_year": config.CPI_BASE_YEAR,
                    "index_base_year": (cpi or {}).get(config.CPI_BASE_YEAR),
                    "years_covered": [min(cpi), max(cpi)] if cpi else None}
    prj = pd.read_parquet(PROJECTS_PATH, columns=["CORE_PROJECT_NUM", "SUBPROJECT_ID", "TOTAL_COST", "FY_FILE", "BUDGET_START", "BUDGET_END"])
    prj = prj[prj["CORE_PROJECT_NUM"].isin(set(pairs["core_project_num"]))].copy()
    prj["budget_start"] = exporter.parse_exporter_date(prj["BUDGET_START"])
    prj["budget_end"] = exporter.parse_exporter_date(prj["BUDGET_END"])
    trials = c[["nct_id", "start_date", "trial_end"]].rename(columns={"start_date": "trial_start"})
    pairs_prim = cohort_projects.loc[~cohort_projects["is_umbrella_n20"], ["nct_id", "core_project_num"]]
    d_prim = trial_dollars(pairs_prim, prj, trials, cpi).add_suffix("_primary").rename(columns={"nct_id_primary": "nct_id"})
    d_all = trial_dollars(pairs[["nct_id", "core_project_num"]], prj, trials, cpi).add_suffix("_all").rename(columns={"nct_id_all": "nct_id"})
    c = c.merge(d_prim, on="nct_id", how="left").merge(d_all, on="nct_id", how="left")
    stats["notes"].append("total_cost_*_primary is NaN for trials with no attributable (non-umbrella, N=20) project; 0 when attributable projects exist but no parent row overlaps the trial window")
    stats["notes"].append("TOTAL_COST null on a parent row counts as 0 and is tallied in n_fy_rows_cost_null_*")
    stats["notes"].append("CPI: FY row deflated with the calendar-year CPI-U annual average of its nominal fiscal year (FY2015 -> CY2015)")

    # --- training-window reports (step 2 and step 4); test window never summarised
    tr = c[c["window"] == "train"]
    stats["train_n"] = int(len(tr))
    rep = {}
    for r in umbrella.RULES:
        no_attr = tr[~tr[f"has_attributable_funding_{r}"]]
        rep[r] = {"no_attributable_funding": _split(no_attr), "with_attributable_funding": _split(tr[tr[f"has_attributable_funding_{r}"]])}
    stats["train_attributable_by_rule"] = rep
    stats["train_a2_flag"] = {"withdrawn_zero_enroll": _split(tr[tr["withdrawn_zero_enroll"]]),
                              "withdrawn_total": _split(tr[tr["overall_status"].str.upper().eq("WITHDRAWN")]),
                              "withdrawn_enrollment_null": int((tr["overall_status"].str.upper().eq("WITHDRAWN") & tr["enrollment"].isna()).sum())}
    stats["train_phase_group"] = {k: int(v) for k, v in tr["phase_group"].value_counts().items()}
    stats["train_condition_group"] = {k: int(v) for k, v in tr["condition_group"].value_counts().items()}
    stats["train_n_top_headings"] = {int(k): int(v) for k, v in tr["n_top_headings"].value_counts().sort_index().items()}
    stats["train_top_heading_carriage"] = {k: int(v) for k, v in tr["top_headings"].str.split(";").explode().replace("", pd.NA).dropna().value_counts().items()}
    stats["train_sponsor_class"] = {k: int(v) for k, v in tr["sponsor_class"].value_counts().items()}
    stats["train_activity_class_primary"] = {k: int(v) for k, v in tr["activity_class_primary"].value_counts().items()}
    stats["train_activity_class_all"] = {k: int(v) for k, v in tr["activity_class_all"].value_counts().items()}
    attr = tr[tr["has_attributable_funding_n20"]]
    stats["train_ic_modal_primary_top15"] = {k: int(v) for k, v in attr["ic_primary"].value_counts().head(15).items()}
    stats["train_ic_modal_all_top15"] = {k: int(v) for k, v in tr["ic_all"].value_counts().head(15).items()}
    stats["train_ic_group_primary"] = {k: int(v) for k, v in tr["ic_group_primary"].value_counts().items()}
    stats["config_ic_top10"] = config.IC_TOP10
    stats["config_ic_top10_matches_train_primary"] = (
        sorted(config.IC_TOP10) == sorted(attr["ic_primary"].value_counts().head(config.IC_TOP_N).index.tolist()))
    dol = {"primary_attributable_set": {"overall": _iqr(attr["total_cost_base_primary"])},
           "all_linked": {"overall": _iqr(tr["total_cost_base_all"])}}
    for ph in priors.PHASE_ORDER:
        if (attr["phase_group"] == ph).any():
            dol["primary_attributable_set"][f"phase {ph}"] = _iqr(attr.loc[attr["phase_group"] == ph, "total_cost_base_primary"])
        if (tr["phase_group"] == ph).any():
            dol["all_linked"][f"phase {ph}"] = _iqr(tr.loc[tr["phase_group"] == ph, "total_cost_base_all"])
    dol["primary_attributable_set"]["nominal_overall"] = _iqr(attr["total_cost_nominal_primary"])
    dol["primary_attributable_set"]["n_fy_overlap"] = _iqr(attr["n_fy_overlap_primary"])
    dol["primary_attributable_set"]["n_projects_overlap"] = _iqr(attr["n_projects_overlap_primary"])
    dol["primary_attributable_set"]["n_attributable_projects"] = _iqr(attr["n_attributable_n20"])
    dol["primary_attributable_set"]["trials_with_zero_dollars"] = int((attr["total_cost_base_primary"] == 0).sum())
    dol["primary_attributable_set"]["trials_with_any_cost_null_row"] = int((attr["n_fy_rows_cost_null_primary"] > 0).sum())
    dol["all_linked"]["n_fy_overlap"] = _iqr(tr["n_fy_overlap_all"])
    dol["all_linked"]["n_projects_overlap"] = _iqr(tr["n_projects_overlap_all"])
    dol["all_linked"]["trials_with_zero_dollars"] = int((tr["total_cost_base_all"] == 0).sum())
    stats["train_dollars_2020"] = dol
    # consistency: step-5 overlap used all rows (incl. sub-projects); parent-only overlap can be 0
    stats["train_zero_parent_overlap_all_linked"] = int((tr["n_fy_rows_overlap_all"] == 0).sum())

    cols = ["nct_id", "window", "start_date", "start_year", "trial_end", "trial_end_source", "overall_status", "status_group",
            "o1_label", "withdrawn_zero_enroll", "enrollment", "enrollment_type", "duration_months",
            "linked_projects", "n_linked_projects",
            "n_attributable_n20", "n_attributable_n50", "n_attributable_activity",
            "has_attributable_funding_n20", "has_attributable_funding_n50", "has_attributable_funding_activity",
            "phase", "phase_group", "condition_group", "n_top_headings", "top_headings", "sponsor_class",
            "activity_class_primary", "activity_class_all", "ic_primary", "ic_group_primary", "ic_all", "ic_group_all",
            "total_cost_nominal_primary", "total_cost_base_primary", "n_fy_overlap_primary", "n_fy_rows_overlap_primary",
            "n_fy_rows_cost_null_primary", "n_projects_overlap_primary",
            "total_cost_nominal_all", "total_cost_base_all", "n_fy_overlap_all", "n_fy_rows_overlap_all",
            "n_fy_rows_cost_null_all", "n_projects_overlap_all"]
    c = c[cols]
    if write:
        c.to_parquet(COHORT_PATH, index=False)
        cohort_projects.to_parquet(COHORT_PROJECTS_PATH, index=False)
        priors.mesh_group_map().to_csv(MESH_MAP_PATH, index=False)
        STATS_PATH.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    return c, stats


if __name__ == "__main__":
    _, s = build()
    print(json.dumps(s, indent=2, default=str))
