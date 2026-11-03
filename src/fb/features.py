"""Phase 3 builder (plan §5, §10; A4): P, F, F₂ feature tables for the TRAINING
window, the version-0 planned-value report, the outcome-date (leakage cut) report, and the
200-string why_stopped sample.

    python -m fb.features

Writes (data/derived/): features_P.parquet (attributable), features_P_all.parquet (all
training trials), features_F.parquet, features_F2.parquet (attributable), features_stats.json,
features_coverage.md, why_stopped_sample.csv.
No outcome label is written to any feature table, no feature is compared with an outcome,
and no test-window feature value is computed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import aact, config, features_f, features_p, leakage, missing, o2, planned
from .cpi import CPI_PATH, load_cpi
from .link import PROJECTS_PATH

COHORT_PATH = config.DATA_DERIVED / "cohort.parquet"
COHORT_PROJECTS_PATH = config.DATA_DERIVED / "cohort_projects.parquet"
PROJECT_COLUMNS = ["CORE_PROJECT_NUM", "SUBPROJECT_ID", "FY_FILE", "APPLICATION_TYPE", "SUFFIX", "SUPPORT_YEAR",
                   "PROJECT_START", "PROJECT_END", "BUDGET_START", "BUDGET_END", "TOTAL_COST", "INDIRECT_COST_AMT",
                   "DIRECT_COST_AMT", "TOTAL_COST_SUB_PROJECT", "ACTIVITY"]


def _aact(table: str, cols: list[str], ids: set[str]) -> pd.DataFrame:
    d = aact.load_table(table, usecols=cols)
    d["nct_id"] = d["nct_id"].str.strip().str.upper()
    return d[d["nct_id"].isin(ids)]


def _vc(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}


def _q(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "min": float(s.min()), "q1": float(s.quantile(.25)), "median": float(s.median()),
            "q3": float(s.quantile(.75)), "p99": float(s.quantile(.99)), "max": float(s.max())}


def planned_report(cohort: pd.DataFrame, v0: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Step 1: apply the planned-value rule to every cohort trial; report the training window, plus the
    identity of the extreme and zero v0 enrollments cohort-wide (registry fields only)."""
    m = cohort[["nct_id", "window", "start_date"]].merge(v0, on="nct_id", how="left")
    pv = planned.planned_values(m)
    pv = pv.merge(cohort[["nct_id", "window"]], on="nct_id")
    tr = pv[pv["window"] == "train"]
    rep = {
        "train_n": int(len(tr)),
        "train_usable_planned_enrollment": int(tr["planned_enrollment"].notna().sum()),
        "train_usable_planned_duration": int(tr["planned_duration_months"].notna().sum()),
        "train_planned_enrollment_reason": _vc(tr["planned_enrollment_reason"]),
        "train_planned_duration_reason": _vc(tr["planned_duration_reason"]),
        "train_planned_start_source": _vc(tr["planned_start_source"]),
        "train_registered_late": int(tr["registered_late"].sum()),
        "train_v0_enrollment_actual": int(tr["v0_enrollment_actual"].sum()),
        "train_late_and_actual": int((tr["registered_late"] & tr["v0_enrollment_actual"]).sum()),
        "train_o34_eligible": int(tr["o34_eligible"].sum()),
        "train_planned_enrollment_dist": _q(tr["planned_enrollment"]),
        "train_planned_duration_dist": _q(tr["planned_duration_months"]),
    }
    enr = pd.to_numeric(m["v0_enrollment"], errors="coerce")
    top = m.assign(e=enr).sort_values("e", ascending=False).head(5)
    rep["cohort_top5_v0_enrollment"] = [
        {"nct_id": r.nct_id, "window": r.window, "v0_enrollment": r.v0_enrollment, "v0_enrollment_type": r.v0_enrollment_type,
         "v0_overall_status": r.v0_overall_status} for r in top.itertuples()]
    z = m[enr.eq(0)]
    rep["cohort_zero_v0_enrollment"] = [
        {"nct_id": r.nct_id, "window": r.window, "v0_enrollment_type": r.v0_enrollment_type, "v0_overall_status": r.v0_overall_status}
        for r in z.itertuples()]
    return pv, rep


def null_cost_report(prj: pd.DataFrame) -> dict:
    """What the parent rows with null TOTAL_COST are (projects behind cohort trials)."""
    par = prj[prj["SUBPROJECT_ID"].isna()]
    nul = par[pd.to_numeric(par["TOTAL_COST"], errors="coerce").isna()]
    return {
        "parent_rows": int(len(par)), "null_total_cost_rows": int(len(nul)),
        "null_rows_with_total_cost_sub_project": int(nul["TOTAL_COST_SUB_PROJECT"].notna().sum()),
        "null_rows_with_direct_cost": int(nul["DIRECT_COST_AMT"].notna().sum()),
        "null_rows_with_indirect_cost": int(nul["INDIRECT_COST_AMT"].notna().sum()),
        "null_rows_by_activity_top10": {k: int(v) for k, v in nul["ACTIVITY"].value_counts().head(10).items()},
        "null_rows_by_fy": {int(k): int(v) for k, v in nul["FY_FILE"].value_counts().sort_index().items()},
        "handling": "no substitute amount exists; row counts as an award action of unknown amount (counts features) and is skipped by dollar features",
    }


def build(write: bool = True) -> dict:
    stats: dict = {}
    cohort = pd.read_parquet(COHORT_PATH)
    cp = pd.read_parquet(COHORT_PROJECTS_PATH)
    train = cohort[cohort["window"] == "train"].reset_index(drop=True)
    ids = set(train["nct_id"])
    attr = train[train["has_attributable_funding_n20"].fillna(False).astype(bool)].reset_index(drop=True)
    stats["train_n"] = int(len(train))
    stats["train_attributable_n"] = int(len(attr))

    # --- step 1: planned values (planned-value rule) -------------------------------------------
    pv, stats["planned"] = planned_report(cohort, planned.load_v0())
    pv_train = pv[pv["window"] == "train"]

    # --- step 2: outcome dates (leakage cut), training window -----------------------
    st = _aact("studies", ["nct_id", "primary_completion_date", "primary_completion_date_type", "completion_date",
                           "completion_date_type", "last_update_submitted_date", "why_stopped"], ids)
    od_in = train[["nct_id", "status_group", "overall_status"]].merge(st, on="nct_id", how="left")
    od = leakage.outcome_dates(od_in)
    od = od.merge(train[["nct_id", "status_group"]], on="nct_id")
    stats["outcome_date"] = {
        "source_by_status_group": {g: _vc(d["outcome_date_source"]) for g, d in od.groupby("status_group")},
        "flagged_by_status_group": {g: int(d["outcome_date_flag"].sum()) for g, d in od.groupby("status_group")},
        "missing": int(od["outcome_date"].isna().sum()),
        "event_by_status": {s: _vc(od_in.loc[od_in["overall_status"].eq(s), "nct_id"].map(od.set_index("nct_id")["outcome_date_source"]))
                            for s in ["TERMINATED", "SUSPENDED", "WITHDRAWN"]},
    }

    # --- step 3: P features ----------------------------------------------------------
    designs = _aact("designs", ["nct_id", "allocation", "masking"], ids)
    interventions = _aact("interventions", ["nct_id", "intervention_type"], ids)
    facilities = _aact("facilities", ["nct_id"], ids)
    countries = _aact("countries", ["nct_id", "name", "removed"], ids)
    p_raw_all = features_p.build_p_raw(train, designs, interventions, facilities, countries, pv_train)
    p_raw_attr = p_raw_all[p_raw_all["nct_id"].isin(set(attr["nct_id"]))].reset_index(drop=True)
    p_all, med_p_all = features_p.fill_p(p_raw_all)
    p_attr, med_p_attr = features_p.fill_p(p_raw_attr)
    stats["p_medians_all"] = med_p_all
    stats["p_medians_attributable"] = med_p_attr

    # --- steps 4–5: F and F₂ ---------------------------------------------------------
    pairs = cp.loc[~cp["is_umbrella_n20"].fillna(False).astype(bool) & cp["nct_id"].isin(set(attr["nct_id"])),
                   ["nct_id", "core_project_num"]].drop_duplicates()
    prj_all = pd.read_parquet(PROJECTS_PATH, columns=PROJECT_COLUMNS)
    stats["null_total_cost"] = null_cost_report(prj_all[prj_all["CORE_PROJECT_NUM"].isin(set(cp["core_project_num"]))])
    prj = prj_all[prj_all["CORE_PROJECT_NUM"].isin(set(pairs["core_project_num"]))]
    del prj_all
    cpi = load_cpi() if CPI_PATH.exists() else None
    if cpi is None:
        raise SystemExit("CPI file missing: run `make download-cpi` first")
    rows = features_f.prepare_rows(prj, cpi)
    trials = attr[["nct_id", "start_date", "trial_end"]].rename(columns={"start_date": "trial_start"}).merge(
        od[["nct_id", "outcome_date"]], on="nct_id", how="left")
    trials["trial_start"] = pd.to_datetime(trials["trial_start"])
    trials["trial_end"] = pd.to_datetime(trials["trial_end"])
    pe = pv_train.set_index("nct_id")["planned_enrollment"]
    f_raw = features_f.compute_features(pairs, rows, trials, pe, "F")
    f2_raw = features_f.compute_features(pairs, rows, trials, pe, "F2")
    f_num = [c for c in features_f.F_COLUMNS]
    f2_num = [c for c in features_f.F2_COLUMNS]
    f_fill, med_f = missing.fill_numeric(f_raw, f_num)
    f2_fill, med_f2 = missing.fill_numeric(f2_raw, f2_num)
    stats["f_medians"] = med_f
    stats["f2_medians"] = med_f2
    stats["f_rows_reason"] = _vc(f_raw["f_rows_reason"])
    stats["f2_rows_reason"] = _vc(f2_raw["f_rows_reason"])
    stats["f_rows_used"] = {"rows_overlap_total": int(f_raw["f_n_rows_overlap"].sum()), "rows_after_cut_total": int(f_raw["f_n_rows"].sum()),
                            "rows_cost_null_after_cut": int(f_raw["f_n_rows_cost_null"].sum()),
                            "trials_with_any_cost_null_row": int((f_raw["f_n_rows_cost_null"] > 0).sum())}
    stats["f_zero_dollar_phase2_definition"] = int((attr["total_cost_base_primary"] == 0).sum())
    stats["f_row_quirks"] = {
        "parent_rows_attributable_projects": int(len(rows)),
        "rows_total_cost_under_1000": int((rows["cost"] < 1000).sum()),
        "rows_total_cost_eq_1": int(rows["cost"].eq(1).sum()),
        "rows_fy2006plus_total_below_direct_plus_indirect": int(((rows["FY_FILE"] >= 2006) & (rows["cost"] + 1 < rows["direct"] + rows["indirect"])).sum()),
        "trials_F_total_below_10k_dollars": int((f_raw["f_total_dollars_2020"] < 10_000).sum()),
    }
    dur = pv_train["planned_duration_months"]
    stats["planned"]["train_planned_duration_over_120_months"] = int((dur > 120).sum())
    stats["planned"]["train_planned_duration_over_240_months"] = [
        {"nct_id": r.nct_id, "months": round(float(r.planned_duration_months), 1)}
        for r in pv_train[dur > 240].itertuples()]
    stats["f_gap_window_n"] = _q(f_raw["f_gap_window_n"])
    stats["f_total_dollars_2020"] = _q(f_raw["f_total_dollars_2020"])
    nce_rows = rows[~rows["app_type"].isin(features_f.COMPETING_TYPES)]
    stats["f_nce_rule"] = {
        "trials_F_nce_1": int((f_raw["F_nce"] == 1).sum()), "trials_F_nce_0": int((f_raw["F_nce"] == 0).sum()),
        "trials_F_nce_missing": int(f_raw["F_nce"].isna().sum()),
        "rows_all_years_meeting_criterion_A": int((pd.to_datetime(nce_rows["project_end"]) > pd.to_datetime(nce_rows["seg_orig_end"])
                                                   + pd.Timedelta(days=config.NCE_EXTENSION_TOLERANCE_DAYS)).sum()),
        "rows_all_years_meeting_criterion_B": int(((rows["support_year"] > rows["seg_term_years"]) & rows["cost"].eq(0)).sum()),
        "rows_all_years_total_cost_zero": int(rows["cost"].eq(0).sum()),
    }

    # --- step 6: O2 sample (training-window O1 events) --------------------------------
    ev = train.loc[train["status_group"].eq("event"), ["nct_id"]].merge(st[["nct_id", "why_stopped"]], on="nct_id", how="left")
    sample = o2.draw_sample(ev)
    stats["o2"] = {"train_events": int(len(ev)), "train_events_with_why_stopped": int(ev["why_stopped"].notna().sum()),
                   "sample_n": int(len(sample)), "seed": config.SEED,
                   "keywords": o2.KEYWORD_PATTERNS}

    # --- step 7: coverage ------------------------------------------------------------
    cov_p_all = missing.coverage_table(p_raw_all, features_p.P_NUMERIC, features_p.P_CATEGORICAL)
    cov_p = missing.coverage_table(p_raw_attr, features_p.P_NUMERIC, features_p.P_CATEGORICAL)
    cov_f = missing.coverage_table(f_raw, f_num, [])
    cov_f2 = missing.coverage_table(f2_raw, f2_num, [])
    md = ["# Phase 3 feature coverage — training window (start 2008–2015)", "",
          f"## P — all training trials (features_P_all.parquet, n = {len(p_raw_all)})", "", missing.to_markdown(cov_p_all), "",
          f"## P — attributable training trials (features_P.parquet, n = {len(p_raw_attr)})", "", missing.to_markdown(cov_p), "",
          f"## F — attributable training trials (features_F.parquet, n = {len(f_raw)})", "", missing.to_markdown(cov_f), "",
          f"## F₂ — attributable training trials (features_F2.parquet, n = {len(f2_raw)})", "", missing.to_markdown(cov_f2), ""]
    stats["shapes"] = {"features_P_all": list(p_all.shape), "features_P": list(p_attr.shape),
                       "features_F": list(f_fill.shape), "features_F2": list(f2_fill.shape)}

    if write:
        p_attr.to_parquet(config.FEATURES_P_PATH, index=False)
        p_all.to_parquet(config.FEATURES_P_ALL_PATH, index=False)
        f_fill.to_parquet(config.FEATURES_F_PATH, index=False)
        f2_fill.to_parquet(config.FEATURES_F2_PATH, index=False)
        write_sample(sample)
        config.FEATURES_COVERAGE_PATH.write_text("\n".join(md))
        config.FEATURES_STATS_PATH.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    stats["_coverage_md"] = "\n".join(md)
    return stats


def write_sample(sample: pd.DataFrame, path=None) -> None:
    """Write the O2 sample once. An existing file holds the blind labels: it is kept when the drawn
    sample has the same NCT IDs in the same order, and a different draw stops the build."""
    path = path or config.WHY_STOPPED_SAMPLE_PATH
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        if old["nct_id"].tolist() != sample["nct_id"].astype(str).tolist():
            raise SystemExit(f"{path.name}: drawn O2 sample differs from the labelled file; labels kept, stopping")
        return
    sample.to_csv(path, index=False)


if __name__ == "__main__":
    s = build()
    md = s.pop("_coverage_md")
    print(json.dumps(s, indent=2, default=str))
    print(md)
