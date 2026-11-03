"""Funded dollars per trial (plan §1 (c); A1): parent rows only, overlapping fiscal
years, attributable projects only, CPI-adjusted to the base year.

Deliberately NOT here: any shape / gap / supplement / NCE quantity (Phase 3 F features).
"""
from __future__ import annotations

import pandas as pd

from . import config, rules
from .cpi import deflator


def trial_dollars(
    pairs: pd.DataFrame,
    projects: pd.DataFrame,
    trials: pd.DataFrame,
    cpi: dict[int, float] | None,
    base_year: int = config.CPI_BASE_YEAR,
) -> pd.DataFrame:
    """Sum TOTAL_COST over parent FY rows (SUBPROJECT_ID null) of the projects in `pairs`
    whose budget period overlaps the trial window.

    pairs:    nct_id, core_project_num  (already restricted to attributable projects)
    projects: CORE_PROJECT_NUM, SUBPROJECT_ID, TOTAL_COST, FY_FILE, budget_start, budget_end
    trials:   nct_id, trial_start, trial_end
    cpi:      {year: index}; None -> nominal dollars (flagged in the output)
    Returns one row per trial in `pairs`: total_cost_nominal, total_cost_base (NaN if nominal
    only), n_fy_overlap (distinct fiscal years), n_fy_rows_overlap, n_fy_rows_cost_null,
    n_projects_overlap (distinct projects with >= 1 overlapping parent row).
    """
    par = projects[projects["SUBPROJECT_ID"].isna()].copy()
    j = pairs.merge(par, left_on="core_project_num", right_on="CORE_PROJECT_NUM", how="inner")
    j = j.merge(trials[["nct_id", "trial_start", "trial_end"]], on="nct_id", how="left")
    j["overlap"] = rules.fy_overlaps_vec(j)
    j = j[j["overlap"]].copy()
    j["cost"] = pd.to_numeric(j["TOTAL_COST"], errors="coerce")
    j["cost_null"] = j["cost"].isna()
    j["cost"] = j["cost"].fillna(0.0)
    if cpi is not None:
        j["cost_base"] = j["cost"] * deflator(cpi, j["FY_FILE"], base_year)
    else:
        j["cost_base"] = float("nan")
    out = j.groupby("nct_id").agg(
        total_cost_nominal=("cost", "sum"),
        total_cost_base=("cost_base", "sum"),
        n_fy_overlap=("FY_FILE", "nunique"),
        n_fy_rows_overlap=("FY_FILE", "size"),
        n_fy_rows_cost_null=("cost_null", "sum"),
        n_projects_overlap=("core_project_num", "nunique"),
    ).reset_index()
    if cpi is None:
        out["total_cost_base"] = float("nan")
    # trials in pairs with no overlapping parent row: zero dollars, zero years
    all_ids = pd.DataFrame({"nct_id": pairs["nct_id"].unique()})
    out = all_ids.merge(out, on="nct_id", how="left")
    for c in ["total_cost_nominal", "n_fy_overlap", "n_fy_rows_overlap", "n_fy_rows_cost_null", "n_projects_overlap"]:
        out[c] = out[c].fillna(0)
    if cpi is not None:
        out["total_cost_base"] = out["total_cost_base"].fillna(0.0)
    for c in ["n_fy_overlap", "n_fy_rows_overlap", "n_fy_rows_cost_null", "n_projects_overlap"]:
        out[c] = out[c].astype(int)
    return out
