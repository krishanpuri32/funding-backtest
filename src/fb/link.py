"""Build data/derived/link.parquet: one row per (NCT ID, core project number).

Phase 1. Reads the ExPORTER Clinical Studies file, deduplicates, attaches
the count of fiscal-year Projects rows per core project, and reports join statistics
plus the two distributions (core projects per NCT; NCT IDs per core project with the
top-20 umbrella candidates).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import aact, config, exporter

LINK_PATH = config.DATA_DERIVED / "link.parquet"
PROJECTS_PATH = config.DATA_DERIVED / "projects_fy.parquet"
STATS_PATH = config.DATA_DERIVED / "link_stats.json"

NCT_RE = re.compile(r"^NCT\d{8}$")


def _norm_nct(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _bucket_projects_per_nct(n: pd.Series) -> pd.Series:
    return pd.cut(n, bins=[0, 1, 2, 5, 10**9], labels=["1", "2", "3-5", ">5"])


def _bucket_ncts_per_project(n: pd.Series) -> pd.Series:
    return pd.cut(n, bins=[0, 1, 5, 20, 10**9], labels=["1", "2-5", "6-20", ">20"])


def build(write: bool = True) -> tuple[pd.DataFrame, dict]:
    cs = exporter.load_clinical_studies()
    stats: dict = {"clinical_studies_columns": list(cs.columns), "clinical_studies_rows_raw": int(len(cs))}
    # column discovery: the file has historically carried Core Project Number, ClinicalTrials.gov ID, Study, Study Status
    col_core = next(c for c in cs.columns if "core" in c.lower() and "project" in c.lower())
    col_nct = next(c for c in cs.columns if "clinicaltrials" in c.lower() or c.lower().replace(" ", "") in ("nctid", "nct_id"))
    stats["col_core"], stats["col_nct"] = col_core, col_nct

    link = pd.DataFrame({
        "nct_id": _norm_nct(cs[col_nct]),
        "core_project_num": cs[col_core].astype(str).str.strip().str.upper(),
    })
    stats["rows_with_malformed_nct"] = int((~link["nct_id"].str.match(NCT_RE)).sum())
    link = link[link["nct_id"].str.match(NCT_RE) & link["core_project_num"].ne("") & link["core_project_num"].ne("NAN")]
    stats["rows_after_malformed_drop"] = int(len(link))
    link = link.drop_duplicates().reset_index(drop=True)
    stats["link_rows_dedup"] = int(len(link))
    stats["distinct_nct"] = int(link["nct_id"].nunique())
    stats["distinct_core_projects"] = int(link["core_project_num"].nunique())

    # fiscal-year Projects rows per core project (all FYs, FY2000..latest)
    prj = exporter.load_projects_all()
    prj["CORE_PROJECT_NUM"] = prj["CORE_PROJECT_NUM"].astype(str).str.strip().str.upper()
    if write:
        prj.to_parquet(PROJECTS_PATH, index=False)
    stats["projects_rows_all_fy"] = int(len(prj))
    stats["projects_distinct_core"] = int(prj["CORE_PROJECT_NUM"].nunique())
    fy_counts = prj.groupby("CORE_PROJECT_NUM").size().rename("n_fy_rows")
    link = link.merge(fy_counts, left_on="core_project_num", right_index=True, how="left")
    link["n_fy_rows"] = link["n_fy_rows"].fillna(0).astype(int)
    stats["link_core_projects_with_zero_fy_rows"] = int(link.loc[link["n_fy_rows"] == 0, "core_project_num"].nunique())
    stats["link_rows_with_zero_fy_rows"] = int((link["n_fy_rows"] == 0).sum())
    # activity / IC for the umbrella probe: modal value across FY rows
    meta = (
        prj.groupby("CORE_PROJECT_NUM")
        .agg(activity=("ACTIVITY", lambda s: s.mode().iat[0] if s.notna().any() else None),
             ic=("ADMINISTERING_IC", lambda s: s.mode().iat[0] if s.notna().any() else None))
    )
    link = link.merge(meta, left_on="core_project_num", right_index=True, how="left")

    # NCT IDs in the link file not found in AACT studies
    studies = aact.load_table("studies", usecols=["nct_id"])
    aact_ids = set(_norm_nct(studies["nct_id"]))
    stats["aact_studies_rows"] = int(len(studies))
    link["in_aact"] = link["nct_id"].isin(aact_ids)
    stats["link_nct_not_in_aact"] = int(link.loc[~link["in_aact"], "nct_id"].nunique())

    # distributions
    per_nct = link.groupby("nct_id")["core_project_num"].nunique()
    stats["core_projects_per_nct"] = _bucket_projects_per_nct(per_nct).value_counts().sort_index().to_dict()
    stats["core_projects_per_nct"] = {str(k): int(v) for k, v in stats["core_projects_per_nct"].items()}
    per_proj = link.groupby("core_project_num")["nct_id"].nunique()
    stats["ncts_per_core_project"] = {str(k): int(v) for k, v in _bucket_ncts_per_project(per_proj).value_counts().sort_index().to_dict().items()}
    stats["ncts_per_core_project_quantiles"] = {q: float(per_proj.quantile(float(q))) for q in ["0.5", "0.9", "0.95", "0.99", "0.999"]}
    stats["ncts_per_core_project_max"] = int(per_proj.max())
    top = per_proj.sort_values(ascending=False).head(20).rename("n_nct").reset_index()
    top = top.merge(meta, left_on="core_project_num", right_index=True, how="left")
    stats["top20_projects_by_nct_count"] = top.to_dict(orient="records")
    # candidate umbrella thresholds: how many NCTs / projects would be touched at each N
    cand = {}
    for n in [5, 10, 20, 50, 100]:
        umb = set(per_proj[per_proj > n].index)
        cand[str(n)] = {
            "projects_over_N": int(len(umb)),
            "nct_linked_to_any_umbrella": int(link.loc[link["core_project_num"].isin(umb), "nct_id"].nunique()),
            "nct_linked_only_to_umbrellas": int(
                (link.groupby("nct_id")["core_project_num"].apply(lambda s: s.isin(umb).all())).sum()
            ),
        }
    stats["umbrella_threshold_candidates"] = cand

    if write:
        LINK_PATH.parent.mkdir(parents=True, exist_ok=True)
        link.to_parquet(LINK_PATH, index=False)
        STATS_PATH.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    return link, stats


if __name__ == "__main__":
    _, s = build()
    print(json.dumps(s, indent=2, default=str))
