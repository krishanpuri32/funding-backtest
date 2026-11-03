"""Amendment A1 umbrella-award flags and Amendment A2 withdrawn/zero-enrollment flag.

Pure functions over small frames; tested on hand-built fixtures (tests/test_umbrella.py).
"""
from __future__ import annotations

import pandas as pd

from . import config

RULES = ("n20", "n50", "activity")


def project_flags(link: pd.DataFrame) -> pd.DataFrame:
    """One row per core project from the WHOLE link file (not the cohort subset).

    Columns: core_project_num, n_nct_linked, activity, is_umbrella_n20 (> 20 NCT IDs),
    is_umbrella_n50 (> 50), is_umbrella_activity (activity code in P30 / U10, any count).
    """
    g = link.groupby("core_project_num").agg(
        n_nct_linked=("nct_id", "nunique"),
        activity=("activity", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    ).reset_index()
    g["is_umbrella_n20"] = g["n_nct_linked"] > config.UMBRELLA_N_PRIMARY
    g["is_umbrella_n50"] = g["n_nct_linked"] > config.UMBRELLA_N_ALT
    g["is_umbrella_activity"] = g["activity"].fillna("").str.upper().isin(config.UMBRELLA_ACTIVITY_CODES)
    return g


def trial_flags(pairs: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Per trial: linked projects, counts and has_attributable_funding_<rule> (>= 1 non-umbrella project).

    `pairs` has nct_id, core_project_num (cohort trials only); `flags` from project_flags.
    """
    j = pairs.merge(flags, on="core_project_num", how="left")
    for r in RULES:
        j[f"is_umbrella_{r}"] = j[f"is_umbrella_{r}"].fillna(False).astype(bool)
    out = j.groupby("nct_id").agg(
        linked_projects=("core_project_num", lambda s: ";".join(sorted(set(s)))),
        n_linked_projects=("core_project_num", "nunique"),
        **{f"n_attributable_{r}": (f"is_umbrella_{r}", lambda s: int((~s).sum())) for r in RULES},
    ).reset_index()
    for r in RULES:
        out[f"has_attributable_funding_{r}"] = out[f"n_attributable_{r}"] > 0
    return out


def withdrawn_zero_enroll(overall_status: pd.Series, enrollment: pd.Series) -> pd.Series:
    """A2 flag: overall_status WITHDRAWN and enrollment exactly 0 (null enrollment -> False)."""
    st = overall_status.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "_")
    enr = pd.to_numeric(enrollment, errors="coerce")
    return (st == "WITHDRAWN") & (enr == 0)
