"""Baseline priors P needed for the base-rate table (plan §5; A3-freeze).

Only the five priors the table needs: phase, condition group, activity class,
administering IC, lead sponsor class (plus start year, which is a date field).
The collapsing rules here are FROZEN once reported in Phase 2; later phases import
these functions and never re-derive them.
"""
from __future__ import annotations

import pandas as pd

from . import config

# --- phase ---------------------------------------------------------------------
PHASE_MAP = {
    "EARLY_PHASE1": "Early 1",
    "PHASE1": "1",
    "PHASE1/PHASE2": "1-2",
    "PHASE2": "2",
    "PHASE2/PHASE3": "2-3",
    "PHASE3": "3",
    "PHASE4": "4",
    "NA": "N/A",
}
PHASE_ORDER = ["Early 1", "1", "1-2", "2", "2-3", "3", "4", "N/A", "missing"]


def collapse_phase(phase: pd.Series) -> pd.Series:
    s = phase.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "")
    out = s.map(PHASE_MAP)
    out[s.eq("")] = "missing"
    out = out.fillna("missing")
    return out


# --- condition group (MeSH top-level tree branch, collapsed to <= 15 groups) ------
# AACT `browse_conditions` rows of mesh_type 'mesh-ancestor' carry every ancestor of a
# trial's listed MeSH conditions, including the top-level C-branch headings and F03
# 'Mental Disorders'. Each top-level heading maps to one group; the rows below are
# (MeSH tree number, heading as spelled in AACT, group). Spellings for C12/C13 cover
# both the pre-2023 and 2023+ MeSH trees because AACT ancestors mix them.
MESH_TOP_LEVEL: list[tuple[str, str, str]] = [
    ("C04", "Neoplasms", "Neoplasms"),
    ("C01", "Infections", "Infections"),
    ("F03", "Mental Disorders", "Mental & behavioral"),
    ("F01", "Behavior", "Mental & behavioral"),  # F01 'Behavior and Behavior Mechanisms' child; see F01 rule below
    ("C14", "Cardiovascular Diseases", "Cardiovascular"),
    ("C10", "Nervous System Diseases", "Nervous system"),
    ("C18", "Nutritional and Metabolic Diseases", "Nutritional, metabolic & endocrine"),
    ("C19", "Endocrine System Diseases", "Nutritional, metabolic & endocrine"),
    ("C08", "Respiratory Tract Diseases", "Respiratory"),
    ("C06", "Digestive System Diseases", "Digestive"),
    ("C12", "Urogenital Diseases", "Urogenital & pregnancy"),
    ("C12", "Male Urogenital Diseases", "Urogenital & pregnancy"),
    ("C13", "Female Urogenital Diseases and Pregnancy Complications", "Urogenital & pregnancy"),
    ("C15", "Hemic and Lymphatic Diseases", "Hemic, lymphatic & immune"),
    ("C20", "Immune System Diseases", "Hemic, lymphatic & immune"),
    ("C05", "Musculoskeletal Diseases", "Musculoskeletal, skin & connective"),
    ("C17", "Skin and Connective Tissue Diseases", "Musculoskeletal, skin & connective"),
    ("C16", "Congenital, Hereditary, and Neonatal Diseases and Abnormalities", "Congenital, hereditary & neonatal"),
    ("C11", "Eye Diseases", "Eye, ENT & stomatognathic"),
    ("C09", "Otorhinolaryngologic Diseases", "Eye, ENT & stomatognathic"),
    ("C07", "Stomatognathic Diseases", "Eye, ENT & stomatognathic"),
    ("C26", "Wounds and Injuries", "Injuries, chemical & environmental"),
    ("C25", "Chemically-Induced Disorders", "Injuries, chemical & environmental"),
    ("C21", "Disorders of Environmental Origin", "Injuries, chemical & environmental"),
    ("C24", "Occupational Diseases", "Injuries, chemical & environmental"),
    ("C23", "Pathological Conditions, Signs and Symptoms", "Other / symptoms only"),
    ("C22", "Animal Diseases", "Other / symptoms only"),
]
# Priority when a trial carries several top-level headings: first match in this order wins.
MESH_GROUP_PRIORITY = [
    "Neoplasms",
    "Infections",
    "Mental & behavioral",
    "Cardiovascular",
    "Nervous system",
    "Nutritional, metabolic & endocrine",
    "Respiratory",
    "Digestive",
    "Urogenital & pregnancy",
    "Hemic, lymphatic & immune",
    "Musculoskeletal, skin & connective",
    "Congenital, hereditary & neonatal",
    "Eye, ENT & stomatognathic",
    "Injuries, chemical & environmental",
    "Other / symptoms only",
]
MESH_GROUP_NONE = "No MeSH term"  # missing level, not a group: trial has no browse_conditions row
MESH_GROUP_OTHER = "Other / symptoms only"
assert len(MESH_GROUP_PRIORITY) <= config.MESH_MAX_GROUPS
_HEADING_TO_GROUP = {h: g for _, h, g in MESH_TOP_LEVEL}
_GROUP_RANK = {g: i for i, g in enumerate(MESH_GROUP_PRIORITY)}


def mesh_group_map() -> pd.DataFrame:
    """The full heading -> group map with priority rank, for review (written to data/derived)."""
    rows = [{"mesh_tree": t, "mesh_heading": h, "group": g, "group_priority": _GROUP_RANK[g] + 1} for t, h, g in MESH_TOP_LEVEL]
    rows.append({"mesh_tree": "(none)", "mesh_heading": "(no browse_conditions row)", "group": MESH_GROUP_NONE, "group_priority": None})
    rows.append({"mesh_tree": "(other)", "mesh_heading": "(MeSH rows, none of the headings above)", "group": MESH_GROUP_OTHER, "group_priority": _GROUP_RANK[MESH_GROUP_OTHER] + 1})
    return pd.DataFrame(rows)


def mesh_groups_for_trials(browse_conditions: pd.DataFrame, nct_ids: pd.Series) -> pd.DataFrame:
    """Assign one condition group per trial.

    Rule: collect the trial's MeSH terms (mesh-list and mesh-ancestor rows alike) that
    are top-level headings in MESH_TOP_LEVEL; map each to its group; take the group with
    the highest priority. Trials with MeSH rows but no top-level heading -> 'Other /
    symptoms only' (in practice these are F01 behaviour terms that AACT lists without
    the F01 root; 'Behavior' itself is in the table so most land in 'Mental & behavioral').
    Trials with no browse_conditions row -> 'No MeSH term'.
    Returns nct_id, condition_group, n_top_headings, top_headings (';'-joined).
    """
    bc = browse_conditions[["nct_id", "mesh_term"]].dropna()
    bc = bc[bc["mesh_term"].isin(_HEADING_TO_GROUP)]
    bc = bc.assign(group=bc["mesh_term"].map(_HEADING_TO_GROUP), rank=lambda d: d["group"].map(_GROUP_RANK))
    agg = bc.groupby("nct_id").agg(
        rank=("rank", "min"),
        n_top_headings=("mesh_term", "nunique"),
        top_headings=("mesh_term", lambda s: ";".join(sorted(set(s)))),
    )
    agg["condition_group"] = agg["rank"].map(lambda r: MESH_GROUP_PRIORITY[int(r)])
    has_any_row = set(browse_conditions["nct_id"].dropna())
    out = pd.DataFrame({"nct_id": pd.Series(nct_ids).astype(str).values})
    out = out.merge(agg[["condition_group", "n_top_headings", "top_headings"]], left_on="nct_id", right_index=True, how="left")
    no_heading = out["condition_group"].isna()
    out.loc[no_heading & out["nct_id"].isin(has_any_row), "condition_group"] = MESH_GROUP_OTHER
    out.loc[no_heading & ~out["nct_id"].isin(has_any_row), "condition_group"] = MESH_GROUP_NONE
    out["n_top_headings"] = out["n_top_headings"].fillna(0).astype(int)
    out["top_headings"] = out["top_headings"].fillna("")
    return out


# --- activity code class ----------------------------------------------------------
ACTIVITY_CLASS_ORDER = ["R", "U", "P", "K", "other"]  # also the tie-break order for the modal class


def activity_class(activity: pd.Series) -> pd.Series:
    """R01-class = any R*, U-class = U*, P-class = P*, K-class = K*, else 'other' (null -> 'other')."""
    first = activity.fillna("").astype(str).str.strip().str.upper().str[:1]
    out = first.where(first.isin(["R", "U", "P", "K"]), "other")
    return out


def modal_activity_class(classes: pd.Series) -> str:
    """Most frequent class; ties broken by ACTIVITY_CLASS_ORDER (R before U before P before K before other)."""
    if classes.empty:
        return "none"
    vc = classes.value_counts()
    top = vc[vc == vc.max()].index.tolist()
    return next(c for c in ACTIVITY_CLASS_ORDER if c in top)


def modal_value(values: pd.Series) -> str | None:
    """Most frequent non-null value; ties broken alphabetically (deterministic)."""
    v = values.dropna()
    if v.empty:
        return None
    vc = v.value_counts()
    return sorted(vc[vc == vc.max()].index.tolist())[0]


def collapse_ic(ic: pd.Series, top: list[str] = config.IC_TOP10) -> pd.Series:
    s = ic.fillna("").astype(str).str.strip().str.upper()
    out = s.where(s.isin(top), "other")
    out[s.eq("")] = "none"
    return out


# --- lead sponsor class -------------------------------------------------------------
def sponsor_class(agency_class: pd.Series) -> pd.Series:
    """AACT `sponsors.agency_class` of the lead sponsor, upper-cased; null -> 'missing'."""
    s = agency_class.fillna("").astype(str).str.strip().str.upper()
    return s.where(s.ne(""), "missing")
