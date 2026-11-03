"""Baseline priors P (plan §5). Pure collapse rules + a builder.

Frozen priors (phase_group, condition_group, sponsor_class, activity_class_primary,
ic_group_primary) are taken as-is from cohort.parquet (fb.priors, A3-freeze); they are
not re-derived here. New here: intervention type, allocation, masking, facilities,
countries, planned enrollment and planned duration (from version 0, fb.planned).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import missing

INTERVENTION_MAP = {
    "DRUG": "drug",
    "BIOLOGICAL": "biological",
    "DEVICE": "device",
    "BEHAVIORAL": "behavioral",
    "PROCEDURE": "procedure",
    "DIETARY_SUPPLEMENT": "dietary",
}  # everything else (RADIATION, GENETIC, DIAGNOSTIC_TEST, COMBINATION_PRODUCT, OTHER) -> 'other'

P_NUMERIC = ["log_n_facilities", "n_countries", "log_planned_enrollment", "planned_duration_months",
             "start_year", "n_attributable_projects"]
P_CATEGORICAL = ["phase_group", "condition_group", "intervention_type", "allocation", "masking",
                 "sponsor_class", "activity_class", "ic_group"]


def collapse_intervention(interventions: pd.DataFrame, nct_ids) -> pd.Series:
    """One intervention class per trial.

    Each AACT intervention_type maps to drug / biological / device / behavioral / procedure /
    dietary, or 'other'. 'other' is ignored when any specific class is present (placebo and
    usual-care arms are typed OTHER); one specific class -> that class; two or more ->
    'multiple'; only 'other' -> 'other'; no intervention row -> NaN (missing level later).
    """
    i = interventions[["nct_id", "intervention_type"]].dropna(subset=["nct_id"]).copy()
    i["cls"] = i["intervention_type"].fillna("").str.strip().str.upper().map(INTERVENTION_MAP).fillna("other")

    def pick(s: pd.Series) -> str:
        spec = sorted(set(s) - {"other"})
        if len(spec) == 1:
            return spec[0]
        if len(spec) > 1:
            return "multiple"
        return "other"

    per = i.groupby("nct_id")["cls"].agg(pick)
    return pd.Series(pd.Index(nct_ids).map(per), index=range(len(nct_ids)), dtype="object")


def collapse_allocation(allocation: pd.Series) -> pd.Series:
    s = allocation.fillna("").astype(str).str.strip().str.upper()
    out = s.map({"RANDOMIZED": "randomized", "NON_RANDOMIZED": "non-randomized", "NA": "N/A"})
    return out.astype("object")  # null -> NaN -> 'missing' level


def collapse_masking(masking: pd.Series) -> pd.Series:
    s = masking.fillna("").astype(str).str.strip().str.upper()
    out = s.map({"NONE": "none", "SINGLE": "single", "DOUBLE": "double+", "TRIPLE": "double+", "QUADRUPLE": "double+"})
    return out.astype("object")


def count_facilities(facilities: pd.DataFrame, nct_ids) -> pd.Series:
    """Facility rows per trial in the current record; 0 when the trial lists none."""
    n = facilities.groupby("nct_id").size()
    return pd.Series(pd.Index(nct_ids).map(n), index=range(len(nct_ids))).fillna(0).astype(int)


def count_countries(countries: pd.DataFrame, nct_ids) -> pd.Series:
    """Distinct countries per trial, excluding rows AACT marks removed ('t'); 0 when none."""
    c = countries[countries["removed"].fillna("f").str.lower().ne("t")]
    n = c.groupby("nct_id")["name"].nunique()
    return pd.Series(pd.Index(nct_ids).map(n), index=range(len(nct_ids))).fillna(0).astype(int)


def build_p_raw(cohort: pd.DataFrame, designs: pd.DataFrame, interventions: pd.DataFrame,
                facilities: pd.DataFrame, countries: pd.DataFrame, planned: pd.DataFrame) -> pd.DataFrame:
    """Raw (unfilled) P table, one row per cohort trial, in cohort order."""
    ids = cohort["nct_id"].tolist()
    p = pd.DataFrame({"nct_id": ids})
    keep = cohort.set_index("nct_id")
    p["phase_group"] = keep.loc[ids, "phase_group"].values
    p["condition_group"] = keep.loc[ids, "condition_group"].values
    p["sponsor_class"] = keep.loc[ids, "sponsor_class"].values
    p["activity_class"] = keep.loc[ids, "activity_class_primary"].values
    p["ic_group"] = keep.loc[ids, "ic_group_primary"].values
    p["start_year"] = keep.loc[ids, "start_year"].astype(float).values
    p["n_attributable_projects"] = keep.loc[ids, "n_attributable_n20"].astype(float).values
    p["intervention_type"] = collapse_intervention(interventions, ids).values
    d = designs.drop_duplicates("nct_id").set_index("nct_id")
    p["allocation"] = collapse_allocation(pd.Series(pd.Index(ids).map(d["allocation"]))).values
    p["masking"] = collapse_masking(pd.Series(pd.Index(ids).map(d["masking"]))).values
    p["log_n_facilities"] = np.log1p(count_facilities(facilities, ids)).values
    p["n_countries"] = count_countries(countries, ids).astype(float).values
    pl = planned.set_index("nct_id")
    enr = pd.Series(pd.Index(ids).map(pl["planned_enrollment"]), dtype=float)
    p["log_planned_enrollment"] = np.log1p(enr).values
    p["planned_duration_months"] = pd.Series(pd.Index(ids).map(pl["planned_duration_months"]), dtype=float).values
    return p


def fill_p(raw: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out, med = missing.fill_numeric(raw, P_NUMERIC, medians)
    out = missing.fill_categorical(out, P_CATEGORICAL)
    return out, med
