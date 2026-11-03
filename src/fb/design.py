"""Design matrices (plan §6): one function for every model.

- P numeric as is (log transforms already in the P table) except planned duration, which
  enters as log(1 + months) (A4c). Each P numeric carries its `_missing` indicator (plan §6).
- P categoricals one-hot over the TRAINING levels (the `missing` level included); a level
  unseen in training is all-zero.
- F / F₂ numeric (the plan §5 features only; `f_*` diagnostics never enter), median-filled
  with TRAINING medians. `F_*_missing` indicators only when `f_missing=True` (S1 sensitivity
  and the hazard model); the primary landmark models carry none (A4b).
- No label, status or date column can enter: `design_matrix` raises if one does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features_f, features_p, missing

FORBIDDEN = {"o1", "o2", "o3", "o4", "o2_rule", "status_group", "overall_status", "outcome_date", "outcome_date_source",
             "months_to_exit", "withdrawn_zero_enroll", "enrollment", "enrollment_type", "y", "event", "f_rows_reason",
             "f_n_rows", "f_n_rows_overlap", "f_n_rows_cost_null", "f_total_dollars_2020", "f_gap_window_n"}


def f_columns(variant: str | None) -> list[str]:
    if variant is None:
        return []
    return list(features_f.F_COLUMNS if variant == "F" else features_f.F2_COLUMNS)


def p_levels(p_raw: pd.DataFrame) -> dict[str, list[str]]:
    """Training levels per P categorical, after the missing-level fill."""
    filled = missing.fill_categorical(p_raw, features_p.P_CATEGORICAL)
    return {c: sorted(filled[c].astype(str).unique()) for c in features_p.P_CATEGORICAL}


def design_matrix(p_ids, p_raw: pd.DataFrame, p_medians: dict, levels: dict[str, list[str]],
                  f_keys=None, f_raw: pd.DataFrame | None = None, f_medians: dict | None = None,
                  variant: str | None = None, f_missing: bool = False, drop_p_missing: bool = False) -> pd.DataFrame:
    """One row per entry of `p_ids` (nct_id; may repeat for person-period rows).

    f_keys: the key of each row in f_raw['nct_id'] (defaults to p_ids). Medians and levels are
    the training ones, passed in; nothing is estimated here.
    drop_p_missing: Phase 4-X exploratory variant — omit the six P `<col>_missing` indicators
    (median-filled values and categorical `missing` levels stay).
    """
    p_ids = list(p_ids)
    p = p_raw.drop_duplicates("nct_id").set_index("nct_id").loc[p_ids].reset_index()
    p, _ = missing.fill_numeric(p, features_p.P_NUMERIC, p_medians)
    p = missing.fill_categorical(p, features_p.P_CATEGORICAL)
    cols: dict[str, np.ndarray] = {}
    for c in features_p.P_NUMERIC:
        v = p[c].astype(float).values
        cols["log_planned_duration" if c == "planned_duration_months" else c] = (
            np.log1p(np.clip(v, 0, None)) if c == "planned_duration_months" else v)
    if not drop_p_missing:
        for c in features_p.P_NUMERIC:
            cols[f"{c}_missing"] = p[f"{c}_missing"].astype(float).values
    for c in features_p.P_CATEGORICAL:
        s = p[c].astype(str).values
        for lvl in levels[c]:
            cols[f"{c}={lvl}"] = (s == lvl).astype(float)
    fc = f_columns(variant)
    if fc:
        if f_raw is None or f_medians is None:
            raise ValueError("variant given without f_raw / f_medians")
        keys = p_ids if f_keys is None else list(f_keys)
        f = f_raw.drop_duplicates("nct_id").set_index("nct_id").loc[keys].reset_index()
        f, _ = missing.fill_numeric(f, fc, f_medians)
        for c in fc:
            cols[c] = f[c].astype(float).values
        if f_missing:
            for c in fc:
                cols[f"{c}_missing"] = f[f"{c}_missing"].astype(float).values
    X = pd.DataFrame(cols)
    bad = FORBIDDEN & set(X.columns)
    if bad:
        raise ValueError(f"label / diagnostic column in design matrix: {sorted(bad)}")
    return X


def f_block(X: pd.DataFrame) -> list[str]:
    """Columns of the funding block (F / F₂ features and their indicators) — what the
    permutation null shuffles."""
    return [c for c in X.columns if c.startswith("F_")]
