"""Phase 3 step 0: the §11 sample-size gate re-stated on the attributable test window (A3).

Counts O1 events and Completed trials among test-window trials with >= 1 attributable
(non-umbrella) linked project, under the primary rule (A1, N = 20) and the two grid rules.
This is the ONLY test-window outcome quantity Phase 3 computes.

    python -m fb.gate
"""
from __future__ import annotations

import json

import pandas as pd

from . import config

COHORT_PATH = config.DATA_DERIVED / "cohort.parquet"
RULE_COLUMNS = {
    "n20": "has_attributable_funding_n20",
    "n50": "has_attributable_funding_n50",
    "activity": "has_attributable_funding_activity",
}


def gate_cell(o1_events: int) -> str:
    """§11 cell for a test-window O1 event count."""
    if o1_events >= config.GATE_FULL:
        return "full plan (>= 300)"
    if o1_events >= config.GATE_POOLED:
        return "pooled O1 only, no strata, no O2-O4 (150-300)"
    return "stop (< 150)"


def gate_counts(cohort: pd.DataFrame) -> dict:
    """Per attribution rule: attributable test trials, O1 events, Completed, and the §11 cell."""
    test = cohort[cohort["window"] == "test"]
    out = {}
    for rule, col in RULE_COLUMNS.items():
        a = test[test[col].fillna(False).astype(bool)]
        ev = int((a["status_group"] == "event").sum())
        out[rule] = {
            "attributable_test_trials": int(len(a)),
            "o1_events": ev,
            "completed": int((a["status_group"] == "control").sum()),
            "cell": gate_cell(ev),
        }
    return out


def main() -> dict:
    c = pd.read_parquet(COHORT_PATH, columns=["nct_id", "window", "status_group", *RULE_COLUMNS.values()])
    res = gate_counts(c)
    config.GATE_PATH.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
