"""Phase 4 step 0: the §11 gate re-stated on the A4b landmark subset.

Landmark subset = attributable trials (A1 N = 20) with >= 1 parent fiscal-year row observable
before the outcome date (`f_rows_reason` in config.LANDMARK_REASONS). Counts trials, O1 events,
Completed for both windows from cohort.parquet + the overlap / cut machinery only; no feature
value is computed. The §11 cell is read on the TEST count (fb.gate.gate_cell).

    python -m fb.landmark
"""
from __future__ import annotations

import json

import pandas as pd

from . import config, tables
from .gate import gate_cell


def landmark_counts(trials: pd.DataFrame, reasons: pd.DataFrame) -> dict:
    """trials: nct_id, status_group; reasons: nct_id, f_rows_reason (fb.tables.observable_reason)."""
    m = trials[["nct_id", "status_group"]].merge(reasons, on="nct_id", how="left")
    lm = m[m["f_rows_reason"].isin(config.LANDMARK_REASONS)]
    return {
        "attributable_trials": int(len(m)),
        "f_rows_reason": {str(k): int(v) for k, v in m["f_rows_reason"].value_counts().items()},
        "landmark_trials": int(len(lm)),
        "o1_events": int(lm["status_group"].eq("event").sum()),
        "completed": int(lm["status_group"].eq("control").sum()),
        "censored": int(lm["status_group"].eq("censored").sum()),
        "unknown": int(lm["status_group"].eq("unknown").sum()),
        "o1_events_dropped_by_landmark": int(m["status_group"].eq("event").sum() - lm["status_group"].eq("event").sum()),
    }


def main() -> dict:
    inp = tables.load_inputs()
    res: dict = {}
    for rule in ("n20", "n50", "activity", "none"):
        res[rule] = {}
        for w in ("train", "test"):
            a = tables.attributable(tables.window_trials(inp, w), rule)
            res[rule][w] = landmark_counts(a, tables.observable_reason(inp, a, rule))
        res[rule]["cell_on_test"] = gate_cell(res[rule]["test"]["o1_events"])
    res["primary_cell"] = res["n20"]["cell_on_test"]
    config.GATE_LANDMARK_PATH.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
