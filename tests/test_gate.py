"""Step-0 gate (A3) on a hand-built cohort."""
import pandas as pd

from fb import gate


def test_gate_cells():
    assert gate.gate_cell(300).startswith("full")
    assert gate.gate_cell(299).startswith("pooled")
    assert gate.gate_cell(150).startswith("pooled")
    assert gate.gate_cell(149).startswith("stop")


def test_gate_counts_only_attributable_test_trials():
    c = pd.DataFrame({
        "window": ["test", "test", "test", "test", "train"],
        "status_group": ["event", "event", "control", "censored", "event"],
        "has_attributable_funding_n20": [True, False, True, True, True],
        "has_attributable_funding_n50": [True, True, True, True, True],
        "has_attributable_funding_activity": [False, False, True, True, True],
    })
    g = gate.gate_counts(c)
    assert g["n20"] == {"attributable_test_trials": 3, "o1_events": 1, "completed": 1, "cell": "stop (< 150)"}
    assert g["n50"]["o1_events"] == 2 and g["activity"]["o1_events"] == 0
