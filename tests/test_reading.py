"""Plan §9 reading function on each cell, a CI straddling 0, and the stability label."""
import pandas as pd
import pytest

from fb import config, reading


@pytest.mark.parametrize("args, want", [
    ((0.05, 0.02, 0.08, 0.03, 0.01, 0.05, 0.65), "R1"),
    ((0.05, 0.02, 0.08, 0.01, -0.01, 0.03, 0.65), "R2"),   # F₂ under 0.02
    ((0.05, 0.02, 0.08, 0.03, -0.001, 0.05, 0.65), "R2"),  # F₂ CI includes 0
    ((0.01, 0.00, 0.02, 0.03, 0.01, 0.05, 0.75), "R3"),    # ΔAUC under 0.03
    ((0.01, 0.00, 0.02, 0.03, 0.01, 0.05, 0.65), "R4"),
    ((0.04, -0.01, 0.09, 0.03, 0.01, 0.05, 0.72), "R3"),   # Δ ≥ 0.03 but CI straddles 0
    ((0.04, -0.01, 0.09, 0.03, 0.01, 0.05, 0.69), "R4"),
    ((0.03, 0.001, 0.06, 0.02, 0.001, 0.04, 0.5), "R1"),   # thresholds are inclusive
    ((0.0299, 0.01, 0.05, 0.03, 0.01, 0.05, 0.70), "R3"),  # P-only AUC 0.70 is ≥ 0.70
])
def test_cells(args, want):
    assert reading.cell(*args) == want


def test_nan_is_conservative():
    nan = float("nan")
    assert reading.cell(nan, nan, nan, nan, nan, nan, 0.6) == "R4"


def test_thresholds_are_the_plan_values():
    assert (config.READ_DELTA_AUC_MIN, config.READ_F2_DELTA_AUC_MIN, config.READ_P_AUC_MIN) == (0.03, 0.02, 0.70)
    assert set(config.READING_CELLS) == {"R1", "R2", "R3", "R4"}


def test_stability():
    assert reading.stability("R3", {"a": "R3", "b": "R3"}) == ("robust", [])
    assert reading.stability("R3", {"a": "R3", "b": "R4"}) == ("fragile", ["b"])


def _m(pop, fs, metric, v, lo=None, hi=None, n=100, ev=10):
    return dict(outcome="O1", population=pop, spec="logistic", featureset=fs, comparison="", stratum="all",
                window="test", metric=metric, value=v, ci_lo=lo, ci_hi=hi, n=n, n_events=ev, note="")


def test_read_from_metrics_table():
    rows = [_m("P_only", "P", "auc", 0.72, 0.70, 0.74), _m("P_only_A2", "P", "auc", 0.68, 0.66, 0.70)]
    for pop in ("primary", "S1", "A2", "N50", "activity", "no_umbrella"):
        rows += [_m(pop, "P", "auc", 0.7, 0.68, 0.72), _m(pop, "P+F", "delta_auc", 0.01, -0.01, 0.03),
                 _m(pop, "P+F2", "delta_auc", 0.0, -0.01, 0.01)]
    text, res = reading.read(pd.DataFrame(rows))
    assert res["primary"]["cell"] == "R3"
    assert res["sensitivity"]["A2 (withdrawn / enrollment-0 excluded)"]["cell"] == "R4"  # its own P-only AUC
    assert res["stability"] == "fragile" and res["disagreeing"] == ["A2 (withdrawn / enrollment-0 excluded)"]
    assert "CELL: R3" in text
