"""Phase 4-X exploratory run "px": the design variant drops exactly the six P indicators,
the exploratory evaluation cannot write any Phase 4 path, and the what-if reading override."""
import pandas as pd
import pytest

from fb import config, design, evaluate, features_p, models, reading
from tests.test_design import P_MED, _f_raw, _p_raw
from tests.test_reading import _m

F_MED = {c: 10.0 for c in design.f_columns("F")}


@pytest.mark.parametrize("variant", [None, "F"])
def test_drop_p_missing_removes_exactly_six_columns(variant):
    p = _p_raw()
    lv = design.p_levels(p)
    kw = dict(f_raw=_f_raw(), f_medians=F_MED, variant=variant) if variant else {}
    full = design.design_matrix(["T1", "T2", "T3"], p, P_MED, lv, **kw)
    px = design.design_matrix(["T1", "T2", "T3"], p, P_MED, lv, drop_p_missing=True, **kw)
    dropped = [c for c in full.columns if c not in px.columns]
    assert dropped == [f"{c}_missing" for c in features_p.P_NUMERIC]
    assert len(dropped) == 6
    assert list(px.columns) == [c for c in full.columns if c not in dropped]   # order kept
    pd.testing.assert_frame_equal(px, full[list(px.columns)])                   # values unchanged
    assert "phase_group=missing" in px.columns and "allocation=missing" in px.columns


def test_px_models_dir_is_outside_phase4():
    assert models.bundle_path("O1", "P_only", "logistic", "P", models.PX_MODELS_DIR).parent == models.PX_MODELS_DIR
    evaluate.guard_write(models.PX_MODELS_DIR / "x.joblib")


@pytest.mark.parametrize("path", [
    config.METRICS_PATH, config.READING_PATH, config.TEST_EVALUATED_MARKER,
    config.MODELS_DIR / "O1__P_only__logistic__P.joblib", config.FIGURES_DIR / "calibration_bins.csv",
    config.PHASE4_DIR / "test_P.parquet", config.MODELS_DIR,
])
def test_guard_refuses_phase4_paths(path):
    with pytest.raises(PermissionError):
        evaluate.guard_write(path)


def test_guard_allows_exploratory_outputs():
    for p in (config.EXPLORATORY_DIR / "metrics_px.csv", config.EXPLORATORY_DIR / "px_evaluated.json",
              config.EXPLORATORY_DIR / "dryrun_metrics_px.csv"):
        evaluate.guard_write(p)


def test_evaluate_px_refuses_phase4_output(monkeypatch):
    # the guard runs before any data is read
    monkeypatch.setattr(evaluate.tables, "load_window", lambda w: (_ for _ in ()).throw(AssertionError("read data")))
    with pytest.raises(PermissionError):
        evaluate.evaluate_px("test", config.METRICS_PATH)


def _table(p_only_auc):
    rows = [_m("P_only", "P", "auc", p_only_auc, 0.0, 1.0), _m("P_only_A2", "P", "auc", 0.75, 0.7, 0.8)]
    for pop in ("primary", "S1", "A2", "N50", "activity", "no_umbrella"):
        rows += [_m(pop, "P", "auc", 0.72, 0.7, 0.74), _m(pop, "P+F", "delta_auc", 0.01, 0.001, 0.02),
                 _m(pop, "P+F2", "delta_auc", 0.0, -0.01, 0.01)]
    return pd.DataFrame(rows)


def test_reading_override_changes_only_the_p_auc():
    _, base = reading.read(_table(0.78))
    assert base["primary"]["cell"] == "R3"
    _, what_if = reading.read(_table(0.78), {"P_only": 0.69})
    assert what_if["primary"]["cell"] == "R4" and what_if["primary"]["p_auc"] == 0.69
    assert what_if["primary"]["delta"] == base["primary"]["delta"]
    # A2's P-only AUC comes from P_only_A2 and is not overridden here
    assert what_if["sensitivity"]["A2 (withdrawn / enrollment-0 excluded)"]["cell"] == "R3"
