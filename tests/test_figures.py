import json

import numpy as np
import pytest

from fb import config, figures, metrics


def test_guard_post_write_allows_post_only():
    figures.guard_post_write(config.FIGURES_POST_DIR / "chart.png")
    for p in (config.FIGURES_DIR / "calibration_bins.csv", config.METRICS_PATH, config.TEST_EVALUATED_MARKER,
              config.MODELS_DIR / "x.joblib"):
        with pytest.raises(PermissionError):
            figures.guard_post_write(p)


def test_append_marker_record_adds_one_key_only(tmp_path):
    m = tmp_path / "marker.json"
    m.write_text(json.dumps({"started_utc": "t0", "frozen_models_sha256": {"a": "1"}}))
    figures.append_marker_record(m, "roc_scoring", {"reason": "r"})
    d = json.loads(m.read_text())
    assert d == {"started_utc": "t0", "frozen_models_sha256": {"a": "1"}, "roc_scoring": {"reason": "r"}}
    with pytest.raises(PermissionError):
        figures.append_marker_record(m, "roc_scoring", {"reason": "again"})


def test_roc_points_auc_matches_metrics_auc_with_ties():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.9, 0.7, 0.5, 0.5, 0.3, 0.2, 0.2])
    pts = figures.roc_points(y, s)
    assert pts.iloc[0].tolist() == [0.0, 0.0] and pts.iloc[-1].tolist() == [1.0, 1.0]
    assert figures.trapezoid_auc(pts) == pytest.approx(metrics.auc(y, s))
