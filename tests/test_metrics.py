"""Phase 4 metrics on hand-computed cases."""
import math

import numpy as np
import pandas as pd
import pytest

from fb import metrics
from fb.stats import wilson_ci

# positives 0.9, 0.8, 0.4; negatives 0.7, 0.3, 0.2, 0.1
Y = np.array([1, 1, 1, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.2, 0.1])


def test_auc_hand():
    # pairs won: 0.9 -> 4, 0.8 -> 4, 0.4 -> 3 of 12
    assert metrics.auc(Y, S) == pytest.approx(11 / 12)


def test_auc_ties_count_half():
    assert metrics.auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert metrics.auc([1, 1, 0], [0.5, 0.7, 0.5]) == pytest.approx((0.5 + 1) / 2)


def test_delong_known_case():
    a, var, lo, hi = metrics.delong(Y, S)
    # placement values: positives [1, 1, 3/4], negatives (share of positives above) [2/3, 1, 1, 1]
    vp = np.array([1, 1, 0.75])
    vn = np.array([2 / 3, 1, 1, 1])
    want = vp.var(ddof=1) / 3 + vn.var(ddof=1) / 4
    assert a == pytest.approx(11 / 12)
    assert var == pytest.approx(want)
    assert want == pytest.approx(0.0208333 / 3 + 0.0277778 / 4, rel=1e-4)
    assert lo == pytest.approx(11 / 12 - 1.959964 * math.sqrt(want), rel=1e-5)
    assert hi == 1.0  # clipped


def test_bootstrap_delta_toy():
    y = np.array([1, 0, 1, 0])
    new = np.array([0.9, 0.1, 0.8, 0.2])   # AUC 1
    base = np.array([0.4, 0.6, 0.7, 0.3])  # AUC 3/4
    idx = np.array([[0, 1, 2, 3],          # delta 0.25
                    [0, 0, 1, 1],          # y 1,1,0,0: new 1, base 0 -> delta 1
                    [1, 1, 3, 3]])         # one class -> skipped
    point, lo, hi, d = metrics.bootstrap_delta(y, new, base, idx)
    assert point == pytest.approx(0.25)
    assert d.tolist() == pytest.approx([0.25, 1.0])
    assert lo == pytest.approx(0.25 + 0.025 * 0.75)
    assert hi == pytest.approx(0.25 + 0.975 * 0.75)


def test_resample_matrix_fixed_seed():
    a = metrics.resample_matrix(10, 5, 7)
    b = metrics.resample_matrix(10, 5, 7)
    assert a.shape == (5, 10) and (a == b).all()


def test_brier_logloss():
    y = np.array([1, 0])
    p = np.array([0.8, 0.4])
    assert metrics.brier(y, p) == pytest.approx((0.04 + 0.16) / 2)
    assert metrics.log_loss(y, p) == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)


def test_calibration_of_a_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    s, i = metrics.calibration_slope_intercept(y, p)
    assert s == pytest.approx(1.0, abs=0.03)
    assert i == pytest.approx(0.0, abs=0.02)
    # predictions twice too extreme on the logit scale -> slope ~ 0.5
    lp = np.log(p / (1 - p))
    s2, _ = metrics.calibration_slope_intercept(y, 1 / (1 + np.exp(-2 * lp)))
    assert s2 == pytest.approx(0.5, abs=0.03)


def test_decile_bins_with_wilson():
    p = np.arange(1, 21) / 20
    y = np.array([0, 0] * 5 + [1, 0] * 3 + [1, 1] * 2)
    b = metrics.decile_bins(y, p, 10)
    assert b["n"].tolist() == [2] * 10
    assert b["observed"].tolist() == [0.0] * 5 + [0.5] * 3 + [1.0] * 2
    assert b.loc[0, "mean_pred"] == pytest.approx(0.075)
    assert (b.loc[5, "obs_lo"], b.loc[5, "obs_hi"]) == pytest.approx(wilson_ci(1, 2))


def test_permute_within_keeps_strata():
    strata = np.array(["a", "a", "b", "b", "b", "c"])
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx = metrics.permute_within(strata, rng)
        assert sorted(idx.tolist()) == list(range(6))
        assert (strata[idx] == strata).all()
        assert idx[5] == 5


def test_rmse_r2():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.rmse(y, [1, 2, 4]) == pytest.approx(math.sqrt(1 / 3))
    assert metrics.r2(y, [1, 2, 4]) == pytest.approx(1 - 1 / 2)
