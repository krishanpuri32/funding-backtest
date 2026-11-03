"""Plan §7–§8 metrics. Pure numpy / scipy, no data access; every function is tested on a
hand-computed case (tests/test_metrics.py).

- auc: Mann–Whitney with mid-ranks (ties count 1/2).
- delong: AUC with the DeLong variance (Sun & Xu 2014 placement values) and a normal 95 % CI.
- bootstrap_delta: paired percentile bootstrap of AUC(new) − AUC(base) over test rows; the
  resample matrix comes from a fixed seed, so every comparison sees the same resamples.
- brier, log_loss; calibration_slope_intercept (logistic recalibration on logit p: slope from
  y ~ a + b·lp, intercept = calibration-in-the-large from y ~ a + offset(lp)).
- decile_bins: equal-count bins of the predicted risk with Wilson CIs on the observed rate.
- permute_within: a row permutation that only moves rows inside their stratum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, rankdata

from .stats import wilson_ci

Z95 = float(norm.ppf(0.975))


def auc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - m * (m + 1) / 2) / (m * n))


def delong(y, s) -> tuple[float, float, float, float]:
    """(auc, variance, ci_lo, ci_hi)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    tx, ty = rankdata(pos), rankdata(neg)
    tz = rankdata(np.concatenate([pos, neg]))
    a = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v_pos = (tz[:m] - tx) / n          # per positive: share of negatives scored lower
    v_neg = 1.0 - (tz[m:] - ty) / m    # per negative: share of positives scored higher
    var = v_pos.var(ddof=1) / m + v_neg.var(ddof=1) / n
    half = Z95 * np.sqrt(var)
    return float(a), float(var), float(max(0.0, a - half)), float(min(1.0, a + half))


def resample_matrix(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_delta(y, s_new, s_base, idx: np.ndarray, fn=auc) -> tuple[float, float, float, np.ndarray]:
    """(point delta on the full set, ci_lo, ci_hi, bootstrap deltas). `idx`: resample matrix;
    resamples with a single class are skipped."""
    y = np.asarray(y)
    s_new = np.asarray(s_new, dtype=float)
    s_base = np.asarray(s_base, dtype=float)
    point = fn(y, s_new) - fn(y, s_base)
    d = []
    for i in idx:
        yy = y[i]
        if fn is auc and (yy.min() == yy.max()):
            continue
        d.append(fn(yy, s_new[i]) - fn(yy, s_base[i]))
    d = np.asarray(d)
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), d


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def log_loss(y, p, eps: float = 1e-15) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit_irls(X: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 100) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ beta + offset)
        w = p * (1 - p)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-12 * np.eye(X.shape[1]), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_slope_intercept(y, p, eps: float = 1e-6) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    one = np.ones_like(lp)
    slope = _logit_irls(np.column_stack([one, lp]), y, np.zeros_like(lp))[1]
    intercept = _logit_irls(one[:, None], y, lp)[0]
    return float(slope), float(intercept)


def decile_bins(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins in order of predicted risk (stable sort): n, mean predicted, observed
    rate and its Wilson 95 % CI."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for b, ix in enumerate(np.array_split(order, bins)):
        k, n = int(y[ix].sum()), len(ix)
        lo, hi = wilson_ci(k, n)
        rows.append({"bin": b + 1, "n": n, "mean_pred": float(p[ix].mean()) if n else float("nan"),
                     "observed": k / n if n else float("nan"), "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows)


def permute_within(strata, rng: np.random.Generator) -> np.ndarray:
    """Index array: a permutation of 0..n-1 that maps every row to a row of the same stratum."""
    strata = pd.Series(np.asarray(strata)).astype(str)
    idx = np.arange(len(strata))
    for _, ix in strata.groupby(strata).indices.items():
        idx[ix] = rng.permutation(ix)
    return idx


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)) ** 2)))


def r2(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - np.asarray(yhat, dtype=float)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
