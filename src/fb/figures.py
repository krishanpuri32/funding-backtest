"""Charts 1–3 for the public write-up -> data/derived/figures/post/ (PNG + SVG).

    python -m fb.figures

Chart 1 — the P-only logistic model, O1, all trials (A3), test window: ROC curve (left) and the
decile calibration already written by fb.evaluate (right). Test-window scores were not persisted
by fb.evaluate, so the frozen `O1__P_only__logistic__P.joblib` is applied to the existing test
tables once; the ROC points are cached in `post/chart1_roc_points.csv` and later runs reuse them.
The AUC recomputed from those scores must equal the metrics.csv value; no metric is written.
The scoring is recorded as a `roc_scoring` key appended to phase4/test_evaluated.json.
Chart 2 — training-window O1 rate (Wilson CI) by stratum from base_rates_all.csv, median funded
dollars per stratum from base_rates_primary.csv (rates all trials, dollars attributable set).
Chart 3 — O1 logistic test ΔAUC (P+F − P and P+F₂ − P, bootstrap CI) for the primary and the
five sensitivity runs from metrics.csv, against 0 and the 0.03 bar.

Every write goes through guard_post_write: only paths under figures/post/ pass; every other path
goes to fb.evaluate.guard_write, which refuses the Phase 4 outputs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .evaluate import guard_write

POST = config.FIGURES_POST_DIR
ROC_POINTS_PATH = POST / "chart1_roc_points.csv"
ROC_REASON = "ROC curve for chart 1 of the public write-up; no metric recomputed"
ROC_MODEL = ("O1", "P_only", "logistic", "P")

# Okabe–Ito (colourblind-safe)
INK, BLUE, ORANGE, SKY, GREY = "#222222", "#0072B2", "#E69F00", "#56B4E9", "#999999"

# Chart 2 strata: (label, stratum_var, stratum value in base_rates_*.csv; NaN = phase N/A)
STRATA = [
    ("All trials", "overall", "all"),
    ("Neoplasms", "condition_group", "Neoplasms"),
    ("Mental & behavioral", "condition_group", "Mental & behavioral"),
    ("Cardiovascular", "condition_group", "Cardiovascular"),
    ("Phase 1", "phase", "1"),
    ("Phase 2", "phase", "2"),
    ("Phase 3", "phase", "3"),
    ("Phase N/A", "phase", None),
    ("P-class awards", "activity_class", "P"),
    ("R-class awards", "activity_class", "R"),
    ("NCI", "ic", "CA"),
    ("NIMH", "ic", "MH"),
]
# Chart 3 runs: (label, population in metrics.csv)
RUNS = [
    ("Primary (N = 20, landmark)", "primary"),
    ("S1: all attributable + F indicators", "S1"),
    ("A2: withdrawn / enrollment-0 excluded", "A2"),
    ("A1 (i): no umbrella exclusion", "no_umbrella"),
    ("A1 (ii): N = 50", "N50"),
    ("A1 (iii): activity rule", "activity"),
]


# --- write guards -----------------------------------------------------------------------
def guard_post_write(path) -> None:
    """Allow a path under figures/post/; anything else must pass fb.evaluate.guard_write."""
    p, post = Path(path).resolve(), POST.resolve()
    if post == p or post in p.parents:
        return
    guard_write(p)


def append_marker_record(marker: Path, key: str, record: dict) -> None:
    """Add one new top-level key to a marker JSON; refuses to replace a key or alter any other."""
    before = json.loads(marker.read_text())
    if key in before:
        raise PermissionError(f"{marker.name} already holds {key!r}")
    after = {**before, key: record}
    assert {k: after[k] for k in before} == before
    marker.write_text(json.dumps(after, indent=1) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- data ---------------------------------------------------------------------------------
def _metric(m: pd.DataFrame, pop: str, fs: str, metric: str) -> pd.Series:
    r = m[(m["outcome"] == "O1") & (m["spec"] == "logistic") & (m["window"] == "test") & (m["population"] == pop)
          & (m["featureset"] == fs) & (m["metric"] == metric) & (m["stratum"] == "all")]
    if len(r) != 1:
        raise KeyError(f"{pop} {fs} {metric}: {len(r)} rows")
    return r.iloc[0]


def roc_points(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """ROC vertices (fpr, tpr) at every distinct score threshold, highest first, from (0, 0) to (1, 1)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.diff(ss) != 0, True]          # last index of each tie block
    tp = np.cumsum(ys)[last]
    fp = np.cumsum(1 - ys)[last]
    P, N = int(y.sum()), int(len(y) - y.sum())
    return pd.DataFrame({"fpr": np.r_[0.0, fp / N], "tpr": np.r_[0.0, tp / P]})


def trapezoid_auc(pts: pd.DataFrame) -> float:
    return float(np.trapezoid(pts["tpr"], pts["fpr"]))


def score_roc(m: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen P-only model to the test tables once; cache the ROC points; record it in the marker."""
    import joblib

    from . import models, tables

    marker = config.TEST_EVALUATED_MARKER
    frozen = json.loads(marker.read_text())["frozen_models_sha256"]
    path = models.bundle_path(*ROC_MODEL)
    if _sha(path) != frozen[path.name]:
        raise SystemExit(f"{path.name} differs from the hash in {marker.name}; stopping")
    b = joblib.load(path)
    rows, X = models.assemble(tables.load_window("test"), "O1", "P_only", "P", b["medians"], b["levels"])
    assert list(X.columns) == b["columns"], "column order differs from the frozen model"
    y = rows["y"].values.astype(int)
    pts = roc_points(y, models.predict(b["model"], X, True))
    ref = _metric(m, "P_only", "P", "auc")
    a = trapezoid_auc(pts)
    if abs(a - ref["value"]) > 1e-9 or len(y) != ref["n"] or int(y.sum()) != ref["n_events"]:
        raise SystemExit(f"rescored AUC {a} / n {len(y)} / events {int(y.sum())} differ from metrics.csv {ref['value']}")
    guard_post_write(ROC_POINTS_PATH)
    POST.mkdir(parents=True, exist_ok=True)
    pts.to_csv(ROC_POINTS_PATH, index=False)
    append_marker_record(marker, "roc_scoring", {
        "utc": datetime.now(timezone.utc).isoformat(), "reason": ROC_REASON, "model": path.name,
        "n": int(len(y)), "n_events": int(y.sum()), "auc_check": a, "output": f"figures/post/{ROC_POINTS_PATH.name}"})
    return pts


def chart2_values() -> pd.DataFrame:
    a = pd.read_csv(config.DATA_DERIVED / "base_rates_all.csv", dtype={"stratum": str})
    p = pd.read_csv(config.DATA_DERIVED / "base_rates_primary.csv", dtype={"stratum": str})

    def pick(df, var, val):
        r = df[(df["stratum_var"] == var) & (df["stratum"].isna() if val is None else df["stratum"] == val)]
        if len(r) != 1:
            raise KeyError(f"{var}={val}: {len(r)} rows")
        return r.iloc[0]

    rows = []
    for label, var, val in STRATA:
        ra, rp = pick(a, var, val), pick(p, var, val)
        rows.append({"label": label, "stratum_var": var, "stratum": val, "n_terminal": int(ra["n_terminal"]),
                     "o1_events": int(ra["o1_events"]), "event_rate": ra["event_rate"], "wilson_lo": ra["wilson_lo"],
                     "wilson_hi": ra["wilson_hi"], "dollars_median_attributable": rp["dollars_median"],
                     "dollars_n_attributable": int(rp["dollars_n"])})
    return pd.DataFrame(rows)


def chart3_values(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, pop in RUNS:
        f, f2 = _metric(m, pop, "P+F", "delta_auc"), _metric(m, pop, "P+F2", "delta_auc")
        rows.append({"label": label, "population": pop, "n": int(f["n"]), "n_events": int(f["n_events"]),
                     "delta_auc_F": f["value"], "F_lo": f["ci_lo"], "F_hi": f["ci_hi"],
                     "delta_auc_F2": f2["value"], "F2_lo": f2["ci_lo"], "F2_hi": f2["ci_hi"]})
    return pd.DataFrame(rows)


# --- drawing ------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "legend.frameon": False,
        "svg.hashsalt": "funding-backtest", "svg.fonttype": "none"})
    return plt


def _save(fig, stem: str) -> list[Path]:
    out = []
    for ext in ("png", "svg"):
        p = POST / f"{stem}.{ext}"
        guard_post_write(p)
        fig.savefig(p, dpi=200, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
        out.append(p)
    return out


def chart1(m: pd.DataFrame, pts: pd.DataFrame, bins: pd.DataFrame) -> list[Path]:
    plt = _style()
    auc = _metric(m, "P_only", "P", "auc")
    slope = _metric(m, "P_only", "P", "cal_slope")["value"]
    icpt = _metric(m, "P_only", "P", "cal_intercept")["value"]
    fig, (l, r) = plt.subplots(1, 2, figsize=(9, 4.3))
    l.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="chance (AUC 0.5)")
    l.plot(pts["fpr"], pts["tpr"], color=BLUE, lw=1.8, label="design priors, logistic")
    l.set_xlim(0, 1), l.set_ylim(0, 1), l.set_aspect("equal")
    l.set_xlabel("false-positive rate (completed trials flagged)")
    l.set_ylabel("true-positive rate (failed trials flagged)")
    l.set_title("Discrimination, 2016–2020 test window")
    l.text(0.97, 0.05, f"AUC {auc['value']:.4f}\n95 % CI {auc['ci_lo']:.4f}–{auc['ci_hi']:.4f}\n"
                       f"n = {int(auc['n']):,}, failures = {int(auc['n_events']):,}", ha="right", va="bottom")
    l.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    lim = float(max(bins["mean_pred"].max(), bins["obs_hi"].max())) * 1.05
    r.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--", label="perfect calibration")
    r.errorbar(bins["mean_pred"], bins["observed"],
               yerr=[(bins["observed"] - bins["obs_lo"]).clip(lower=0), (bins["obs_hi"] - bins["observed"]).clip(lower=0)],
               fmt="o-", ms=4, lw=1.2, capsize=2, color=ORANGE, label="observed by decile (Wilson 95 % CI)")
    r.set_xlim(0, lim), r.set_ylim(0, lim), r.set_aspect("equal")
    r.set_xlabel("mean predicted failure risk (decile)")
    r.set_ylabel("observed failure rate")
    r.set_title("Calibration, same model and window")
    r.text(0.97 * lim, 0.05 * lim, f"intercept {icpt:+.2f}\nslope {slope:.2f}", ha="right", va="bottom")
    r.legend(loc="upper left")
    fig.tight_layout()
    out = _save(fig, "chart1_model")
    plt.close(fig)
    return out


def chart2(v: pd.DataFrame) -> list[Path]:
    plt = _style()
    groups = [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]
    ys, y, prev = [], 0.0, None
    for g in groups:
        if prev is not None:
            y += 1.0 + (0.6 if g != prev else 0.0)
        ys.append(y)
        prev = g
    ys = -np.asarray(ys)
    import matplotlib.transforms as mtransforms
    fig, ax = plt.subplots(figsize=(9, 5.2))
    overall = v.iloc[0]["event_rate"] * 100
    ax.axvline(overall, color=GREY, lw=1, ls=":")
    ax.errorbar(v["event_rate"] * 100, ys, xerr=[(v["event_rate"] - v["wilson_lo"]) * 100, (v["wilson_hi"] - v["event_rate"]) * 100],
                fmt="o", ms=5, color=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=2.5)
    xmax = float(v["wilson_hi"].max() * 100) * 1.08
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)  # x in axes fraction, y in data
    for yy, r in zip(ys, v.itertuples()):
        ax.text(1.04, yy, f"{r.event_rate:.1%}  ({r.o1_events:,} / {r.n_terminal:,})", transform=tr, va="center", fontsize=8.5)
        ax.text(1.56, yy, f"${r.dollars_median_attributable / 1e6:,.2f} M", transform=tr, va="center", ha="right", fontsize=8.5)
    ax.text(1.04, ys[0] + 1.1, "rate (failed / n)", transform=tr, fontsize=8, color=GREY, va="bottom")
    ax.text(1.56, ys[0] + 1.1, "median funded", transform=tr, fontsize=8, color=GREY, va="bottom", ha="right")
    ax.set_yticks(ys, v["label"])
    ax.set_ylim(ys[-1] - 0.8, ys[0] + 1.6)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("operational failure rate, % (Wilson 95 % CI)")
    ax.set_title("Operational failure by stratum, training window 2008–2015\n"
                 "rate from all trials; median funded dollars (CPI-2020) from the attributable set", fontsize=9, loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    out = _save(fig, "chart2_base_rates")
    plt.close(fig)
    return out


def chart3(v: pd.DataFrame) -> list[Path]:
    plt = _style()
    ys = -np.arange(len(v), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axvline(0, color=INK, lw=0.8)
    ax.axvline(config.READ_DELTA_AUC_MIN, color=ORANGE, lw=1.4, ls="--")
    ax.text(config.READ_DELTA_AUC_MIN, ys[0] + 0.75, f" pre-registered bar {config.READ_DELTA_AUC_MIN:.2f}",
            color=ORANGE, fontsize=8.5, va="bottom")
    ax.errorbar(v["delta_auc_F2"], ys - 0.18, xerr=[v["delta_auc_F2"] - v["F2_lo"], v["F2_hi"] - v["delta_auc_F2"]],
                fmt="s", ms=4, color=SKY, ecolor=SKY, elinewidth=1.2, capsize=2, label="first two funded years only (P + F₂)")
    ax.errorbar(v["delta_auc_F"], ys + 0.12, xerr=[v["delta_auc_F"] - v["F_lo"], v["F_hi"] - v["delta_auc_F"]],
                fmt="o", ms=5, color=BLUE, ecolor=BLUE, elinewidth=1.5, capsize=2.5, label="full funding trajectory (P + F)")
    ax.set_yticks(ys, v["label"])
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.2)
    ax.set_xlim(-0.01, 0.04)
    ax.set_xlabel("ΔAUC over design priors, 2016–2020 test window (bootstrap 95 % CI)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.tight_layout()
    out = _save(fig, "chart3_sensitivity")
    plt.close(fig)
    return out


def main() -> None:
    m = pd.read_csv(config.METRICS_PATH)
    POST.mkdir(parents=True, exist_ok=True)
    if ROC_POINTS_PATH.exists():
        pts = pd.read_csv(ROC_POINTS_PATH)
        print(f"chart 1: reusing {ROC_POINTS_PATH.name} (test window already scored once)")
    else:
        pts = score_roc(m)
        print(f"chart 1: frozen model scored on the test tables; marker key roc_scoring appended")
    b = pd.read_csv(config.FIGURES_DIR / "calibration_bins.csv")
    o, pop, spec, fs = ROC_MODEL
    bins = b[(b["outcome"] == o) & (b["population"] == pop) & (b["spec"] == spec) & (b["featureset"] == fs)].sort_values("bin")
    v2, v3 = chart2_values(), chart3_values(m)
    for name, df in (("chart1_calibration_values.csv", bins), ("chart2_values.csv", v2), ("chart3_values.csv", v3)):
        guard_post_write(POST / name)
        df.to_csv(POST / name, index=False)
    paths = chart1(m, pts, bins) + chart2(v2) + chart3(v3)
    pd.set_option("display.width", 200)
    print(f"chart 1 ROC: {len(pts)} vertices, trapezoid AUC {trapezoid_auc(pts):.6f}")
    print(bins.to_string(index=False))
    print(v2.to_string(index=False))
    print(v3.to_string(index=False))
    for p in paths:
        print("wrote", p.relative_to(config.DATA_DERIVED))


if __name__ == "__main__":
    main()
