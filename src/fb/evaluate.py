"""Phase 4 step 7: evaluate the frozen models on the TEST window — once.

    python -m fb.evaluate --dry-run     # exercises every code path on the TRAINING window (in-sample);
                                        # writes phase4/dryrun_metrics.csv, touches no test row
    python -m fb.evaluate               # the single test evaluation -> metrics.csv, figures/, marker

The real run refuses to start if phase4/test_evaluated.json exists (the test window is read
once); `--rerun-reason "<text>"` overrides it and the reason is recorded in the marker
(a post-step-7 bug fix is reported with both runs' numbers).

metrics.csv is long format: outcome, population, spec, featureset, comparison, stratum,
window, metric, value, ci_lo, ci_hi, n, n_events, note.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, design, hazard, metrics, models, tables

COLUMNS = ["outcome", "population", "spec", "featureset", "comparison", "stratum", "window", "metric", "value",
           "ci_lo", "ci_hi", "n", "n_events", "note"]


class Rows(list):
    def add(self, **kw):
        r = {c: kw.get(c) for c in COLUMNS}
        r["stratum"] = r["stratum"] or "all"
        self.append(r)


def _load(outcome, pop, spec, fs):
    return joblib.load(models.bundle_path(outcome, pop, spec, fs))


def _calibration_figure(curves: dict, title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    lim = max(0.05, max(float(max(b["mean_pred"].max(), b["obs_hi"].max())) for b in curves.values()) * 1.05)
    ax.plot([0, lim], [0, lim], color="#999999", lw=1, ls="--", label="perfect")
    for (fs, b), c in zip(curves.items(), ["#1f77b4", "#d62728", "#2ca02c"]):
        ax.errorbar(b["mean_pred"], b["observed"],
                    yerr=[(b["observed"] - b["obs_lo"]).clip(lower=0), (b["obs_hi"] - b["observed"]).clip(lower=0)],
                    fmt="o-", ms=4, lw=1, capsize=2, color=c, label=fs)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted risk (decile)")
    ax.set_ylabel("observed rate (Wilson 95 % CI)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def binary_metrics(out: Rows, outcome, pop, spec, preds: dict, y: np.ndarray, rows: pd.DataFrame, window: str,
                   idx: np.ndarray, stratum: str = "all") -> None:
    n, ev = int(len(y)), int(y.sum())
    base = dict(outcome=outcome, population=pop, spec=spec, window=window, n=n, n_events=ev, stratum=stratum)
    for fs, p in preds.items():
        a, _, lo, hi = metrics.delong(y, p)
        out.add(**base, featureset=fs, metric="auc", value=a, ci_lo=lo, ci_hi=hi, note="DeLong 95% CI")
        if stratum != "all":
            continue
        out.add(**base, featureset=fs, metric="brier", value=metrics.brier(y, p))
        out.add(**base, featureset=fs, metric="log_loss", value=metrics.log_loss(y, p))
        s, i = metrics.calibration_slope_intercept(y, p)
        out.add(**base, featureset=fs, metric="cal_slope", value=s)
        out.add(**base, featureset=fs, metric="cal_intercept", value=i, note="calibration-in-the-large (offset logit p)")
    for fs in ("P+F", "P+F2"):
        if fs in preds and "P" in preds:
            d, lo, hi, _ = metrics.bootstrap_delta(y, preds[fs], preds["P"], idx)
            out.add(**base, featureset=fs, comparison=f"{fs} − P", metric="delta_auc", value=d, ci_lo=lo, ci_hi=hi,
                    note=f"paired bootstrap {len(idx)} resamples, percentile 95% CI")


def run_binary(out: Rows, W: dict, window: str, outcome: str, pop: str, fsets, curves_out: dict, perm: bool,
               strata_out: dict) -> None:
    for spec in models.SPECS["binary"]:
        preds, rows = {}, None
        Xs = {}
        for fs in fsets:
            b = _load(outcome, pop, spec, fs)
            rows, X = models.assemble(W, outcome, pop, fs, b["medians"], b["levels"])
            assert list(X.columns) == b["columns"], "column order differs from the frozen model"
            Xs[fs] = X
            preds[fs] = models.predict(b["model"], X, True)
        y = rows["y"].values.astype(int)
        idx = metrics.resample_matrix(len(y), config.BOOTSTRAP_N, config.SEED)
        binary_metrics(out, outcome, pop, spec, preds, y, rows, window, idx)
        curves_out[(outcome, pop, spec)] = {fs: metrics.decile_bins(y, p, config.CALIBRATION_BINS) for fs, p in preds.items()}

        if outcome == "O1" and pop == "primary":
            # stratified rows (§7 floors on the test stratum of this population)
            for col in ("phase_group", "condition_group"):
                for lvl, g in rows.groupby(col):
                    nn, ee = len(g), int(g["y"].sum())
                    strata_out.setdefault(spec, []).append({"stratum": f"{col}={lvl}", "n": nn, "events": ee,
                                                            "passes": nn >= config.STRATUM_MIN_TRIALS and ee >= config.STRATUM_MIN_EVENTS})
                    if nn >= config.STRATUM_MIN_TRIALS and ee >= config.STRATUM_MIN_EVENTS:
                        ix = g.index.values
                        sidx = metrics.resample_matrix(len(ix), config.BOOTSTRAP_N, config.SEED)
                        binary_metrics(out, outcome, pop, spec, {k: v[ix] for k, v in preds.items()}, y[ix], g, window,
                                       sidx, stratum=f"{col}={lvl}")
            if perm:
                strata = rows["phase_group"].astype(str) + "×" + rows["start_year"].astype(str)
                for fs in ("P+F", "P+F2"):
                    b = _load(outcome, pop, spec, fs)
                    X = Xs[fs]
                    fcols = design.f_block(X)
                    rng = np.random.default_rng(config.SEED)
                    a_p = metrics.auc(y, preds["P"])
                    obs = metrics.auc(y, preds[fs]) - a_p
                    null = []
                    for _ in range(config.PERMUTATION_N):
                        pi = metrics.permute_within(strata.values, rng)
                        Xp = X.copy()
                        Xp[fcols] = X[fcols].values[pi]
                        null.append(metrics.auc(y, models.predict(b["model"], Xp, True)) - a_p)
                    null = np.asarray(null)
                    base = dict(outcome=outcome, population=pop, spec=spec, featureset=fs, comparison=f"{fs} − P",
                                window=window, n=len(y), n_events=int(y.sum()))
                    note = f"{config.PERMUTATION_N} permutations of the F block within phase × start-year strata; model frozen"
                    out.add(**base, metric="perm_null_p95", value=float(np.percentile(null, 95)), note=note)
                    out.add(**base, metric="perm_null_mean", value=float(null.mean()), note=note)
                    out.add(**base, metric="perm_null_share_ge_observed", value=float((null >= obs).mean()), note=note)


def run_continuous(out: Rows, W: dict, window: str, outcome: str, pop: str, fsets) -> None:
    for spec in models.SPECS["continuous"]:
        preds, rows = {}, None
        for fs in fsets:
            b = _load(outcome, pop, spec, fs)
            rows, X = models.assemble(W, outcome, pop, fs, b["medians"], b["levels"])
            assert list(X.columns) == b["columns"]
            preds[fs] = models.predict(b["model"], X, False)
        y = rows["y"].values
        base = dict(outcome=outcome, population=pop, spec=spec, window=window, n=int(len(y)))
        for fs, p in preds.items():
            out.add(**base, featureset=fs, metric="rmse", value=metrics.rmse(y, p))
            out.add(**base, featureset=fs, metric="r2", value=metrics.r2(y, p))
        idx = metrics.resample_matrix(len(y), config.BOOTSTRAP_N, config.SEED)
        d, lo, hi, _ = metrics.bootstrap_delta(y, preds["P+F"], preds["P"], idx, fn=metrics.r2)
        out.add(**base, featureset="P+F", comparison="P+F − P", metric="delta_r2", value=d, ci_lo=lo, ci_hi=hi,
                note=f"paired bootstrap {len(idx)} resamples")
        d, lo, hi, _ = metrics.bootstrap_delta(y, preds["P+F"], preds["P"], idx, fn=metrics.rmse)
        out.add(**base, featureset="P+F", comparison="P+F − P", metric="delta_rmse", value=d, ci_lo=lo, ci_hi=hi,
                note=f"paired bootstrap {len(idx)} resamples")


def run_hazard(out: Rows, W: dict, window: str) -> None:
    pp = pd.read_parquet(tables.path(window, "hazard_pp"))
    ht = pd.read_parquet(tables.path(window, "hazard_trials"))
    if window == "train":  # dry run: training table holds at-risk periods only
        pp = pp[pp["at_risk"]].reset_index(drop=True)
    start = pd.Series(pd.to_datetime(ht["start_date"]).values, index=ht["nct_id"].values)
    risks = {}
    for fs, with_f in (("P", False), ("P+F", True)):
        b = _load("O5", "hazard", "logistic", fs)
        X = models.hazard_design(pp, W["P"], b["medians"], b["levels"], with_f)
        assert list(X.columns) == b["columns"]
        h = b["model"].predict_proba(X)[:, 1]
        risks[fs] = {hz: hazard.cumulative_risk(pp, h, start, hz) for hz in config.HAZARD_HORIZONS_MONTHS}
    for hz in config.HAZARD_HORIZONS_MONTHS:
        t = ht.set_index("nct_id")
        case = (t["event"].eq(1) & (t["months_to_exit"] <= hz)).astype(int)
        ids = t.index
        y = case.loc[ids].values
        base = dict(outcome="O5", population="hazard", spec="logistic", window=window, n=int(len(y)), n_events=int(y.sum()))
        preds = {fs: risks[fs][hz].reindex(ids).values for fs in risks}
        for fs, p in preds.items():
            a, _, lo, hi = metrics.delong(y, p)
            out.add(**base, featureset=fs, metric=f"td_auc_{hz}m", value=a, ci_lo=lo, ci_hi=hi,
                    note="cumulative/dynamic: cases terminated by h; controls all others (no censoring before 36 m)")
            out.add(**base, featureset=fs, metric=f"td_brier_{hz}m", value=metrics.brier(y, p))
        idx = metrics.resample_matrix(len(y), config.BOOTSTRAP_N, config.SEED)
        d, lo, hi, _ = metrics.bootstrap_delta(y, preds["P+F"], preds["P"], idx)
        out.add(**base, featureset="P+F", comparison="P+F − P", metric=f"delta_td_auc_{hz}m", value=d, ci_lo=lo, ci_hi=hi,
                note=f"paired bootstrap {len(idx)} resamples of trials")


def lr_rows(out: Rows) -> None:
    hyper = json.loads((config.MODELS_DIR / "hyperparams.json").read_text())
    for k, v in hyper.items():
        parts = k.split("|")
        if len(parts) == 4 and parts[2] == "lr_test":
            base = dict(outcome=parts[0], population=parts[1], spec="logistic", featureset=parts[3],
                        comparison=f"{parts[3]} − P", window="train", note="unpenalised MLE, nested pair, training window")
            out.add(**base, metric="lr_chi2", value=v["chi2"])
            out.add(**base, metric="lr_df", value=v["df"])
            out.add(**base, metric="lr_p", value=v["p"])


def evaluate(window: str, metrics_path, figures_dir, perm: bool = True) -> pd.DataFrame:
    W = tables.load_window(window)
    out = Rows()
    curves: dict = {}
    strata: dict = {}
    for outcome, pop, fsets in models.RUNS:
        if not models.gate_allows(outcome):
            continue
        if outcome in models.BINARY:
            run_binary(out, W, window, outcome, pop, fsets, curves, perm, strata)
        else:
            run_continuous(out, W, window, outcome, pop, fsets)
        print("evaluated", outcome, pop, flush=True)
    if models.gate_allows("O5"):
        run_hazard(out, W, window)
        print("evaluated O5 hazard", flush=True)
    lr_rows(out)
    m = pd.DataFrame(out, columns=COLUMNS)
    m.to_csv(metrics_path, index=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    bins = []
    for (outcome, pop, spec), cv in curves.items():
        _calibration_figure(cv, f"{outcome} · {pop} · {spec} · {window} window", figures_dir / f"calibration_{outcome}_{pop}_{spec}.png")
        for fs, b in cv.items():
            bins.append(b.assign(outcome=outcome, population=pop, spec=spec, featureset=fs))
    pd.concat(bins).to_csv(figures_dir / "calibration_bins.csv", index=False)
    (figures_dir / "strata_floors.json").write_text(json.dumps(strata, indent=2) + "\n")
    return m


# --- Phase 4-X exploratory run "px" (plan §14: labelled exploratory) -------------------------
def guard_write(path) -> None:
    """Refuse any write to a Phase 4 primary path (file or anything under a protected dir)."""
    p = Path(path).resolve()
    for q in config.PHASE4_PROTECTED:
        q = Path(q).resolve()
        if p == q or q in p.parents:
            raise PermissionError(f"exploratory run may not write Phase 4 path {p}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase4_hashes() -> dict:
    out = {f"models/{p.name}": _sha(p) for p in sorted(config.MODELS_DIR.glob("*"))}
    for p in (config.METRICS_PATH, config.READING_PATH, config.TEST_EVALUATED_MARKER):
        out[p.name] = _sha(p)
    return out


def evaluate_px(window: str, out_path) -> pd.DataFrame:
    """Primary P-only models (frozen, Phase 4) vs the px models (no P indicators), O1, on `window`."""
    guard_write(out_path)
    W = tables.load_window(window)
    out = Rows()
    for pop in models.PX_POPULATIONS:
        for spec in models.SPECS["binary"]:
            preds, y = {}, None
            for fs, base, drop in (("P", None, False), ("P-px", models.PX_MODELS_DIR, True)):
                b = joblib.load(models.bundle_path("O1", pop, spec, "P", base))
                rows, X = models.assemble(W, "O1", pop, "P", b["medians"], b["levels"], drop_p_missing=drop)
                assert list(X.columns) == b["columns"], "column order differs from the frozen model"
                yy = rows["y"].values.astype(int)
                assert y is None or (yy == y).all()
                y = yy
                preds[fs] = models.predict(b["model"], X, True)
            n, ev = int(len(y)), int(y.sum())
            base = dict(outcome="O1", population=pop, spec=spec, window=window, n=n, n_events=ev)
            for fs, p in preds.items():
                a, _, lo, hi = metrics.delong(y, p)
                out.add(**base, featureset=fs, metric="auc", value=a, ci_lo=lo, ci_hi=hi, note="DeLong 95% CI")
                out.add(**base, featureset=fs, metric="brier", value=metrics.brier(y, p))
                out.add(**base, featureset=fs, metric="log_loss", value=metrics.log_loss(y, p))
                s, i = metrics.calibration_slope_intercept(y, p)
                out.add(**base, featureset=fs, metric="cal_slope", value=s)
                out.add(**base, featureset=fs, metric="cal_intercept", value=i, note="calibration-in-the-large (offset logit p)")
            idx = metrics.resample_matrix(n, config.BOOTSTRAP_N, config.SEED)  # = the Phase 4 matrix for this population
            d, lo, hi, _ = metrics.bootstrap_delta(y, preds["P-px"], preds["P"], idx)
            out.add(**base, featureset="P-px", comparison="P-px − P", metric="delta_auc", value=d, ci_lo=lo, ci_hi=hi,
                    note=f"exploratory (plan §14); paired bootstrap {len(idx)} resamples, percentile 95% CI")
    m = pd.DataFrame(out, columns=COLUMNS)
    m.to_csv(out_path, index=False)
    return m


def main_px(a) -> None:
    d = config.EXPLORATORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        m = evaluate_px("train", d / "dryrun_metrics_px.csv")
        print(m.to_string())
        return
    marker = d / "px_evaluated.json"
    guard_write(marker)
    if not a.rerun_reason:
        raise SystemExit("an exploratory test read needs --rerun-reason (plan §14)")
    if marker.exists():
        raise SystemExit(f"px already evaluated ({marker.read_text().strip()[:200]})")
    frozen = json.loads(config.TEST_EVALUATED_MARKER.read_text())["frozen_models_sha256"]
    before = phase4_hashes()
    if {k.removeprefix("models/"): v for k, v in before.items() if k.startswith("models/")} != frozen:
        raise SystemExit("Phase 4 model files differ from the hashes in test_evaluated.json; stopping")
    started = datetime.now(timezone.utc).isoformat()
    marker.write_text(json.dumps({"started_utc": started, "rerun_reason": a.rerun_reason}) + "\n")
    m = evaluate_px("test", d / "metrics_px.csv")
    after = phase4_hashes()
    rec = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "rerun_reason": a.rerun_reason,
           "rows": len(m), "phase4_files_unchanged": before == after, "phase4_models_match_marker": True,
           "px_models_sha256": {p.name: _sha(p) for p in sorted(models.PX_MODELS_DIR.glob("*"))}}
    marker.write_text(json.dumps(rec, indent=1) + "\n")
    print(m.to_string())
    print("Phase 4 files unchanged:", before == after)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--exploratory", choices=["px"], default=None)
    a = ap.parse_args()
    if a.exploratory == "px":
        main_px(a)
        return
    if a.dry_run:
        d = config.PHASE4_DIR / "dryrun"
        d.mkdir(parents=True, exist_ok=True)
        m = evaluate("train", d / "dryrun_metrics.csv", d / "figures", perm=not a.no_perm)
        print(m.to_string())
        return
    marker = config.TEST_EVALUATED_MARKER
    if marker.exists() and not a.rerun_reason:
        raise SystemExit(f"test window already evaluated ({marker.read_text().strip()}); pass --rerun-reason to override")
    prior = json.loads(marker.read_text()) if marker.exists() else None
    started = datetime.now(timezone.utc).isoformat()
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(config.MODELS_DIR.glob("*"))}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"started_utc": started, "rerun_reason": a.rerun_reason, "prior": prior}) + "\n")
    m = evaluate("test", config.METRICS_PATH, config.FIGURES_DIR)
    marker.write_text(json.dumps({"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                                  "rerun_reason": a.rerun_reason, "prior": prior, "rows": len(m),
                                  "frozen_models_sha256": frozen}, indent=1) + "\n")
    print(m.to_string())


if __name__ == "__main__":
    main()
