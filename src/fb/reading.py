"""Plan §9 reading rules, applied by code (Phase 4 step 8).

    python -m fb.reading            # reads data/derived/metrics.csv, prints and writes reading.txt

The cell is computed from config thresholds and never chosen. A1 / A2 / A4b stability: the
primary cell comes from the primary run only; it is *robust* if every sensitivity run lands in
the same cell and *fragile* otherwise, naming the disagreeing runs.

Inputs taken from metrics.csv (outcome O1, spec logistic, window test):
  ΔAUC (P+F − P) and its bootstrap CI, F₂ ΔAUC (P+F2 − P) and its CI — per population;
  P-only AUC — the all-trials P-only model (A3), or its A2 version for the A2 run.
"""
from __future__ import annotations

import pandas as pd

from . import config

SENSITIVITY_RUNS = {  # run -> (population for ΔAUC, population for the P-only AUC)
    "S1 (A4b, all attributable + F indicators)": ("S1", "P_only"),
    "A2 (withdrawn / enrollment-0 excluded)": ("A2", "P_only_A2"),
    "A1 grid (ii) N = 50": ("N50", "P_only"),
    "A1 grid (iii) activity rule": ("activity", "P_only"),
    "A1 grid (i) no umbrella exclusion": ("no_umbrella", "P_only"),
}


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def cell(delta: float, lo: float, hi: float, f2_delta: float, f2_lo: float, f2_hi: float, p_auc: float) -> str:
    """R1–R4 by plan §9. Any NaN input on the primary ΔAUC fails its threshold (conservative)."""
    primary_ok = (delta >= config.READ_DELTA_AUC_MIN) and ci_excludes_zero(lo, hi)
    f2_ok = (f2_delta >= config.READ_F2_DELTA_AUC_MIN) and ci_excludes_zero(f2_lo, f2_hi)
    if primary_ok and f2_ok:
        return "R1"
    if primary_ok:
        return "R2"
    return "R3" if p_auc >= config.READ_P_AUC_MIN else "R4"


def stability(primary: str, others: dict[str, str]) -> tuple[str, list[str]]:
    diff = [k for k, v in others.items() if v != primary]
    return ("robust" if not diff else "fragile"), diff


def _get(m: pd.DataFrame, population: str, featureset: str, metric: str) -> pd.Series:
    r = m[(m["outcome"] == "O1") & (m["spec"] == "logistic") & (m["window"] == "test") & (m["population"] == population)
          & (m["featureset"] == featureset) & (m["metric"] == metric) & (m["stratum"] == "all")]
    if len(r) != 1:
        raise KeyError(f"{population} {featureset} {metric}: {len(r)} rows")
    return r.iloc[0]


def run_cell(m: pd.DataFrame, population: str, p_population: str, p_override: dict | None = None) -> dict:
    """p_override: {p_population: AUC} replaces the P-only AUC (exploratory what-if; CI unknown)."""
    d = _get(m, population, "P+F", "delta_auc")
    f2 = _get(m, population, "P+F2", "delta_auc")
    p = _get(m, p_population, "P", "auc")
    p_auc, p_lo, p_hi = p["value"], p["ci_lo"], p["ci_hi"]
    if p_override and p_population in p_override:
        p_auc, p_lo, p_hi = float(p_override[p_population]), float("nan"), float("nan")
    pin = _get(m, population, "P", "auc")
    c = cell(d["value"], d["ci_lo"], d["ci_hi"], f2["value"], f2["ci_lo"], f2["ci_hi"], p_auc)
    return {"population": population, "delta": d["value"], "lo": d["ci_lo"], "hi": d["ci_hi"],
            "f2": f2["value"], "f2_lo": f2["ci_lo"], "f2_hi": f2["ci_hi"], "p_auc": p_auc,
            "p_auc_lo": p_lo, "p_auc_hi": p_hi, "p_auc_in_population": pin["value"],
            "n": int(d["n"]), "events": int(d["n_events"]), "cell": c}


def read(m: pd.DataFrame, p_override: dict | None = None) -> tuple[str, dict]:
    prim = run_cell(m, "primary", "P_only", p_override)
    others = {name: run_cell(m, pop, ppop, p_override) for name, (pop, ppop) in SENSITIVITY_RUNS.items()}
    label, diff = stability(prim["cell"], {k: v["cell"] for k, v in others.items()})
    perm = {}
    for fs in ("P+F", "P+F2"):
        try:
            perm[fs] = (_get(m, "primary", fs, "perm_null_p95")["value"],
                        _get(m, "primary", fs, "perm_null_share_ge_observed")["value"])
        except KeyError:
            pass
    f = lambda x: f"{x:+.4f}"
    lines = [
        "READING (plan §9, computed by fb.reading from metrics.csv; thresholds from fb.config)",
        f"Primary ΔAUC(O1) [P+F − P, logistic, test, primary landmark population, n = {prim['n']}, events = {prim['events']}]: "
        f"{f(prim['delta'])}  (bootstrap 95 % CI {f(prim['lo'])} to {f(prim['hi'])})",
        f"F₂ ΔAUC(O1) [P+F₂ − P]: {f(prim['f2'])}  (CI {f(prim['f2_lo'])} to {f(prim['f2_hi'])})",
        f"P-only AUC(O1) [all trials, A3]: {prim['p_auc']:.4f}  (DeLong 95 % CI {prim['p_auc_lo']:.4f}–{prim['p_auc_hi']:.4f});"
        f" P AUC inside the primary population: {prim['p_auc_in_population']:.4f}",
        f"Thresholds: ΔAUC ≥ {config.READ_DELTA_AUC_MIN} · F₂ ΔAUC ≥ {config.READ_F2_DELTA_AUC_MIN} · P-only AUC ≥ {config.READ_P_AUC_MIN}",
        "",
        f"CELL: {prim['cell']} — {config.READING_CELLS[prim['cell']]}",
        f"STABILITY: {label}" + (f" — disagreeing runs: {', '.join(diff)}" if diff else " — every sensitivity run lands in the same cell"),
        "",
        "Sensitivity runs (same rule, same metric):",
    ]
    for name, r in others.items():
        lines.append(f"  {name}: ΔAUC {f(r['delta'])} ({f(r['lo'])} to {f(r['hi'])}) · F₂ ΔAUC {f(r['f2'])} "
                     f"({f(r['f2_lo'])} to {f(r['f2_hi'])}) · P-only AUC {r['p_auc']:.4f} · n {r['n']} / events {r['events']} → {r['cell']}")
    lines.append("")
    for fs, (p95, share) in perm.items():
        obs = _get(m, "primary", fs, "delta_auc")["value"]
        lines.append(f"Permutation null ({fs} − P, {config.PERMUTATION_N} reps, F permuted within phase × start-year strata of the test set): "
                     f"95th percentile {f(p95)} vs observed {f(obs)}; share of null ≥ observed {share:.3f}")
    return "\n".join(lines), {"primary": prim, "sensitivity": others, "stability": label, "disagreeing": diff}


def main() -> str:
    """Bare: the primary reading -> reading.txt. With --p-only-auc (and optionally
    --p-only-a2-auc): an EXPLORATORY what-if cell (plan §14) printed and written to
    exploratory/reading_px.txt only; reading.txt is never touched in that mode."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--p-only-auc", type=float, default=None)
    ap.add_argument("--p-only-a2-auc", type=float, default=None)
    a = ap.parse_args()
    m = pd.read_csv(config.METRICS_PATH)
    override = {k: v for k, v in (("P_only", a.p_only_auc), ("P_only_A2", a.p_only_a2_auc)) if v is not None}
    if not override:
        text, _ = read(m)
        config.READING_PATH.write_text(text + "\n")
        print(text)
        return text
    text, _ = read(m, override)
    head = ("EXPLORATORY (plan §14) — not the primary reading. P-only AUC replaced by: "
            + ", ".join(f"{k} = {v:.4f}" for k, v in override.items()) + "; its CI is not carried here.")
    text = head + "\n" + text
    config.EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    (config.EXPLORATORY_DIR / "reading_px.txt").write_text(text + "\n")
    print(text)
    return text


if __name__ == "__main__":
    main()
