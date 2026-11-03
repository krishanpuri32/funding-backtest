"""Base-rate table (plan §1 secondary (c), §13).

Training window only, terminal-status trials (O1 events + Completed); unknown and
censored counted beside each stratum. Three variants: primary (A1 N = 20 attributable
set), all (no umbrella exclusion), a2 (primary minus Withdrawn/enrollment-0).
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from . import config, priors
from .cohort import COHORT_PATH
from .stats import wilson_ci

HISTORY_MANUAL_PATH = config.DATA_DERIVED / "history_probe_manual.csv"
VARIANTS = {
    "primary": {"label": "primary — A1 N = 20 attributable set", "dollar": "total_cost_base_primary",
                "activity": "activity_class_primary", "ic": "ic_group_primary"},
    "all": {"label": "all trials — no umbrella exclusion", "dollar": "total_cost_base_all",
            "activity": "activity_class_all", "ic": "ic_group_all"},
    "a2": {"label": "A2 excluded — primary set minus Withdrawn/enrollment-0", "dollar": "total_cost_base_primary",
           "activity": "activity_class_primary", "ic": "ic_group_primary"},
}
STRATA = [("phase", "phase_group"), ("condition_group", "condition_group"), ("activity_class", None),
          ("ic", None), ("start_year", "start_year")]


def select_variant(cohort: pd.DataFrame, variant: str) -> pd.DataFrame:
    tr = cohort[cohort["window"] == "train"].copy()
    if variant in ("primary", "a2"):
        tr = tr[tr["has_attributable_funding_n20"]]
    if variant == "a2":
        tr = tr[~tr["withdrawn_zero_enroll"]]
    v = VARIANTS[variant]
    tr["dollars"] = tr[v["dollar"]]
    tr["activity_class"] = tr[v["activity"]]
    tr["ic"] = tr[v["ic"]]
    return tr


def _q(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


def stratum_row(df: pd.DataFrame) -> dict:
    """One base-rate row for the trials in df (all statuses); rates on terminal rows only."""
    term = df[df["status_group"].isin(["event", "control"])]
    k = int((term["status_group"] == "event").sum())
    n = int(len(term))
    lo, hi = wilson_ci(k, n)
    return {
        "n_terminal": n, "o1_events": k, "event_rate": (k / n if n else float("nan")),
        "wilson_lo": lo, "wilson_hi": hi,
        "n_unknown": int((df["status_group"] == "unknown").sum()),
        "n_censored": int((df["status_group"] == "censored").sum()),
        "dollars_n": int(term["dollars"].notna().sum()),
        "dollars_median": _q(term["dollars"], 0.5), "dollars_q1": _q(term["dollars"], 0.25), "dollars_q3": _q(term["dollars"], 0.75),
        "enrollment_median": _q(term["enrollment"], 0.5),
        "duration_months_median": _q(term["duration_months"], 0.5),
        "meets_floor": bool(n >= config.STRATUM_MIN_TRIALS and k >= config.STRATUM_MIN_EVENTS),
    }


def _order(var: str, values: list) -> list:
    if var == "phase":
        return [p for p in priors.PHASE_ORDER if p in values]
    if var == "condition_group":
        return [g for g in priors.MESH_GROUP_PRIORITY + [priors.MESH_GROUP_NONE] if g in values]
    if var == "activity_class":
        return [a for a in priors.ACTIVITY_CLASS_ORDER + ["none"] if a in values]
    if var == "ic":
        return [i for i in config.IC_TOP10 + ["other", "none"] if i in values]
    return sorted(values)


def table_for(tr: pd.DataFrame) -> pd.DataFrame:
    rows = [{"stratum_var": "overall", "stratum": "all", **stratum_row(tr)}]
    for var, col in STRATA:
        col = col or var
        for val in _order(var, tr[col].dropna().unique().tolist()):
            rows.append({"stratum_var": var, "stratum": str(val), **stratum_row(tr[tr[col] == val])})
    return pd.DataFrame(rows)


def phase_x_condition(tr: pd.DataFrame) -> pd.DataFrame:
    """(f): cells meeting the §7 floor on their own; every other cell pooled into 'other'."""
    cells = []
    pooled = pd.Series(False, index=tr.index)
    for ph in _order("phase", tr["phase_group"].unique().tolist()):
        for cg in _order("condition_group", tr["condition_group"].unique().tolist()):
            m = (tr["phase_group"] == ph) & (tr["condition_group"] == cg)
            if not m.any():
                continue
            r = stratum_row(tr[m])
            if r["meets_floor"]:
                cells.append({"phase": ph, "condition_group": cg, **r})
            else:
                pooled |= m
    if pooled.any():
        cells.append({"phase": "other", "condition_group": "other (pooled cells below floor)", **stratum_row(tr[pooled])})
    return pd.DataFrame(cells)


def history_summary(path=HISTORY_MANUAL_PATH, cohort: pd.DataFrame | None = None) -> dict:
    """Step 6: summarise the Phase 1 hand pull if any row is filled."""
    if not path.exists():
        return {"status": "file missing"}
    m = pd.read_csv(path, dtype=str)
    data_cols = [c for c in m.columns if c not in ("nct_id", "url")]
    filled = m[m[data_cols].notna().any(axis=1)]
    if filled.empty:
        return {"status": "blank", "rows": int(len(m))}
    out = {"status": "filled", "rows": int(len(m)), "n_filled": int(len(filled))}
    enr = pd.to_numeric(filled.get("v1_enrollment"), errors="coerce")
    pcd = pd.to_datetime(filled.get("v1_primary_completion"), errors="coerce")
    out["n_with_v1_enrollment"] = int(enr.notna().sum())
    out["n_with_v1_primary_completion"] = int(pcd.notna().sum())
    if cohort is not None:
        cur = cohort.set_index("nct_id")
        j = filled.set_index("nct_id").join(cur[["enrollment", "trial_end", "trial_end_source"]], how="left")
        e1 = pd.to_numeric(j["v1_enrollment"], errors="coerce")
        out["enrollment_differs_from_current"] = int(((e1.notna()) & (j["enrollment"].notna()) & (e1 != j["enrollment"])).sum())
        p1 = pd.to_datetime(j["v1_primary_completion"], errors="coerce")
        pc_current = pd.to_datetime(j["trial_end"]).where(j["trial_end_source"] == "primary_completion")
        out["primary_completion_differs_from_current"] = int(((p1.notna()) & (pc_current.notna()) & (p1 != pc_current)).sum())
    return out


def build(write: bool = True) -> dict:
    cohort = pd.read_parquet(COHORT_PATH)
    out: dict = {}
    for variant in VARIANTS:
        tr = select_variant(cohort, variant)
        t = table_for(tr)
        f = phase_x_condition(tr)
        out[variant] = {"n_train_rows": int(len(tr)), "table": t, "phase_x_condition": f}
        if write:
            t.to_csv(config.DATA_DERIVED / f"base_rates_{variant}.csv", index=False)
            f.to_csv(config.DATA_DERIVED / f"base_rates_{variant}_phase_x_condition.csv", index=False)
    out["history"] = history_summary(cohort=cohort)
    return out


# --- markdown rendering for the report ---------------------------------------------
def _fmt_money(v) -> str:
    return "—" if pd.isna(v) else f"{v / 1e6:,.2f}"


def _fmt(v, nd=1) -> str:
    return "—" if pd.isna(v) else f"{v:,.{nd}f}"


def md_table(t: pd.DataFrame) -> str:
    hdr = "| stratum | n terminal | O1 events | rate % | Wilson 95 % | unknown | censored | $ n | $M median [IQR] (2020) | enrol. median | dur. mo median | floor |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [hdr]
    for _, r in t.iterrows():
        name = r["stratum"] if r["stratum_var"] == "overall" else f"{r['stratum_var']}: {r['stratum']}"
        lines.append(
            f"| {name} | {r['n_terminal']:,} | {r['o1_events']:,} | {_fmt(100 * r['event_rate'])} | "
            f"{_fmt(100 * r['wilson_lo'])}–{_fmt(100 * r['wilson_hi'])} | {r['n_unknown']:,} | {r['n_censored']:,} | {r['dollars_n']:,} | "
            f"{_fmt_money(r['dollars_median'])} [{_fmt_money(r['dollars_q1'])}–{_fmt_money(r['dollars_q3'])}] | "
            f"{_fmt(r['enrollment_median'], 0)} | {_fmt(r['duration_months_median'])} | {'✓' if r['meets_floor'] else '✗'} |")
    return "\n".join(lines)


def md_phase_x_condition(f: pd.DataFrame, top: int = 5) -> str:
    f2 = f.copy()
    f2["stratum_var"] = "phase × condition"
    f2["stratum"] = f2["phase"] + " × " + f2["condition_group"]
    real = f2[f2["phase"] != "other"].sort_values("n_terminal", ascending=False).head(top)
    pooled = f2[f2["phase"] == "other"]
    return md_table(pd.concat([real, pooled]))


def render_markdown(res: dict) -> str:
    parts = []
    for variant, v in VARIANTS.items():
        r = res[variant]
        parts.append(f"### Base rates — {v['label']} (training-window rows: {r['n_train_rows']:,})\n")
        parts.append(md_table(r["table"]))
        f = r["phase_x_condition"]
        n_cells = int((f["phase"] != "other").sum())
        parts.append(f"\n(f) phase × condition group: `data/derived/base_rates_{variant}_phase_x_condition.csv` — {n_cells} cells meet the §7 floor; five largest plus the pooled remainder:\n")
        parts.append(md_phase_x_condition(f))
        parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    res = build()
    print(render_markdown(res))
    print("\nhistory sample:", json.dumps(res["history"]))
