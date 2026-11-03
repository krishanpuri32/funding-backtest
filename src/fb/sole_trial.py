"""Sole-trial dollars table (plan §1 secondary (c); Amendment A6, descriptive, after the reading).

    python -m fb.sole_trial      # -> data/derived/sole_trial.csv; prints the markdown table

Population: cohort trials with exactly one attributable project under A1 N = 20, where that
project (non-umbrella) links to exactly one NCT ID. No attribution rule is needed and no dollar
is counted twice. Terminal = O1 label 0 / 1. Dollars = `total_cost_base_primary` (CPI-2020,
parent rows, FYs overlapping the trial window — the cohort convention; nothing recomputed).
Beside each window, a comparison row for the full attributable N = 20 set (its dollars are sums
over trials that may share an award, so its totals double count; shown for the rate only).
"""
from __future__ import annotations

import pandas as pd

from . import config
from .cohort import COHORT_PATH, COHORT_PROJECTS_PATH
from .stats import wilson_ci

SOLE_TRIAL_PATH = config.DATA_DERIVED / "sole_trial.csv"
COLUMNS = ["window", "population", "n_terminal", "o1_events", "event_rate", "wilson_lo", "wilson_hi",
           "dollars_n", "dollars_total_base", "dollars_median_base", "dollars_noncomplete_base", "noncomplete_share"]


def sole_trial_ids(cohort: pd.DataFrame, cohort_projects: pd.DataFrame) -> set[str]:
    """NCT IDs whose single attributable (non-umbrella N = 20) project links to that trial only."""
    c = cohort[cohort["has_attributable_funding_n20"] & (cohort["n_attributable_n20"] == 1)]
    attr = cohort_projects[~cohort_projects["is_umbrella_n20"] & cohort_projects["nct_id"].isin(c["nct_id"])]
    one = attr.groupby("nct_id").filter(lambda g: len(g) == 1)
    return set(one.loc[one["n_nct_linked"] == 1, "nct_id"])


def summary_row(df: pd.DataFrame) -> dict:
    """Rate and dollars over the terminal rows of df (o1_label in {0, 1})."""
    t = df[df["o1_label"].isin([0, 1])]
    y = t["o1_label"].astype(int)
    n, k = int(len(t)), int(y.sum())
    lo, hi = wilson_ci(k, n)
    d = pd.to_numeric(t["total_cost_base_primary"], errors="coerce")
    total = float(d.sum())
    nonc = float(d[y == 1].sum())
    return {"n_terminal": n, "o1_events": k, "event_rate": (k / n if n else float("nan")), "wilson_lo": lo, "wilson_hi": hi,
            "dollars_n": int(d.notna().sum()), "dollars_total_base": total,
            "dollars_median_base": float(d.median()) if d.notna().any() else float("nan"),
            "dollars_noncomplete_base": nonc, "noncomplete_share": (nonc / total if total else float("nan"))}


def build(cohort: pd.DataFrame, cohort_projects: pd.DataFrame) -> pd.DataFrame:
    ids = sole_trial_ids(cohort, cohort_projects)
    rows = []
    for w in ("train", "test"):
        cw = cohort[cohort["window"] == w]
        rows.append({"window": w, "population": "sole_trial", **summary_row(cw[cw["nct_id"].isin(ids)])})
        rows.append({"window": w, "population": "attributable_n20", **summary_row(cw[cw["has_attributable_funding_n20"]])})
    return pd.DataFrame(rows, columns=COLUMNS)


def markdown(t: pd.DataFrame) -> str:
    out = ["| window | population | n terminal | events | rate (Wilson 95 % CI) | obligated, CPI-2020 | median | to trials that did not complete |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples():
        out.append(f"| {r.window} | {r.population} | {r.n_terminal:,} | {r.o1_events:,} | "
                   f"{r.event_rate:.1%} ({r.wilson_lo:.1%}–{r.wilson_hi:.1%}) | ${r.dollars_total_base / 1e6:,.0f} M | "
                   f"${r.dollars_median_base / 1e6:,.2f} M | ${r.dollars_noncomplete_base / 1e6:,.0f} M ({r.noncomplete_share:.1%}) |")
    out.append("")
    out.append("attributable_n20 dollar totals sum trials that may share an award (double counted); its rate is the comparison.")
    return "\n".join(out)


def main() -> pd.DataFrame:
    t = build(pd.read_parquet(COHORT_PATH), pd.read_parquet(COHORT_PROJECTS_PATH))
    t.to_csv(SOLE_TRIAL_PATH, index=False)
    print(markdown(t))
    return t


if __name__ == "__main__":
    main()
