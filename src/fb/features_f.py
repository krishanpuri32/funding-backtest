"""Funding-trajectory features F and the early-window variant F₂ (plan §5; A4).

Rows used: ExPORTER *parent* fiscal-year rows (SUBPROJECT_ID null) of a trial's
attributable projects (A1, N = 20), whose budget period overlaps the trial window
[start, trial_end] (fb.rules.fy_overlaps_vec) AND whose BUDGET_END precedes the trial's
outcome date (fb.leakage.cut_rows). Dollars are CPI-U 2020 (fb.cpi), by the row's FY.

Exact rules (documented for the record):
- A row with a null TOTAL_COST is an award action of unknown amount (VA intramural, N01
  contracts): it counts for F_years, F_gap, F_supp, F_nce and F_noncompeting_share, and is
  skipped by every dollar feature.
- Annual series: CPI-2020 dollars summed per FY_FILE over known-cost rows, from the first
  to the last FY with a positive obligation, zero-filled in between.
- F_total  = log(1 + sum of the annual series); missing if no known-cost row.
- F_years  = distinct FY_FILE among the rows.
- F_shape  = OLS slope of the annual series on FY index (0, 1, ...) ÷ its total; missing
  if fewer than 2 FYs carry a positive obligation.
- F_yoy_min = largest proportional year-on-year cut in the annual series,
  max(0, (y[t-1] − y[t]) / y[t-1]) over t with y[t-1] > 0; 0 if never cut; missing as F_shape.
- F_gap / F_gap_count = whether / how many FYs in the gap window carry no obligation on any
  attributable project. Gap window: FY(trial start) .. FY(min(outcome date, trial end)) − 2
  (fb.leakage.gap_window_fys — the FYs fully observable before the cut). A FY is covered if
  a row has that FY_FILE, or an earlier row's budget period runs to 30 Sep of that FY
  (multi-year budgets and stretched final years are not gaps). Missing if the window is empty.
- F_supp   = rows with APPLICATION_TYPE 3 or SUFFIX starting 'S'.
- F_nce    = 1 if any attributable project shows (A) a row that is not a competing action
  (type 1 / 2) whose PROJECT_END is more than NCE_EXTENSION_TOLERANCE_DAYS later than the
  PROJECT_END on the first row of its award segment (segment = rows from a type 1 / 2 row
  up to the next one), or (B) a row with SUPPORT_YEAR beyond the segment's original term in
  years and TOTAL_COST exactly 0. Else 0.
- F_indirect_ratio = mean over FYs >= 2006 of (sum indirect ÷ sum (direct + indirect)) on
  rows where both are known and their sum > 0; missing if none. The denominator is the
  budget's own direct + indirect, not TOTAL_COST: in 184 parent rows behind attributable
  cohort projects TOTAL_COST is below direct + indirect (mostly U10 / U01 cooperative
  agreements), which pushed the ratio to ~2e5; direct + indirect equals TOTAL_COST in 83 %
  of rows and keeps the ratio in [0, 1].
- F_per_enrollee = log(1 + total CPI-2020 dollars ÷ planned enrollment from version 0);
  missing if either is missing or planned enrollment is 0.
- F_noncompeting_share = type-5 rows ÷ all rows.
- A trial with no row after overlap and cut has every feature missing (plan §6: indicator +
  median, never zero); `f_rows_reason` says why.
F₂: same functions on the rows of the first two fiscal years (f0, f0 + 1), f0 = the earliest
FY_FILE among the trial's rows after overlap and cut; gap window {f0, f0 + 1} limited to the
observable FYs; F_years becomes years funded of the two; F_nce is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, exporter, leakage, rules
from .cpi import deflator

F_COLUMNS = ["F_total", "F_years", "F_shape", "F_yoy_min", "F_gap", "F_gap_count", "F_supp", "F_nce",
             "F_indirect_ratio", "F_per_enrollee", "F_noncompeting_share"]
F2_COLUMNS = [c for c in F_COLUMNS if c != "F_nce"]
COMPETING_TYPES = {"1", "2"}


# --- row preparation ----------------------------------------------------------------
def prepare_rows(projects: pd.DataFrame, cpi: dict[int, float] | None,
                 base_year: int = config.CPI_BASE_YEAR) -> pd.DataFrame:
    """Parent rows with parsed dates, numeric costs, CPI-2020 dollars and award-segment references.

    projects: ExPORTER columns CORE_PROJECT_NUM, SUBPROJECT_ID, FY_FILE, APPLICATION_TYPE,
    SUFFIX, SUPPORT_YEAR, PROJECT_START, PROJECT_END, BUDGET_START, BUDGET_END, TOTAL_COST,
    INDIRECT_COST_AMT.
    """
    p = projects[projects["SUBPROJECT_ID"].isna()].copy()
    r = pd.DataFrame({
        "core_project_num": p["CORE_PROJECT_NUM"].values,
        "FY_FILE": pd.to_numeric(p["FY_FILE"], errors="coerce").astype(int).values,
        "app_type": p["APPLICATION_TYPE"].fillna("").astype(str).str.strip().values,
        "suffix": p["SUFFIX"].fillna("").astype(str).str.strip().str.upper().values,
        "support_year": pd.to_numeric(p["SUPPORT_YEAR"], errors="coerce").values,
        "project_start": exporter.parse_exporter_date(p["PROJECT_START"]).values,
        "project_end": exporter.parse_exporter_date(p["PROJECT_END"]).values,
        "budget_start": exporter.parse_exporter_date(p["BUDGET_START"]).values,
        "budget_end": exporter.parse_exporter_date(p["BUDGET_END"]).values,
        "cost": pd.to_numeric(p["TOTAL_COST"], errors="coerce").values,
        "indirect": pd.to_numeric(p["INDIRECT_COST_AMT"], errors="coerce").values,
        "direct": pd.to_numeric(p["DIRECT_COST_AMT"], errors="coerce").values if "DIRECT_COST_AMT" in p else np.nan,
    })
    if cpi is not None:
        r["cost_base"] = r["cost"] * deflator(cpi, r["FY_FILE"], base_year).values
    else:
        r["cost_base"] = r["cost"]
    # award segments: a competing action (type 1 / 2) opens a new segment
    r["_rank"] = np.where(r["app_type"].isin(COMPETING_TYPES), 0, 1)
    r = r.sort_values(["core_project_num", "FY_FILE", "_rank", "budget_start"]).reset_index(drop=True)
    r["seg"] = r["app_type"].isin(COMPETING_TYPES).groupby(r["core_project_num"]).cumsum()
    grp = r.groupby(["core_project_num", "seg"])
    r["seg_orig_end"] = grp["project_end"].transform("first")
    seg_start = grp["project_start"].transform("first")
    term = ((r["seg_orig_end"] - seg_start).dt.days / 365.25).round()
    r["seg_term_years"] = term.clip(lower=1)
    return r.drop(columns=["_rank"])


# --- one function per feature (g = one trial's rows after overlap and cut) -----------
def annual_series(g: pd.DataFrame) -> pd.Series:
    k = g[g["cost_base"].notna()]
    if k.empty:
        return pd.Series(dtype=float)
    a = k.groupby("FY_FILE")["cost_base"].sum()
    a = a[a > 0]
    if a.empty:
        return pd.Series(dtype=float)
    return a.reindex(range(int(a.index.min()), int(a.index.max()) + 1), fill_value=0.0)


def f_total(g: pd.DataFrame) -> float:
    a = annual_series(g)
    return float(np.log1p(a.sum())) if len(a) else np.nan


def f_years(g: pd.DataFrame) -> float:
    return float(g["FY_FILE"].nunique()) if len(g) else np.nan


def f_shape(g: pd.DataFrame) -> float:
    a = annual_series(g)
    if (a > 0).sum() < 2:
        return np.nan
    x = np.arange(len(a), dtype=float)
    y = a.values.astype(float)
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return float(slope / y.sum())


def f_yoy_min(g: pd.DataFrame) -> float:
    a = annual_series(g)
    if (a > 0).sum() < 2:
        return np.nan
    prev, cur = a.values[:-1], a.values[1:]
    ok = prev > 0
    cuts = (prev[ok] - cur[ok]) / prev[ok]
    return float(max(0.0, cuts.max())) if len(cuts) else 0.0


def f_gap(g: pd.DataFrame, window_fys: list[int]) -> tuple[float, float]:
    if not window_fys:
        return np.nan, np.nan
    fy = set(int(v) for v in g["FY_FILE"])
    gaps = 0
    for f in window_fys:
        if f in fy:
            continue
        fy_end = pd.Timestamp(year=f, month=9, day=30)
        earlier = g[(g["FY_FILE"] < f) & (pd.to_datetime(g["budget_end"]) >= fy_end)]
        if earlier.empty:
            gaps += 1
    return float(gaps > 0), float(gaps)


def f_supp(g: pd.DataFrame) -> float:
    return float((g["app_type"].eq("3") | g["suffix"].str.startswith("S")).sum())


def f_nce(g: pd.DataFrame, tol_days: int = config.NCE_EXTENSION_TOLERANCE_DAYS) -> float:
    pe = pd.to_datetime(g["project_end"])
    orig = pd.to_datetime(g["seg_orig_end"])
    ext = ~g["app_type"].isin(COMPETING_TYPES) & (pe > orig + pd.Timedelta(days=tol_days))
    zero = (g["support_year"] > g["seg_term_years"]) & g["cost"].eq(0)
    return float((ext | zero).any())


def f_indirect_ratio(g: pd.DataFrame, min_fy: int = config.F_INDIRECT_MIN_FY) -> float:
    k = g[(g["FY_FILE"] >= min_fy) & g["indirect"].notna() & g["direct"].notna()]
    k = k.assign(budget=k["direct"] + k["indirect"])
    k = k[k["budget"] > 0]
    if k.empty:
        return np.nan
    by = k.groupby("FY_FILE")[["indirect", "budget"]].sum()
    return float((by["indirect"] / by["budget"]).mean())


def f_per_enrollee(total_dollars: float, planned_enrollment: float) -> float:
    if pd.isna(total_dollars) or pd.isna(planned_enrollment) or planned_enrollment <= 0:
        return np.nan
    return float(np.log1p(total_dollars / planned_enrollment))


def f_noncompeting_share(g: pd.DataFrame) -> float:
    return float(g["app_type"].eq("5").mean()) if len(g) else np.nan


# --- per-trial assembly -------------------------------------------------------------
def trial_rows(pairs: pd.DataFrame, rows: pd.DataFrame, trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overlapping rows, overlapping rows after the leakage cut), joined to trial fields.

    pairs: nct_id, core_project_num (attributable projects only)
    trials: nct_id, trial_start, trial_end, outcome_date
    """
    j = pairs[["nct_id", "core_project_num"]].drop_duplicates().merge(rows, on="core_project_num", how="inner")
    j = j.merge(trials[["nct_id", "trial_start", "trial_end", "outcome_date"]], on="nct_id", how="inner")
    ov = j[rules.fy_overlaps_vec(j)]
    return ov, leakage.cut_rows(ov)


def first_two_fys(cut: pd.DataFrame) -> pd.DataFrame:
    """F₂ rows: those of FY f0 and f0 + 1, f0 = each trial's earliest FY_FILE after the cut."""
    if cut.empty:
        return cut
    f0 = cut.groupby("nct_id")["FY_FILE"].transform("min")
    return cut[cut["FY_FILE"] <= f0 + 1]


def _features_one(g: pd.DataFrame, window_fys: list[int], planned_enrollment: float, variant: str) -> dict:
    a = annual_series(g)
    total_dollars = float(a.sum()) if len(a) else np.nan
    gap, gap_n = f_gap(g, window_fys)
    out = {
        "F_total": f_total(g), "F_years": f_years(g), "F_shape": f_shape(g), "F_yoy_min": f_yoy_min(g),
        "F_gap": gap, "F_gap_count": gap_n, "F_supp": f_supp(g),
        "F_indirect_ratio": f_indirect_ratio(g), "F_per_enrollee": f_per_enrollee(total_dollars, planned_enrollment),
        "F_noncompeting_share": f_noncompeting_share(g),
        "f_total_dollars_2020": total_dollars,
    }
    if variant == "F":
        out["F_nce"] = f_nce(g)
    return out


def compute_features(pairs: pd.DataFrame, rows: pd.DataFrame, trials: pd.DataFrame,
                     planned_enrollment: pd.Series | None = None, variant: str = "F") -> pd.DataFrame:
    """One row per trial in `trials` (all assumed attributable), raw values (NaN = missing).

    planned_enrollment: Series indexed by nct_id (version-0 rule applied), or None.
    Extra columns: f_rows_reason ('ok' | 'no_overlap' | 'all_cut' | 'cost_unknown'),
    f_n_rows, f_n_rows_overlap, f_n_rows_cost_null, f_total_dollars_2020, f_gap_window_n.
    """
    if variant not in ("F", "F2"):
        raise ValueError(variant)
    cols = F_COLUMNS if variant == "F" else F2_COLUMNS
    ov, cut = trial_rows(pairs, rows, trials)
    use = first_two_fys(cut) if variant == "F2" else cut
    by_trial = dict(tuple(use.groupby("nct_id")))
    n_ov = ov.groupby("nct_id").size()
    pe = planned_enrollment if planned_enrollment is not None else pd.Series(dtype=float)
    recs = []
    for t in trials.itertuples(index=False):
        g = by_trial.get(t.nct_id)
        cut_date = min(d for d in (t.outcome_date, t.trial_end) if pd.notna(d)) if (pd.notna(t.outcome_date) or pd.notna(t.trial_end)) else pd.NaT
        rec = {"nct_id": t.nct_id, "f_n_rows_overlap": int(n_ov.get(t.nct_id, 0))}
        if g is None or g.empty:
            rec.update({c: np.nan for c in cols})
            rec.update(f_n_rows=0, f_n_rows_cost_null=0, f_total_dollars_2020=np.nan, f_gap_window_n=0,
                       f_rows_reason="no_overlap" if rec["f_n_rows_overlap"] == 0 else "all_cut")
            recs.append(rec)
            continue
        full_window = leakage.gap_window_fys(t.trial_start, cut_date)
        if variant == "F2":
            f0 = int(g["FY_FILE"].min())
            hi = (max(full_window) if full_window else -1)
            window = [f for f in (f0, f0 + 1) if f <= hi]
        else:
            window = full_window
        rec.update(_features_one(g, window, pe.get(t.nct_id, np.nan), variant))
        rec.update(f_n_rows=int(len(g)), f_n_rows_cost_null=int(g["cost"].isna().sum()), f_gap_window_n=len(window),
                   f_rows_reason="ok" if g["cost"].notna().any() else "cost_unknown")
        recs.append(rec)
    out = pd.DataFrame(recs)
    return out[["nct_id", *cols, "f_rows_reason", "f_n_rows", "f_n_rows_overlap", "f_n_rows_cost_null",
                "f_total_dollars_2020", "f_gap_window_n"]]
