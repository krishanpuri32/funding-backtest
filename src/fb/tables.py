"""Phase 4 tables (plan §5–§6; A4): raw P / F / F₂ tables and outcome labels for
either window and any attribution rule, built by calling the Phase 3 functions unchanged.

    python -m fb.tables          # step 2: test-window feature tables + coverage (no outcome join)

Raw tables are cached under data/derived/phase4/ ({window}_P.parquet, {window}_F_{rule}.parquet,
{window}_F2_{rule}.parquet, {window}_labels.parquet). Filling (indicator + median) happens later
with TRAINING medians only (fb.design.Medians); test medians are never computed.

Attribution rules: 'n20' (A1 primary), 'n50' and 'activity' (A1 grid ii, iii) drop umbrella
projects' rows; 'none' (A1 grid i) keeps every linked project.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import aact, config, features_f, features_p, leakage, missing, o2, planned
from .cpi import CPI_PATH, load_cpi
from .features import COHORT_PATH, COHORT_PROJECTS_PATH, PROJECT_COLUMNS
from .link import PROJECTS_PATH

RULES = ("n20", "n50", "activity", "none")
STATUS_KEEP = ("event", "control", "censored", "unknown")


def umbrella_col(rule: str) -> str | None:
    return None if rule == "none" else f"is_umbrella_{rule}"


def attributable_col(rule: str) -> str | None:
    return None if rule == "none" else f"has_attributable_funding_{rule}"


@dataclass
class Inputs:
    cohort: pd.DataFrame
    cp: pd.DataFrame
    pv: pd.DataFrame          # fb.planned.planned_values for every cohort trial
    v0_pc: pd.Series          # v0 primary completion date (parsed), by nct_id
    studies: pd.DataFrame
    od: pd.DataFrame          # fb.leakage.outcome_dates for every cohort trial
    designs: pd.DataFrame
    interventions: pd.DataFrame
    facilities: pd.DataFrame
    countries: pd.DataFrame
    rows: pd.DataFrame        # fb.features_f.prepare_rows over every cohort project


def _aact(table: str, cols: list[str], ids: set[str]) -> pd.DataFrame:
    d = aact.load_table(table, usecols=cols)
    d["nct_id"] = d["nct_id"].str.strip().str.upper()
    return d[d["nct_id"].isin(ids)]


def load_inputs() -> Inputs:
    cohort = pd.read_parquet(COHORT_PATH)
    cp = pd.read_parquet(COHORT_PROJECTS_PATH)
    ids = set(cohort["nct_id"])
    v0 = planned.load_v0()
    m = cohort[["nct_id", "window", "start_date"]].merge(v0, on="nct_id", how="left")
    pv = planned.planned_values(m)
    v0_pc = pd.Series(planned.parse_registry_date(m["v0_primary_completion"]).values, index=m["nct_id"].values)
    st = _aact("studies", ["nct_id", "primary_completion_date", "primary_completion_date_type", "completion_date",
                           "completion_date_type", "last_update_submitted_date", "why_stopped"], ids)
    od = leakage.outcome_dates(cohort[["nct_id", "status_group", "overall_status"]].merge(st, on="nct_id", how="left"))
    prj = pd.read_parquet(PROJECTS_PATH, columns=PROJECT_COLUMNS)
    prj = prj[prj["CORE_PROJECT_NUM"].isin(set(cp["core_project_num"]))]
    if not CPI_PATH.exists():
        raise SystemExit("CPI file missing: run `make download-cpi` first")
    rows = features_f.prepare_rows(prj, load_cpi())
    return Inputs(
        cohort=cohort, cp=cp, pv=pv, v0_pc=v0_pc, studies=st, od=od,
        designs=_aact("designs", ["nct_id", "allocation", "masking"], ids),
        interventions=_aact("interventions", ["nct_id", "intervention_type"], ids),
        facilities=_aact("facilities", ["nct_id"], ids),
        countries=_aact("countries", ["nct_id", "name", "removed"], ids),
        rows=rows,
    )


# --- populations ---------------------------------------------------------------------
def window_trials(inp: Inputs, window: str) -> pd.DataFrame:
    return inp.cohort[inp.cohort["window"] == window].reset_index(drop=True)


def attributable(trials: pd.DataFrame, rule: str) -> pd.DataFrame:
    col = attributable_col(rule)
    if col is None:
        return trials.reset_index(drop=True)
    return trials[trials[col].fillna(False).astype(bool)].reset_index(drop=True)


def pairs_for(inp: Inputs, ids, rule: str) -> pd.DataFrame:
    cp = inp.cp[inp.cp["nct_id"].isin(set(ids))]
    col = umbrella_col(rule)
    if col is not None:
        cp = cp[~cp[col].fillna(False).astype(bool)]
    return cp[["nct_id", "core_project_num"]].drop_duplicates()


def trials_frame(inp: Inputs, trials: pd.DataFrame) -> pd.DataFrame:
    t = trials[["nct_id", "start_date", "trial_end"]].rename(columns={"start_date": "trial_start"}).merge(
        inp.od[["nct_id", "outcome_date"]], on="nct_id", how="left")
    t["trial_start"] = pd.to_datetime(t["trial_start"])
    t["trial_end"] = pd.to_datetime(t["trial_end"])
    return t


def observable_reason(inp: Inputs, trials: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Step 0 (A4b landmark): per trial, why F rows are / are not observable — computed from
    the overlap + cut machinery only, no feature value. Same labels as fb.features_f
    (`ok` | `cost_unknown` | `all_cut` | `no_overlap`)."""
    tf = trials_frame(inp, trials)
    ov, cut = features_f.trial_rows(pairs_for(inp, trials["nct_id"], rule), inp.rows, tf)
    n_ov = ov.groupby("nct_id").size()
    n_cut = cut.groupby("nct_id").size()
    any_cost = cut.groupby("nct_id")["cost"].apply(lambda s: bool(s.notna().any()))
    ids = tf["nct_id"]
    reason = np.where(ids.map(n_cut).fillna(0) > 0,
                      np.where(ids.map(any_cost).fillna(False).astype(bool), "ok", "cost_unknown"),
                      np.where(ids.map(n_ov).fillna(0) > 0, "all_cut", "no_overlap"))
    return pd.DataFrame({"nct_id": ids.values, "f_rows_reason": reason})


# --- raw tables ------------------------------------------------------------------------
def p_raw(inp: Inputs, trials: pd.DataFrame) -> pd.DataFrame:
    pv = inp.pv[inp.pv["nct_id"].isin(set(trials["nct_id"]))]
    return features_p.build_p_raw(trials, inp.designs, inp.interventions, inp.facilities, inp.countries, pv)


def f_raw(inp: Inputs, trials: pd.DataFrame, rule: str, variant: str) -> pd.DataFrame:
    pe = inp.pv.set_index("nct_id")["planned_enrollment"]
    return features_f.compute_features(pairs_for(inp, trials["nct_id"], rule), inp.rows, trials_frame(inp, trials),
                                       pe, variant)


def labels(inp: Inputs, trials: pd.DataFrame) -> pd.DataFrame:
    """Outcome labels O1–O5 for the given trials (plan §4; A2, A5).

    O1: 1 event / 0 control / NA otherwise. O2: 1 when O1 = 1 and fb.o2's rule fires on
    why_stopped; 0 for every other O1-eligible trial (controls, events whose reason does not
    match or is blank). O3: log(1 + actual enrollment) − log(1 + planned v0 enrollment), only
    for O1-eligible trials eligible for O3/O4 (not late, v0 not ACTUAL) with an ACTUAL current
    enrollment and planned enrollment > 0. O4: actual primary completion − planned v0
    primary completion, months, same eligibility, current primary completion typed ACTUAL.
    O5 inputs: outcome_date (exit) and months from start to exit.
    """
    t = trials[["nct_id", "window", "status_group", "overall_status", "withdrawn_zero_enroll", "start_date",
                "enrollment", "enrollment_type", "phase_group", "condition_group", "start_year"]].copy()
    t = t.merge(inp.od[["nct_id", "outcome_date", "outcome_date_source"]], on="nct_id", how="left")
    st = inp.studies.drop_duplicates("nct_id").set_index("nct_id")
    ws = pd.Series(t["nct_id"].map(st["why_stopped"]).values, index=t.index)
    t["o1"] = pd.Series(np.where(t["status_group"].eq("event"), 1, np.where(t["status_group"].eq("control"), 0, -1)),
                        index=t.index).replace(-1, pd.NA).astype("Int64")
    rl = o2.rule_label(ws)
    t["o2_rule"] = rl
    t["o2"] = t["o1"].where(t["o1"].isna(), ((t["o1"] == 1) & rl.fillna(0).eq(1)).astype(int)).astype("Int64")
    pv = inp.pv.set_index("nct_id")
    elig = t["nct_id"].map(pv["o34_eligible"]).fillna(False).astype(bool) & t["o1"].notna()
    pe = t["nct_id"].map(pv["planned_enrollment"]).astype(float)
    actual_enr = t["enrollment"].where(t["enrollment_type"].fillna("").str.upper().eq("ACTUAL"))
    ok3 = elig & pe.gt(0) & actual_enr.notna()
    t["o3"] = (np.log1p(actual_enr) - np.log1p(pe)).where(ok3)
    pcd = pd.to_datetime(t["nct_id"].map(st["primary_completion_date"]), errors="coerce")
    pcd_act = pcd.where(t["nct_id"].map(st["primary_completion_date_type"]).fillna("").str.upper().eq("ACTUAL"))
    v0pc = pd.to_datetime(t["nct_id"].map(inp.v0_pc), errors="coerce")
    ok4 = elig & pcd_act.notna() & v0pc.notna()
    t["o4"] = ((pcd_act - v0pc).dt.days / planned.DAYS_PER_MONTH).where(ok4)
    t["months_to_exit"] = (pd.to_datetime(t["outcome_date"]) - pd.to_datetime(t["start_date"])).dt.days / planned.DAYS_PER_MONTH
    return t


# --- cache -----------------------------------------------------------------------------
def path(window: str, name: str):
    return config.PHASE4_DIR / f"{window}_{name}.parquet"


def build_window(inp: Inputs, window: str, rules=RULES, write: bool = True) -> dict:
    trials = window_trials(inp, window)
    out = {"P": p_raw(inp, trials), "labels": labels(inp, trials)}
    for rule in rules:
        a = attributable(trials, rule)
        out[f"F_{rule}"] = f_raw(inp, a, rule, "F")
        out[f"F2_{rule}"] = f_raw(inp, a, rule, "F2")
    if write:
        config.PHASE4_DIR.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            v.to_parquet(path(window, k), index=False)
    return out


def load_window(window: str) -> dict:
    out = {"P": pd.read_parquet(path(window, "P")), "labels": pd.read_parquet(path(window, "labels"))}
    for rule in RULES:
        for v in ("F", "F2"):
            p = path(window, f"{v}_{rule}")
            if p.exists():
                out[f"{v}_{rule}"] = pd.read_parquet(p)
    return out


def phase3_medians() -> dict:
    s = json.loads(config.FEATURES_STATS_PATH.read_text())
    return {"P_all": s["p_medians_all"], "P_attr": s["p_medians_attributable"], "F": s["f_medians"], "F2": s["f2_medians"]}


def main() -> dict:
    """Step 2: raw tables for both windows (training rebuilt to check it reproduces Phase 3),
    test-window filled tables with training medians, test coverage. No outcome is joined."""
    inp = load_inputs()
    rep: dict = {}
    tr = build_window(inp, "train")
    te = build_window(inp, "test")
    med = phase3_medians()

    # reproduction check: training n20 tables equal the Phase 3 files
    ph3 = {"F": pd.read_parquet(config.FEATURES_F_PATH), "F2": pd.read_parquet(config.FEATURES_F2_PATH),
           "P_all": pd.read_parquet(config.FEATURES_P_ALL_PATH)}
    chk = {}
    for v, cols in (("F", features_f.F_COLUMNS), ("F2", features_f.F2_COLUMNS)):
        new, _ = missing.fill_numeric(tr[f"{v}_n20"], cols, med[v])
        a = new.set_index("nct_id").sort_index()
        b = ph3[v].set_index("nct_id").sort_index()
        chk[v] = bool(a.index.equals(b.index) and np.allclose(a[cols].values, b[cols].values, equal_nan=True))
    p_new, _ = features_p.fill_p(tr["P"], med["P_all"])
    a = p_new.set_index("nct_id").sort_index()
    b = ph3["P_all"].set_index("nct_id").sort_index()
    chk["P_all"] = bool(a.index.equals(b.index) and np.allclose(a[features_p.P_NUMERIC].values, b[features_p.P_NUMERIC].values)
                        and (a[features_p.P_CATEGORICAL] == b[features_p.P_CATEGORICAL]).all().all())
    rep["train_reproduces_phase3"] = chk

    # test-window filled tables (training medians), attributable = A1 N = 20
    test = window_trials(inp, "test")
    attr_ids = set(attributable(test, "n20")["nct_id"])
    p_all, _ = features_p.fill_p(te["P"], med["P_all"])
    p_raw_attr = te["P"][te["P"]["nct_id"].isin(attr_ids)].reset_index(drop=True)
    p_attr, _ = features_p.fill_p(p_raw_attr, med["P_attr"])
    f_fill, _ = missing.fill_numeric(te["F_n20"], features_f.F_COLUMNS, med["F"])
    f2_fill, _ = missing.fill_numeric(te["F2_n20"], features_f.F2_COLUMNS, med["F2"])
    for name, df in (("P_all", p_all), ("P", p_attr), ("F", f_fill), ("F2", f2_fill)):
        df.to_parquet(config.DATA_DERIVED / f"features_{name}_test.parquet", index=False)
    rep["shapes_test"] = {"features_P_all_test": list(p_all.shape), "features_P_test": list(p_attr.shape),
                          "features_F_test": list(f_fill.shape), "features_F2_test": list(f2_fill.shape)}
    rep["f_rows_reason_test"] = {v: {str(k): int(n) for k, n in te[f"{v}_n20"]["f_rows_reason"].value_counts().items()}
                                 for v in ("F", "F2")}
    rep["f_rows_reason_by_rule"] = {w: {r: {str(k): int(n) for k, n in d[f"F_{r}"]["f_rows_reason"].value_counts().items()}
                                        for r in RULES} for w, d in (("train", tr), ("test", te))}
    rep["test_trials_by_rule"] = {r: int(len(te[f"F_{r}"])) for r in RULES}
    rep["test_rows_used"] = {"F": int(te["F_n20"]["f_n_rows"].sum()), "F2": int(te["F2_n20"]["f_n_rows"].sum())}
    rep["test_planned_enrollment_max"] = float(np.expm1(te["P"]["log_planned_enrollment"].max()))

    cov = [f"# Phase 4 feature coverage — TEST window (start 2016–2020), raw values before filling", "",
           "Filled tables use training medians (features_stats.json); no test median is computed.", ""]
    for title, raw, num, cat in (
            (f"P — all test trials (features_P_all_test.parquet, n = {len(te['P'])})", te["P"], features_p.P_NUMERIC, features_p.P_CATEGORICAL),
            (f"P — attributable test trials (features_P_test.parquet, n = {len(p_raw_attr)})", p_raw_attr, features_p.P_NUMERIC, features_p.P_CATEGORICAL),
            (f"F — attributable test trials (features_F_test.parquet, n = {len(te['F_n20'])})", te["F_n20"], features_f.F_COLUMNS, []),
            (f"F₂ — attributable test trials (features_F2_test.parquet, n = {len(te['F2_n20'])})", te["F2_n20"], features_f.F2_COLUMNS, [])):
        cov += [f"## {title}", "", missing.to_markdown(missing.coverage_table(raw, num, cat)), ""]
    config.FEATURES_COVERAGE_TEST_PATH.write_text("\n".join(cov))
    (config.PHASE4_DIR / "tables_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))
    print("\n".join(cov))
    return rep


if __name__ == "__main__":
    main()
