"""Discrete-time hazard data for O5 (plan §6.3; A5 (a)).

Person-period rows by NIH fiscal year. A trial is at risk from FY(start) to FY(exit), exit =
the outcome date (termination for O1 events, primary completion for Completed trials — the
cause-specific hazard of termination treats completion as leaving the risk set) or the
snapshot for censored trials. y = 1 only on the last period of an O1 event.

Time-varying F: the row for FY f sees only parent FY rows whose BUDGET_END precedes the start
of FY f (1 Oct of calendar year f − 1) — "only years observed so far" — and never the outcome
date (cut = min(start of FY f, outcome date)), computed by fb.features_f.compute_features
unchanged. P is fixed per trial. Unknown-status trials are excluded (as in O1).

For evaluation at a horizon h months, a trial's periods are extended (at_risk = False) up to
the FY containing start + h; information stays frozen at the outcome date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features_f, leakage, tables


def fy_start(fy: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(fy) - 1, month=10, day=1)


def person_periods(trials: pd.DataFrame, horizon_months: int | None = None) -> pd.DataFrame:
    """trials: nct_id, start_date, exit_date, event (0/1).
    Returns nct_id, key ('NCT|FY'), fy, period (0-based), at_risk, y, cut_date."""
    recs = []
    for t in trials.itertuples(index=False):
        f0 = int(leakage.nih_fy([t.start_date]).iloc[0])
        fx = max(f0, int(leakage.nih_fy([t.exit_date]).iloc[0]))
        last = fx
        if horizon_months is not None:
            h_end = pd.Timestamp(t.start_date) + pd.DateOffset(months=horizon_months) - pd.Timedelta(days=1)
            last = max(fx, int(leakage.nih_fy([h_end]).iloc[0]))
        for f in range(f0, last + 1):
            at_risk = f <= fx
            recs.append({"nct_id": t.nct_id, "key": f"{t.nct_id}|{f}", "fy": f, "period": f - f0, "at_risk": at_risk,
                         "y": int(at_risk and f == fx and int(t.event) == 1),
                         "cut_date": min(fy_start(f), pd.Timestamp(t.exit_date))})
    return pd.DataFrame(recs)


def hazard_trials(lab: pd.DataFrame, ids) -> pd.DataFrame:
    """Hazard population from fb.tables.labels: event / control / censored among `ids`."""
    t = lab[lab["nct_id"].isin(set(ids)) & lab["status_group"].isin(["event", "control", "censored"])]
    return pd.DataFrame({"nct_id": t["nct_id"].values, "start_date": pd.to_datetime(t["start_date"]).values,
                         "exit_date": pd.to_datetime(t["outcome_date"]).values,
                         "event": t["status_group"].eq("event").astype(int).values,
                         "months_to_exit": t["months_to_exit"].values})


def pp_features(inp: tables.Inputs, pp: pd.DataFrame, rule: str = "n20") -> pd.DataFrame:
    """Time-varying F per person-period key (raw, NaN = not observable yet)."""
    ids = pp["nct_id"].unique()
    pairs = tables.pairs_for(inp, ids, rule).merge(pp[["nct_id", "key"]], on="nct_id")
    pairs = pairs[["key", "core_project_num"]].rename(columns={"key": "nct_id"})
    tr = inp.cohort.set_index("nct_id").loc[pp["nct_id"], ["start_date", "trial_end"]]
    trials = pd.DataFrame({"nct_id": pp["key"].values, "trial_start": pd.to_datetime(tr["start_date"].values),
                           "trial_end": pd.to_datetime(tr["trial_end"].values),
                           "outcome_date": pd.to_datetime(pp["cut_date"].values)})
    pe = inp.pv.set_index("nct_id")["planned_enrollment"]
    pe_k = pd.Series(pp["nct_id"].map(pe).values, index=pp["key"].values)
    return features_f.compute_features(pairs, inp.rows, trials, pe_k, "F")


def period_dummies(period) -> pd.DataFrame:
    p = np.minimum(np.asarray(period, dtype=int), config.HAZARD_MAX_PERIOD)
    labels = [f"period={k}" for k in range(config.HAZARD_MAX_PERIOD)] + [f"period={config.HAZARD_MAX_PERIOD}+"]
    return pd.DataFrame({lab: (p == k).astype(float) for k, lab in enumerate(labels)})


def cumulative_risk(pp: pd.DataFrame, hazard: np.ndarray, start_dates: pd.Series, horizon_months: int) -> pd.Series:
    """1 − Π(1 − h) over each trial's periods whose FY starts before start + horizon."""
    d = pp[["nct_id", "fy"]].copy()
    d["h"] = hazard
    st = d["nct_id"].map(start_dates)
    h_end = st + pd.DateOffset(months=horizon_months)
    keep = pd.to_datetime(d["fy"].map(lambda f: fy_start(f))) < h_end
    # the first period always counts (its FY starts on or before the trial start)
    keep |= d.groupby("nct_id")["fy"].transform("min").eq(d["fy"])
    d = d[keep]
    surv = d.assign(s=np.log1p(-np.clip(d["h"], 0, 1 - 1e-12))).groupby("nct_id")["s"].sum()
    return 1 - np.exp(surv)


def build(window: str, inp: tables.Inputs | None = None, write: bool = True) -> pd.DataFrame:
    """Person-period table for the attributable (N = 20) trials of a window, with raw F.
    Training: at-risk periods only. Test: extended to the largest horizon for evaluation."""
    inp = inp or tables.load_inputs()
    trials = tables.attributable(tables.window_trials(inp, window), "n20")
    lab = tables.labels(inp, trials)
    ht = hazard_trials(lab, trials["nct_id"])
    pp = person_periods(ht, None if window == "train" else max(config.HAZARD_HORIZONS_MONTHS))
    f = pp_features(inp, pp)
    out = pp.merge(f.rename(columns={"nct_id": "key"}), on="key", how="left")
    if write:
        config.PHASE4_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(tables.path(window, "hazard_pp"), index=False)
        ht.to_parquet(tables.path(window, "hazard_trials"), index=False)
    return out


if __name__ == "__main__":
    inp = tables.load_inputs()
    for w in ("train", "test"):
        o = build(w, inp)
        print(w, o.shape, "at-risk rows", int(o["at_risk"].sum()), "events", int(o["y"].sum()),
              "trials", o["nct_id"].nunique(), "F_total observable", int(o["F_total"].notna().sum()))
