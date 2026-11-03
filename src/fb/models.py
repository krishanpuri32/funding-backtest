"""Phase 4 step 4: fit the pre-registered specifications on the TRAINING window and freeze them.

    python -m fb.models

Reads only data/derived/phase4/train_*.parquet. Writes one joblib bundle per
outcome × population × specification × feature set under data/derived/models/ (model, column
order, training levels and medians) and models/hyperparams.json (C chosen by CV, boosting
iterations, training LR tests). Nothing here reads a test-window row.

Populations (A1–A4):
  P_only       all trials (A3), P only
  P_only_A2    all trials minus Withdrawn / enrollment-0 (A2), P only — the A2 run's P-only AUC
  primary      A1 N = 20 attributable trials with >= 1 observable FY row (A4b landmark); no F indicators
  S1           all N = 20 attributable trials, with F_*_missing indicators (A4b sensitivity)
  A2           primary minus Withdrawn / enrollment-0 (A2)
  N50          landmark subset under the N = 50 rule (A1 grid ii)
  activity     landmark subset under the P30 / U10 rule (A1 grid iii)
  no_umbrella  landmark subset with every linked project's rows (A1 grid i)
Binary models use status ∈ {event, control}; the hazard model adds censored trials.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, design, features_p, hazard, missing, tables
from .metrics import log_loss


@dataclass(frozen=True)
class Population:
    rule: str | None          # attribution rule; None = every trial, P only
    landmark: bool = False
    a2: bool = False
    f_missing: bool = False


POPULATIONS = {
    "P_only": Population(None),
    "P_only_A2": Population(None, a2=True),
    "primary": Population("n20", landmark=True),
    "S1": Population("n20", f_missing=True),
    "A2": Population("n20", landmark=True, a2=True),
    "N50": Population("n50", landmark=True),
    "activity": Population("activity", landmark=True),
    "no_umbrella": Population("none", landmark=True),
}
FS_ALL = ("P", "P+F", "P+F2")
RUNS = [  # (outcome, population, feature sets)
    ("O1", "P_only", ("P",)), ("O1", "P_only_A2", ("P",)),
    *[("O1", p, FS_ALL) for p in ("primary", "S1", "A2", "N50", "activity", "no_umbrella")],
    ("O2", "P_only", ("P",)), ("O2", "primary", FS_ALL),
    ("O3", "primary", ("P", "P+F")), ("O4", "primary", ("P", "P+F")),
]
BINARY = {"O1", "O2"}
SPECS = {"binary": ("logistic", "hgb"), "continuous": ("ridge", "hgb")}
VARIANT = {"P": None, "P+F": "F", "P+F2": "F2"}


# --- population assembly -----------------------------------------------------------------
def members(W: dict, pop: Population) -> list[str]:
    lab = W["labels"]
    if pop.rule is None:
        ids = lab["nct_id"]
    else:
        f = W[f"F_{pop.rule}"]
        if pop.landmark:
            f = f[f["f_rows_reason"].isin(config.LANDMARK_REASONS)]
        ids = f["nct_id"]
    keep = set(ids)
    if pop.a2:
        keep -= set(lab.loc[lab["withdrawn_zero_enroll"].astype(bool), "nct_id"])
    return [i for i in lab["nct_id"] if i in keep]


def training_medians(Wtr: dict, pop: Population) -> dict:
    """P / F / F₂ medians from the TRAINING window. N = 20 populations reuse the Phase 3 medians
    (features_stats.json); the grid rules compute theirs on their own attributable training set."""
    ph3 = tables.phase3_medians()
    if pop.rule is None:
        return {"P": ph3["P_all"], "F": None, "F2": None}
    if pop.rule == "n20":
        return {"P": ph3["P_attr"], "F": ph3["F"], "F2": ph3["F2"]}
    ids = set(Wtr[f"F_{pop.rule}"]["nct_id"])
    _, mp = missing.fill_numeric(Wtr["P"][Wtr["P"]["nct_id"].isin(ids)], features_p.P_NUMERIC)
    _, mf = missing.fill_numeric(Wtr[f"F_{pop.rule}"], design.f_columns("F"))
    _, mf2 = missing.fill_numeric(Wtr[f"F2_{pop.rule}"], design.f_columns("F2"))
    return {"P": mp, "F": mf, "F2": mf2}


def outcome_rows(lab: pd.DataFrame, ids: list[str], outcome: str) -> pd.DataFrame:
    col = {"O1": "o1", "O2": "o2", "O3": "o3", "O4": "o4"}[outcome]
    l = lab.set_index("nct_id").loc[ids].reset_index()
    l = l[l[col].notna()].reset_index(drop=True)
    l["y"] = l[col].astype(float)
    return l


def assemble(W: dict, outcome: str, pop_name: str, fs: str, med: dict, levels: dict,
             drop_p_missing: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows with y and strata columns, X) for one window / outcome / population / feature set."""
    pop = POPULATIONS[pop_name]
    rows = outcome_rows(W["labels"], members(W, pop), outcome)
    v = VARIANT[fs]
    X = design.design_matrix(rows["nct_id"], W["P"], med["P"], levels,
                             f_raw=W[f"{v}_{pop.rule}"] if v else None, f_medians=med[v] if v else None,
                             variant=v, f_missing=pop.f_missing, drop_p_missing=drop_p_missing)
    return rows, X


# --- specifications ---------------------------------------------------------------------
def fit_logistic(X: pd.DataFrame, y: np.ndarray, groups=None) -> tuple[Pipeline, dict]:
    pipe = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))])
    cv = (GroupKFold(n_splits=config.CV_FOLDS) if groups is not None
          else StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED))
    gs = GridSearchCV(pipe, {"lr__C": config.LOGIT_C_GRID}, scoring=config.CV_SCORING, cv=cv, refit=True)
    gs.fit(X, y, **({"groups": groups} if groups is not None else {}))
    return gs.best_estimator_, {"C": float(gs.best_params_["lr__C"]), "cv_neg_log_loss": float(gs.best_score_),
                                "cv_by_C": {f"{c:g}": float(s) for c, s in zip(config.LOGIT_C_GRID, gs.cv_results_["mean_test_score"])}}


def fit_hgb(X: pd.DataFrame, y: np.ndarray, regression: bool = False):
    cls = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
    params = dict(config.HGB_PARAMS)
    m = cls(**params, random_state=config.SEED).fit(X, y)
    return m, {"n_iter": int(m.n_iter_), **{k: v for k, v in params.items()}}


def fit_ridge(X: pd.DataFrame, y: np.ndarray):
    alphas = np.logspace(-2, 6, 17)  # widened from 1e-3..1e3 after the first training fit hit 1e3 at the edge
    pipe = Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=alphas, cv=KFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)))])
    pipe.fit(X, y)
    return pipe, {"alpha": float(pipe.named_steps["ridge"].alpha_)}


def lr_test(X0: pd.DataFrame, X1: pd.DataFrame, y: np.ndarray) -> dict:
    """Likelihood-ratio test of the nested logistic pair, unpenalised MLE on the training window."""
    def ll(X):
        m = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=np.inf, max_iter=20000))]).fit(X, y)
        return -log_loss(y, m.predict_proba(X)[:, 1]) * len(y)
    l0, l1 = ll(X0), ll(X1)
    df = X1.shape[1] - X0.shape[1]
    stat = max(0.0, 2 * (l1 - l0))
    return {"ll_P": l0, "ll_PF": l1, "chi2": stat, "df": int(df), "p": float(chi2.sf(stat, df))}


def predict(model, X: pd.DataFrame, binary: bool) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if binary else model.predict(X)


def bundle_path(outcome: str, pop: str, spec: str, fs: str, base=None):
    return (base or config.MODELS_DIR) / f"{outcome}__{pop}__{spec}__{fs.replace('+', '')}.joblib"


# --- Phase 4-X exploratory run "px": P-only without the six P _missing indicators ------
PX_POPULATIONS = ("P_only", "P_only_A2", "primary")
PX_MODELS_DIR = config.EXPLORATORY_DIR / "models"


def fit_exploratory_px() -> dict:
    """Training only. Same C grid / CV / seed / HGB settings, same training medians and levels
    as the frozen Phase 4 P models; only the P indicators are dropped. Writes under
    data/derived/exploratory/models/ and never under data/derived/models/."""
    PX_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Wtr = tables.load_window("train")
    hyper: dict = {}
    for pop_name in PX_POPULATIONS:
        pop = POPULATIONS[pop_name]
        med = training_medians(Wtr, pop)
        levels = design.p_levels(Wtr["P"][Wtr["P"]["nct_id"].isin(set(members(Wtr, pop)))])
        rows, X = assemble(Wtr, "O1", pop_name, "P", med, levels, drop_p_missing=True)
        y = rows["y"].values.astype(int)
        for spec in SPECS["binary"]:
            m, h = fit_logistic(X, y) if spec == "logistic" else fit_hgb(X, y)
            h.update(n=int(len(y)), n_events=int(y.sum()), n_cols=int(X.shape[1]))
            hyper[f"O1|{pop_name}|{spec}|P|px"] = h
            joblib.dump({"model": m, "columns": list(X.columns), "levels": levels, "medians": med, "drop_p_missing": True},
                        bundle_path("O1", pop_name, spec, "P", PX_MODELS_DIR))
            print(pop_name, spec, {k: v for k, v in h.items() if k != "cv_by_C"}, flush=True)
    (PX_MODELS_DIR / "hyperparams_px.json").write_text(json.dumps(hyper, indent=2, default=str) + "\n")
    return hyper


# --- hazard (O5) ------------------------------------------------------------------------
def hazard_design(pp: pd.DataFrame, P: pd.DataFrame, med: dict, levels: dict, with_f: bool) -> pd.DataFrame:
    X = design.design_matrix(pp["nct_id"], P, med["P"], levels, f_keys=pp["key"],
                             f_raw=pp.rename(columns={"nct_id": "_trial", "key": "nct_id"}) if with_f else None,
                             f_medians=med["F"] if with_f else None, variant="F" if with_f else None, f_missing=True)
    return pd.concat([X.reset_index(drop=True), hazard.period_dummies(pp["period"]).reset_index(drop=True)], axis=1)


def fit_hazard(Wtr: dict, hyper: dict) -> None:
    pp = pd.read_parquet(tables.path("train", "hazard_pp"))
    pp = pp[pp["at_risk"]].reset_index(drop=True)
    med = training_medians(Wtr, POPULATIONS["S1"])
    ht = pd.read_parquet(tables.path("train", "hazard_trials"))
    levels = design.p_levels(Wtr["P"][Wtr["P"]["nct_id"].isin(set(ht["nct_id"]))])
    y = pp["y"].values.astype(int)
    for fs, with_f in (("P", False), ("P+F", True)):
        X = hazard_design(pp, Wtr["P"], med, levels, with_f)
        m, h = fit_logistic(X, y, groups=pp["nct_id"].values)
        h.update(n_rows=int(len(pp)), n_trials=int(pp["nct_id"].nunique()), n_events=int(y.sum()), n_cols=int(X.shape[1]))
        hyper[f"O5|hazard|logistic|{fs}"] = h
        joblib.dump({"model": m, "columns": list(X.columns), "levels": levels, "medians": med, "with_f": with_f},
                    bundle_path("O5", "hazard", "logistic", fs))


# --- main ---------------------------------------------------------------------------------
def gate_allows(outcome: str) -> bool:
    cell = json.loads(config.GATE_LANDMARK_PATH.read_text())["primary_cell"]
    if cell.startswith("stop"):
        return False
    return outcome == "O1" or cell.startswith("full")


def main() -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Wtr = tables.load_window("train")
    hyper: dict = {}
    for outcome, pop_name, fsets in RUNS:
        if not gate_allows(outcome):
            hyper[f"{outcome}|{pop_name}"] = "not modelled: §11 cell"
            continue
        pop = POPULATIONS[pop_name]
        med = training_medians(Wtr, pop)
        ids = members(Wtr, pop)
        levels = design.p_levels(Wtr["P"][Wtr["P"]["nct_id"].isin(set(ids))])
        binary = outcome in BINARY
        Xs = {}
        for fs in fsets:
            rows, X = assemble(Wtr, outcome, pop_name, fs, med, levels)
            Xs[fs] = X
            y = rows["y"].values
            for spec in SPECS["binary" if binary else "continuous"]:
                if spec == "logistic":
                    m, h = fit_logistic(X, y.astype(int))
                elif spec == "ridge":
                    m, h = fit_ridge(X, y)
                else:
                    m, h = fit_hgb(X, y.astype(int) if binary else y, regression=not binary)
                h.update(n=int(len(y)), n_events=int(y.sum()) if binary else None, n_cols=int(X.shape[1]))
                hyper[f"{outcome}|{pop_name}|{spec}|{fs}"] = h
                joblib.dump({"model": m, "columns": list(X.columns), "levels": levels, "medians": med},
                            bundle_path(outcome, pop_name, spec, fs))
                print(outcome, pop_name, spec, fs, {k: v for k, v in h.items() if k != "cv_by_C"}, flush=True)
        if binary and "P+F" in Xs:
            yb = rows["y"].values.astype(int)
            for fs in ("P+F", "P+F2"):
                hyper[f"{outcome}|{pop_name}|lr_test|{fs}"] = lr_test(Xs["P"], Xs[fs], yb)
                print(outcome, pop_name, "LR test", fs, hyper[f"{outcome}|{pop_name}|lr_test|{fs}"], flush=True)
    if gate_allows("O5"):
        fit_hazard(Wtr, hyper)
        print("O5 hazard", {k: {kk: vv for kk, vv in v.items() if kk != "cv_by_C"} for k, v in hyper.items() if k.startswith("O5")})
    (config.MODELS_DIR / "hyperparams.json").write_text(json.dumps(hyper, indent=2, default=str) + "\n")
    return hyper


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--exploratory", "px"]:
        fit_exploratory_px()
    elif sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}; use --exploratory px or nothing")
    else:
        main()
