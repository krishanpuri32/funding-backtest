"""Single home for paths, pinned snapshot identifiers, windows and seeds."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_DERIVED = REPO_ROOT / "data" / "derived"
MANIFEST_PATH = DATA_RAW / "manifest.json"

# --- seeds: the only place a seed is defined -------------------------------
SEED = 20261019

# --- pinned snapshots -------------------------------------------------------
AACT_SNAPSHOT_DATE = "20261001"  # monthly static flat-file export, YYYYMMDD
AACT_EXPORT_URL = (
    "https://aact.ctti-clinicaltrials.org/static/exported_files/monthly/"
    f"{AACT_SNAPSHOT_DATE}_export.zip"
)
AACT_TABLES = [
    "studies",
    "calculated_values",
    "browse_conditions",
    "designs",
    "interventions",
    "facilities",
    "sponsors",
    "countries",
]

EXPORTER_FY_FIRST = 2000
EXPORTER_FY_LAST = 2025  # latest file served on 2026-10-19 (FY2026 -> 404)
EXPORTER_PROJECTS_URL = "https://reporter.nih.gov/exporter/projects/download/{fy}"
EXPORTER_CLINICAL_STUDIES_URL = "https://reporter.nih.gov/exporter/clinicalstudies/download"

# --- cohort windows (plan §3, §7) -------------------------------------------
COHORT_START = "2008-01-01"
COHORT_END = "2020-12-31"
TRAIN_YEARS = (2008, 2015)
TEST_YEARS = (2016, 2020)

# --- outcome O1 status mapping (plan §4) ------------------------------------
O1_EVENT_STATUSES = {"TERMINATED", "SUSPENDED", "WITHDRAWN"}
O1_CONTROL_STATUSES = {"COMPLETED"}
CENSORED_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
}
UNKNOWN_STATUSES = {"UNKNOWN"}

# --- sample-size gate (plan §11) --------------------------------------------
GATE_FULL = 300
GATE_POOLED = 150

# --- Phase 2: umbrella rule (plan A1), A2, CPI, strata floors (plan §7) ---------
UMBRELLA_N_PRIMARY = 20            # A1 primary: > 20 NCT IDs linked -> umbrella
UMBRELLA_N_ALT = 50                # A1 grid (ii)
UMBRELLA_ACTIVITY_CODES = {"P30", "U10"}  # A1 grid (iii)
CPI_SERIES = "CPIAUCSL"            # FRED: CPI-U all items, US city average, annual average
CPI_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={CPI_SERIES}&fq=Annual&fam=avg"
CPI_BASE_YEAR = 2020
STRATUM_MIN_TRIALS = 100           # plan §7 stratum floor (trials)
STRATUM_MIN_EVENTS = 30            # plan §7 stratum floor (O1 events)
IC_TOP_N = 10
MESH_MAX_GROUPS = 15

# Frozen 2026-10-20 (Phase 2) on the training window, primary attributable set:
# the ten most frequent modal administering ICs. Reused as-is by later phases.
IC_TOP10 = ["CA", "MH", "DA", "HL", "DK", "HD", "AG", "AT", "AA", "NR"]

# --- cohort v0 history pull (pre-Phase 3; PLAN.md Note 2026-10-21) -------------------
# Access rationale: repo CLAUDE.md "Registry record history". robots.txt allows /api/int/,
# Crawl-delay 1 (fetched 2026-10-21); pacing below stays above that.
HISTORY_V0_IDS_PATH = DATA_DERIVED / "history_v0_cohort_ids.txt"
HISTORY_V0_CSV_PATH = DATA_DERIVED / "history_v0_cohort.csv"
HISTORY_PULL_LANDING = "https://clinicaltrials.gov/study/NCT01178112?tab=history"
HISTORY_PULL_DELAY_S = 1.5       # between requests
HISTORY_PULL_MIN_DELAY_S = 1.0   # floor; --delay is clamped to this
HISTORY_PULL_BACKOFF_S = 60      # on 429 / 5xx / network error
HISTORY_PULL_MAX_RETRIES = 3
HISTORY_PULL_TIMEOUT_MS = 60_000

# --- Phase 3: features (plan §5, §10) ------------------------------
AACT_SNAPSHOT_TS = "2026-10-01"  # outcome / censoring date for trials with no terminal status
# Registry status at version 0 that means the record was first submitted after
# recruitment had closed: planned values are missing for these (planned-value rule, PLAN.md Note 2026-10-21, ratified 2026-10-22).
V0_LATE_STATUSES = {"COMPLETED", "ACTIVE_NOT_RECRUITING", "TERMINATED", "WITHDRAWN", "SUSPENDED"}
NCE_EXTENSION_TOLERANCE_DAYS = 31   # PROJECT_END must move by more than this to count as an extension
F_INDIRECT_MIN_FY = 2006            # DIRECT/INDIRECT_COST_AMT do not exist before FY2006
O2_SAMPLE_N = 200
FEATURES_P_PATH = DATA_DERIVED / "features_P.parquet"
FEATURES_P_ALL_PATH = DATA_DERIVED / "features_P_all.parquet"
FEATURES_F_PATH = DATA_DERIVED / "features_F.parquet"
FEATURES_F2_PATH = DATA_DERIVED / "features_F2.parquet"
FEATURES_STATS_PATH = DATA_DERIVED / "features_stats.json"
FEATURES_COVERAGE_PATH = DATA_DERIVED / "features_coverage.md"
WHY_STOPPED_SAMPLE_PATH = DATA_DERIVED / "why_stopped_sample.csv"
GATE_PATH = DATA_DERIVED / "gate_attributable.json"

# --- Phase 4: models and reading (plan §6–§9; A1–A4) ------------------
PHASE4_DIR = DATA_DERIVED / "phase4"          # cached raw tables per window / rule
MODELS_DIR = DATA_DERIVED / "models"
FIGURES_DIR = DATA_DERIVED / "figures"
GATE_LANDMARK_PATH = DATA_DERIVED / "gate_landmark.json"
O2_VALIDATION_PATH = DATA_DERIVED / "o2_validation.json"
FEATURES_COVERAGE_TEST_PATH = DATA_DERIVED / "features_coverage_test.md"
METRICS_PATH = DATA_DERIVED / "metrics.csv"
READING_PATH = DATA_DERIVED / "reading.txt"
TEST_EVALUATED_MARKER = DATA_DERIVED / "phase4" / "test_evaluated.json"  # step 7 runs once

# Phase 4-X exploratory runs (plan §14; labelled exploratory, never replace the primary).
EXPLORATORY_DIR = DATA_DERIVED / "exploratory"
# Paths an exploratory run may read but never write.
PHASE4_PROTECTED = (METRICS_PATH, READING_PATH, TEST_EVALUATED_MARKER, MODELS_DIR, FIGURES_DIR, PHASE4_DIR)
# Charts for the public write-up (fb.figures): the only sub-path of FIGURES_DIR anything may write after step 7.
FIGURES_POST_DIR = FIGURES_DIR / "post"

# Landmark (A4b): a trial is in the primary F / F₂ population when >= 1 parent FY row of an
# attributable project survives overlap + the outcome-date cut.
LANDMARK_REASONS = {"ok", "cost_unknown"}

# Specifications (plan §6). Frozen before the test window is read.
LOGIT_C_GRID = [10.0 ** k for k in (-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)]
CV_FOLDS = 5
CV_SCORING = "neg_log_loss"
# sklearn defaults (max_leaf_nodes 31, min_samples_leaf 20, learning_rate 0.1) with early
# stopping on an inner 20 % validation fold; nothing tuned.
HGB_PARAMS = {"learning_rate": 0.1, "max_iter": 500, "max_leaf_nodes": 31, "min_samples_leaf": 20,
              "l2_regularization": 0.0, "early_stopping": True, "validation_fraction": 0.2,
              "n_iter_no_change": 10, "scoring": "loss"}
HAZARD_MAX_PERIOD = 7                 # FY-period dummies 0..6 and 7+ (baseline hazard)
HAZARD_HORIZONS_MONTHS = (12, 24, 36)

# Metrics (plan §7–§8)
BOOTSTRAP_N = 1000
PERMUTATION_N = 200
CALIBRATION_BINS = 10

# Reading rules (plan §9) — thresholds and cell texts are data; never edited after ratification.
READ_DELTA_AUC_MIN = 0.03             # ΔAUC(O1), logistic, test window, bootstrap CI excluding 0
READ_F2_DELTA_AUC_MIN = 0.02          # F₂ early-window ΔAUC, CI excluding 0
READ_P_AUC_MIN = 0.70                 # P-only AUC
READING_CELLS = {
    "R1": "ΔAUC(O1) ≥ 0.03, bootstrap CI excludes 0, and the F₂ early-window ΔAUC ≥ 0.02 with CI excluding 0 — "
          "Money-flow carries signal beyond priors even at annual resolution, and early",
    "R2": "ΔAUC(O1) ≥ 0.03 with CI excluding 0, but F₂ fails its threshold — Signal exists but is only visible late",
    "R3": "ΔAUC(O1) < 0.03 or CI includes 0, and P-only AUC ≥ 0.70 — Failure is predictable from design, not money",
    "R4": "ΔAUC(O1) < 0.03 or CI includes 0, and P-only AUC < 0.70 — Inconclusive on the fine-grained question — "
          "annual obligations too coarse to adjudicate",
}
