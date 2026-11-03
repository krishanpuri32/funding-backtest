# funding-backtest

A pre-registered retrospective study: does the NIH funding trajectory of a clinical trial add
predictive skill for operational failure beyond stage-level priors known at award? Public data
only: ClinicalTrials.gov (via AACT) and NIH ExPORTER. The analysis plan, with every amendment
dated, is in [`PLAN.md`](PLAN.md).

## Question and finding

The cohort is 13,983 NIH-funded interventional trials registered 2008–2020 and linked to RePORTER
(train 7,436 / test 6,547). Operational failure = terminated, suspended or withdrawn vs completed.
The overall O1 rate is 11.2 % (776 / 6,931; Wilson 10.5–12.0), and Phase 2 × Neoplasms is
24.6 % (164 / 666). Design priors known at award predict operational failure with AUC 0.7768
(95 % CI 0.7592–0.7944) on a held-out 2016–2020 test window. The priors are phase, condition,
award mechanism, institute, sponsor class, planned size and duration, site footprint, and start year.

Adding the trial's annual NIH funding trajectory raises that by +0.0086 AUC (bootstrap 95 % CI
+0.0014 to +0.0161). That is statistically non-zero and operationally small: under a third of the
pre-registered 0.03 bar for a material gain. Restricted to the first two funded fiscal years, the
gain is +0.0053 (CI −0.0016 to +0.0122), not distinguishable from zero. Under the plan's reading
rules this is R3, robust: failure is predictable from design, not from annual money-flow, and every
sensitivity run lands in the same cell. The study is predictive by design. Annual funder
obligations are not investigator spend, and the study says nothing about transaction-level
funding data; that question is untested here, not refuted.

The computed reading is [`data/derived/reading.txt`](data/derived/reading.txt); every metric is in
[`data/derived/metrics.csv`](data/derived/metrics.csv); the charts are in
[`data/derived/figures/post/`](data/derived/figures/post/).

## How this code was written

The code in this repository was written with an AI coding assistant (Anthropic's Claude) under
the author's direction and review. The analysis plan was also drafted with AI assistance (see
`PLAN.md`). The 200 `why_stopped` labels used to validate the O2 keyword rule were assigned by the
AI assistant under the author's direction, blind to the rule's output
(`data/derived/why_stopped_sample.csv`, columns `label` vs `rule_label`).

## What this repository is

This public repository is a snapshot assembled after the study from the working repository. It
reproduces every figure in the write-up. It also produces outcomes the write-up does not quote:
the pre-registered secondary outcomes O2 (recruitment-related failure), O3 (enrollment shortfall)
and O4 (timeline overrun), and one labelled exploratory run (Phase 4-X, P-only without the P
missing indicators, `data/derived/exploratory/`). The write-up leaves them out because O2–O3 are
under-powered secondary reads, O4's gain from funding features is mechanical (PLAN.md A5 (c)),
and the 4-X run is exploratory by the plan's own rule (§14).

The working repository's history is not public. The freeze-before-read record is
`data/derived/phase4/test_evaluated.json`: it holds the SHA-256 of every frozen model, the UTC
time of the single test-window read, and one later `roc_scoring` entry (chart 1's ROC curve; no
metric recomputed). The amendments in `PLAN.md` are dated and say whether each one preceded the
result it affects.

## Data sources

| source | what | pinned |
|---|---|---|
| AACT (CTTI) | ClinicalTrials.gov registry, monthly static flat-file export — `https://aact.ctti-clinicaltrials.org/static/exported_files/monthly/20261001_export.zip` | 2026-10-01 export |
| NIH ExPORTER | Projects by fiscal year, `https://reporter.nih.gov/exporter/projects/download/{FY}` | FY2000–FY2025, downloaded 2026-10-19 |
| NIH ExPORTER | Clinical Studies link file (core project ↔ NCT ID), `https://reporter.nih.gov/exporter/clinicalstudies/download` | downloaded 2026-10-19 |
| ClinicalTrials.gov record history | version 0 (original registration) of every cohort trial, `https://clinicaltrials.gov/api/int/studies/{nct}/history/0` | pulled 2026-10-21/22 |
| FRED | `CPIAUCSL` annual average, `https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&fq=Annual&fam=avg` | downloaded 2026-10-20; base year 2020 |

`data/raw/manifest.json` records the URL, UTC download time, size, SHA-256 and server
`Last-Modified` of every raw file; `make verify` re-hashes them. Data quirks and the record-history
access rationale are in [`NOTES.md`](NOTES.md).

## Reproduce

Needs Python ≥ 3.11 (`python3`), ~4 GB of disk for raw files, and `rsync`/`make`.

```
make venv          # .venv with the package, pytest and the model dependencies
make test          # unit tests on hand-built fixtures
make download      # ExPORTER zips + Clinical Studies, the AACT export (~2.5 GB), CPI -> data/raw/
make verify        # re-hash every raw file against data/raw/manifest.json
make all           # link → coverage → cohort → base-rates → gate → features → landmark-gate →
                   # o2-validate → test-tables → hazard-tables → fit → evaluate-dry → sole-trial → figures
```

Nothing has to be run to inspect the results. These are committed: `metrics.csv`, `reading.txt`,
`base_rates_*.csv` and `base_rates_report.md`, `sole_trial.csv`, the gate and validation JSON files,
`figures/` (Phase 4 calibration plots and `post/` charts with their plotted values), and the
evaluate marker.

`make evaluate`, the single read of the 2016–2020 test window, is not part of `make all`. It
refuses to run when `data/derived/phase4/test_evaluated.json` exists, by design. A refit with
`make fit` produces models equivalent to the frozen ones, but a byte-level match with the hashes in
the marker is not guaranteed. `make figures` reuses the committed ROC points for chart 1 and does
not read the test window again.

Two inputs are committed because they cannot be regenerated cheaply. `history_v0_cohort.csv` comes
from `make history-pull`: one browser request per trial at 1.5 s pacing, 13,983 trials, about nine
hours (`pip install -e '.[pull]'` and `python -m playwright install chromium` first). The other is
`why_stopped_sample.csv`, which holds the 200 blind labels; `make features` keeps an existing labelled
file and stops if the drawn sample differs.

## Models

- Temporal split, fixed before any data was downloaded: training = trials starting 2008–2015,
  test = 2016–2020. Hyperparameters by 5-fold CV inside the training window only.
- **L2 logistic regression** (standardised features; C chosen from 10⁻³…10² by 5-fold CV on
  training log-loss).
- **HistGradientBoosting** (scikit-learn defaults: 31 leaves, min 20 samples per leaf, learning
  rate 0.1; early stopping on an inner 20 % validation fold; nothing tuned), as a robustness check.
- **Discrete-time hazard** for time to termination: person-period rows by fiscal year, P fixed,
  F time-varying using only rows whose budget period ended before the start of that year.
- Feature sets: P (design priors), P + F (funding trajectory), P + F₂ (first two funded fiscal
  years). Missing values: indicator + training median.
- Metrics: AUC with DeLong CI; ΔAUC with a paired bootstrap (1,000 resamples); permutation null
  (200 permutations of F within phase × start-year strata); Brier, log-loss, calibration slope and
  intercept, decile reliability.
- The test window was read once from the frozen models; `fb.evaluate` writes the marker and
  refuses a second run.

## Layout of `src/fb/`

| module | role |
|---|---|
| `config.py` | paths, pinned snapshot ids, windows, seeds, thresholds, frozen IC top-10 |
| `download.py`, `manifest.py` | pinned raw downloads; manifest and `verify` |
| `exporter.py`, `aact.py` | ExPORTER and AACT loaders (date-format change at FY2006 handled here) |
| `link.py` | NCT ID ↔ core project link table |
| `coverage.py` | plan §3 applied step by step; the §11 sample-size cell |
| `rules.py` | pure cohort rules: start window, O1 mapping, fiscal-year overlap, trial-end fallback |
| `umbrella.py` | A1 umbrella-award flags (N = 20, N = 50, P30/U10) and the A2 flag |
| `priors.py` | frozen base-rate priors: phase, MeSH condition group, activity class, IC, sponsor class |
| `cpi.py`, `dollars.py`, `stats.py` | CPI deflator, parent-row dollar sums, Wilson CI |
| `cohort.py` | cohort table, one row per trial |
| `base_rates.py` | training-window base-rate tables (all trials, attributable set, A2) |
| `sole_trial.py` | sole-trial dollars table (A6) |
| `history_pull.py` | registry version-0 pull (Playwright, resumable) |
| `planned.py` | planned enrollment / duration from version 0; late-registration rule |
| `gate.py`, `landmark.py` | the §11 gate re-stated on the attributable set and on the A4b landmark subset |
| `leakage.py` | outcome date, `BUDGET_END` cut, observable gap window |
| `features_p.py`, `features_f.py`, `missing.py`, `features.py` | P, F, F₂ features; missing-value rule; Phase 3 builder |
| `o2.py` | O2 keyword rule, the 200-string sample, validation |
| `tables.py`, `hazard.py`, `design.py` | raw tables per window and attribution rule; person-period rows; the design matrix |
| `models.py` | populations, specifications, fit and freeze (training only) |
| `metrics.py` | AUC, DeLong, bootstrap ΔAUC, permutation helpers, calibration |
| `evaluate.py` | the single test-window evaluation, marker-guarded; dry run on training |
| `reading.py` | plan §9 reading cell and robust / fragile label from `metrics.csv` |
| `figures.py` | charts 1–3 for the write-up |

## Licence

- **Code** (`src/`, `tests/`, `Makefile`, `pyproject.toml`, `requirements.txt`): MIT, see [`LICENSE`](LICENSE).
- **Derived data, figures, labels and documentation** (everything under `data/derived/`, the 200
  `why_stopped` labels, `PLAN.md`, `NOTES.md` and this README): CC BY 4.0, see
  [`LICENSE-data.md`](LICENSE-data.md).
- **Upstream data** (ClinicalTrials.gov/AACT, NIH ExPORTER, FRED CPI-U) are public-domain US
  government data and are not relicensed here; CC BY 4.0 applies only to what this project adds.
