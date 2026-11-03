---
plan_status: ratified
plan_version: 0.1 (A1, A2 2026-10-19; A3 + priors freeze 2026-10-21; A4 2026-10-23; A5 2026-10-24)
---

# Funding-trajectory backtest — pre-registered analysis plan

**Status: `ratified`** — v0.1, written 2026-10-15 (drafted with AI assistance; see README) before any data was downloaded; ratified by Krishan 2026-10-18, still before any data was downloaded. From that point every change is logged under *Amendments* with a date and reason, and the ratified text is never edited in place. The reading rules (§9) are the point of pre-registering: they are fixed before results exist so the result cannot be rationalized afterwards.



## 1. Research question

**Primary.** Among NIH-funded interventional clinical trials, does the funding trajectory add predictive skill for operational failure over a baseline of stage-level priors known at award?

**Secondary.** (a) Is any added skill present when only the first two fiscal years of funding are visible? (b) Which failure mode is predictable — recruitment-related termination or other? (c) What are the base rates of operational failure and funded cost by trial type?

The study is **predictive, not causal.** No statement of the form "funding pattern X causes failure" is licensed by this design.

## 2. Data sources

| Source                            | What it provides                                                                                                                                                                                                                                                                                                                        | Access                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| AACT (CTTI)                       | Full ClinicalTrials.gov registry as Postgres / flat files: `studies`, `designs`, `conditions`, `interventions`, `facilities`, `sponsors`, `calculated_values`, `outcomes`                                                                                                                                                               | Monthly static snapshot download; pin the snapshot date                                      |
| ClinicalTrials.gov record history | Per-record version history, needed for *originally planned* enrollment and primary-completion date                                                                                                                                                                                                                                      | ClinicalTrials.gov API v2 history endpoints; coverage and rate limits are a feasibility item |
| NIH ExPORTER                      | Projects by fiscal year (FY2000→): `CORE_PROJECT_NUM`, `APPLICATION_TYPE`, `ACTIVITY`, `IC`, `TOTAL_COST`, `DIRECT_COST_AMT`, `INDIRECT_COST_AMT`, `PROJECT_START`, `PROJECT_END`, `BUDGET_START`, `BUDGET_END`, `SUPPORT_YEAR`, `SUFFIX`; Clinical Studies link file (`CORE_PROJECT_NUM` ↔ NCT ID); publication and patent link tables | Bulk CSV download from the RePORTER ExPORTER page; pin the file dates                        |

**Known limitation stated up front:** ExPORTER amounts are annual *obligations by the funder*, not investigator expenditures. The study observes the funding envelope at yearly resolution. Burn rate, cost transfers, cost-per-enrolled-patient and any per-transaction quantity are **not observable** and are not claimed.

## 3. Cohort

Unit of analysis: one ClinicalTrials.gov record (NCT ID) linked to ≥ 1 NIH core project number.

Inclusion:
- `study_type` = Interventional.
- Start date 2008-01-01 → 2020-12-31 (FDAAA 2007 makes registration materially more complete from 2008; the 2020 cap leaves ≥ 5 years of follow-up to the snapshot so most trials have reached a terminal status).
- Linked to NIH funding through the ExPORTER Clinical Studies file.
- At least one ExPORTER fiscal-year record for a linked project whose budget period overlaps the trial window.

Exclusion:
- Status `Withdrawn` with enrollment 0 and no funded fiscal year overlapping the trial window (never started; no trajectory to observe) — reported separately in the coverage table.
- Records with no start date.
- Trials whose linked projects are multi-trial umbrella awards (program projects, cooperative agreements funding > N trials — threshold set at the feasibility gate and recorded here as an amendment) when the funding cannot be attributed to the trial. Reported, and a sensitivity analysis keeps them.

Censoring: trials still `Recruiting`, `Active, not recruiting`, `Enrolling by invitation`, or `Not yet recruiting` at the snapshot are censored at the snapshot date in the survival analysis and excluded from the binary-outcome models.

## 4. Outcomes

Primary and secondary outcomes are fixed here.

- **O1 (primary, binary) — operational failure:** `overall_status` ∈ {Terminated, Suspended, Withdrawn} vs {Completed}. Unknown-status records are excluded from O1 and counted in the coverage table.
- **O2 (secondary, binary) — recruitment-related failure:** O1 = 1 and `why_stopped` classified as enrollment/accrual-related. Classification: keyword rules (accrual, enrol, recruit, participants, slow, low, insufficient, eligible, and variants) validated against a random sample of 200 `why_stopped` strings labelled by the AI assistant under the author's direction, blind to the rule's output; the rule set and its precision/recall on that sample are recorded (`data/derived/o2_validation.json`) before O2 models are run.
- **O3 (secondary, continuous) — enrollment shortfall:** actual enrollment ÷ originally planned enrollment (from record history). Analyzed as log ratio. Computed only where the planned value is recoverable; coverage reported.
- **O4 (secondary, continuous) — timeline overrun:** actual primary-completion date − originally planned primary-completion date, in months. Same recoverability caveat.
- **O5 (descriptive) — time to termination:** months from start to termination, for the hazard model in §6.

## 5. Features

**Baseline priors (P) — known at award:** phase (incl. N/A), condition category (MeSH tree top-level from AACT `browse_conditions`, collapsed to ≤ 15 groups), intervention type, allocation (randomized / non-randomized / N/A), masking level, number of facilities (log), number of countries, lead sponsor class, planned enrollment (log), planned duration (months), activity code (collapsed: R01-class, U-class cooperative agreements, P-class program, K-class career, other), administering IC, start year, number of distinct linked core projects.

**Funding trajectory (F) — the test set:**
- `F_total` — total obligated across linked projects within the trial window (log, CPI-adjusted to 2020 dollars).
- `F_years` — count of fiscal years with an obligation inside the trial window.
- `F_shape` — slope of obligations over fiscal years, normalized by total (front- vs back-loaded).
- `F_yoy_min` — the largest single-year proportional cut.
- `F_gap` — any fiscal year inside the trial window with no obligation on any linked project (binary) and count.
- `F_supp` — count of supplements (application type 3 or suffix S*).
- `F_nce` — evidence of no-cost extension: `PROJECT_END` later than the end implied by the original support-year count, or a fiscal year with `SUPPORT_YEAR` beyond the original award term and zero obligation.
- `F_indirect_ratio` — indirect ÷ total, averaged across years.
- `F_per_enrollee` — total obligated ÷ planned enrollment (log).
- `F_noncompeting_share` — share of fiscal-year records that are non-competing continuations (type 5).

**Early-window variant (F₂):** every F feature recomputed using only the first two fiscal years overlapping the trial window; later years masked. `F_years` becomes "years funded of the first two"; `F_nce` is undefined and dropped.

All F and F₂ features are computed from fiscal-year records whose `BUDGET_END` precedes the outcome date (termination date for O1/O2/O5; primary-completion date otherwise). This is the leakage control for reverse causation — a trial that dies stops drawing money, and funding observed after death must never enter a feature.

## 6. Models

Three specifications per outcome, each fitted twice (P only; P + F) and, for O1/O2, a third time with P + F₂:

1. **Regularized logistic regression** (L2, penalty chosen by inner CV) — interpretable coefficients; the nested P vs P + F comparison supports a likelihood-ratio test.
2. **Gradient-boosted trees** (LightGBM or sklearn HistGradientBoosting, default-ish depth, early stopping on the inner fold) — captures interactions; no claim of superiority is made, it is a robustness check.
3. **Discrete-time hazard model** for O5 — person-period data by fiscal year with F features time-varying (only years observed so far), P features fixed. This is the closest analogue to the product's "futility marker" and the version that answers §1(a) most directly.

Missing values: indicator + median (continuous) / explicit "missing" level (categorical). No imputation of outcomes.

## 7. Validation

- **Temporal split, fixed now:** training = trials starting 2008–2015; test = 2016–2020. No random splits (era effects leak). Hyperparameters chosen by 5-fold CV inside the training window only.
- **Stratified reporting** by phase and condition group in the test window, only where the stratum holds ≥ 100 trials and ≥ 30 O1 events; otherwise pooled only.
- **Permutation null:** F features permuted within phase × start-year strata, 200 repetitions, to give the reference distribution for ΔAUC under no signal.

## 8. Metrics

- Discrimination: AUC with DeLong 95% CI; ΔAUC (P + F minus P) with bootstrap 95% CI (1,000 resamples of the test set).
- Calibration: Brier score, calibration slope and intercept, reliability plot in deciles.
- Likelihood: log-loss; likelihood-ratio test on the nested logistic models.
- Hazard model: time-dependent AUC at 12, 24, 36 months; Brier at the same horizons.
- O3/O4: RMSE and R² of P vs P + F, same split.

Primary metric for the reading rules: **ΔAUC for O1, logistic specification, test window, with its bootstrap CI.** Everything else is supporting.

## 9. Reading rules — fixed before results

| Result | Reading |
|---|---|
| R1 — ΔAUC(O1) ≥ 0.03, bootstrap CI excludes 0, **and** the F₂ early-window ΔAUC ≥ 0.02 with CI excluding 0 | Money-flow carries signal beyond priors even at annual resolution, and early |
| R2 — ΔAUC(O1) ≥ 0.03 with CI excluding 0, but F₂ fails its threshold | Signal exists but is only visible late |
| R3 — ΔAUC(O1) < 0.03 or CI includes 0, and P-only AUC ≥ 0.70 | Failure is predictable from design, not money |
| R4 — ΔAUC(O1) < 0.03 or CI includes 0, and P-only AUC < 0.70 | Inconclusive on the fine-grained question — annual obligations too coarse to adjudicate |



A result that lands between cells is reported as the more conservative cell. Secondary outcomes do not override the primary reading; they qualify it (e.g. "signal concentrated in recruitment failure").

## 10. Leakage and bias controls

- Features computed only from fiscal years ending before the outcome date (§5).
- Planned enrollment and completion taken from the earliest registry version, not the current record (§2, §4); where history is unavailable the trial is excluded from O3/O4 and flagged.
- Linkage coverage reported: linked vs unlinked NIH trials compared on all P features; a large imbalance is reported as a limitation, not corrected.
- `why_stopped` classifier validated before use (§4).
- No feature is derived from publications, patents, or results posting (all post-outcome).
- Snapshot dates for AACT and ExPORTER files are pinned (`src/fb/config.py`, `data/raw/manifest.json`); the analysis is reproducible from those files.

## 11. Sample-size gate

Before any model is fitted, the feasibility pass reports: trials linked · with terminal status · with ≥ 1 overlapping funded fiscal year · with recoverable planned enrollment · O1 events in training and test windows.

Proceed to the full plan only if the test window holds **≥ 300 O1 events**. Between 150 and 300: pooled O1 analysis only, no stratified reporting, no O2–O4. Below 150: stop, record the result, and treat the question as unanswerable from public data at this granularity.

## 12. Software and reproducibility

Python; `pandas` / `polars`, `scikit-learn`, `lightgbm`, `lifelines` or a hand-rolled person-period logistic for the hazard model, `statsmodels` for the LR test. Raw downloads are kept immutable under `data/raw/` with a manifest of file names, dates and SHA-256; everything derived is regenerable from a single `make`-style entry point. Seeds fixed. The repo's `CLAUDE.md` is authoritative for how to run it.

## 13. Reporting

Results go into a results note in this folder, written from the session records. Report the coverage table, the base-rate table, the P-only and P + F metrics with CIs, the permutation null, the calibration plots, the F₂ comparison, the stratified results where permitted, and the reading cell. Negative or inconclusive readings are reported with the same prominence as positive ones.

## 14. Deviations policy

Any departure from §3–§11 after ratification is an amendment: logged below with date, reason, and whether it was decided before or after seeing the affected result. A deviation decided after seeing results is labelled exploratory in the report.

## Amendments

**A1 — 2026-10-19 — Umbrella-award threshold (§3) set: N = 20, with a pre-specified sensitivity grid.** Decided *after* the Phase 1 feasibility counts (coverage table and trials-per-project distribution only; no feature, model or outcome analysis had been run), as §3 itself anticipated. Rule: a core project linked to **more than 20** NCT IDs in the ExPORTER Clinical Studies file is an umbrella award; its fiscal-year records are **not** used for any F / F₂ feature. A trial whose only linked projects are umbrella awards has no attributable funding and is **excluded from the F models** (it stays in the coverage table, the base-rate table and the P-only models). A trial with at least one non-umbrella linked project keeps its F features computed from the non-umbrella projects only. Feasibility figures behind the choice: 112 projects exceed 20 (median 1, p99 10, p99.9 ≈ 115, max 774), almost all NCI P30 cancer-center and U10 cooperative-group awards — a different mechanism, not the tail of a continuum. At N = 20 the rule touches 7,255 linked trials and removes 6,328 that have no other link (whole link file, pre-cohort). **Sensitivity grid, fixed now:** (i) no umbrella exclusion; (ii) N = 50; (iii) activity-code rule — exclude P30 and U10 projects regardless of count. Each grid run reports the primary metric (ΔAUC for O1, logistic, test window, bootstrap CI) in one table beside the primary. **Stability criterion:** the primary reading cell (§9) comes from the N = 20 run only; it is reported as *robust* if every grid run lands in the same cell and *fragile* if any does not — the disagreement is reported, never resolved by choosing among runs. The same stability rule applies to the A2 sensitivity run.

**Note — 2026-10-21 — Record-history source for §4 (O3/O4) and §5 (planned enrollment, planned duration).** Not an amendment: the plan already specifies the earliest registry version (§10). Recording the route and its authorisation: ClinicalTrials.gov's public v2 API serves the current version only; the per-version history is served by the site's internal endpoint `/api/int/studies/{nct}/history/{n}`, which the site's robots.txt explicitly allows for all user agents (`Allow: /api/int/`, `Crawl-delay: 1`, fetched 2026-10-21) but which answers only browser clients. The study therefore reads version 0 for every cohort trial through a real browser (Playwright Chromium, same-origin fetch from a study page), one request per trial at 1.5 s spacing, no rendered-page scraping. Probe of 100 trials on 2026-10-20: 0 errors; version 0 carries enrollment for 100 % and primary completion for 95 %. Trials whose version 0 was submitted after completion or closure to enrollment (8 % in the probe; 909 = 6.5 % in the full pull, 2026-10-22) are excluded from O3/O4 and their planned values treated as missing for §5. **Ratified extension 2026-10-22:** the same handling applies when the version-0 enrollment is typed ACTUAL (665 trials) — an actual count at first registration is not a plan. The pull itself precedes Phase 3 and touches no outcome.

**A6 — 2026-10-30 — descriptive addition, after the reading:** sole-trial dollars table (§1 secondary (c)). Population: trials with exactly one attributable (non-umbrella, N = 20) linked project, where that project links to exactly one NCT ID, so no attribution rule is needed and no dollar is counted twice. Terminal = O1 label 0 / 1; dollars = CPI-2020 parent-row obligations in fiscal years overlapping the trial window (the cohort convention). Reported per window beside the full attributable set: terminal trials, O1 events, rate with Wilson CI, total, median and non-complete dollars, non-complete share. No threshold, no reading; no effect on any cell.

**A5 — 2026-10-24 — Clarifications recorded after the Phase 4 reading (R3, robust); none changes the primary run.** (a) **Hazard-model covariate cut** (§6.3): F features for fiscal year *f* use rows with `BUDGET_END` before the **start** of *f* (= end of the previous FY) and before the outcome date — "only years observed so far"; an end-of-year cut would let survival years see a full year's money while the termination year sees part of one. (b) **A1 stability label** uses all three grid runs the amendment names — no exclusion, N = 50, activity rule — plus the A2 and A4(b)-S1 sensitivity runs; the Phase 4 run instructions had listed two. (c) **O3/O4 reporting rule:** the O4 (timeline-overrun) gain from F is mechanical — a trial that overruns accumulates more funded fiscal years before its late primary completion, so `F_years` and `F_total` track the realised duration; O3/O4 are reported as secondary with the mechanism stated and are never quoted as predictive skill. (d) **Exploratory run 4-X** (§14): P-only models without the P missing-indicators, decided after the reading was seen, labelled exploratory; result 0.7767 vs 0.7768, cell unchanged. Ratified by Krishan 2026-10-24.

**A4 — 2026-10-23 — Feature definitions fixed after the Phase 3 build (training window only; no model fitted, no feature compared with an outcome).** (a) **`F_indirect_ratio` = indirect ÷ (direct + indirect)**, not ÷ `TOTAL_COST`: in 4,701 FY2006+ ExPORTER rows `TOTAL_COST` is below direct + indirect and the literal rule produced ratios above 1 for 15 trials; direct + indirect is the budget's total. (b) **Landmark for the F / F₂ models.** 466 attributable training trials have every funding row ending after their outcome date — they stopped before their first budget period closed — so their F features are missing by construction and the missing-indicators would mark early failure (reverse causation through missingness). Rule: the **primary** P + F and P + F₂ models use trials with ≥ 1 fiscal-year row observable before the outcome date (a landmark at the close of the first funded fiscal year); the §11 gate is re-stated on that subset at the start of Phase 4 and the §11 cell rules apply to it; the all-attributable-trials version with missing-indicators is a **pre-specified sensitivity run** reported beside the primary under the A1 stability rule. The discrete-time hazard model (§6.3) is unchanged — it conditions on survival to each fiscal year by construction. The P-only models are unchanged (all trials, per A3). (c) **Planned duration enters as log(1 + months)** (plan §5 listed it untransformed; max 1,046 months). (d) **Caveat, not a change:** number of facilities and number of countries are current-record values (sites are added during recruitment, so an early stop lists fewer); kept as the only site-footprint priors available and named as a limitation in the write-up. Decided before any outcome was seen by a model; ratified by Krishan 2026-10-23.

**A3 — 2026-10-21 — Scope of the funding-trajectory models, and the §11 gate re-stated on the attributable set.** Decided after the Phase 2 base-rate table (training window only; no feature, model or test-window outcome had been computed). Phase 2 showed that the A1 attribution rule removes the trials with the highest failure rate: in the training window, 1,523 trials have no attributable project, hold 359 of 776 O1 events (26.5 % event rate vs 7.5 % in the attributable set), and are 82 % oncology funded through NCI P30 / U10 / U01 / UM1 / UL1 awards. Rule: **the F / F₂ models and the §9 reading apply to trials with ≥ 1 attributable (non-umbrella) linked project.** Umbrella-only trials remain in the base-rate table and in the P-only models; they are outside the scope of any funding-trajectory claim, and the write-up states this as a pre-committed limitation: the result does not speak to cooperative-group or cancer-centre oncology trials. **Gate:** the §11 sample-size gate was reported on all trials (800 test-window events); it is re-stated on the attributable test window as the first query of Phase 3, before any feature is built, and the §11 cell rules apply to that count: ≥ 300 → full plan; 150–300 → pooled O1 only, no stratified reporting, no O2–O4; < 150 → stop and record. **Base-rate reporting rule (not an amendment to §3–§11):** rates are quoted from the all-trials table, dollars from the attributable set.

**A3-freeze — 2026-10-21 — Priors frozen.** The MeSH condition map (15 groups + "No MeSH term", priority order Neoplasms → Infections → Mental & behavioral → Cardiovascular → Nervous → Nutritional/metabolic/endocrine → Respiratory → Digestive → Urogenital/pregnancy → Hemic/immune → Musculoskeletal/skin → Congenital → Eye/ENT/stomatognathic → Injuries/chemical/environmental → Other; F01 Behavior folded into Mental & behavioral), the phase collapse, the activity-code class, and `IC_TOP10` = {CA, MH, DA, HL, DK, HD, AG, AT, AA, NR} are frozen as of the Phase 2 record in Phase 2 — Base-rate table and `src/fb/priors.py` / `fb.config`. Later phases import them and do not re-derive.

**A2 — 2026-10-19 — Withdrawn / zero-enrollment trials: clarification of §3–§4 plus a pre-specified sensitivity run.** Decided *after* the Phase 1 feasibility counts (coverage table only; no feature, model or outcome analysis had been run). The feasibility pass showed that the §3 "never started" exclusion as worded removes almost nothing (10 of 364 withdrawn trials), because the overlap test uses the registry's planned window and the award was obligated for that period. The ratified text stands: a Withdrawn trial with an overlapping funded fiscal year is an O1 event (a funded program that never enrolled is an operational failure the funder paid for). Added: a pre-specified **sensitivity run of every O1 model with Withdrawn/enrollment-0 trials excluded**, reported in the same table as the primary with the same metric and CI. The primary reading cell comes from the primary run; the sensitivity run qualifies it (robust if same cell, fragile otherwise) and never overrides it. Reason: 200 of the 800 test-window events are of this kind, and a reader will reasonably ask whether the result depends on them.

## Glossary (added at ratification — terminology only, no change to §3–§11)

- **IC** (ExPORTER) — NIH Institute or Center that administers the award (e.g. NCI, NHLBI, NIDDK). Not *information coefficient*.
- **MeSH** — Medical Subject Headings, the NLM controlled vocabulary; AACT maps each trial's conditions onto the MeSH tree, and the plan collapses those to top-level branches for the condition-category prior.
- **AUC** — area under the ROC curve: the probability that a randomly chosen failed trial is scored riskier than a randomly chosen completed one (0.5 = chance, 1.0 = perfect). ΔAUC is the gain from adding funding features to the priors.
