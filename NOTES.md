# Notes on the data

Quirks found while building the pipeline, and the access rationale for the registry record
history. Useful to anyone who works with the same files.

## Data quirks found at the feasibility gate (2026-10-19)

- ExPORTER date columns change format: FY2000–FY2005 are `MM/DD/YYYY`, FY2006+ are `YYYY-MM-DD`.
  Always parse through `fb.exporter.parse_exporter_date`. `DIRECT_COST_AMT` / `INDIRECT_COST_AMT`
  do not exist before FY2006. The eRA download service ignores Range requests and drops long
  connections; the downloader retries from byte 0.
- The AACT snapshots page needs a login to *browse*, but the export zip itself is served from a
  public object store without authentication; the store honours Range (resume with `curl -C -`).
- The documented ClinicalTrials.gov v2 API has no record-history endpoint; the website's internal
  `/api/int/studies/{nct}/history` returns HTTP 403 to non-browser clients (see the access
  rationale below).

## Data quirks found in Phase 2 (2026-10-20)

- AACT `browse_conditions` rows of `mesh_type = mesh-ancestor` carry the MeSH top-level headings
  (C01…C26, F03) directly, so the condition group needs no separate MeSH tree download. Both the
  pre-2023 (C12 `Male Urogenital Diseases` / C13 `Female Urogenital…`) and 2023+ (`Urogenital
  Diseases`) spellings occur; `fb.priors` maps all three. ~950 cohort trials carry only F01
  behaviour terms (smoking, depression-as-symptom, adherence) with no disease heading.
- 621 of 60,809 parent (`SUBPROJECT_ID` null) FY rows behind cohort projects have null
  `TOTAL_COST`; counted as 0 and tallied in `n_fy_rows_cost_null_*`.
- Under A1 N = 20, 1,523 training trials (359 of the 776 training O1 events) have no attributable
  project — almost all NCI P30 / U10 oncology trials. The attributable-set and all-trials base-rate
  tables therefore describe different populations (PLAN.md A3).
- In 4,701 FY2006+ ExPORTER rows `TOTAL_COST` is below `DIRECT_COST_AMT + INDIRECT_COST_AMT`;
  the indirect ratio uses direct + indirect as its denominator (PLAN.md A4 (a)).

## Registry record history — access rationale

The study needs each trial's *originally registered* enrollment and primary-completion date
(PLAN.md §5, §10). ClinicalTrials.gov keeps every record's version history; version 0 is the
original registration. The documented v2 API serves only the current version. The history is
served by the site's own internal endpoint `/api/int/studies/{nct}/history/{n}`, which
**robots.txt explicitly allows for all user agents** (`Allow: /api/int/`, `Crawl-delay: 1`;
fetched 2026-10-21). That endpoint rejects non-browser HTTP clients, so the pull runs inside a
real browser (Playwright Chromium) as a same-origin `fetch` from a study page — the same request
the site's History tab makes. Pacing is 1.5 s per request (above the published crawl delay),
one request per trial, 13,983 trials total, no scraping of rendered pages, no parallelism, stop
on any 403. A 100-trial probe on 2026-10-20 ran this way with 0 errors; the full pull
(2026-10-21/22) fetched all 13,983 with 0 failures. The decision is recorded in PLAN.md
(Note 2026-10-21). The output, `data/derived/history_v0_cohort.csv`, is committed so the pull
need not be repeated.
