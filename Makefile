# funding-backtest — pipeline entry points. Needs Python >= 3.11 as `python3`.
PY := .venv/bin/python
EXPORT_DIR := ../funding-backtest-public

.PHONY: venv download download-exporter download-aact download-cpi verify test all export

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e '.[dev,models]'

test:
	$(PY) -m pytest -q

# --- downloads (manifested; `verify` re-hashes every raw file) -------------------------------
download: download-exporter download-aact download-cpi

download-exporter:
	$(PY) -m fb.download exporter

download-aact:
	$(PY) -m fb.download aact

download-cpi:
	$(PY) -m fb.download cpi

verify:
	$(PY) -m fb.download verify

# --- full chain after the downloads, in dependency order ------------------------------------
# `evaluate` (the single test-window read) is deliberately not in `all`: it refuses a rerun.
# Its output (metrics.csv) and marker (phase4/test_evaluated.json) are committed.
all: link coverage cohort base-rates gate features landmark-gate o2-validate test-tables hazard-tables fit \
     evaluate-dry sole-trial figures

# --- Phase 1: link and coverage ---------------------------------------------------------------
.PHONY: link coverage
link:
	$(PY) -m fb.link

coverage:
	$(PY) -m fb.coverage

# --- Phase 2: cohort and base rates -----------------------------------------------------------
.PHONY: cohort base-rates sole-trial
cohort:
	$(PY) -m fb.cohort

base-rates:
	$(PY) -m fb.base_rates

sole-trial:
	$(PY) -m fb.sole_trial

# --- pre-Phase 3: cohort record-history v0 pull (resumable; ~8-9 h) -----------
# Needs: pip install -e '.[pull]' && python -m playwright install chromium (once).
# Rerun = continue. ARGS passes flags, e.g. make history-pull ARGS="--limit 50 --headless".
# Its output, data/derived/history_v0_cohort.csv, is committed. On macOS, prefix the command
# with `caffeinate -i` to keep the machine awake during the pull (optional).
.PHONY: history-pull history-summary
history-pull:
	$(PY) -m fb.history_pull $(ARGS)

history-summary:
	$(PY) -m fb.history_pull --summary

# --- Phase 3: gate re-statement (A3) and feature build (training window only) --------
.PHONY: gate features
gate:
	$(PY) -m fb.gate

features:
	$(PY) -m fb.features

# --- Phase 4: models and reading (test window read once, in `evaluate`) ---------------
.PHONY: landmark-gate o2-validate test-tables hazard-tables fit evaluate-dry evaluate reading figures
landmark-gate:
	$(PY) -m fb.landmark

o2-validate:
	$(PY) -m fb.o2

test-tables:
	$(PY) -m fb.tables

hazard-tables:
	$(PY) -m fb.hazard

fit:
	$(PY) -m fb.models

evaluate-dry:
	$(PY) -m fb.evaluate --dry-run

evaluate:
	$(PY) -m fb.evaluate

reading:
	$(PY) -m fb.reading

# charts 1-3 -> data/derived/figures/post/ (chart 1's ROC points are cached; see fb.figures)
figures:
	$(PY) -m fb.figures

# --- public snapshot: tracked + unignored files, minus workflow files, into a plain directory ---
export:
	@if [ -e "$(EXPORT_DIR)/.git" ]; then echo "$(EXPORT_DIR) is a git repository; refusing to overwrite"; exit 1; fi
	rm -rf "$(EXPORT_DIR)"
	mkdir -p "$(EXPORT_DIR)"
	git ls-files --cached --others --exclude-standard \
	  | grep -vE '^(CLAUDE\.md|\.claude/)' \
	  | while IFS= read -r f; do [ -e "$$f" ] && printf '%s\n' "$$f"; done \
	  | rsync -a --files-from=- . "$(EXPORT_DIR)/"
	@echo "exported $$(find "$(EXPORT_DIR)" -type f | wc -l | tr -d ' ') files to $(EXPORT_DIR)"
