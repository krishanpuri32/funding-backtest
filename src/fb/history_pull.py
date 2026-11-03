"""Cohort record-history pull, version 0 (pre-Phase 3; PLAN.md Note 2026-10-21).

Access rationale: repo CLAUDE.md, "Registry record history". In short: version 0 of a
ClinicalTrials.gov record is the original registration; it is served by the site's own
endpoint `/api/int/studies/{nct}/history/0`, which robots.txt allows (`Allow: /api/int/`,
`Crawl-delay: 1`). The endpoint answers browser requests only, so the request is the same
same-origin `fetch` the site's History tab makes, run inside a Playwright Chromium page.
One request per trial, 1.5 s apart, no parallelism, no per-trial navigation, no HTML parsing,
stop on any 403.

Output: data/derived/history_v0_cohort.csv, one row appended (flushed + fsynced) per attempt.
Resumable: IDs whose latest row has v0_status 200 are skipped; anything else is retried.
The file is compacted to one row per ID (ID-file order) at the start and end of each run.

    python -m fb.history_pull [--limit N] [--delay 1.5] [--headed | --headless] [--only NCT..,NCT..]
    python -m fb.history_pull --summary
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config

COLUMNS = [
    "nct_id", "v0_status", "v0_overall_status", "v0_start", "v0_enrollment",
    "v0_enrollment_type", "v0_primary_completion", "v0_primary_completion_type",
    "v0_completion", "fetched_utc", "error",
]
REGISTERED_LATE = {"COMPLETED", "ACTIVE_NOT_RECRUITING", "TERMINATED", "WITHDRAWN", "SUSPENDED"}

# Runs inside the page. Returns {status, body, error}; never throws.
FETCH_JS = """
async ([nct, timeoutMs]) => {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch('/api/int/studies/' + nct + '/history/0',
                          {headers: {'Accept': 'application/json'}, signal: ctl.signal});
    const text = await r.text();
    let body = null, error = null;
    try { body = JSON.parse(text); } catch (e) { error = 'non-json body: ' + text.slice(0, 120); }
    return {status: r.status, body: body, error: r.ok ? error : (error || 'http ' + r.status)};
  } catch (e) {
    return {status: 0, body: null, error: 'fetch failed: ' + String(e).slice(0, 200)};
  } finally { clearTimeout(t); }
}
"""


# --- pure helpers (unit-tested, no network) ---------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse_v0(nct: str, status: int, body, error: str | None = None, fetched_utc: str | None = None) -> dict:
    """One CSV row from a version-0 response. Missing fields -> empty string."""
    row = {c: "" for c in COLUMNS}
    row.update(nct_id=nct, v0_status=str(status), fetched_utc=fetched_utc or utc_now(), error=error or "")
    ps = _get(body, "study", "protocolSection")
    if not isinstance(ps, dict):
        if status == 200:
            row["error"] = row["error"] or "no study.protocolSection in body"
        return row
    enr = _get(ps, "designModule", "enrollmentInfo") or {}
    sm = ps.get("statusModule") or {}
    vals = {
        "v0_overall_status": sm.get("overallStatus"),
        "v0_start": _get(sm, "startDateStruct", "date"),
        "v0_enrollment": enr.get("count"),
        "v0_enrollment_type": enr.get("type"),
        "v0_primary_completion": _get(sm, "primaryCompletionDateStruct", "date"),
        "v0_primary_completion_type": _get(sm, "primaryCompletionDateStruct", "type"),
        "v0_completion": _get(sm, "completionDateStruct", "date"),
    }
    row.update({k: "" if v is None else str(v) for k, v in vals.items()})
    return row


def read_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tok in path.read_text().split():
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def latest_by_id(rows: list[dict]) -> dict[str, dict]:
    """Latest row per ID, except that a 200 row is never displaced by a later failure."""
    out: dict[str, dict] = {}
    for r in rows:
        prev = out.get(r["nct_id"])
        if prev is not None and prev["v0_status"] == "200" and r["v0_status"] != "200":
            continue
        out[r["nct_id"]] = r
    return out


def done_ids(csv_path: Path) -> set[str]:
    return {k for k, r in latest_by_id(read_rows(csv_path)).items() if r["v0_status"] == "200"}


def pending_ids(ids: list[str], csv_path: Path) -> list[str]:
    done = done_ids(csv_path)
    return [i for i in ids if i not in done]


def compact(csv_path: Path, order: list[str] | None = None) -> int:
    """Rewrite the CSV atomically with one row per ID, in `order` where known."""
    rows = latest_by_id(read_rows(csv_path))
    if not rows:
        return 0
    rank = {k: i for i, k in enumerate(order or [])}
    keys = sorted(rows, key=lambda k: (rank.get(k, len(rank)), k))
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for k in keys:
            w.writerow({c: rows[k].get(c, "") for c in COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)
    return len(keys)


class RowWriter:
    """Append rows to the CSV (header if new), flushing and fsyncing after each row."""

    def __init__(self, csv_path: Path):
        new = not csv_path.exists() or csv_path.stat().st_size == 0
        self.f = open(csv_path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNS)
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row: dict) -> None:
        self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


# --- fetch with back-off ------------------------------------------------------------
class RouteChanged(RuntimeError):
    """HTTP 403: the route has changed or the browser is refused. The run stops."""


def fetch_one(page, nct: str, sleep=time.sleep) -> dict:
    """Fetch v0 for one ID. 429 / 5xx / network error: back off and retry up to the cap."""
    attempt = 0
    while True:
        res = page.evaluate(FETCH_JS, [nct, config.HISTORY_PULL_TIMEOUT_MS])
        st = int(res.get("status") or 0)
        if st == 403:
            raise RouteChanged(f"HTTP 403 on {nct}: {res.get('error')}")
        retryable = st == 429 or st >= 500 or st == 0
        if retryable and attempt < config.HISTORY_PULL_MAX_RETRIES:
            attempt += 1
            print(f"  {nct}: HTTP {st} ({res.get('error')}); sleeping {config.HISTORY_PULL_BACKOFF_S}s, "
                  f"retry {attempt}/{config.HISTORY_PULL_MAX_RETRIES}", flush=True)
            sleep(config.HISTORY_PULL_BACKOFF_S)
            continue
        err = res.get("error")
        if attempt and err:
            err = f"{err} (after {attempt} retries)"
        return parse_v0(nct, st, res.get("body"), err)


# --- summary --------------------------------------------------------------------------
def summarize(csv_path: Path = config.HISTORY_V0_CSV_PATH, ids_path: Path = config.HISTORY_V0_IDS_PATH,
              cohort_path: Path | None = None) -> str:
    rows = latest_by_id(read_rows(csv_path))
    ok = [r for r in rows.values() if r["v0_status"] == "200"]
    ids = read_ids(ids_path) if ids_path.exists() else []
    okset = {r["nct_id"] for r in ok}
    n = len(ok) or 1
    enr = sum(r["v0_enrollment"] != "" for r in ok)
    pcd = sum(r["v0_primary_completion"] != "" for r in ok)
    late = sum(r["v0_overall_status"] in REGISTERED_LATE for r in ok)
    out = [
        f"rows (one per ID): {len(rows)}",
        f"cohort IDs: {len(ids)}  |  status 200: {len(ok)}  |  not yet 200: {len(set(ids) - okset)}",
        f"HTTP status histogram: {dict(sorted(Counter(r['v0_status'] for r in rows.values()).items()))}",
        f"v0 enrollment present: {enr} / {len(ok)} ({100 * enr / n:.1f} %)",
        f"v0 primary completion present: {pcd} / {len(ok)} ({100 * pcd / n:.1f} %)",
        f"registered late (v0 status in {sorted(REGISTERED_LATE)}): {late} / {len(ok)} ({100 * late / n:.1f} %)",
        f"v0 overall-status distribution: {dict(Counter(r['v0_overall_status'] for r in ok).most_common())}",
        f"v0 enrollment type: {dict(Counter(r['v0_enrollment_type'] for r in ok).most_common())}",
        f"v0 primary completion type: {dict(Counter(r['v0_primary_completion_type'] for r in ok).most_common())}",
    ]
    fails = [r for r in rows.values() if r["v0_status"] != "200"]
    if fails:
        out.append(f"failures ({len(fails)}; first 20): "
                   + "; ".join(f"{r['nct_id']} {r['v0_status']} {r['error']}" for r in fails[:20]))
    cohort_path = cohort_path or (config.DATA_DERIVED / "cohort.parquet")
    if cohort_path.exists():
        import pandas as pd
        c = pd.read_parquet(cohort_path, columns=["nct_id", "window"])
        for w, g in c.groupby("window"):
            got = int(g["nct_id"].isin(okset).sum())
            out.append(f"coverage {w}: {got} / {len(g)} ({100 * got / len(g):.1f} %)")
    return "\n".join(out)


# --- browser loop ---------------------------------------------------------------------
def _open_page(pw, headless: bool):
    browser = pw.chromium.launch(headless=headless)
    page = browser.new_page()
    resp = page.goto(config.HISTORY_PULL_LANDING, wait_until="load", timeout=config.HISTORY_PULL_TIMEOUT_MS)
    print(f"landing page: HTTP {resp.status if resp else '?'} {page.url}", flush=True)
    return browser, page


def run(limit: int | None, delay: float, headless: bool, only: list[str] | None,
        ids_path: Path = config.HISTORY_V0_IDS_PATH, csv_path: Path = config.HISTORY_V0_CSV_PATH) -> int:
    from playwright.sync_api import sync_playwright

    delay = max(delay, config.HISTORY_PULL_MIN_DELAY_S)
    all_ids = read_ids(ids_path)
    compact(csv_path, all_ids)
    todo = pending_ids(only if only else all_ids, csv_path)
    if limit is not None:
        todo = todo[:limit]
    print(f"[{utc_now()}] {len(all_ids)} cohort IDs; {len(done_ids(csv_path))} already 200; "
          f"{len(todo)} to fetch (delay {delay}s, {'headless' if headless else 'headed'})", flush=True)
    if not todo:
        return 0
    writer = RowWriter(csv_path)
    n_ok = n_bad = 0
    code = 0
    t0 = time.time()
    try:
        with sync_playwright() as pw:
            browser, page = _open_page(pw, headless)
            try:
                for i, nct in enumerate(todo, 1):
                    try:
                        row = fetch_one(page, nct)
                    except RouteChanged:
                        raise
                    except Exception as e:  # page crashed or navigated away: reopen once, retry once
                        print(f"  {nct}: page error {e!r}; reopening browser", flush=True)
                        try:
                            browser.close()
                        except Exception:
                            pass
                        browser, page = _open_page(pw, headless)
                        row = fetch_one(page, nct)
                    writer.write(row)
                    if row["v0_status"] == "200":
                        n_ok += 1
                    else:
                        n_bad += 1
                    if i % 100 == 0 or i == len(todo):
                        el = time.time() - t0
                        print(f"[{utc_now()}] {i}/{len(todo)}  ok {n_ok}  failed {n_bad}  "
                              f"{el / i:.2f} s/ID  eta {(len(todo) - i) * el / i / 3600:.1f} h", flush=True)
                    if i < len(todo):
                        time.sleep(delay)
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
    except RouteChanged as e:
        print(f"STOPPED: {e}. Rows so far are kept; rerun resumes.", flush=True)
        code = 3
    finally:
        writer.close()
        compact(csv_path, all_ids)
    print(f"[{utc_now()}] run finished: ok {n_ok}, failed {n_bad}; file has {len(done_ids(csv_path))} IDs at 200",
          flush=True)
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=config.HISTORY_PULL_DELAY_S)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--headed", action="store_true", help="visible browser window (the default)")
    g.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--only", default=None, help="comma-separated NCT IDs (IDs already at 200 are still skipped)")
    ap.add_argument("--summary", action="store_true", help="print the coverage summary and exit")
    a = ap.parse_args(argv)
    if a.summary:
        print(summarize())
        return 0
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    return run(a.limit, a.delay, a.headless, only)


if __name__ == "__main__":
    sys.exit(main())
