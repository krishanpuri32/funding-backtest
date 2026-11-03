"""Download the pinned raw files into data/raw/ and record them in the manifest.

Usage:
    python -m fb.download exporter   # Projects FY2000..FY2025 + ClinicalStudies.csv
    python -m fb.download aact       # AACT monthly export zip (large)
    python -m fb.download manifest-aact  # (re)hash an AACT zip placed by hand
    python -m fb.download cpi        # FRED CPIAUCSL annual averages -> data/raw/cpi/
    python -m fb.download verify     # re-hash every raw file against the manifest
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

from . import config
from .cpi import CPI_PATH
from .manifest import load_manifest, record_file, save_manifest, utc_now_iso, verify_manifest

EXPORTER_DIR = config.DATA_RAW / "exporter"
AACT_DIR = config.DATA_RAW / "aact"


def _stream_download(url: str, dest: Path, timeout: int = 600, attempts: int = 6) -> dict:
    """Stream url to dest; return server headers of the final response.

    The eRA document service does not honour Range requests, so a dropped
    connection restarts the file from byte 0 (attempts with exponential backoff).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                headers = {k.lower(): v for k, v in r.headers.items()}
            tmp.replace(dest)
            return headers
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            last_err = e
            wait = 5 * (2 ** i)
            print(f"retry {i + 1}/{attempts} for {dest.name} after {type(e).__name__}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_err


def _server_filename(headers: dict) -> str | None:
    cd = headers.get("content-disposition", "")
    if "filename=" in cd:
        return cd.split("filename=", 1)[1].strip().strip('"')
    return None


def download_exporter() -> None:
    m = load_manifest()
    for fy in range(config.EXPORTER_FY_FIRST, config.EXPORTER_FY_LAST + 1):
        url = config.EXPORTER_PROJECTS_URL.format(fy=fy)
        dest = EXPORTER_DIR / f"RePORTER_PRJ_C_FY{fy}.zip"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        t0 = utc_now_iso()
        h = _stream_download(url, dest)
        record_file(
            m, key=f"exporter/projects/FY{fy}", path=dest, url=url, downloaded_utc=t0,
            server_last_modified=h.get("last-modified"), server_filename=_server_filename(h),
        )
        save_manifest(m)
        print(f"ok   {dest.name} {dest.stat().st_size:,} bytes")
        time.sleep(1.0)
    url = config.EXPORTER_CLINICAL_STUDIES_URL
    dest = EXPORTER_DIR / "ClinicalStudies.csv"
    if not dest.exists():
        t0 = utc_now_iso()
        h = _stream_download(url, dest)
        record_file(
            m, key="exporter/clinical_studies", path=dest, url=url, downloaded_utc=t0,
            server_last_modified=h.get("last-modified"), server_filename=_server_filename(h),
        )
        save_manifest(m)
        print(f"ok   {dest.name} {dest.stat().st_size:,} bytes")


def download_aact() -> None:
    m = load_manifest()
    dest = AACT_DIR / f"{config.AACT_SNAPSHOT_DATE}_export.zip"
    if not dest.exists():
        t0 = utc_now_iso()
        h = _stream_download(config.AACT_EXPORT_URL, dest, timeout=3600)
        record_file(
            m, key="aact/export", path=dest, url=config.AACT_EXPORT_URL, downloaded_utc=t0,
            server_last_modified=h.get("last-modified"), server_filename=_server_filename(h),
        )
        save_manifest(m)
    print(f"ok   {dest.name} {dest.stat().st_size:,} bytes")


def manifest_aact(downloaded_utc: str | None = None, last_modified: str | None = None) -> None:
    """Record an AACT zip that was downloaded outside this module (e.g. by curl)."""
    m = load_manifest()
    dest = AACT_DIR / f"{config.AACT_SNAPSHOT_DATE}_export.zip"
    record_file(
        m, key="aact/export", path=dest, url=config.AACT_EXPORT_URL,
        downloaded_utc=downloaded_utc or utc_now_iso(), server_last_modified=last_modified,
        server_filename=dest.name,
        note="monthly static flat-file export; snapshot date pinned in fb.config",
    )
    save_manifest(m)
    print(f"ok   {dest.name} {dest.stat().st_size:,} bytes")


def download_cpi(force: bool = False) -> None:
    """CPI-U annual averages from FRED (CPIAUCSL, fq=Annual, fam=avg); small csv, manifested."""
    m = load_manifest()
    dest = CPI_PATH
    if dest.exists() and not force:
        print(f"skip {dest.name} (exists)")
        return
    t0 = utc_now_iso()
    h = _stream_download(config.CPI_URL, dest, timeout=60)
    record_file(
        m, key="cpi/cpiaucsl_annual", path=dest, url=config.CPI_URL, downloaded_utc=t0,
        server_last_modified=h.get("last-modified"), server_filename=_server_filename(h),
        note="FRED CPIAUCSL, annual average of monthly CPI-U (all items, US city average, SA); deflator base year in fb.config",
    )
    save_manifest(m)
    print(f"ok   {dest.name} {dest.stat().st_size:,} bytes")


def verify() -> int:
    """Print one line per file: status, size, key. Exit code 1 if anything is not a match."""
    rows = verify_manifest()
    bad = 0
    for r in rows:
        extra = ""
        if r["status"] == "mismatch":
            extra = f"  expected {r['expected_size']:,} B {r['expected_sha256'][:12]}… got {r['actual_size']:,} B {r['actual_sha256'][:12]}…"
        if r["status"] != "match":
            bad += 1
        print(f"{r['status']:<12} {r['file']}{extra}")
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("match", "mismatch", "missing", "unmanifested")}
    print("summary:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 1 if bad else 0


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "exporter"
    if what == "exporter":
        download_exporter()
    elif what == "aact":
        download_aact()
    elif what == "manifest-aact":
        manifest_aact(*(sys.argv[2:4]))
    elif what == "cpi":
        download_cpi()
    elif what == "verify":
        raise SystemExit(verify())
    else:
        raise SystemExit(f"unknown target {what}")
