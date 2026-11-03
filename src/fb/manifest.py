"""Raw-download manifest: file name, URL, UTC timestamp, size, SHA-256, server dates.

Writes are atomic (temp file + rename) and merge over whatever is on disk at save
time, so two downloader processes recording different keys cannot clobber each other
(the Phase 1 write race: `fb.download exporter` and `manifest-aact` both loaded the
manifest at start and the later save dropped the other's entries).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_RAW, MANIFEST_PATH


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"files": {}}


def save_manifest(m: dict, path: Path = MANIFEST_PATH) -> dict:
    """Merge `m["files"]` over the manifest currently on disk and write atomically.

    Entries in `m` win on key collision; entries only on disk are kept. Returns the
    merged manifest (and updates `m` in place so the caller's view matches the file).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    on_disk = load_manifest(path)
    merged_files = dict(on_disk.get("files", {}))
    merged_files.update(m.get("files", {}))
    merged = {**on_disk, **{k: v for k, v in m.items() if k != "files"}, "files": merged_files}
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(merged, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    m.clear()
    m.update(merged)
    return merged


def record_file(
    m: dict,
    *,
    key: str,
    path: Path,
    url: str,
    downloaded_utc: str,
    server_last_modified: str | None = None,
    server_filename: str | None = None,
    note: str | None = None,
) -> dict:
    entry = {
        "file": path.name,
        "relpath": str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else path.name,
        "url": url,
        "downloaded_utc": downloaded_utc,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "server_last_modified": server_last_modified,
        "server_filename": server_filename,
    }
    if note:
        entry["note"] = note
    m.setdefault("files", {})[key] = entry
    return entry


def verify_manifest(path: Path = MANIFEST_PATH, raw_dir: Path = DATA_RAW) -> list[dict]:
    """Re-hash every manifested file and list raw files the manifest does not know.

    Returns one dict per file: key, file, status in {match, mismatch, missing,
    unmanifested}, and for mismatches the expected/actual size and SHA-256.
    `relpath` in the manifest is relative to `data/` (e.g. `raw/aact/x.zip`).
    """
    m = load_manifest(path)
    data_dir = raw_dir.parent
    rows: list[dict] = []
    seen: set[Path] = set()
    for key, e in sorted(m.get("files", {}).items()):
        p = data_dir / e["relpath"]
        seen.add(p.resolve())
        row = {"key": key, "file": e["relpath"], "expected_sha256": e["sha256"], "expected_size": e["size_bytes"]}
        if not p.exists():
            row["status"] = "missing"
        else:
            size = p.stat().st_size
            digest = sha256_of(p)
            row["actual_size"] = size
            row["actual_sha256"] = digest
            row["status"] = "match" if (size == e["size_bytes"] and digest == e["sha256"]) else "mismatch"
        rows.append(row)
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or p == path:
            continue
        if p.suffix in {".headers", ".part"} or p.name.endswith(("download_started_utc", "download_finished_utc")):
            continue  # downloader side-files, not data
        if p.resolve() not in seen:
            rows.append({"key": None, "file": str(p.relative_to(data_dir)), "status": "unmanifested"})
    return rows
