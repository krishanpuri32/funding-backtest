import hashlib
import json

from fb import manifest


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "data" / "raw" / "x.bin"
    p.parent.mkdir(parents=True)
    payload = b"abc" * 100_000 + b"tail"
    p.write_bytes(payload)
    assert manifest.sha256_of(p, chunk=7) == hashlib.sha256(payload).hexdigest()


def test_record_and_roundtrip_manifest(tmp_path):
    p = tmp_path / "data" / "raw" / "f.csv"
    p.parent.mkdir(parents=True)
    p.write_text("a,b\n1,2\n")
    m = manifest.load_manifest(tmp_path / "manifest.json")
    e = manifest.record_file(
        m, key="k", path=p, url="https://example.org/f.csv", downloaded_utc="2026-10-19T00:00:00Z",
        server_last_modified="Mon, 12 Oct 2026 08:00:39 GMT", server_filename="f.csv",
    )
    assert e["size_bytes"] == 8
    assert e["sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    manifest.save_manifest(m, tmp_path / "manifest.json")
    back = json.loads((tmp_path / "manifest.json").read_text())
    assert back["files"]["k"]["sha256"] == e["sha256"]
    assert back["files"]["k"]["url"] == "https://example.org/f.csv"


def test_save_manifest_merges_over_disk_and_is_atomic(tmp_path):
    path = tmp_path / "manifest.json"
    # process A loads an empty manifest, process B records key b and saves first
    a = manifest.load_manifest(path)
    b = manifest.load_manifest(path)
    b["files"]["b"] = {"sha256": "bb", "relpath": "raw/b", "size_bytes": 1}
    manifest.save_manifest(b, path)
    a["files"]["a"] = {"sha256": "aa", "relpath": "raw/a", "size_bytes": 1}
    manifest.save_manifest(a, path)   # must not drop b
    back = json.loads(path.read_text())
    assert set(back["files"]) == {"a", "b"}
    assert set(a["files"]) == {"a", "b"}  # caller's view updated to the merged file
    assert not list(tmp_path.glob(".manifest-*"))  # temp file cleaned up


def test_verify_manifest_statuses(tmp_path):
    raw = tmp_path / "data" / "raw"
    (raw / "x").mkdir(parents=True)
    good = raw / "x" / "good.bin"; good.write_bytes(b"good")
    bad = raw / "x" / "bad.bin"; bad.write_bytes(b"original")
    extra = raw / "x" / "extra.bin"; extra.write_bytes(b"extra")
    (raw / "x" / "bad.bin.headers").write_text("ignored side-file")
    path = raw / "manifest.json"
    m = manifest.load_manifest(path)
    for k, p in (("good", good), ("bad", bad)):
        manifest.record_file(m, key=k, path=p, url="u", downloaded_utc="t")
    m["files"]["gone"] = {"relpath": "raw/x/gone.bin", "sha256": "0" * 64, "size_bytes": 3}
    manifest.save_manifest(m, path)
    bad.write_bytes(b"tampered!")
    rows = {r["file"]: r["status"] for r in manifest.verify_manifest(path, raw)}
    assert rows == {"raw/x/good.bin": "match", "raw/x/bad.bin": "mismatch", "raw/x/gone.bin": "missing", "raw/x/extra.bin": "unmanifested"}
