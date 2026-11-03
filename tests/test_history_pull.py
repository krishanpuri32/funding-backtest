"""fb.history_pull: parser, resume, append-and-flush, back-off. No network."""
import csv

import pytest

from fb import config
from fb import history_pull as hp

V0 = {"study": {"protocolSection": {
    "designModule": {"enrollmentInfo": {"count": 224, "type": "ESTIMATED"}},
    "statusModule": {"overallStatus": "NOT_YET_RECRUITING",
                     "startDateStruct": {"date": "2007-12"},
                     "primaryCompletionDateStruct": {"date": "2009-06", "type": "ANTICIPATED"},
                     "completionDateStruct": {"date": "2009-12"}}}}}


def _row(nct, status):
    if status == 200:
        return hp.parse_v0(nct, 200, V0, fetched_utc="t")
    return hp.parse_v0(nct, status, None, f"http {status}", "t")


def _write(path, rows):
    w = hp.RowWriter(path)
    for r in rows:
        w.write(r)
    w.close()


def test_parse_v0_full_fixture():
    r = hp.parse_v0("NCT1", 200, V0, fetched_utc="2026-10-21T00:00:00.000Z")
    assert list(r) == hp.COLUMNS
    assert r == {"nct_id": "NCT1", "v0_status": "200", "v0_overall_status": "NOT_YET_RECRUITING",
                 "v0_start": "2007-12", "v0_enrollment": "224", "v0_enrollment_type": "ESTIMATED",
                 "v0_primary_completion": "2009-06", "v0_primary_completion_type": "ANTICIPATED",
                 "v0_completion": "2009-12", "fetched_utc": "2026-10-21T00:00:00.000Z", "error": ""}


def test_parse_v0_missing_fields_are_blank():
    body = {"study": {"protocolSection": {"statusModule": {"overallStatus": "RECRUITING"}}}}
    r = hp.parse_v0("NCT2", 200, body)
    assert r["v0_overall_status"] == "RECRUITING"
    assert r["v0_enrollment"] == "" and r["v0_primary_completion"] == "" and r["error"] == ""


def test_parse_v0_error_and_bad_body():
    assert hp.parse_v0("NCT3", 500, None, "http 500")["error"] == "http 500"
    assert hp.parse_v0("NCT4", 200, {"x": 1})["error"] == "no study.protocolSection in body"


def test_append_and_flush_visible_before_close(tmp_path):
    p = tmp_path / "out.csv"
    w = hp.RowWriter(p)
    w.write(_row("NCT1", 200))
    with open(p, newline="") as f:  # writer still open: header + row are already on disk
        assert [r["nct_id"] for r in csv.DictReader(f)] == ["NCT1"]
    w.write(_row("NCT2", 200))
    w.close()
    w2 = hp.RowWriter(p)  # reopening appends without a second header
    w2.write(_row("NCT3", 500))
    w2.close()
    lines = p.read_text().splitlines()
    assert sum(line.startswith("nct_id,") for line in lines) == 1 and len(lines) == 4


def test_resume_skips_200_rows_and_retries_500(tmp_path):
    p = tmp_path / "out.csv"
    _write(p, [_row("NCT1", 200), _row("NCT2", 200), _row("NCT3", 500)])
    assert hp.pending_ids(["NCT1", "NCT2", "NCT3", "NCT4"], p) == ["NCT3", "NCT4"]


def test_latest_attempt_wins_but_never_displaces_200(tmp_path):
    p = tmp_path / "out.csv"
    _write(p, [_row("NCT3", 500), _row("NCT1", 200), _row("NCT3", 200), _row("NCT1", 429)])
    latest = hp.latest_by_id(hp.read_rows(p))
    assert latest["NCT3"]["v0_status"] == "200" and latest["NCT1"]["v0_status"] == "200"
    assert hp.compact(p, ["NCT1", "NCT3"]) == 2
    assert [r["nct_id"] for r in hp.read_rows(p)] == ["NCT1", "NCT3"]


def test_read_ids_dedupes_and_ignores_missing_trailing_newline(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("NCT2\nNCT1\nNCT2\nNCT3")
    assert hp.read_ids(p) == ["NCT2", "NCT1", "NCT3"]


class FakePage:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def evaluate(self, js, args):
        self.calls += 1
        st = self.statuses.pop(0)
        return {"status": st, "body": V0 if st == 200 else None, "error": None if st == 200 else f"http {st}"}


def test_backoff_on_429_and_5xx_then_success():
    slept = []
    page = FakePage([429, 503, 200])
    r = hp.fetch_one(page, "NCT1", sleep=slept.append)
    assert r["v0_status"] == "200" and page.calls == 3
    assert slept == [config.HISTORY_PULL_BACKOFF_S] * 2


def test_gives_up_after_max_retries_and_records_status():
    page = FakePage([500] * (config.HISTORY_PULL_MAX_RETRIES + 1))
    r = hp.fetch_one(page, "NCT1", sleep=lambda s: None)
    assert r["v0_status"] == "500" and page.calls == config.HISTORY_PULL_MAX_RETRIES + 1
    assert "after 3 retries" in r["error"]


def test_403_stops_the_run():
    with pytest.raises(hp.RouteChanged):
        hp.fetch_one(FakePage([403]), "NCT1", sleep=lambda s: None)


def test_404_recorded_without_retry():
    page = FakePage([404])
    r = hp.fetch_one(page, "NCT1", sleep=lambda s: None)
    assert r["v0_status"] == "404" and page.calls == 1


def test_delay_floor_is_one_second():
    assert config.HISTORY_PULL_MIN_DELAY_S >= 1.0 and config.HISTORY_PULL_DELAY_S >= config.HISTORY_PULL_MIN_DELAY_S
