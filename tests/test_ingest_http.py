"""Offline tests for the SEC client and the raw archive: a scripted transport stands in for the
network and a fake clock advances only through the fake sleep, so nothing waits or connects."""
import gzip
import hashlib
import io
import json
import random
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone
from http.client import HTTPMessage

import pytest

from fairbench.ingest.archive import RawArchive
from fairbench.ingest.http import (ALLOWED_HOSTS, USER_AGENT_ENV, FetchFailed, HostNotAllowed,
                                   MissingUserAgent, SecClient, _AllowedRedirects, _decode,
                                   urllib_transport)

UA = "FairBench Tests tests@example.org"
FILING = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK0000320193.json"
BODY = b"\x00\xff<html>\r\n  not   valid utf-8: \xc3\x28 \r\n</html>\r\n"


class FakeTime:
    """Clock that moves only when something sleeps (or the test advances it by hand)."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Script:
    """Transport that replays (status, headers, body) tuples and records every call."""

    def __init__(self, *responses, time=None, latency=0.0):
        self.responses, self.calls, self.starts = list(responses), [], []
        self.time, self.latency = time, latency

    def __call__(self, url, headers):
        assert self.responses, f"unexpected request to {url}"
        self.calls.append((url, dict(headers)))
        if self.time is not None:
            self.starts.append(self.time.clock())
            self.time.now += self.latency
        return self.responses.pop(0)


class MaxRng(random.Random):
    """Full jitter pinned to its upper bound, so delays are exactly base * 2**attempt."""

    def uniform(self, a, b):
        return b


def ok(body=BODY, **headers):
    return 200, headers, body


def make(tmp_path, *responses, latency=0.0, **kw):
    t = FakeTime()
    transport = Script(*responses, time=t, latency=latency)
    client = SecClient(RawArchive(tmp_path / "raw"), user_agent=UA, transport=transport,
                       sleep=t.sleep, clock=t.clock, rng=MaxRng(), **kw)
    return client, transport, t


# ---------------------------------------------------------------- user agent
def test_missing_user_agent_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)
    with pytest.raises(MissingUserAgent) as exc:
        SecClient(RawArchive(tmp_path), transport=Script())
    assert USER_AGENT_ENV in str(exc.value) and "contact@example.org" in str(exc.value)


@pytest.mark.parametrize("ua", ["FairBench", "   ", "FairBench contact.example.org"])
def test_user_agent_without_email_raises(tmp_path, monkeypatch, ua):
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)
    with pytest.raises(MissingUserAgent):
        SecClient(RawArchive(tmp_path), user_agent=ua, transport=Script())


def test_user_agent_taken_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(USER_AGENT_ENV, "Env Project env@example.org")
    transport = Script(ok())
    SecClient(RawArchive(tmp_path), transport=transport).get(SUBMISSIONS)
    headers = transport.calls[0][1]
    assert headers["User-Agent"] == "Env Project env@example.org"
    assert headers["Accept-Encoding"] == "gzip, deflate"


def test_real_transport_is_the_default(tmp_path):
    client = SecClient(RawArchive(tmp_path), user_agent=UA)
    assert client._transport is urllib_transport


# ---------------------------------------------------------------- hosts
@pytest.mark.parametrize("url", [
    "https://example.com/Archives/edgar/data/1/x.htm",
    "http://www.sec.gov/Archives/edgar/data/1/x.htm",
    "ftp://www.sec.gov/x",
    "https://www.sec.gov.evil.example/x",
    "https://evil.example/x?u=https://www.sec.gov/x",
    "https://www.sec.gov@evil.example/x",
    "https://evil.example@www.sec.gov/x",
    "https://evil.example\\@www.sec.gov/x",
    "https://www.sec.gov:8443/x",
    "//www.sec.gov/x",
    "www.sec.gov/x",
    "",
])
def test_non_sec_urls_rejected_before_any_request(tmp_path, url):
    client, transport, _ = make(tmp_path)
    with pytest.raises(HostNotAllowed):
        client.get(url)
    assert transport.calls == [] and client.requests_made == 0


def test_allowed_hosts_are_exactly_the_two_sec_hosts():
    assert ALLOWED_HOSTS == {"www.sec.gov", "data.sec.gov"}


def test_redirects_are_followed_only_to_allowed_hosts():
    handler = _AllowedRedirects()
    req = urllib.request.Request(SUBMISSIONS)
    away = handler.redirect_request(req, None, 302, "Found", {}, "https://evil.example/x")
    downgrade = handler.redirect_request(req, None, 302, "Found", {}, "http://data.sec.gov/x")
    within = handler.redirect_request(req, None, 301, "Moved", {}, "https://www.sec.gov/x")
    assert away is None and downgrade is None
    assert within is not None and within.full_url == "https://www.sec.gov/x"


# ---------------------------------------------------------------- archive of bodies
def test_body_is_stored_unmodified(tmp_path):
    headers = {"ETag": '"abc"', "last-modified": "Tue, 01 Oct 2024 10:00:00 GMT",
               "CONTENT-TYPE": "text/html"}
    client, transport, _ = make(tmp_path, ok(BODY, **headers))
    rec = client.get(SUBMISSIONS)
    sha = hashlib.sha256(BODY).hexdigest()
    assert rec.sha256 == sha and rec.size == len(BODY) and rec.status == 200
    assert rec.path == tmp_path / "raw" / "blobs" / sha[:2] / sha
    assert rec.path.read_bytes() == BODY
    assert client.read(SUBMISSIONS, revalidate=False) == BODY
    assert (rec.etag, rec.last_modified, rec.content_type) == (
        '"abc"', "Tue, 01 Oct 2024 10:00:00 GMT", "text/html")
    assert rec.not_modified is False and rec.retrieved_at.endswith("Z")
    datetime.strptime(rec.retrieved_at, "%Y-%m-%dT%H:%M:%SZ")
    (line,) = (tmp_path / "raw" / "index.jsonl").read_text().splitlines()
    assert json.loads(line) == {
        "url": SUBMISSIONS, "retrieved_at": rec.retrieved_at, "status": 200, "sha256": sha,
        "size": len(BODY), "etag": '"abc"', "last_modified": "Tue, 01 Oct 2024 10:00:00 GMT",
        "content_type": "text/html", "not_modified": False}


def test_missing_headers_are_recorded_as_none(tmp_path):
    client, _, _ = make(tmp_path, ok(b"x"))
    rec = client.get(SUBMISSIONS)
    assert (rec.etag, rec.last_modified, rec.content_type) == (None, None, None)


def test_identical_content_is_stored_once(tmp_path):
    other = "https://data.sec.gov/submissions/CIK0000789019.json"
    client, _, _ = make(tmp_path, ok(BODY), ok(BODY))
    a, b = client.get(SUBMISSIONS), client.get(other)
    assert a.sha256 == b.sha256 and a.path == b.path
    blobs = [p for p in (tmp_path / "raw" / "blobs").rglob("*") if p.is_file()]
    assert blobs == [a.path]
    assert [r.url for r in client.archive.records()] == [SUBMISSIONS, other]


def test_gzip_and_deflate_bodies_are_decoded_before_archiving():
    raw = b"<xml>filing</xml>" * 50
    assert _decode({"Content-Encoding": "gzip"}, gzip.compress(raw)) == raw
    assert _decode({"content-encoding": "GZIP"}, gzip.compress(raw)) == raw
    assert _decode({"Content-Encoding": "deflate"}, zlib.compress(raw)) == raw
    assert _decode({"Content-Encoding": "deflate"}, zlib.compress(raw)[2:-4]) == raw  # raw deflate
    assert _decode({}, raw) == raw


def test_real_transport_maps_failures_to_statuses_without_a_network(monkeypatch):
    """urllib's opener is replaced by a fake: no socket is ever opened."""
    def message(**headers):
        m = HTTPMessage()
        for k, v in headers.items():
            m[k.replace("_", "-")] = v
        return m

    class Resp:
        status = 200
        headers = message(Content_Encoding="gzip", ETag='"z"')

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return gzip.compress(BODY)

    seen = {}

    def install(outcome):
        class Opener:
            def open(self, request, timeout=None):
                seen.update(request=request, timeout=timeout)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())

    headers = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    install(Resp())
    status, got, body = urllib_transport(SUBMISSIONS, headers)
    assert (status, body, got["ETag"]) == (200, BODY, '"z"')
    assert seen["timeout"] == 60 and seen["request"].get_header("User-agent") == UA

    denied = urllib.error.HTTPError(SUBMISSIONS, 403, "Forbidden", message(Retry_After="9"),
                                    io.BytesIO(b"denied"))
    install(denied)
    assert urllib_transport(SUBMISSIONS, headers) == (403, {"Retry-After": "9"}, b"denied")

    for failure in (urllib.error.URLError("dns"), TimeoutError(), ConnectionResetError(), EOFError()):
        install(failure)
        assert urllib_transport(SUBMISSIONS, headers) == (599, {}, b"")


# ---------------------------------------------------------------- caching
def test_immutable_filing_is_requested_once(tmp_path):
    client, transport, _ = make(tmp_path, ok(BODY, ETag='"f"'))
    first = client.get(FILING)
    second = client.get(FILING)
    assert second == first and client.read(FILING) == BODY
    assert len(transport.calls) == 1 == client.requests_made
    assert len(client.archive.history(FILING)) == 1


def test_immutable_filing_can_be_revalidated_on_request(tmp_path):
    client, transport, _ = make(tmp_path, ok(BODY, ETag='"f"'), (304, {}, b""))
    first = client.get(FILING)
    again = client.get(FILING, revalidate=True)
    assert again == first
    assert transport.calls[1][1]["If-None-Match"] == '"f"'
    assert [r.not_modified for r in client.archive.history(FILING)] == [False, True]


def test_mutable_url_revalidates_with_validators_and_304_returns_earlier_body(tmp_path):
    lm = "Tue, 01 Oct 2024 10:00:00 GMT"
    client, transport, _ = make(tmp_path, ok(BODY, ETag='"v1"', **{"Last-Modified": lm}),
                                (304, {"ETag": '"v1"'}, b""))
    first = client.get(SUBMISSIONS)
    second = client.get(SUBMISSIONS)
    sent_first, sent_second = transport.calls[0][1], transport.calls[1][1]
    assert "If-None-Match" not in sent_first and "If-Modified-Since" not in sent_first
    assert sent_second["If-None-Match"] == '"v1"' and sent_second["If-Modified-Since"] == lm
    assert second == first and client.archive.read(second) == BODY

    hist = client.archive.history(SUBMISSIONS)
    assert [(r.status, r.not_modified) for r in hist] == [(200, False), (304, True)]
    assert hist[1].sha256 == first.sha256 and hist[1].size == len(BODY)
    assert client.archive.latest(SUBMISSIONS) == first  # the 304 does not become the body


def test_only_validators_the_server_gave_are_sent(tmp_path):
    client, transport, _ = make(tmp_path, ok(BODY, ETag='"only"'), ok(BODY, **{"Last-Modified": "lm"}),
                                ok(b"none"))
    client.get(SUBMISSIONS)
    client.get(SUBMISSIONS)
    client.get(SUBMISSIONS)
    assert [sorted(h for h in c[1] if h.startswith("If-")) for c in transport.calls] == [
        [], ["If-None-Match"], ["If-Modified-Since"]]


def test_changed_mutable_url_stores_new_body_and_latest_follows(tmp_path):
    client, _, _ = make(tmp_path, ok(b"old", ETag='"1"'), ok(b"new", ETag='"2"'))
    client.get(SUBMISSIONS)
    new = client.get(SUBMISSIONS)
    assert client.archive.latest(SUBMISSIONS) == new and client.archive.read(new) == b"new"
    assert len(client.archive.history(SUBMISSIONS)) == 2


def test_revalidate_false_skips_the_request_when_archived(tmp_path):
    client, transport, _ = make(tmp_path, ok(BODY))
    first = client.get(SUBMISSIONS)
    assert client.get(SUBMISSIONS, revalidate=False) == first
    assert len(transport.calls) == 1


def test_revalidate_false_still_fetches_when_nothing_is_archived(tmp_path):
    client, transport, _ = make(tmp_path, ok(BODY))
    client.get(SUBMISSIONS, revalidate=False)
    assert len(transport.calls) == 1


def test_304_to_an_unconditional_request_fails(tmp_path):
    client, _, _ = make(tmp_path, (304, {}, b""))
    with pytest.raises(FetchFailed):
        client.get(SUBMISSIONS)
    assert list(client.archive.records()) == []


# ---------------------------------------------------------------- rate limit
def test_requests_are_spaced_at_least_one_over_max_rps(tmp_path):
    n, rps = 6, 2.0
    client, transport, t = make(tmp_path, *[ok(bytes([i])) for i in range(n)], max_rps=rps)
    for i in range(n):
        client.get(f"https://data.sec.gov/submissions/CIK{i:010d}.json")
    assert client.requests_made == n
    assert sum(t.sleeps) >= (n - 1) / rps - 1e-9
    gaps = [b - a for a, b in zip(transport.starts, transport.starts[1:])]
    assert all(g >= 1 / rps - 1e-9 for g in gaps)


def test_no_extra_sleep_when_requests_are_already_slow(tmp_path):
    client, _, t = make(tmp_path, ok(b"a"), ok(b"b"), ok(b"c"), max_rps=2.0, latency=0.6)
    for i in range(3):
        client.get(f"https://data.sec.gov/submissions/CIK{i:010d}.json")
    assert t.sleeps == []


def test_sustained_rate_at_the_sec_ceiling(tmp_path):
    client, transport, _ = make(tmp_path, *[ok(bytes([i])) for i in range(21)], max_rps=10)
    for i in range(21):
        client.get(f"https://data.sec.gov/submissions/CIK{i:010d}.json")
    elapsed = transport.starts[-1] - transport.starts[0]
    assert elapsed >= 20 / 10 - 1e-9  # 21 requests need at least 2 s, so never above 10/s


@pytest.mark.parametrize("rps", [0, -1, 10.5, 11, 100, float("nan")])
def test_max_rps_outside_sec_limit_rejected(tmp_path, rps):
    with pytest.raises(ValueError):
        SecClient(RawArchive(tmp_path), user_agent=UA, max_rps=rps, transport=Script())


def test_max_rps_at_ceiling_accepted(tmp_path):
    SecClient(RawArchive(tmp_path), user_agent=UA, max_rps=10, transport=Script())


# ---------------------------------------------------------------- retries
@pytest.mark.parametrize("status", [403, 429, 500, 503, 599])
def test_transient_status_is_retried_with_exponential_backoff(tmp_path, status):
    client, transport, t = make(tmp_path, (status, {}, b"busy"), (status, {}, b"busy"), ok(BODY))
    rec = client.get(SUBMISSIONS)
    assert client.archive.read(rec) == BODY
    assert client.requests_made == 3 == len(transport.calls)
    assert t.sleeps == [1.0, 2.0]
    assert [r.status for r in client.archive.records()] == [200]  # failures are not archived


def test_backoff_is_capped_at_60_seconds(tmp_path):
    client, _, t = make(tmp_path, *[(503, {}, b"")] * 9, ok(BODY), max_retries=9)
    client.get(SUBMISSIONS)
    assert t.sleeps == [1, 2, 4, 8, 16, 32, 60, 60, 60]


def test_jitter_is_drawn_from_zero_to_the_backoff_window(tmp_path):
    class Recording(random.Random):
        draws = []

        def uniform(self, a, b):
            self.draws.append((a, b))
            return super().uniform(a, b)

    t = FakeTime()
    rng = Recording(7)
    client = SecClient(RawArchive(tmp_path), user_agent=UA, transport=Script(*[(503, {}, b"")] * 5, ok(BODY)),
                       sleep=t.sleep, clock=t.clock, rng=rng)
    client.get(SUBMISSIONS)
    assert rng.draws == [(0.0, 1.0), (0.0, 2.0), (0.0, 4.0), (0.0, 8.0), (0.0, 16.0)]
    assert sum(t.sleeps) <= 1 + 2 + 4 + 8 + 16 + 5 * 0.5  # backoff plus pacing, nothing more


def test_retry_after_is_a_minimum(tmp_path):
    client, _, t = make(tmp_path, (429, {"Retry-After": "30"}, b""), (503, {"retry-after": "1"}, b""),
                        (503, {}, b""), ok(BODY))
    client.get(SUBMISSIONS)
    assert t.sleeps == [30.0, 2.0, 4.0]  # 30 beats the 1 s window; 1 loses to the 2 s window


def test_non_numeric_retry_after_is_ignored(tmp_path):
    client, _, t = make(tmp_path, (503, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, b""), ok(BODY))
    client.get(SUBMISSIONS)
    assert t.sleeps == [1.0]


def test_gives_up_after_max_retries(tmp_path):
    client, transport, t = make(tmp_path, *[(503, {}, b"")] * 4, max_retries=3)
    with pytest.raises(FetchFailed) as exc:
        client.get(SUBMISSIONS)
    assert exc.value.status == 503 and exc.value.url == SUBMISSIONS
    assert len(transport.calls) == 4 == client.requests_made  # first try + 3 retries
    assert t.sleeps == [1.0, 2.0, 4.0]
    assert list(client.archive.records()) == []


def test_final_403_message_points_at_user_agent_and_rate(tmp_path):
    client, _, _ = make(tmp_path, *[(403, {}, b"<title>Request Rate Threshold Exceeded</title>")] * 3,
                        max_retries=2)
    with pytest.raises(FetchFailed) as exc:
        client.get(SUBMISSIONS)
    assert exc.value.status == 403
    assert "User-Agent" in str(exc.value) and "request rate" in str(exc.value)


def test_max_retries_zero_means_a_single_attempt(tmp_path):
    client, transport, t = make(tmp_path, (503, {}, b""), max_retries=0)
    with pytest.raises(FetchFailed):
        client.get(SUBMISSIONS)
    assert len(transport.calls) == 1 and t.sleeps == []


@pytest.mark.parametrize("status", [404, 400, 401, 301, 410])
def test_other_statuses_fail_without_retry(tmp_path, status):
    client, transport, t = make(tmp_path, (status, {}, b"nope"))
    with pytest.raises(FetchFailed) as exc:
        client.get(FILING)
    assert exc.value.status == status
    assert len(transport.calls) == 1 and t.sleeps == []
    assert list(client.archive.records()) == []


# ---------------------------------------------------------------- RawArchive
def put(arc, url=SUBMISSIONS, body=BODY, **kw):
    return arc.put(url, 200, kw.pop("headers", {}), body, **kw)


def test_archive_root_is_created_lazily(tmp_path):
    arc = RawArchive(tmp_path / "a" / "b")
    assert not (tmp_path / "a").exists()
    assert arc.latest(SUBMISSIONS) is None and list(arc.records()) == []
    put(arc)
    assert (tmp_path / "a" / "b" / "index.jsonl").exists()


def test_archive_persists_across_instances(tmp_path):
    rec = put(RawArchive(tmp_path))
    again = RawArchive(tmp_path)
    assert again.latest(SUBMISSIONS) == rec and again.read(rec) == BODY
    assert [r.url for r in again.records()] == [SUBMISSIONS]


def test_putting_the_same_content_twice_is_a_no_op_for_the_blob(tmp_path):
    arc = RawArchive(tmp_path)
    a, b = put(arc), put(arc)
    assert a.path == b.path and a.path.read_bytes() == BODY
    assert len(arc.history(SUBMISSIONS)) == 2
    assert [p.name for p in (tmp_path / "blobs").rglob("*") if p.is_file()] == [a.sha256]


def test_retrieved_at_is_utc_with_z(tmp_path):
    arc = RawArchive(tmp_path)
    plus2 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2)))
    assert put(arc, retrieved_at=plus2).retrieved_at == "2026-01-02T01:04:05Z"
    assert put(arc, retrieved_at=datetime(2026, 1, 2, 3, 4, 5)).retrieved_at == "2026-01-02T03:04:05Z"


def test_not_modified_record_needs_the_earlier_sha(tmp_path):
    arc = RawArchive(tmp_path)
    first = put(arc)
    with pytest.raises(ValueError):
        arc.put(SUBMISSIONS, 304, {}, b"", not_modified=True)
    with pytest.raises(ValueError):
        arc.put(SUBMISSIONS, 304, {}, b"data", not_modified=True, sha256=first.sha256)
    with pytest.raises(FileNotFoundError):
        arc.put(SUBMISSIONS, 304, {}, b"", not_modified=True, sha256="0" * 64)
    ref = arc.put(SUBMISSIONS, 304, {}, b"", not_modified=True, sha256=first.sha256)
    assert ref.not_modified and ref.sha256 == first.sha256 and arc.read(ref) == BODY
    assert arc.latest(SUBMISSIONS) == first


def test_latest_matches_the_exact_url(tmp_path):
    arc = RawArchive(tmp_path)
    put(arc, SUBMISSIONS, b"a")
    assert arc.latest(SUBMISSIONS + "?x=1") is None
    assert arc.latest(SUBMISSIONS.upper()) is None


def test_index_survives_a_truncated_last_line(tmp_path):
    arc = RawArchive(tmp_path)
    a, b = put(arc, SUBMISSIONS, b"one"), put(arc, FILING, b"two")
    index = tmp_path / "index.jsonl"
    index.write_bytes(index.read_bytes() + b'{"url": "https://data.sec.gov/trunc", "retrieved_a')
    reloaded = RawArchive(tmp_path)
    assert [r.sha256 for r in reloaded.records()] == [a.sha256, b.sha256]
    assert reloaded.skipped_lines == 1
    assert reloaded.latest(SUBMISSIONS) == a


def test_appending_after_a_truncated_line_does_not_corrupt_the_new_record(tmp_path):
    arc = RawArchive(tmp_path)
    put(arc, SUBMISSIONS, b"one")
    index = tmp_path / "index.jsonl"
    index.write_bytes(index.read_bytes() + b'{"url": "cut')
    again = RawArchive(tmp_path)
    c = put(again, FILING, b"three")
    final = RawArchive(tmp_path)
    assert [r.url for r in final.records()] == [SUBMISSIONS, FILING]
    assert final.latest(FILING) == c and final.skipped_lines == 1


def test_index_truncated_mid_character_is_tolerated(tmp_path):
    arc = RawArchive(tmp_path)
    put(arc)
    index = tmp_path / "index.jsonl"
    index.write_bytes(index.read_bytes() + '{"url": "café'.encode()[:-1])
    assert len(list(RawArchive(tmp_path).records())) == 1


def test_read_detects_a_corrupted_blob(tmp_path):
    arc = RawArchive(tmp_path)
    rec = put(arc)
    rec.path.write_bytes(BODY[:-1] + b"X")
    with pytest.raises(ValueError, match="corrupt"):
        arc.read(rec)
    rec.path.write_bytes(b"")
    with pytest.raises(ValueError):
        arc.read(rec)


def test_archive_records_are_immutable(tmp_path):
    rec = put(RawArchive(tmp_path))
    with pytest.raises(Exception):
        rec.sha256 = "0" * 64
