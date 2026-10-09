"""Polite HTTP client for the US SEC (EDGAR), the only hosts FairBench is allowed to contact.

SEC's fair-access policy asks automated clients to declare who they are, stay under 10
requests/second and not download what they already have. ``SecClient`` enforces that, plus
the project's own rules:

* The User-Agent must contain a contact email and has no default. It comes from the
  constructor or ``FAIRBENCH_SEC_USER_AGENT`` ("Project Name contact@example.org").
* Only https URLs on ``ALLOWED_HOSTS`` are fetched, checked before any request and again on
  every redirect. ``max_rps`` is capped at 10, SEC's own limit, and requests of one client are
  spaced at least ``1/max_rps`` seconds apart.
* Every 200 body goes to the ``RawArchive`` unmodified before anything parses it.
* Filings under ``/Archives/edgar/data/`` never change, so one already archived is not
  requested again. Any other URL is revalidated with If-None-Match / If-Modified-Since from the
  archived response; a 304 downloads nothing and is logged as a ``not_modified`` record.
* 403, 429 and 5xx (and network errors, status 599) are retried with exponential backoff and
  full jitter (base 1 s, cap 60 s, a numeric Retry-After is a minimum). SEC answers undeclared
  or too-fast clients with 403 "Request Rate Threshold Exceeded", so 403 is not permanent.
  Any other status (404, ...) fails at once with ``FetchFailed``.

The network is reached only through an injectable ``Transport``, so tests run offline.
"""
from __future__ import annotations

import gzip
import os
import random
import time
import urllib.error
import urllib.request
import zlib
from typing import Callable, Mapping
from urllib.parse import urlsplit

from .archive import ArchiveRecord, RawArchive

USER_AGENT_ENV = "FAIRBENCH_SEC_USER_AGENT"
ALLOWED_HOSTS = {"www.sec.gov", "data.sec.gov"}
MAX_RPS = 10.0  # SEC's published ceiling
IMMUTABLE_PREFIX = "/Archives/edgar/data/"
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
TIMEOUT = 60.0

Transport = Callable[[str, dict[str, str]], tuple[int, dict[str, str], bytes]]


class MissingUserAgent(RuntimeError):
    """No User-Agent with a contact email was given."""


class HostNotAllowed(ValueError):
    """The URL is not https on an SEC host."""


class FetchFailed(RuntimeError):
    def __init__(self, url: str, status: int, detail: str = ""):
        self.url, self.status = url, status
        super().__init__(f"{url}: HTTP {status}{'; ' + detail if detail else ''}")


# ---------------------------------------------------------------- transport
def check_url(url: str) -> str:
    """Return the URL's path if it is https on an allowed host, else raise HostNotAllowed."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise HostNotAllowed(f"unparseable URL {url!r}") from e
    # userinfo, odd ports and backslashes are the usual ways to make a URL read as one host
    # and connect to another.
    if (parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username is not None
            or port not in (None, 443) or "\\" in url):
        raise HostNotAllowed(f"{url!r} is not an https URL on {sorted(ALLOWED_HOSTS)}")
    return parts.path


class _AllowedRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to allowed hosts; others surface as the 3xx status."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            check_url(newurl)
        except HostNotAllowed:
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode(headers: Mapping[str, str], body: bytes) -> bytes:
    encoding = next((v for k, v in headers.items() if k.lower() == "content-encoding"), "")
    encoding = encoding.strip().lower()
    if encoding in ("gzip", "x-gzip"):
        return gzip.decompress(body)
    if encoding == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)  # raw deflate, which some servers send
    return body


def urllib_transport(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
    """The real transport. Never raises for HTTP or network failures: they come back as a
    status (the HTTP code, or 599 for a network error / timeout / undecodable body)."""
    opener = urllib.request.build_opener(_AllowedRedirects)
    request = urllib.request.Request(url, headers=headers)
    try:
        try:
            with opener.open(request, timeout=TIMEOUT) as resp:
                status, resp_headers, body = resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as e:
            status, resp_headers, body = e.code, dict(e.headers.items()), e.read()
            e.close()
        return status, resp_headers, _decode(resp_headers, body)
    except (OSError, EOFError, zlib.error):  # URLError, timeouts, bad gzip are all OSError/EOF/zlib
        return 599, {}, b""


# ---------------------------------------------------------------- client
def _retry_after(headers: Mapping[str, str]) -> float:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:  # an HTTP-date; the exponential delay is used instead
                return 0.0
    return 0.0


def _retryable(status: int) -> bool:
    return status in (403, 429) or 500 <= status <= 599


class SecClient:
    def __init__(self, archive: RawArchive, user_agent: str | None = None, max_rps: float = 2.0,
                 max_retries: int = 5, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, rng: random.Random | None = None):
        ua = (user_agent or os.environ.get(USER_AGENT_ENV) or "").strip()
        if "@" not in ua:
            raise MissingUserAgent(
                f"SEC requires a User-Agent with a contact email. Set {USER_AGENT_ENV} or pass "
                f'user_agent, in the form "Project Name contact@example.org".')
        if not 0 < max_rps <= MAX_RPS:
            raise ValueError(f"max_rps must satisfy 0 < max_rps <= {MAX_RPS:g} (SEC's limit), got {max_rps}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self.archive = archive
        self.user_agent = ua
        self.max_retries = max_retries
        self.requests_made = 0  # transport calls actually made, retries included
        self._min_interval = 1.0 / max_rps
        self._transport = transport or urllib_transport
        self._sleep, self._clock = sleep, clock
        self._rng = rng or random.Random()
        self._last_start: float | None = None

    def get(self, url: str, *, revalidate: bool | None = None) -> ArchiveRecord:
        """Archive and return the record for ``url``.

        ``revalidate=None`` uses the policy in the module docstring; False never requests
        a URL already archived; True always asks the server (conditionally, when it can)."""
        path = check_url(url)
        prior = self.archive.latest(url)
        if prior is not None and (revalidate is False or
                                  (revalidate is None and path.startswith(IMMUTABLE_PREFIX))):
            return prior
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        if prior is not None:
            if prior.etag:
                headers["If-None-Match"] = prior.etag
            if prior.last_modified:
                headers["If-Modified-Since"] = prior.last_modified
        status, resp_headers, body = self._request(url, headers)
        if status == 304:
            if prior is None:
                raise FetchFailed(url, 304, "server answered 304 to an unconditional request")
            self.archive.put(url, 304, resp_headers, b"", not_modified=True, sha256=prior.sha256)
            return prior
        return self.archive.put(url, status, resp_headers, body)

    def read(self, url: str, *, revalidate: bool | None = None) -> bytes:
        return self.archive.read(self.get(url, revalidate=revalidate))

    # ------------------------------------------------------------ internals
    def _pace(self) -> None:
        """Keep consecutive request starts at least 1/max_rps apart."""
        if self._last_start is not None:
            wait = self._last_start + self._min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_start = self._clock()

    def _request(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        """Send with retries; returns only for 200 or 304."""
        for attempt in range(self.max_retries + 1):
            self._pace()
            self.requests_made += 1
            status, resp_headers, body = self._transport(url, headers)
            if status in (200, 304):
                return status, resp_headers, body
            if not _retryable(status):
                raise FetchFailed(url, status)
            if attempt == self.max_retries:
                break
            ceiling = min(BACKOFF_CAP, BACKOFF_BASE * 2 ** attempt)
            self._sleep(max(self._rng.uniform(0.0, ceiling), _retry_after(resp_headers)))
        hint = ("; SEC answers a missing or undeclared User-Agent and too many requests with 403, "
                "so check the User-Agent contact email and the request rate") if status == 403 else ""
        raise FetchFailed(url, status, f"gave up after {self.max_retries} retries{hint}")
