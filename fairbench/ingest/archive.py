"""Content-addressed raw archive: every downloaded body is kept byte-for-byte before parsing.

Layout under ``root``::

    blobs/<sha256[:2]>/<sha256>     the body exactly as received (after transport gzip decoding)
    index.jsonl                     append-only, one JSON object per retrieval

A body is stored once however many times it is retrieved. Each retrieval, including a 304
revalidation that downloaded nothing, adds one index line (``ArchiveRecord``); a 304 record
(``not_modified=True``) carries the sha256 of the earlier body it confirms. The archive never
alters bytes: ``read`` re-hashes what it reads and raises ``ValueError`` if a blob changed on
disk. The index is read lazily and kept in memory; a line that is not a complete record (for
example the tail of a write cut short by a crash) is skipped and counted in ``skipped_lines``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

_FIELDS = ("url", "retrieved_at", "status", "sha256", "size", "etag", "last_modified",
           "content_type", "not_modified")


@dataclass(frozen=True)
class ArchiveRecord:
    """One retrieval. ``root`` is the archive root (not written to the index); it only
    locates the blob for ``path``."""
    url: str
    retrieved_at: str
    status: int
    sha256: str
    size: int
    etag: str | None = None
    last_modified: str | None = None
    content_type: str | None = None
    not_modified: bool = False
    root: Path = field(default=Path("."), compare=False, repr=False)

    @property
    def path(self) -> Path:
        return _blob_path(self.root, self.sha256)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in _FIELDS}


def _blob_path(root: Path, sha256: str) -> Path:
    return root / "blobs" / sha256[:2] / sha256


def _header(headers: Mapping[str, str], name: str) -> str | None:
    name = name.lower()
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _utc_iso(when: datetime | None) -> str:
    when = datetime.now(timezone.utc) if when is None else when
    when = when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


class RawArchive:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.skipped_lines = 0
        self._records: list[ArchiveRecord] | None = None

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    # ------------------------------------------------------------ reading
    def _load(self) -> list[ArchiveRecord]:
        if self._records is None:
            records: list[ArchiveRecord] = []
            self.skipped_lines = 0
            if self.index_path.exists():
                with self.index_path.open(encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.strip():
                            rec = self._parse_line(line)
                            if rec is None:
                                self.skipped_lines += 1
                            else:
                                records.append(rec)
            self._records = records
        return self._records

    def _parse_line(self, line: str) -> ArchiveRecord | None:
        try:
            obj = json.loads(line)
            return ArchiveRecord(**{k: obj[k] for k in _FIELDS}, root=self.root)
        except (ValueError, KeyError, TypeError):
            return None

    def records(self) -> Iterator[ArchiveRecord]:
        return iter(list(self._load()))

    def history(self, url: str) -> list[ArchiveRecord]:
        return [r for r in self._load() if r.url == url]

    def latest(self, url: str) -> ArchiveRecord | None:
        """Most recent record for exactly this url that holds a body (not a 304 pointer)."""
        for r in reversed(self._load()):
            if r.url == url and not r.not_modified and 200 <= r.status < 300:
                return r
        return None

    def read(self, record: ArchiveRecord) -> bytes:
        data = _blob_path(self.root, record.sha256).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != record.sha256:
            raise ValueError(f"archive blob {record.sha256} is corrupted: its bytes hash to {got}")
        return data

    # ------------------------------------------------------------ writing
    def put(self, url: str, status: int, headers: Mapping[str, str], body: bytes,
            retrieved_at: datetime | None = None, not_modified: bool = False,
            sha256: str | None = None) -> ArchiveRecord:
        """Store ``body`` (once) and append one index record.

        For a 304, pass the earlier record's ``sha256`` and ``body=b""``: nothing is stored and
        the new record points at the existing blob."""
        if not_modified and sha256 is None:
            raise ValueError("a not_modified record must name the sha256 of the earlier body")
        if sha256 is None:
            sha256 = hashlib.sha256(body).hexdigest()
            self._write_blob(sha256, body)
            size = len(body)
        else:
            if body:
                raise ValueError("when sha256 is given the body must be empty (it is a reference)")
            size = _blob_path(self.root, sha256).stat().st_size  # FileNotFoundError if absent
        record = ArchiveRecord(
            url=url, retrieved_at=_utc_iso(retrieved_at), status=int(status), sha256=sha256,
            size=size, etag=_header(headers, "etag"), last_modified=_header(headers, "last-modified"),
            content_type=_header(headers, "content-type"), not_modified=not_modified, root=self.root)
        self._append(record)
        return record

    def _write_blob(self, sha256: str, body: bytes) -> None:
        path = _blob_path(self.root, sha256)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{sha256}.{os.getpid()}.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)  # a blob is never visible half-written

    def _append(self, record: ArchiveRecord) -> None:
        records = self._load()
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_json(), sort_keys=True) + "\n"
        # A crash can leave the last line without its newline; start a fresh line so this
        # record is not glued onto the fragment.
        if self.index_path.exists() and self.index_path.stat().st_size > 0:
            with self.index_path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    line = "\n" + line
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        records.append(record)
