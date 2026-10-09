"""SQLite storage layer: connect, migrate, insert helpers, read helpers, provenance checks.

Contract
    * Every stored fact can be traced to its source. Each fact table (``PROVENANCE_TABLES``)
      carries ``document_id, locator, extraction_run_id, confidence, status``; the URL, hash,
      retrieval time, filing date and report date come from joining ``documents`` to
      ``filings`` (see ``provenance`` and the ``v_<table>_provenance`` views). ``status`` is
      exactly one of ``STATUSES`` and ``confidence`` lies in [0, 1] (both enforced by CHECKs).
    * Missing values are stored as NULL (NaN, NaT, None, pd.NA all become NULL). Nothing is
      defaulted to a plausible-looking value; ``status`` has no default, and the only
      timestamp the helpers fill in is ``started_at`` in ``start_run`` (the moment it runs).
    * The schema is versioned: ``schema.sql`` is applied once by ``migrate`` (one transaction)
      and its version is recorded in ``schema_migrations``. ``schema.sql`` is read from next
      to this file, so it does not depend on package-data settings in ``pyproject.toml``.
    * Mapping helpers (``add_*``, ``start_run``) take a mapping and/or keyword arguments whose
      keys are column names. An unknown key raises ``KeyError`` naming it, so a typo cannot
      silently drop a value. A missing optional key is NULL; a missing required key raises
      ``KeyError``. Upserting helpers (``add_source``, ``add_fund``, ``add_share_class``,
      ``add_filing``, ``add_document``) overwrite a stored column only with a non-NULL value.
    * Helpers never commit. Call ``conn.commit()`` (or use ``with conn:``) when done. The
      multi-row helpers (``add_snapshot``, ``add_fund_returns``, ``add_fund_flows``) are
      all-or-nothing: on any error none of their rows remain.
    * ``check_provenance`` is the acceptance test for an ingest: after loading a filing every
      count must be zero.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

STATUSES = ("disclosed", "derived", "assumed", "unknown")
SCHEMES = ("cusip", "isin", "ticker", "figi", "lei", "other")
PROVENANCE_TABLES = ("constraints", "holding_snapshots", "holdings", "prices", "corporate_actions",
                     "fund_returns", "fund_flows", "esg_observations", "benchmark_constituents",
                     "observed_holding_changes")

RETURNS_LOCATOR = "fundInfo/returnInfo/monthlyTotReturns"
FLOWS_LOCATOR = "fundInfo/returnInfo/mon[1-3]Flow"
SNAPSHOT_LOCATOR = "fundInfo"

_LEI = re.compile(r"[A-Z0-9]{20}")

_SOURCE = ("name", "licence", "may_redistribute", "terms_url", "terms_checked_on")
_FUND = ("fund_id", "name", "registrant_cik", "registrant_name", "adviser", "asset_class", "currency",
         "fiscal_year_end", "inception_date", "end_date", "benchmark_id")
_SHARE_CLASS = ("class_id", "fund_id", "ticker", "isin", "cusip", "name")
_FILING = ("accession", "cik", "form_type", "filing_date", "report_date", "is_amendment", "supersedes")
_DOCUMENT = ("filing_id", "url", "sha256", "retrieved_at", "http_status", "etag", "media_type",
             "archive_path", "source_id")
_EVIDENCE = ("document_id", "locator", "evidence_text", "char_start", "char_end")
_RUN = ("method", "tool_version", "model", "prompt_hash", "started_at", "note")
_MANDATE = ("mandate_version_id", "fund_id", "effective_from", "effective_to", "effective_date_basis",
            "review_state", "reviewer", "note")
_PROV = ("document_id", "locator", "extraction_run_id", "confidence", "status")
_CONSTRAINT = ("constraint_id", "mandate_version_id", "metric", "aggregation", "operator", "value_json",
               "unit", "reference", "benchmark_id", "scope", "group_by", "group_values_json",
               "weight_basis", "constraint_type", "modality_terms_json", "observability",
               "effective_from", "effective_to", "review_state", "compile_target", "note",
               "evidence_id") + _PROV
# Python-object spellings accepted by add_constraint and returned by get_constraints.
_JSON_FIELDS = {"value": "value_json", "group_values": "group_values_json",
                "modality_terms": "modality_terms_json"}

# Columns accepted in the DataFrames given to add_snapshot / add_fund_returns / add_fund_flows.
_HOLDING_FRAME = ("row_index", "issuer_name", "issuer_lei", "title", "cusip", "isin", "ticker", "balance",
                  "units", "currency", "value_usd", "pct_net_assets", "payoff_profile", "asset_category",
                  "issuer_category", "country", "is_restricted", "fair_value_level", "is_loaned",
                  "security_key")
_RETURN_FRAME = ("class_id", "month_end", "return_pct")
_FLOW_FRAME = ("month_end", "sales", "reinvestment", "redemption")

_HOLDING_COLUMNS = ["holding_id", "snapshot_id", "row_index", "security_id", "security_key", "issuer_name",
                    "title", "shares", "units", "market_value", "pct_net_assets", "payoff_profile",
                    "asset_category", "issuer_category", "country", "currency", "fair_value_level",
                    "is_restricted", "is_loaned", "locator", "confidence", "status"]
_HOLDING_FLOATS = ("shares", "market_value", "pct_net_assets", "confidence")
_HOLDING_INTS = ("security_id", "is_restricted", "is_loaned")


# ---------------------------------------------------------------- connection and migration
def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and ``sqlite3.Row`` rows."""
    conn = sqlite3.connect(path if isinstance(path, str) else str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        conn.close()
        raise RuntimeError("SQLite could not enforce foreign keys")
    return conn


def _applied_version(conn: sqlite3.Connection) -> int:
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'")
    if has.fetchone() is None:
        return 0
    v = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    return 0 if v is None else int(v)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply ``schema.sql`` if ``schema_migrations`` is absent or empty, record the version, and
    return the current version. Idempotent. The schema and its version row are applied in one
    transaction (any open transaction on ``conn`` is committed first). A database written by
    a newer schema version raises ``RuntimeError``."""
    version = _applied_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema version {version} is newer than this code ({SCHEMA_VERSION})")
    if version == SCHEMA_VERSION:
        return version
    record = ("INSERT INTO schema_migrations (version, applied_at, description) VALUES "
              f"({SCHEMA_VERSION}, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), 'initial schema');")
    try:
        conn.executescript("BEGIN;\n" + SCHEMA_PATH.read_text(encoding="utf-8") + "\n" + record + "\nCOMMIT;")
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    return SCHEMA_VERSION


def open_db(path: str | Path = ":memory:") -> sqlite3.Connection:
    """``connect`` then ``migrate``."""
    conn = connect(path)
    migrate(conn)
    return conn


# ---------------------------------------------------------------- value coercion
def _isnull(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    return isinstance(v, (float, np.floating)) and math.isnan(v)


def _py(v: Any) -> Any:
    """A value sqlite3 accepts: NaN / NaT / None / pd.NA -> None, numpy scalars -> Python,
    dates and timestamps -> ISO-8601 text."""
    if _isnull(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


def _text(v: Any) -> str | None:
    v = _py(v)
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))  # a "1" stored in a float column next to NaN
    v = str(v).strip()
    return v or None


def _day(v: Any) -> str | None:
    """A calendar date as YYYY-MM-DD text (timestamps lose their time of day)."""
    if not _isnull(v) and isinstance(v, datetime):
        return v.date().isoformat()
    return _text(v)


def _num(v: Any) -> float | None:
    v = _py(v)
    return None if v is None else float(v)


def _flag(v: Any) -> int | None:
    v = _py(v)
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("y", "yes", "true", "1"):
            return 1
        if s in ("n", "no", "false", "0"):
            return 0
        raise ValueError(f"cannot read {v!r} as a yes/no flag")
    if v in (0, 1):
        return int(v)
    raise ValueError(f"cannot read {v!r} as a yes/no flag")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _take(table: str, row: Mapping[str, Any] | None, fields: Mapping[str, Any],
          allowed: tuple[str, ...], required: tuple[str, ...] = ()) -> dict[str, Any]:
    """Merge ``row`` and keyword ``fields``; return every allowed column (None if absent)."""
    if row is not None and not isinstance(row, Mapping):
        raise TypeError(f"{table}: expected a mapping, got {type(row).__name__}")
    merged = {**(row or {}), **fields}
    bad = [k for k in merged if k not in allowed]
    if bad:
        raise KeyError(f"unknown key(s) for {table}: {', '.join(map(repr, bad))}; allowed: {', '.join(allowed)}")
    out = {c: _py(merged.get(c)) for c in allowed}
    missing = [c for c in required if out[c] is None]
    if missing:
        raise KeyError(f"missing required key(s) for {table}: {', '.join(map(repr, missing))}")
    return out


def _check_frame(name: str, df: pd.DataFrame, allowed: tuple[str, ...], required: tuple[str, ...]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name}: expected a DataFrame, got {type(df).__name__}")
    bad = [c for c in df.columns if c not in allowed]
    if bad:
        raise KeyError(f"unknown column(s) for {name}: {', '.join(map(repr, bad))}; allowed: {', '.join(allowed)}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required column(s) for {name}: {', '.join(map(repr, missing))}")


def _insert(conn: sqlite3.Connection, table: str, vals: Mapping[str, Any], pk: str) -> Any:
    given = {c: v for c, v in vals.items() if v is not None}  # absent columns take the schema default
    cur = conn.execute(f"INSERT INTO {table} ({', '.join(given)}) VALUES ({', '.join('?' * len(given))})",
                       list(given.values()))
    return given[pk] if pk in given else cur.lastrowid


def _upsert(conn: sqlite3.Connection, table: str, vals: Mapping[str, Any], key: tuple[str, ...],
            pk: str, keep: tuple[str, ...] = ()) -> Any:
    """Insert, or update the row with the same ``key`` using only the non-NULL values given
    (columns in ``keep`` are never updated). Returns the primary key ``pk``."""
    given = {c: v for c, v in vals.items() if v is not None}
    update = [c for c in given if c not in key and c not in keep]
    action = ("DO UPDATE SET " + ", ".join(f"{c} = excluded.{c}" for c in update)) if update else "DO NOTHING"
    conn.execute(f"INSERT INTO {table} ({', '.join(given)}) VALUES ({', '.join('?' * len(given))}) "
                 f"ON CONFLICT ({', '.join(key)}) {action}", list(given.values()))
    where = " AND ".join(f"{k} = ?" for k in key)
    return conn.execute(f"SELECT {pk} FROM {table} WHERE {where}", [given[k] for k in key]).fetchone()[0]


@contextmanager
def _atomic(conn: sqlite3.Connection) -> Iterator[None]:
    """All-or-nothing block that never commits: nested in the caller's transaction (one is
    opened if none is), undone to its starting point if the block raises."""
    if not conn.in_transaction:
        conn.execute("BEGIN")
    conn.execute("SAVEPOINT fb_atomic")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK TO fb_atomic")
        conn.execute("RELEASE fb_atomic")
        raise
    conn.execute("RELEASE fb_atomic")


def _records(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------- insert helpers
def add_source(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> int:
    """Upsert a data source by ``name``; returns ``source_id``. Required: name."""
    return _upsert(conn, "sources", _take("sources", row, fields, _SOURCE, ("name",)),
                   ("name",), "source_id")


def add_fund(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> str:
    """Upsert a fund by ``fund_id`` (the SEC series id); returns it. Required: fund_id."""
    return _upsert(conn, "funds", _take("funds", row, fields, _FUND, ("fund_id",)),
                   ("fund_id",), "fund_id")


def add_share_class(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> str:
    """Upsert a share class by ``class_id``; returns it. Required: class_id, fund_id."""
    return _upsert(conn, "share_classes", _take("share_classes", row, fields, _SHARE_CLASS,
                                                ("class_id", "fund_id")), ("class_id",), "class_id")


def add_filing(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> int:
    """Upsert a filing by ``accession``; returns ``filing_id``. Required: accession, cik, form_type."""
    return _upsert(conn, "filings", _take("filings", row, fields, _FILING, ("accession", "cik", "form_type")),
                   ("accession",), "filing_id")


def add_document(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> int:
    """Upsert a retrieved document by ``(url, sha256)``; returns ``document_id``. Required: url,
    sha256, retrieved_at. Fetching the same bytes again keeps the first ``retrieved_at``."""
    return _upsert(conn, "documents", _take("documents", row, fields, _DOCUMENT,
                                            ("url", "sha256", "retrieved_at")),
                   ("url", "sha256"), "document_id", keep=("retrieved_at",))


def start_run(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> int:
    """Record an extraction run; returns ``run_id``. Required: method. ``started_at`` defaults
    to the current UTC time."""
    vals = _take("extraction_runs", row, fields, _RUN, ("method",))
    if vals["started_at"] is None:
        vals["started_at"] = _now()
    return _insert(conn, "extraction_runs", vals, "run_id")


def add_evidence(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> int:
    """Store a verbatim passage of a document; returns ``evidence_id``.
    Required: document_id, locator, evidence_text."""
    return _insert(conn, "document_evidence", _take("document_evidence", row, fields, _EVIDENCE,
                                                    ("document_id", "locator", "evidence_text")),
                   "evidence_id")


def add_mandate_version(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /,
                        **fields: Any) -> str:
    """Insert a mandate version (``review_state`` defaults to 'draft'); returns its id.
    Required: mandate_version_id, fund_id."""
    return _insert(conn, "mandate_versions", _take("mandate_versions", row, fields, _MANDATE,
                                                   ("mandate_version_id", "fund_id")), "mandate_version_id")


def _dump(v: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"cannot store {type(o).__name__} as JSON")
    return json.dumps(v, default=default, allow_nan=False)


def add_constraint(conn: sqlite3.Connection, row: Mapping[str, Any] | None = None, /, **fields: Any) -> str:
    """Insert a constraint; returns ``constraint_id``. Keys are the ``constraints`` columns, and
    ``value``, ``group_values`` and ``modality_terms`` may be given as Python objects (stored in
    the ``*_json`` columns). Required: constraint_id, mandate_version_id, metric, operator,
    constraint_type, observability, status."""
    if row is not None and not isinstance(row, Mapping):
        raise TypeError(f"constraints: expected a mapping, got {type(row).__name__}")
    merged = {**(row or {}), **fields}
    objects = {name: merged.pop(name) for name in _JSON_FIELDS if name in merged}
    vals = _take("constraints", merged, {}, _CONSTRAINT + tuple(_JSON_FIELDS),
                 ("constraint_id", "mandate_version_id", "metric", "operator", "constraint_type",
                  "observability", "status"))
    for name in _JSON_FIELDS:
        del vals[name]
    for name, obj in objects.items():
        column = _JSON_FIELDS[name]
        if obj is not None:
            if vals[column] is not None:
                raise ValueError(f"give either {name!r} or {column!r}, not both")
            vals[column] = _dump(obj)
    return _insert(conn, "constraints", vals, "constraint_id")


def get_constraints(conn: sqlite3.Connection, mandate_version_id: str) -> list[dict[str, Any]]:
    """The constraints of one mandate version in insertion order, with the ``*_json`` columns
    decoded into ``value``, ``group_values`` and ``modality_terms`` (None when NULL). Each dict
    can be passed back to ``add_constraint``."""
    cur = conn.execute("SELECT * FROM constraints WHERE mandate_version_id = ? ORDER BY rowid",
                       (mandate_version_id,))
    out = []
    for rec in _records(cur):
        for name, column in _JSON_FIELDS.items():
            raw = rec.pop(column)
            rec[name] = None if raw is None else json.loads(raw)
        out.append(rec)
    return out


def ensure_issuer(conn: sqlite3.Connection, lei: Any, *, name: str | None = None) -> int | None:
    """Get-or-create an issuer by LEI; returns ``issuer_id``. A value that is not a 20-character
    LEI (missing, 'N/A', ...) returns None and stores nothing. A missing name is filled in."""
    lei = _text(lei)
    lei = lei.upper() if lei else None
    if lei is None or not _LEI.fullmatch(lei):
        return None
    conn.execute("INSERT INTO issuers (name, lei) VALUES (?, ?) "
                 "ON CONFLICT (lei) DO UPDATE SET name = COALESCE(name, excluded.name)", (_text(name), lei))
    return conn.execute("SELECT issuer_id FROM issuers WHERE lei = ?", (lei,)).fetchone()[0]


def ensure_security(conn: sqlite3.Connection, security_key: str, *, title: str | None = None,
                    asset_type: str | None = None, currency: str | None = None,
                    identifiers: Mapping[str, str] | None = None, valid_from: str | None = None,
                    issuer_id: int | None = None) -> int:
    """Get-or-create a security by ``security_key`` ("scheme:value"); returns ``security_id``.
    Title, asset type, currency and issuer are filled in only where still NULL. Each entry of
    ``identifiers`` ({scheme: value}, scheme in ``SCHEMES``) is added if not already present;
    ``valid_from`` applies to the identifier rows added by this call."""
    key = _text(security_key)
    if key is None:
        raise ValueError("security_key is empty")
    attrs = (_text(title), _text(asset_type), _text(currency), _py(issuer_id))
    row = conn.execute("SELECT security_id FROM securities WHERE security_key = ?", (key,)).fetchone()
    if row is None:
        sid = conn.execute("INSERT INTO securities (security_key, title, asset_type, currency, issuer_id) "
                           "VALUES (?, ?, ?, ?, ?)", (key, *attrs)).lastrowid
    else:
        sid = row[0]
        conn.execute("UPDATE securities SET title = COALESCE(title, ?), asset_type = COALESCE(asset_type, ?), "
                     "currency = COALESCE(currency, ?), issuer_id = COALESCE(issuer_id, ?) "
                     "WHERE security_id = ?", (*attrs, sid))
    for scheme, value in (identifiers or {}).items():
        if scheme not in SCHEMES:
            raise ValueError(f"unknown identifier scheme {scheme!r}; allowed: {', '.join(SCHEMES)}")
        value = _text(value)
        if value is not None:
            conn.execute("INSERT INTO security_identifiers (security_id, scheme, value, valid_from) "
                         "VALUES (?, ?, ?, ?) ON CONFLICT (security_id, scheme, value) DO NOTHING",
                         (sid, scheme, value, _day(valid_from)))
    return sid


def add_snapshot(conn: sqlite3.Connection, *, fund_id: str, report_date: Any, filing_id: int | None,
                 document_id: int | None, run_id: int | None, net_assets: float | None,
                 total_assets: float | None, total_liabilities: float | None, holdings: pd.DataFrame,
                 status: str = "disclosed", confidence: float | None = 1.0) -> int:
    """Store one reported portfolio and its rows; returns ``snapshot_id``.

    ``holdings`` has the N-PORT parser's columns (``_HOLDING_FRAME``; only ``row_index`` is
    required, an unknown column raises KeyError). ``balance`` is stored as ``shares`` and
    ``value_usd`` as ``market_value``. A row whose ``security_key`` is missing or starts with
    "unknown:" gets ``security_id`` NULL and is still stored; other rows get a security (with
    their cusip / isin / ticker identifiers, and an issuer when ``issuer_lei`` is a valid LEI).
    Each row's locator is ``invstOrSecs/invstOrSec[<row_index + 1>]``. Storing the same
    ``(fund_id, report_date, filing_id)`` again raises ValueError. All-or-nothing."""
    _check_frame("holdings", holdings, _HOLDING_FRAME, ("row_index",))
    report_date = _day(report_date)
    if report_date is None:
        raise ValueError("report_date is missing")
    dup = conn.execute("SELECT snapshot_id FROM holding_snapshots "
                       "WHERE fund_id = ? AND report_date = ? AND filing_id IS ?",
                       (fund_id, report_date, _py(filing_id))).fetchone()
    if dup is not None:
        raise ValueError(f"snapshot {dup[0]} already holds fund {fund_id!r} at {report_date} "
                         f"for filing {filing_id!r}")
    doc, run, conf = _py(document_id), _py(run_id), _num(confidence)
    with _atomic(conn):
        snapshot_id = conn.execute(
            "INSERT INTO holding_snapshots (fund_id, report_date, filing_id, net_assets, total_assets, "
            "total_liabilities, n_rows, document_id, locator, extraction_run_id, confidence, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fund_id, report_date, _py(filing_id), _num(net_assets), _num(total_assets),
             _num(total_liabilities), len(holdings), doc, SNAPSHOT_LOCATOR, run, conf, status)).lastrowid
        rows = []
        for rec in holdings.to_dict("records"):
            if _isnull(rec["row_index"]):
                raise ValueError("row_index is missing in a holdings row")
            idx = int(rec["row_index"])
            rows.append((snapshot_id, idx, _security_id(conn, rec),
                         _text(rec.get("issuer_name")), _text(rec.get("title")), _num(rec.get("balance")),
                         _text(rec.get("units")), _num(rec.get("value_usd")), _num(rec.get("pct_net_assets")),
                         _text(rec.get("payoff_profile")), _text(rec.get("asset_category")),
                         _text(rec.get("issuer_category")), _text(rec.get("country")),
                         _text(rec.get("currency")), _text(rec.get("fair_value_level")),
                         _flag(rec.get("is_restricted")), _flag(rec.get("is_loaned")),
                         doc, f"invstOrSecs/invstOrSec[{idx + 1}]", run, conf, status))
        conn.executemany(
            "INSERT INTO holdings (snapshot_id, row_index, security_id, issuer_name, title, shares, units, "
            "market_value, pct_net_assets, payoff_profile, asset_category, issuer_category, country, "
            "currency, fair_value_level, is_restricted, is_loaned, document_id, locator, "
            "extraction_run_id, confidence, status) VALUES (" + ", ".join("?" * 22) + ")", rows)
    return snapshot_id


def _security_id(conn: sqlite3.Connection, rec: Mapping[str, Any]) -> int | None:
    key = _text(rec.get("security_key"))
    if key is None or key.startswith("unknown:"):
        return None
    issuer_id = ensure_issuer(conn, rec.get("issuer_lei"), name=rec.get("issuer_name"))
    ids = {s: rec.get(s) for s in ("cusip", "isin", "ticker")}
    return ensure_security(conn, key, title=_text(rec.get("title")), asset_type=_text(rec.get("asset_category")),
                           currency=_text(rec.get("currency")), identifiers=ids, issuer_id=issuer_id)


def add_fund_returns(conn: sqlite3.Connection, fund_id: str, monthly_returns: pd.DataFrame,
                     document_id: int | None, run_id: int | None, *, status: str = "disclosed",
                     confidence: float | None = 1.0) -> int:
    """Store monthly NAV total returns; returns the number of rows. ``monthly_returns`` has
    ``class_id, month_end, return_pct`` (percent): ``nav_return`` is ``return_pct / 100`` as a
    fraction, and a NaN return is stored as NULL. All-or-nothing."""
    _check_frame("monthly_returns", monthly_returns, _RETURN_FRAME, _RETURN_FRAME)
    rows = []
    for rec in monthly_returns.to_dict("records"):
        pct = _num(rec["return_pct"])
        rows.append((_text(rec["class_id"]), fund_id, _day(rec["month_end"]), None if pct is None else pct / 100,
                     _py(document_id), RETURNS_LOCATOR, _py(run_id), _num(confidence), status))
    with _atomic(conn):
        conn.executemany("INSERT INTO fund_returns (class_id, fund_id, period_end, nav_return, document_id, "
                         "locator, extraction_run_id, confidence, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         rows)
    return len(rows)


def add_fund_flows(conn: sqlite3.Connection, fund_id: str, flows: pd.DataFrame, document_id: int | None,
                   run_id: int | None, *, status: str = "disclosed",
                   confidence: float | None = 1.0) -> int:
    """Store monthly fund flows; returns the number of rows. ``flows`` has ``month_end, sales,
    reinvestment, redemption`` (stored as ``reinvestments`` and ``redemptions``); NaN is stored
    as NULL. All-or-nothing."""
    _check_frame("flows", flows, _FLOW_FRAME, _FLOW_FRAME)
    rows = [(fund_id, _day(rec["month_end"]), _num(rec["sales"]), _num(rec["redemption"]),
             _num(rec["reinvestment"]), _py(document_id), FLOWS_LOCATOR, _py(run_id), _num(confidence), status)
            for rec in flows.to_dict("records")]
    with _atomic(conn):
        conn.executemany("INSERT INTO fund_flows (fund_id, month_end, sales, redemptions, reinvestments, "
                         "document_id, locator, extraction_run_id, confidence, status) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


# ---------------------------------------------------------------- read helpers
def snapshot_dates(conn: sqlite3.Connection, fund_id: str) -> list[str]:
    """Report dates for which the fund has a stored snapshot, oldest first."""
    cur = conn.execute("SELECT DISTINCT report_date FROM holding_snapshots WHERE fund_id = ? "
                       "ORDER BY report_date", (fund_id,))
    return [r[0] for r in cur.fetchall()]


def load_holdings(conn: sqlite3.Connection, fund_id: str, report_date: Any) -> pd.DataFrame:
    """The holdings of one fund at one report date, ordered by ``row_index``, with the
    ``security_key`` of each row (NULL where the security is unresolved). If several filings
    cover the date, the one with the latest ``filing_date`` is used (a snapshot with no
    filing date ranks last). An absent date gives an empty frame with the same columns."""
    snap = conn.execute(
        "SELECT hs.snapshot_id FROM holding_snapshots hs LEFT JOIN filings f ON f.filing_id = hs.filing_id "
        "WHERE hs.fund_id = ? AND hs.report_date = ? ORDER BY f.filing_date DESC, hs.snapshot_id DESC LIMIT 1",
        (fund_id, _day(report_date))).fetchone()
    rows = [] if snap is None else conn.execute(
        "SELECT h.holding_id, h.snapshot_id, h.row_index, h.security_id, s.security_key, h.issuer_name, "
        "h.title, h.shares, h.units, h.market_value, h.pct_net_assets, h.payoff_profile, h.asset_category, "
        "h.issuer_category, h.country, h.currency, h.fair_value_level, h.is_restricted, h.is_loaned, "
        "h.locator, h.confidence, h.status FROM holdings h "
        "LEFT JOIN securities s ON s.security_id = h.security_id WHERE h.snapshot_id = ? "
        "ORDER BY h.row_index", (snap[0],)).fetchall()
    df = pd.DataFrame([tuple(r) for r in rows], columns=_HOLDING_COLUMNS)
    for c in _HOLDING_FLOATS:
        df[c] = df[c].astype(float)
    for c in _HOLDING_INTS:
        df[c] = df[c].astype("Int64")
    return df


def provenance(conn: sqlite3.Connection, table: str, pk: Any) -> dict[str, Any]:
    """Where one stored row came from, via ``v_<table>_provenance``: its primary key plus
    source_url, document_sha256, retrieved_at, form_type, filing_date, report_date, locator,
    extraction_method, tool_version, confidence and status (NULL where unknown).

    ``pk`` is a single value, a tuple / list in primary-key column order, or a mapping of
    column to value. Raises ValueError for a table with no provenance view and KeyError for
    a row that does not exist."""
    if table not in PROVENANCE_TABLES:
        raise ValueError(f"{table!r} has no provenance view; tables with one: {', '.join(PROVENANCE_TABLES)}")
    info = sorted((r for r in conn.execute(f"PRAGMA table_info({table})").fetchall() if r[5]), key=lambda r: r[5])
    cols = [r[1] for r in info]
    if isinstance(pk, Mapping):
        if set(pk) != set(cols):
            raise ValueError(f"{table} is keyed by {cols}, got {sorted(pk)}")
        vals = [pk[c] for c in cols]
    else:
        vals = list(pk) if isinstance(pk, (tuple, list)) else [pk]
    if len(vals) != len(cols):
        raise ValueError(f"{table} is keyed by {cols}, got {len(vals)} value(s)")
    where = " AND ".join(f"{c} IS ?" for c in cols)
    cur = conn.execute(f"SELECT * FROM v_{table}_provenance WHERE {where}", [_py(v) for v in vals])
    found = _records(cur)
    if not found:
        raise KeyError(f"no {table} row with key {dict(zip(cols, vals))}")
    return found[0]


def check_provenance(conn: sqlite3.Connection) -> dict[str, int]:
    """For each table in ``PROVENANCE_TABLES``, the number of rows whose ``document_id`` is NULL
    or does not resolve to a ``documents`` row with a url and retrieval time. After ingesting a
    filing every value must be 0."""
    return {t: conn.execute(
        f"SELECT COUNT(*) FROM {t} t LEFT JOIN documents d ON d.document_id = t.document_id "
        "WHERE d.document_id IS NULL OR d.url IS NULL OR d.retrieved_at IS NULL").fetchone()[0]
        for t in PROVENANCE_TABLES}
