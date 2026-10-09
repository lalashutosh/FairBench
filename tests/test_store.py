"""Storage layer: schema versioning, constraints, insert helpers, provenance. All offline."""
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fairbench.store.db import (PROVENANCE_TABLES, SCHEMA_VERSION, add_constraint, add_document, add_evidence,
                                add_filing, add_fund, add_fund_flows, add_fund_returns,
                                add_mandate_version, add_share_class, add_snapshot, add_source,
                                check_provenance, connect, ensure_security, get_constraints,
                                load_holdings, migrate, open_db, provenance, snapshot_dates, start_run)

URL = "https://www.sec.gov/Archives/edgar/data/1/0000000001-24-000001.xml"
SHA = "a" * 64
PROV_COLUMNS = ["source_url", "document_sha256", "retrieved_at", "form_type", "filing_date", "report_date",
                "locator", "extraction_method", "tool_version", "confidence", "status"]


@pytest.fixture
def conn():
    c = open_db(":memory:")
    yield c
    c.close()


def count(conn, table, where="1"):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def make_source(conn):
    """source -> filing -> document -> run -> fund; returns their ids."""
    source_id = add_source(conn, {"name": "SEC EDGAR", "licence": "public domain", "may_redistribute": 1})
    filing_id = add_filing(conn, {"accession": "0000000001-24-000001", "cik": "1", "form_type": "NPORT-P",
                                  "filing_date": "2024-03-01", "report_date": "2024-01-31"})
    document_id = add_document(conn, {"filing_id": filing_id, "url": URL, "sha256": SHA,
                                      "retrieved_at": "2024-03-02T10:00:00Z", "http_status": 200,
                                      "source_id": source_id})
    run_id = start_run(conn, {"method": "xml_parse", "tool_version": "fairbench-0.1"})
    add_fund(conn, {"fund_id": "S000000001", "name": "Test Fund", "registrant_cik": "1"})
    return {"source_id": source_id, "filing_id": filing_id, "document_id": document_id, "run_id": run_id,
            "fund_id": "S000000001"}


def holdings_frame():
    """Five rows in the N-PORT parser's layout: row 2 has an unknown security_key (and an "N/A"
    LEI), row 3 has no reported value."""
    rows = [
        dict(row_index=0, issuer_name="Apple Inc", issuer_lei="HWUPKR0MPOU8FGXBT394", title="Apple Inc",
             cusip="037833100", isin="US0378331005", ticker="AAPL", balance=100.0, units="NS", currency="USD",
             value_usd=15000.0, pct_net_assets=1.5, payoff_profile="Long", asset_category="EC",
             issuer_category="CORP", country="US", is_restricted=False, fair_value_level="1",
             is_loaned=False, security_key="isin:US0378331005"),
        dict(row_index=1, issuer_name="Microsoft Corp", issuer_lei="INR2EJN1ERAN0W5ZP974", title="Microsoft Corp",
             cusip="594918104", isin="US5949181045", ticker="MSFT", balance=50.0, units="NS", currency="USD",
             value_usd=21000.0, pct_net_assets=2.1, payoff_profile="Long", asset_category="EC",
             issuer_category="CORP", country="US", is_restricted=False, fair_value_level="1",
             is_loaned=True, security_key="isin:US5949181045"),
        dict(row_index=2, issuer_name="Mystery Holdings", issuer_lei="N/A", title="Mystery note",
             cusip=None, isin=None, ticker=None, balance=10.0, units="PA", currency="EUR", value_usd=900.0,
             pct_net_assets=0.09, payoff_profile="Long", asset_category="DBT", issuer_category="CORP",
             country="DE", is_restricted=True, fair_value_level="2", is_loaned=False,
             security_key="unknown:abc123"),
        dict(row_index=3, issuer_name="Nestle SA", issuer_lei=None, title="Nestle SA", cusip=None,
             isin="CH0038863350", ticker=None, balance=20.0, units="NS", currency="CHF", value_usd=np.nan,
             pct_net_assets=np.nan, payoff_profile="Long", asset_category="EC", issuer_category="CORP",
             country="CH", is_restricted=False, fair_value_level="1", is_loaned=False,
             security_key="isin:CH0038863350"),
        dict(row_index=4, issuer_name="Treasury", issuer_lei=None, title="UST 4% 2030", cusip="91282CJL6",
             isin=None, ticker=None, balance=1000.0, units="PA", currency="USD", value_usd=990.0,
             pct_net_assets=0.1, payoff_profile="Long", asset_category="UST", issuer_category="UST",
             country="US", is_restricted=False, fair_value_level="2", is_loaned=False,
             security_key="cusip:91282CJL6"),
    ]
    return pd.DataFrame(rows)


def add_demo_snapshot(conn, ids, df=None, report_date="2024-01-31", **kw):
    return add_snapshot(conn, fund_id=ids["fund_id"], report_date=report_date, filing_id=ids["filing_id"],
                        document_id=ids["document_id"], run_id=ids["run_id"], net_assets=1000000.0,
                        total_assets=1010000.0, total_liabilities=10000.0,
                        holdings=holdings_frame() if df is None else df, **kw)


def raw_esg(conn, status="disclosed", confidence=None):
    conn.execute("INSERT INTO esg_observations (subject_type, subject_id, metric, value, status, confidence) "
                 "VALUES ('fund', 'S1', 'carbon', 1.0, ?, ?)", (status, confidence))


# ---------------------------------------------------------------- migration and constraints
def test_migrate_is_idempotent_and_records_version(tmp_path):
    path = tmp_path / "fb.db"
    conn = open_db(path)
    assert migrate(conn) == SCHEMA_VERSION == 1
    rows = conn.execute("SELECT version, applied_at, description FROM schema_migrations").fetchall()
    assert len(rows) == 1 and rows[0]["version"] == 1 and rows[0]["applied_at"].endswith("Z")
    n_objects = count(conn, "sqlite_master")
    add_fund(conn, fund_id="S1")
    conn.commit()
    conn.close()
    conn = open_db(path)  # a second open must not touch the data or the schema
    assert migrate(conn) == 1
    assert count(conn, "schema_migrations") == 1 and count(conn, "sqlite_master") == n_objects
    assert count(conn, "funds") == 1
    conn.close()


def test_migrate_reapplies_when_the_version_table_is_empty(conn):
    conn.execute("DELETE FROM schema_migrations")
    conn.commit()
    assert migrate(conn) == 1
    assert count(conn, "schema_migrations") == 1


def test_migrate_refuses_a_newer_database(conn):
    conn.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (99, '2030-01-01T00:00:00Z')")
    with pytest.raises(RuntimeError, match="99"):
        migrate(conn)


def test_connect_enforces_foreign_keys(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connect(":memory:").row_factory is sqlite3.Row
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute("INSERT INTO share_classes (class_id, fund_id) VALUES ('C1', 'S-missing')")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_document(conn, filing_id=12345, url=URL, sha256=SHA, retrieved_at="2024-03-02T10:00:00Z")


def test_status_check_accepts_the_four_labels_and_rejects_others(conn):
    for status in ("disclosed", "derived", "assumed", "unknown"):
        raw_esg(conn, status)
    assert count(conn, "esg_observations") == 4
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        raw_esg(conn, "guessed")
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        raw_esg(conn, None)  # status has no default


def test_confidence_check(conn):
    raw_esg(conn, confidence=0.0)
    raw_esg(conn, confidence=1.0)
    raw_esg(conn, confidence=None)
    for bad in (1.5, -0.1):
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            raw_esg(conn, confidence=bad)
    assert count(conn, "esg_observations") == 3


def test_every_fact_table_has_the_five_provenance_columns_and_a_view(conn):
    assert len(PROVENANCE_TABLES) == 10
    for table in PROVENANCE_TABLES:
        cols = {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")}
        assert {"document_id", "locator", "extraction_run_id", "confidence", "status"} <= set(cols), table
        assert cols["status"]["notnull"] == 1 and cols["status"]["dflt_value"] is None
        pk = [r["name"] for r in sorted(cols.values(), key=lambda r: r["pk"]) if r["pk"]]
        view_cols = [r["name"] for r in conn.execute(f"PRAGMA table_info(v_{table}_provenance)")]
        assert view_cols == pk + PROV_COLUMNS, table


# ---------------------------------------------------------------- helpers
def test_add_helpers_reject_unknown_keys(conn):
    ids = make_source(conn)
    cases = [
        (add_source, {"name": "X"}), (add_fund, {"fund_id": "S2"}),
        (add_share_class, {"class_id": "C1", "fund_id": ids["fund_id"]}),
        (add_filing, {"accession": "A", "cik": "1", "form_type": "NPORT-P"}),
        (add_document, {"url": "u", "sha256": "s", "retrieved_at": "t"}),
        (start_run, {"method": "manual"}),
        (add_evidence, {"document_id": ids["document_id"], "locator": "p1", "evidence_text": "t"}),
        (add_mandate_version, {"mandate_version_id": "M1", "fund_id": ids["fund_id"]}),
        (add_constraint, {"constraint_id": "K1", "mandate_version_id": "M1", "metric": "m", "operator": "<=",
                          "constraint_type": "hard", "observability": "observable", "status": "disclosed"}),
    ]
    add_mandate_version(conn, mandate_version_id="M1", fund_id=ids["fund_id"])
    for helper, ok in cases:
        before = [count(conn, t) for t in ("sources", "funds", "share_classes", "filings", "documents",
                                           "extraction_runs", "document_evidence", "constraints")]
        with pytest.raises(KeyError, match="typo_key"):
            helper(conn, {**ok, "typo_key": 1})
        with pytest.raises(KeyError, match="typo_key"):
            helper(conn, ok, typo_key=1)
        after = [count(conn, t) for t in ("sources", "funds", "share_classes", "filings", "documents",
                                          "extraction_runs", "document_evidence", "constraints")]
        assert before == after, helper.__name__
    with pytest.raises(KeyError, match="shiares"):
        add_snapshot(conn, fund_id=ids["fund_id"], report_date="2024-01-31", filing_id=None, document_id=None,
                     run_id=None, net_assets=None, total_assets=None, total_liabilities=None,
                     holdings=holdings_frame().rename(columns={"balance": "shiares"}))
    assert count(conn, "holding_snapshots") == 0
    with pytest.raises(KeyError, match="amount"):
        add_fund_flows(conn, ids["fund_id"], pd.DataFrame({"month_end": ["2024-01-31"], "amount": [1.0]}),
                       ids["document_id"], ids["run_id"])


def test_add_helpers_require_their_required_keys(conn):
    with pytest.raises(KeyError, match="accession"):
        add_filing(conn, cik="1", form_type="NPORT-P")
    with pytest.raises(KeyError, match="url"):
        add_document(conn, sha256=SHA, retrieved_at="t")
    with pytest.raises(KeyError, match="method"):
        start_run(conn, tool_version="x")
    with pytest.raises(KeyError, match="status"):  # no default status for a constraint
        add_constraint(conn, constraint_id="K", mandate_version_id="M", metric="m", operator="<=",
                       constraint_type="hard", observability="observable")


def test_upserts_return_the_same_key_and_never_null_out_a_column(conn):
    assert add_fund(conn, fund_id="S1", name="A", adviser="Adviser Co") == "S1"
    assert add_fund(conn, fund_id="S1", name="B", adviser=None, currency=np.nan) == "S1"
    row = conn.execute("SELECT name, adviser, currency FROM funds").fetchone()
    assert (row["name"], row["adviser"], row["currency"]) == ("B", "Adviser Co", None)
    assert count(conn, "funds") == 1

    f1 = add_filing(conn, accession="A-1", cik="1", form_type="NPORT-P", filing_date="2024-03-01")
    f2 = add_filing(conn, accession="A-1", cik="1", form_type="NPORT-P", report_date="2024-01-31")
    assert f1 == f2 and count(conn, "filings") == 1
    row = conn.execute("SELECT * FROM filings").fetchone()
    assert (row["filing_date"], row["report_date"], row["is_amendment"]) == ("2024-03-01", "2024-01-31", 0)

    d1 = add_document(conn, url=URL, sha256=SHA, retrieved_at="2024-03-02T10:00:00Z", http_status=np.int64(200))
    d2 = add_document(conn, url=URL, sha256=SHA, retrieved_at="2024-05-01T00:00:00Z", etag="abc")
    d3 = add_document(conn, url=URL, sha256="b" * 64, retrieved_at="2024-05-01T00:00:00Z")
    assert d1 == d2 != d3 and count(conn, "documents") == 2
    row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (d1,)).fetchone()
    assert (row["retrieved_at"], row["etag"], row["http_status"]) == ("2024-03-02T10:00:00Z", "abc", 200)

    assert add_source(conn, name="S") == add_source(conn, name="S", licence="L")
    assert conn.execute("SELECT licence, may_redistribute FROM sources").fetchone()[:] == ("L", None)
    add_share_class(conn, class_id="C1", fund_id="S1", ticker="AAA")
    assert add_share_class(conn, class_id="C1", fund_id="S1", isin="US0000000000") == "C1"
    assert conn.execute("SELECT ticker, isin FROM share_classes").fetchone()[:] == ("AAA", "US0000000000")


def test_start_run_defaults_only_the_start_time(conn):
    run = start_run(conn, method="llm", model="m", prompt_hash="h")
    row = conn.execute("SELECT * FROM extraction_runs WHERE run_id = ?", (run,)).fetchone()
    assert row["started_at"].endswith("Z") and row["tool_version"] is None and row["note"] is None
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        start_run(conn, method="vibes")


def test_ensure_security_is_get_or_create_and_accumulates_identifiers(conn):
    a = ensure_security(conn, "isin:US0378331005", title="Apple Inc", identifiers={"isin": "US0378331005"})
    b = ensure_security(conn, "isin:US0378331005", asset_type="EC", currency="USD",
                        identifiers={"isin": "US0378331005", "cusip": "037833100", "ticker": "AAPL", "figi": None},
                        valid_from="2024-01-31")
    c = ensure_security(conn, "isin:US0378331005", title="Something else", identifiers={"ticker": "AAPL"})
    assert a == b == c and count(conn, "securities") == 1
    row = conn.execute("SELECT * FROM securities").fetchone()
    assert (row["title"], row["asset_type"], row["currency"]) == ("Apple Inc", "EC", "USD")
    ids = conn.execute("SELECT scheme, value, valid_from FROM security_identifiers ORDER BY scheme").fetchall()
    assert [tuple(r) for r in ids] == [("cusip", "037833100", "2024-01-31"), ("isin", "US0378331005", None),
                                       ("ticker", "AAPL", "2024-01-31")]
    assert ensure_security(conn, "cusip:594918104") != a
    with pytest.raises(ValueError, match="scheme"):
        ensure_security(conn, "cusip:1", identifiers={"sedol": "123"})
    with pytest.raises(ValueError):
        ensure_security(conn, " ")


# ---------------------------------------------------------------- snapshots and provenance
def test_full_path_every_holding_traces_to_its_source(conn):
    ids = make_source(conn)
    snapshot_id = add_demo_snapshot(conn, ids, confidence=0.95)
    assert snapshot_dates(conn, ids["fund_id"]) == ["2024-01-31"]
    assert count(conn, "holdings") == 5
    assert count(conn, "holdings", "security_id IS NULL") == 1
    assert count(conn, "holdings", "market_value IS NULL") == 1
    assert count(conn, "holdings", "pct_net_assets IS NULL") == 1
    assert count(conn, "securities") == 4  # the unknown row has no security
    assert count(conn, "issuers") == 2  # "N/A" is not a LEI
    assert conn.execute("SELECT n_rows, net_assets FROM holding_snapshots").fetchone()[:] == (5, 1000000.0)

    holding_ids = [r[0] for r in conn.execute("SELECT holding_id FROM holdings ORDER BY row_index")]
    for row_index, holding_id in enumerate(holding_ids):
        p = provenance(conn, "holdings", holding_id)
        assert p == {
            "holding_id": holding_id, "source_url": URL, "document_sha256": SHA,
            "retrieved_at": "2024-03-02T10:00:00Z", "form_type": "NPORT-P", "filing_date": "2024-03-01",
            "report_date": "2024-01-31", "locator": f"invstOrSecs/invstOrSec[{row_index + 1}]",
            "extraction_method": "xml_parse", "tool_version": "fairbench-0.1", "confidence": 0.95,
            "status": "disclosed"}
    p = provenance(conn, "holding_snapshots", snapshot_id)
    assert p["source_url"] == URL and p["status"] == "disclosed" and p["extraction_method"] == "xml_parse"

    apple = conn.execute("SELECT s.title, s.asset_type, s.currency, i.lei FROM securities s "
                         "JOIN issuers i USING (issuer_id) WHERE security_key = 'isin:US0378331005'").fetchone()
    assert tuple(apple) == ("Apple Inc", "EC", "USD", "HWUPKR0MPOU8FGXBT394")
    h = conn.execute("SELECT * FROM holdings WHERE row_index = 1").fetchone()
    assert (h["shares"], h["units"], h["market_value"], h["is_loaned"], h["is_restricted"]) == \
        (50.0, "NS", 21000.0, 1, 0)

    assert check_provenance(conn) == {t: 0 for t in PROVENANCE_TABLES}
    conn.execute("INSERT INTO holdings (snapshot_id, row_index, status) VALUES (?, 99, 'unknown')", (snapshot_id,))
    bad = check_provenance(conn)
    assert bad["holdings"] == 1 and sum(bad.values()) == 1
    assert provenance(conn, "holdings", conn.execute("SELECT MAX(holding_id) FROM holdings").fetchone()[0]) \
        ["source_url"] is None


def test_provenance_lookup_forms_and_errors(conn):
    ids = make_source(conn)
    add_fund_returns(conn, ids["fund_id"], pd.DataFrame({"class_id": ["C1"], "month_end": ["2024-01-31"],
                                                         "return_pct": [1.0]}), ids["document_id"], ids["run_id"])
    key = ("C1", "2024-01-31", ids["document_id"])
    by_tuple = provenance(conn, "fund_returns", key)
    assert by_tuple == provenance(conn, "fund_returns", list(key))
    assert by_tuple == provenance(conn, "fund_returns", {"class_id": "C1", "period_end": "2024-01-31",
                                                         "document_id": ids["document_id"]})
    assert by_tuple["locator"] == "fundInfo/returnInfo/monthlyTotReturns" and by_tuple["source_url"] == URL
    with pytest.raises(ValueError, match="no provenance view"):
        provenance(conn, "funds", "S000000001")
    with pytest.raises(ValueError, match="no provenance view"):
        provenance(conn, "holdings; DROP TABLE funds", 1)
    with pytest.raises(ValueError):
        provenance(conn, "fund_returns", "C1")
    with pytest.raises(KeyError):
        provenance(conn, "holdings", 12345)


def test_provenance_of_a_row_without_a_document_is_all_null(conn):
    raw_esg(conn)
    p = provenance(conn, "esg_observations", 1)
    assert p["status"] == "disclosed" and p["source_url"] is None and p["retrieved_at"] is None
    assert check_provenance(conn)["esg_observations"] == 1


def test_duplicate_snapshot_raises_and_changes_nothing(conn):
    ids = make_source(conn)
    add_demo_snapshot(conn, ids)
    counts = [count(conn, t) for t in ("holding_snapshots", "holdings", "securities")]
    with pytest.raises(ValueError, match="already"):
        add_demo_snapshot(conn, ids)
    # without a filing the table's UNIQUE cannot see the duplicate (NULLs differ), the helper does
    kw = dict(fund_id=ids["fund_id"], report_date="2024-02-29", filing_id=None, document_id=ids["document_id"],
              run_id=ids["run_id"], net_assets=1.0, total_assets=1.0, total_liabilities=0.0,
              holdings=holdings_frame())
    add_snapshot(conn, **kw)
    with pytest.raises(ValueError, match="already"):
        add_snapshot(conn, **kw)
    assert [count(conn, t) for t in ("holding_snapshots", "holdings", "securities")] == \
        [counts[0] + 1, counts[1] + 5, counts[2]]


def test_failed_snapshot_leaves_no_partial_rows(conn):
    ids = make_source(conn)
    df = holdings_frame()
    df.loc[4, "row_index"] = 0  # clashes with row 0
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        add_demo_snapshot(conn, ids, df)
    bad_status = holdings_frame()
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        add_demo_snapshot(conn, ids, bad_status, status="guessed")
    assert [count(conn, t) for t in ("holding_snapshots", "holdings", "securities", "issuers")] == [0, 0, 0, 0]
    add_demo_snapshot(conn, ids)  # and the same snapshot can still be stored afterwards
    assert count(conn, "holdings") == 5


def test_snapshot_stores_nan_as_null_and_flags_as_integers(conn):
    ids = make_source(conn)
    df = holdings_frame().assign(is_restricted=["Y", "N", None, np.nan, True],
                                 fair_value_level=[1.0, 2.0, np.nan, np.nan, 3.0])
    add_snapshot(conn, fund_id=ids["fund_id"], report_date=pd.Timestamp("2024-01-31"), filing_id=ids["filing_id"],
                 document_id=ids["document_id"], run_id=ids["run_id"], net_assets=np.nan, total_assets=None,
                 total_liabilities=np.float64(3.5), holdings=df)
    snap = conn.execute("SELECT * FROM holding_snapshots").fetchone()
    assert (snap["report_date"], snap["net_assets"], snap["total_assets"], snap["total_liabilities"]) == \
        ("2024-01-31", None, None, 3.5)
    rows = conn.execute("SELECT is_restricted, fair_value_level, market_value FROM holdings "
                        "ORDER BY row_index").fetchall()
    assert [r["is_restricted"] for r in rows] == [1, 0, None, None, 1]
    assert [r["fair_value_level"] for r in rows] == ["1", "2", None, None, "3"]
    assert rows[3]["market_value"] is None
    assert count(conn, "holdings", "market_value = 0") == 0


def test_load_holdings_returns_the_stored_rows(conn):
    ids = make_source(conn)
    add_demo_snapshot(conn, ids)
    df = load_holdings(conn, ids["fund_id"], "2024-01-31")
    assert df["row_index"].tolist() == [0, 1, 2, 3, 4]
    assert df["security_key"].tolist()[:2] == ["isin:US0378331005", "isin:US5949181045"]
    assert pd.isna(df.loc[2, "security_key"]) and pd.isna(df.loc[2, "security_id"])
    assert math.isnan(df.loc[3, "market_value"]) and df.loc[0, "market_value"] == 15000.0
    assert df.loc[1, "is_loaned"] == 1 and df.loc[0, "is_loaned"] == 0
    assert (df["status"] == "disclosed").all()
    empty = load_holdings(conn, ids["fund_id"], "2000-01-01")
    assert empty.empty and list(empty.columns) == list(df.columns)


def test_load_holdings_picks_the_latest_filing(conn):
    ids = make_source(conn)
    later = add_filing(conn, accession="0000000001-24-000009", cik="1", form_type="NPORT-P/A",
                       filing_date="2024-04-15", report_date="2024-01-31", is_amendment=True,
                       supersedes="0000000001-24-000001")
    doc2 = add_document(conn, filing_id=later, url=URL + "2", sha256="c" * 64, retrieved_at="2024-04-16T00:00:00Z")
    amended = holdings_frame().iloc[:2].assign(balance=[111.0, 222.0])
    # stored first, so insertion order cannot explain the choice
    add_snapshot(conn, fund_id=ids["fund_id"], report_date="2024-01-31", filing_id=later, document_id=doc2,
                 run_id=ids["run_id"], net_assets=1.0, total_assets=1.0, total_liabilities=0.0, holdings=amended)
    add_demo_snapshot(conn, ids)
    assert snapshot_dates(conn, ids["fund_id"]) == ["2024-01-31"]
    df = load_holdings(conn, ids["fund_id"], "2024-01-31")
    assert df["shares"].tolist() == [111.0, 222.0]
    assert provenance(conn, "holdings", int(df.loc[0, "holding_id"]))["form_type"] == "NPORT-P/A"


def test_snapshot_dates_are_sorted_and_per_fund(conn):
    ids = make_source(conn)
    add_fund(conn, fund_id="S000000002")
    for d in ("2024-03-31", "2024-01-31", "2024-02-29"):
        add_demo_snapshot(conn, ids, holdings_frame().iloc[:1], report_date=d)
    assert snapshot_dates(conn, ids["fund_id"]) == ["2024-01-31", "2024-02-29", "2024-03-31"]
    assert snapshot_dates(conn, "S000000002") == []


# ---------------------------------------------------------------- returns, flows
def test_fund_returns_are_stored_as_fractions_with_null_for_nan(conn):
    ids = make_source(conn)
    frame = pd.DataFrame({"class_id": ["C1", "C1", "C1"],
                          "month_end": ["2023-11-30", pd.Timestamp("2023-12-31"), "2024-01-31"],
                          "return_pct": [3.25, np.nan, -1.5]})
    assert add_fund_returns(conn, ids["fund_id"], frame, ids["document_id"], ids["run_id"]) == 3
    rows = conn.execute("SELECT period_end, nav_return, market_return, locator, status FROM fund_returns "
                        "ORDER BY period_end").fetchall()
    assert [r["period_end"] for r in rows] == ["2023-11-30", "2023-12-31", "2024-01-31"]
    assert rows[0]["nav_return"] == pytest.approx(0.0325)
    assert rows[1]["nav_return"] is None
    assert rows[2]["nav_return"] == pytest.approx(-0.015)
    assert {r["locator"] for r in rows} == {"fundInfo/returnInfo/monthlyTotReturns"}
    assert {r["market_return"] for r in rows} == {None}
    assert count(conn, "fund_returns", "nav_return = 0") == 0
    with pytest.raises(sqlite3.IntegrityError):  # the same document cannot report a month twice
        add_fund_returns(conn, ids["fund_id"], frame.iloc[:1], ids["document_id"], ids["run_id"])
    assert count(conn, "fund_returns") == 3
    assert check_provenance(conn)["fund_returns"] == 0


def test_fund_flows(conn):
    ids = make_source(conn)
    flows = pd.DataFrame({"month_end": ["2023-12-31", "2024-01-31"], "sales": [100.0, np.nan],
                          "reinvestment": [5.0, 6.0], "redemption": [40.0, 50.0]})
    assert add_fund_flows(conn, ids["fund_id"], flows, ids["document_id"], ids["run_id"]) == 2
    rows = conn.execute("SELECT * FROM fund_flows ORDER BY month_end").fetchall()
    assert (rows[0]["sales"], rows[0]["reinvestments"], rows[0]["redemptions"]) == (100.0, 5.0, 40.0)
    assert rows[1]["sales"] is None and rows[1]["redemptions"] == 50.0
    p = provenance(conn, "fund_flows", (ids["fund_id"], "2024-01-31", ids["document_id"]))
    assert p["source_url"] == URL and p["extraction_method"] == "xml_parse" and p["status"] == "disclosed"


# ---------------------------------------------------------------- constraints and other tables
def test_constraints_json_round_trip(conn):
    ids = make_source(conn)
    add_mandate_version(conn, mandate_version_id="M1", fund_id=ids["fund_id"], effective_from="2023-01-01",
                        effective_date_basis="printed in document")
    assert conn.execute("SELECT review_state FROM mandate_versions").fetchone()[0] == "draft"
    evidence_id = add_evidence(conn, document_id=ids["document_id"], locator="p.12 para 3",
                               evidence_text="No more than 15% in any one sector.", char_start=10, char_end=44)
    base = dict(mandate_version_id="M1", constraint_type="hard", observability="observable", status="derived",
                document_id=ids["document_id"], extraction_run_id=ids["run_id"], confidence=0.8,
                locator="p.12 para 3", evidence_id=evidence_id)
    add_constraint(conn, constraint_id="K1", metric="sector_weight", aggregation="max", operator="<=",
                   value=0.15, unit="fraction", group_by="sector", group_values=["Tech", "Energy"],
                   modality_terms=["no more than"], **base)
    add_constraint(conn, {"constraint_id": "K2", "metric": "holdings", "operator": "between",
                          "value": {"min": 20, "max": np.int64(40)}, **base})
    add_constraint(conn, constraint_id="K3", metric="exclusion", operator="excludes", value=None,
                   group_values=np.array(["tobacco"]), **base)
    add_constraint(conn, constraint_id="K4", metric="flag", operator="==", value=False, **base)
    got = {c["constraint_id"]: c for c in get_constraints(conn, "M1")}
    assert list(got) == ["K1", "K2", "K3", "K4"]
    assert got["K1"]["value"] == 0.15 and got["K1"]["group_values"] == ["Tech", "Energy"]
    assert got["K1"]["modality_terms"] == ["no more than"] and got["K1"]["review_state"] == "draft"
    assert got["K2"]["value"] == {"min": 20, "max": 40} and got["K2"]["group_values"] is None
    assert got["K3"]["value"] is None and got["K3"]["group_values"] == ["tobacco"]
    assert got["K4"]["value"] is False
    assert not any(k.endswith("_json") for k in got["K1"])
    assert conn.execute("SELECT value_json FROM constraints WHERE constraint_id = 'K1'").fetchone()[0] == "0.15"
    # what comes out goes back in
    conn.execute("DELETE FROM constraints WHERE constraint_id = 'K1'")
    add_constraint(conn, got["K1"])
    assert get_constraints(conn, "M1")[-1] == got["K1"]
    assert get_constraints(conn, "nope") == []
    assert provenance(conn, "constraints", "K2")["source_url"] == URL
    assert check_provenance(conn)["constraints"] == 0


def test_constraint_validation(conn):
    ids = make_source(conn)
    add_mandate_version(conn, mandate_version_id="M1", fund_id=ids["fund_id"])
    ok = dict(constraint_id="K1", mandate_version_id="M1", metric="m", operator="<=", status="assumed",
              constraint_type="hard", observability="proxy")
    with pytest.raises(ValueError, match="not both"):
        add_constraint(conn, value=1, value_json="2", **ok)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        add_constraint(conn, **{**ok, "constraint_type": "maybe"})
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        add_constraint(conn, **{**ok, "observability": "visible"})
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        add_constraint(conn, **{**ok, "review_state": "pending"})
    with pytest.raises(ValueError):
        add_constraint(conn, value=float("nan"), **ok)  # NaN is not valid JSON, and is not a value
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_constraint(conn, **{**ok, "mandate_version_id": "M-missing"})
    assert count(conn, "constraints") == 0
    add_constraint(conn, **ok)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        add_constraint(conn, **ok)


def test_esg_observation_records_missing_data_as_null_with_a_reason(conn):
    conn.execute("INSERT INTO esg_observations (subject_type, subject_id, metric, value, provider, data_date, "
                 "missing_reason, status) VALUES ('issuer', 'LEI1', 'carbon_intensity', NULL, 'ProviderX', "
                 "'2024-01-31', 'issuer not covered by provider', 'unknown')")
    row = conn.execute("SELECT value, missing_reason, status FROM esg_observations").fetchone()
    assert tuple(row) == (None, "issuer not covered by provider", "unknown")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):  # a missing value needs its reason
        conn.execute("INSERT INTO esg_observations (subject_type, subject_id, metric, status) "
                     "VALUES ('fund', 'S1', 'esg', 'unknown')")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute("INSERT INTO esg_observations (subject_type, subject_id, metric, value, status) "
                     "VALUES ('country', 'US', 'esg', 1.0, 'disclosed')")
