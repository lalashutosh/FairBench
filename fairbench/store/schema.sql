-- FairBench storage schema, version 1.
--
-- Dates are ISO-8601 TEXT. A missing value is NULL, never a stand-in number.
-- Tables marked [P] are fact tables and share five provenance columns:
--     document_id, locator, extraction_run_id, confidence, status
-- The source URL, document hash, retrieval time, filing date and report date are reached
-- through documents -> filings; each [P] table has a view v_<table>_provenance that does
-- the join. Foreign keys are enforced by connect() (PRAGMA foreign_keys = ON).
-- Applied by db.migrate(), which wraps this file in one transaction and records the version.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL,
    description TEXT
);

-- ---------------------------------------------------------------- reference data
CREATE TABLE IF NOT EXISTS sources (
    source_id         INTEGER PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    licence           TEXT,
    may_redistribute  INTEGER CHECK (may_redistribute IN (0, 1)),   -- NULL: not established
    terms_url         TEXT,
    terms_checked_on  TEXT
);

-- funds.benchmark_id (and constraints.benchmark_id) are plain TEXT, not foreign keys, so a
-- fund can be loaded before its benchmark; benchmarks.proxy_fund_id points back at funds.
CREATE TABLE IF NOT EXISTS funds (
    fund_id          TEXT PRIMARY KEY,        -- SEC series id, e.g. S000000856
    name             TEXT,
    registrant_cik   TEXT,
    registrant_name  TEXT,
    adviser          TEXT,
    asset_class      TEXT,
    currency         TEXT,
    fiscal_year_end  TEXT,                    -- MMDD
    inception_date   TEXT,
    end_date         TEXT,
    benchmark_id     TEXT
);

CREATE TABLE IF NOT EXISTS share_classes (
    class_id  TEXT PRIMARY KEY,
    fund_id   TEXT NOT NULL REFERENCES funds (fund_id),
    ticker    TEXT,
    isin      TEXT,
    cusip     TEXT,
    name      TEXT
);

CREATE TABLE IF NOT EXISTS issuers (
    issuer_id  INTEGER PRIMARY KEY,
    name       TEXT,
    lei        TEXT UNIQUE,
    cik        TEXT,
    sic_code   TEXT,
    country    TEXT
);

CREATE TABLE IF NOT EXISTS securities (
    security_id   INTEGER PRIMARY KEY,
    security_key  TEXT NOT NULL UNIQUE,       -- "scheme:value" from the identifier normaliser
    issuer_id     INTEGER REFERENCES issuers (issuer_id),
    title         TEXT,
    asset_type    TEXT,
    currency      TEXT
);

CREATE TABLE IF NOT EXISTS security_identifiers (
    security_id  INTEGER NOT NULL REFERENCES securities (security_id),
    scheme       TEXT NOT NULL CHECK (scheme IN ('cusip', 'isin', 'ticker', 'figi', 'lei', 'other')),
    value        TEXT NOT NULL,
    valid_from   TEXT,
    valid_to     TEXT,
    PRIMARY KEY (security_id, scheme, value)
);

-- ---------------------------------------------------------------- documents and evidence
CREATE TABLE IF NOT EXISTS filings (
    filing_id     INTEGER PRIMARY KEY,
    accession     TEXT NOT NULL UNIQUE,
    cik           TEXT NOT NULL,
    form_type     TEXT NOT NULL,
    filing_date   TEXT,
    report_date   TEXT,
    is_amendment  INTEGER NOT NULL DEFAULT 0 CHECK (is_amendment IN (0, 1)),
    supersedes    TEXT                        -- accession of the filing this one replaces
);

CREATE TABLE IF NOT EXISTS documents (
    document_id   INTEGER PRIMARY KEY,
    filing_id     INTEGER REFERENCES filings (filing_id),   -- NULL: not an SEC filing
    url           TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    http_status   INTEGER,
    etag          TEXT,
    media_type    TEXT,
    archive_path  TEXT,
    source_id     INTEGER REFERENCES sources (source_id),
    UNIQUE (url, sha256)
);

CREATE TABLE IF NOT EXISTS document_evidence (
    evidence_id    INTEGER PRIMARY KEY,
    document_id    INTEGER NOT NULL REFERENCES documents (document_id),
    locator        TEXT NOT NULL,             -- page / table+row / XPath / paragraph
    evidence_text  TEXT NOT NULL,
    char_start     INTEGER,
    char_end       INTEGER
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id        INTEGER PRIMARY KEY,
    method        TEXT NOT NULL CHECK (method IN
                  ('xml_parse', 'xbrl_tag', 'html_parse', 'regex', 'llm', 'manual', 'computed')),
    tool_version  TEXT,
    model         TEXT,
    prompt_hash   TEXT,
    started_at    TEXT NOT NULL,
    note          TEXT
);

-- ---------------------------------------------------------------- mandates
CREATE TABLE IF NOT EXISTS mandate_versions (
    mandate_version_id    TEXT PRIMARY KEY,
    fund_id               TEXT NOT NULL REFERENCES funds (fund_id),
    effective_from        TEXT,
    effective_to          TEXT,
    effective_date_basis  TEXT,               -- e.g. 'printed in document', 'filing effective date'
    review_state          TEXT NOT NULL DEFAULT 'draft'
                          CHECK (review_state IN ('draft', 'reviewed', 'approved', 'rejected')),
    reviewer              TEXT,
    note                  TEXT
);

-- [P]
CREATE TABLE IF NOT EXISTS constraints (
    constraint_id       TEXT PRIMARY KEY,
    mandate_version_id  TEXT NOT NULL REFERENCES mandate_versions (mandate_version_id),
    metric              TEXT NOT NULL,
    aggregation         TEXT,
    operator            TEXT NOT NULL,
    value_json          TEXT,
    unit                TEXT,
    reference           TEXT,
    benchmark_id        TEXT,
    scope               TEXT,
    group_by            TEXT,
    group_values_json   TEXT,
    weight_basis        TEXT,
    constraint_type     TEXT NOT NULL CHECK (constraint_type IN ('hard', 'soft', 'regulatory')),
    modality_terms_json TEXT,
    observability       TEXT NOT NULL CHECK (observability IN ('observable', 'proxy', 'unobservable')),
    effective_from      TEXT,
    effective_to        TEXT,
    review_state        TEXT NOT NULL DEFAULT 'draft'
                        CHECK (review_state IN ('draft', 'reviewed', 'approved', 'rejected')),
    compile_target      TEXT,
    note                TEXT,
    evidence_id         INTEGER REFERENCES document_evidence (evidence_id),
    document_id         INTEGER REFERENCES documents (document_id),
    locator             TEXT,
    extraction_run_id   INTEGER REFERENCES extraction_runs (run_id),
    confidence          REAL CHECK (confidence BETWEEN 0 AND 1),
    status              TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown'))
);

-- ---------------------------------------------------------------- holdings
-- [P]
CREATE TABLE IF NOT EXISTS holding_snapshots (
    snapshot_id        INTEGER PRIMARY KEY,
    fund_id            TEXT NOT NULL REFERENCES funds (fund_id),
    report_date        TEXT NOT NULL,
    filing_id          INTEGER REFERENCES filings (filing_id),
    net_assets         REAL,
    total_assets       REAL,
    total_liabilities  REAL,
    n_rows             INTEGER,
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    UNIQUE (fund_id, report_date, filing_id)
);

-- [P]  market_value is in USD (N-PORT valUSD); currency is the instrument's own currency.
CREATE TABLE IF NOT EXISTS holdings (
    holding_id         INTEGER PRIMARY KEY,
    snapshot_id        INTEGER NOT NULL REFERENCES holding_snapshots (snapshot_id),
    row_index          INTEGER NOT NULL,
    security_id        INTEGER REFERENCES securities (security_id),   -- NULL: identity unresolved
    issuer_name        TEXT,
    title              TEXT,
    shares             REAL,
    units              TEXT,
    market_value       REAL,
    pct_net_assets     REAL,
    payoff_profile     TEXT,
    asset_category     TEXT,
    issuer_category    TEXT,
    country            TEXT,
    currency           TEXT,
    fair_value_level   TEXT,
    is_restricted      INTEGER CHECK (is_restricted IN (0, 1)),
    is_loaned          INTEGER CHECK (is_loaned IN (0, 1)),
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    UNIQUE (snapshot_id, row_index)
);

-- ---------------------------------------------------------------- market data
-- [P]
CREATE TABLE IF NOT EXISTS prices (
    security_id        INTEGER NOT NULL REFERENCES securities (security_id),
    date               TEXT NOT NULL,
    close              REAL,
    total_return_index REAL,
    source_id          INTEGER REFERENCES sources (source_id),
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    PRIMARY KEY (security_id, date, source_id)
);

-- [P]
CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id          INTEGER PRIMARY KEY,
    security_id        INTEGER NOT NULL REFERENCES securities (security_id),
    ex_date            TEXT,
    type               TEXT CHECK (type IN
                       ('dividend', 'split', 'merger', 'spin_off', 'identifier_change', 'delisting', 'other')),
    ratio_or_amount    REAL,
    note               TEXT,
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown'))
);

-- ---------------------------------------------------------------- fund-level facts
-- [P]  A NULL document_id is not deduplicated by the primary key (SQLite allows NULL in a
-- composite key); check_provenance() reports such rows.
CREATE TABLE IF NOT EXISTS fund_returns (
    class_id           TEXT NOT NULL,
    fund_id            TEXT NOT NULL REFERENCES funds (fund_id),
    period_end         TEXT NOT NULL,
    nav_return         REAL,                  -- fraction: 0.0325 = +3.25%
    market_return      REAL,
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    PRIMARY KEY (class_id, period_end, document_id)
);

-- [P]
CREATE TABLE IF NOT EXISTS fund_flows (
    fund_id            TEXT NOT NULL REFERENCES funds (fund_id),
    month_end          TEXT NOT NULL,
    sales              REAL,
    redemptions        REAL,
    reinvestments      REAL,
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    PRIMARY KEY (fund_id, month_end, document_id)
);

-- [P]  A missing value is recorded as value NULL plus a missing_reason, never as 0.
CREATE TABLE IF NOT EXISTS esg_observations (
    observation_id     INTEGER PRIMARY KEY,
    subject_type       TEXT NOT NULL CHECK (subject_type IN ('security', 'issuer', 'fund')),
    subject_id         TEXT NOT NULL,
    metric             TEXT NOT NULL,
    value              REAL,
    unit               TEXT,
    provider           TEXT,
    data_date          TEXT,
    missing_reason     TEXT,
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    CHECK (value IS NOT NULL OR missing_reason IS NOT NULL)
);

-- ---------------------------------------------------------------- benchmarks
CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id   TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    provider       TEXT,
    proxy_fund_id  TEXT REFERENCES funds (fund_id),
    is_proxy       INTEGER NOT NULL DEFAULT 0 CHECK (is_proxy IN (0, 1)),
    note           TEXT
);

-- [P]
CREATE TABLE IF NOT EXISTS benchmark_constituents (
    benchmark_id       TEXT NOT NULL REFERENCES benchmarks (benchmark_id),
    date               TEXT NOT NULL,
    security_id        INTEGER NOT NULL REFERENCES securities (security_id),
    weight             REAL,
    document_id        INTEGER REFERENCES documents (document_id),
    locator            TEXT,
    extraction_run_id  INTEGER REFERENCES extraction_runs (run_id),
    confidence         REAL CHECK (confidence BETWEEN 0 AND 1),
    status             TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown')),
    PRIMARY KEY (benchmark_id, date, security_id)
);

-- ---------------------------------------------------------------- observed changes
-- [P]
CREATE TABLE IF NOT EXISTS observed_holding_changes (
    change_id               INTEGER PRIMARY KEY,
    fund_id                 TEXT NOT NULL REFERENCES funds (fund_id),
    security_id             INTEGER REFERENCES securities (security_id),
    from_date               TEXT NOT NULL,
    to_date                 TEXT NOT NULL,
    change_type             TEXT NOT NULL CHECK (change_type IN
                            ('observed_new_position', 'observed_exit', 'observed_holding_increase',
                             'observed_holding_decrease', 'unchanged', 'observed_sector_change',
                             'observed_exposure_change')),
    shares_delta            REAL,
    weight_delta            REAL,
    drift_component         REAL,
    active_component        REAL,
    split_suspected         INTEGER CHECK (split_suspected IN (0, 1)),
    flow_consistent         INTEGER CHECK (flow_consistent IN (0, 1)),
    decision_observability  TEXT NOT NULL CHECK (decision_observability IN
                            ('direct', 'observed_position_change', 'inferred', 'unavailable')),
    explanation_evidence_id INTEGER REFERENCES document_evidence (evidence_id),
    document_id             INTEGER REFERENCES documents (document_id),
    locator                 TEXT,
    extraction_run_id       INTEGER REFERENCES extraction_runs (run_id),
    confidence              REAL CHECK (confidence BETWEEN 0 AND 1),
    status                  TEXT NOT NULL CHECK (status IN ('disclosed', 'derived', 'assumed', 'unknown'))
);

-- ---------------------------------------------------------------- samples and results
CREATE TABLE IF NOT EXISTS reference_universes (
    universe_id    INTEGER PRIMARY KEY,
    fund_id        TEXT NOT NULL REFERENCES funds (fund_id),
    as_of          TEXT NOT NULL,
    definition     TEXT NOT NULL,
    n_securities   INTEGER,
    array_path     TEXT,
    array_sha256   TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_samples (
    sample_id             INTEGER PRIMARY KEY,
    universe_id           INTEGER NOT NULL REFERENCES reference_universes (universe_id),
    mandate_version_id    TEXT REFERENCES mandate_versions (mandate_version_id),
    distribution          TEXT NOT NULL,
    weighting_policy      TEXT NOT NULL,
    sampler               TEXT NOT NULL,
    seed                  INTEGER,
    n_samples             INTEGER,
    n_feasible_estimate   REAL,
    n_feasible_is_exact   INTEGER CHECK (n_feasible_is_exact IN (0, 1)),
    acceptance_rate       REAL,
    n_violations          INTEGER,
    effective_sample_size REAL,
    array_path            TEXT,
    array_sha256          TEXT,
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attribution_results (
    result_id              INTEGER PRIMARY KEY,
    sample_id              INTEGER NOT NULL REFERENCES portfolio_samples (sample_id),
    period_start           TEXT NOT NULL,
    period_end             TEXT NOT NULL,
    realised_frozen_return REAL,
    realised_nav_return    REAL,
    percentile             REAL,
    percentile_se          REAL,
    median                 REAL,
    p05                    REAL,
    p95                    REAL,
    labels_version         TEXT,
    manifest_json          TEXT
);

CREATE TABLE IF NOT EXISTS validation_runs (
    validation_id       INTEGER PRIMARY KEY,
    sampler             TEXT NOT NULL,
    instance            TEXT NOT NULL,
    distribution        TEXT,
    feasibility_rate    REAL,
    violation_rate      REAL,
    tv                  REAL,
    tv_noise_floor      REAL,
    kl                  REAL,
    kl_is_smoothed      INTEGER CHECK (kl_is_smoothed IN (0, 1)),
    marginal_error_max  REAL,
    marginal_error_rmse REAL,
    return_dist_ks      REAL,
    return_dist_w1      REAL,
    percentile_error    REAL,
    runtime_s           REAL,
    qubits              INTEGER,
    depth               INTEGER,
    two_qubit_gates     INTEGER,
    shots               INTEGER,
    seed                INTEGER,
    created_at          TEXT NOT NULL,
    note                TEXT
);

-- ---------------------------------------------------------------- indexes
CREATE INDEX IF NOT EXISTS idx_share_classes_fund     ON share_classes (fund_id);
CREATE INDEX IF NOT EXISTS idx_securities_issuer      ON securities (issuer_id);
CREATE INDEX IF NOT EXISTS idx_identifiers_lookup     ON security_identifiers (scheme, value);
CREATE INDEX IF NOT EXISTS idx_documents_filing       ON documents (filing_id);
CREATE INDEX IF NOT EXISTS idx_evidence_document      ON document_evidence (document_id);
CREATE INDEX IF NOT EXISTS idx_mandates_fund          ON mandate_versions (fund_id);
CREATE INDEX IF NOT EXISTS idx_constraints_mandate    ON constraints (mandate_version_id);
CREATE INDEX IF NOT EXISTS idx_holdings_security      ON holdings (security_id);
CREATE INDEX IF NOT EXISTS idx_fund_returns_fund      ON fund_returns (fund_id, period_end);
CREATE INDEX IF NOT EXISTS idx_corp_actions_security  ON corporate_actions (security_id, ex_date);
CREATE INDEX IF NOT EXISTS idx_esg_subject            ON esg_observations (subject_type, subject_id, metric);
CREATE INDEX IF NOT EXISTS idx_changes_fund           ON observed_holding_changes (fund_id, from_date, to_date);
CREATE INDEX IF NOT EXISTS idx_universes_fund         ON reference_universes (fund_id, as_of);
CREATE INDEX IF NOT EXISTS idx_samples_universe       ON portfolio_samples (universe_id);
CREATE INDEX IF NOT EXISTS idx_results_sample         ON attribution_results (sample_id);

-- ---------------------------------------------------------------- provenance views
-- One per [P] table: the table's primary key, then where the value came from. LEFT JOINs, so
-- a row with missing provenance still appears (with NULL source columns) instead of vanishing.

CREATE VIEW IF NOT EXISTS v_constraints_provenance AS
SELECT t.constraint_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM constraints t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_holding_snapshots_provenance AS
SELECT t.snapshot_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM holding_snapshots t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_holdings_provenance AS
SELECT t.holding_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM holdings t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_prices_provenance AS
SELECT t.security_id, t.date, t.source_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM prices t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_corporate_actions_provenance AS
SELECT t.action_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM corporate_actions t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_fund_returns_provenance AS
SELECT t.class_id, t.period_end, t.document_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM fund_returns t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_fund_flows_provenance AS
SELECT t.fund_id, t.month_end, t.document_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM fund_flows t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_esg_observations_provenance AS
SELECT t.observation_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM esg_observations t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_benchmark_constituents_provenance AS
SELECT t.benchmark_id, t.date, t.security_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM benchmark_constituents t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;

CREATE VIEW IF NOT EXISTS v_observed_holding_changes_provenance AS
SELECT t.change_id,
       d.url AS source_url, d.sha256 AS document_sha256, d.retrieved_at,
       f.form_type, f.filing_date, f.report_date,
       t.locator, r.method AS extraction_method, r.tool_version, t.confidence, t.status
FROM observed_holding_changes t
LEFT JOIN documents d ON d.document_id = t.document_id
LEFT JOIN filings f ON f.filing_id = d.filing_id
LEFT JOIN extraction_runs r ON r.run_id = t.extraction_run_id;
