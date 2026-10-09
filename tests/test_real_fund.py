"""End to end on the synthetic N-PORT example: returns, universe, the period runner, storage."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fairbench.apps.real_fund import (CAVEATS, build_period_case, history_table, keyed_sleeve, run_history,
                                      run_hold_to_end, run_period, single_index_cov, summarise_percentiles)
from fairbench.constraints import PositionBound, WeightRuleSet
from fairbench.ingest.nport import parse_nport
from fairbench.ingest.nport_synth import example_history
from fairbench.ingest.pipeline import ingest_nport_document
from fairbench.ingest.prices import (ChainedReturns, CsvReturns, NportImpliedReturns, nav_period_return,
                                     net_flow_ratio)
from fairbench.store import db


@pytest.fixture(scope="module")
def ex():
    h = example_history(seed=0)
    dates = sorted(h["fund"])
    return dict(meta=h["meta"], dates=dates, raw=h,
                parent={d: parse_nport(h["parent"][d]) for d in dates},
                fund={d: parse_nport(h["fund"][d]) for d in dates})


def _case(ex, i, **kw):
    d = ex["dates"]
    return build_period_case(ex["parent"][d[i]], ex["parent"][d[i + 1]], ex["fund"][d[i]], ex["fund"][d[i + 1]], **kw)


# ----------------------------------------------------------------- returns
def test_implied_returns_match_the_planted_truth(ex):
    for i, truth in enumerate(ex["meta"]["periods"]):
        case = _case(ex, i)
        true_r = np.array([truth["asset_price_returns"][k] for k in case.keys])
        assert case.asset_returns == pytest.approx(true_r, abs=2e-4)      # only share rounding separates them
        assert case.fund_frozen_return == pytest.approx(truth["fund_frozen_price_return"], abs=2e-4)
        assert case.return_kind == "price" and case.return_source == "nport_implied"


def test_split_is_adjusted_not_read_as_a_crash(ex):
    case = _case(ex, 2)                                                    # the 2-for-1 split happens here
    key = ex["meta"]["split"]["key"]
    assert case.coverage["n_split_adjusted"] == 1 and key in case.keys
    truth = ex["meta"]["periods"][2]["asset_price_returns"][key]
    assert case.asset_returns[case.keys.index(key)] == pytest.approx(truth, abs=2e-4)
    assert _case(ex, 1).coverage["n_split_adjusted"] == 0


def test_security_that_left_the_index_has_no_invented_return(ex):
    gone = ex["meta"]["left_index"]["key"]
    drop = _case(ex, 3)
    assert drop.coverage["n_unknown_return"] == 1 and gone not in drop.keys
    assert drop.coverage["n_universe"] == drop.coverage["n_universe_before"] - 1
    assert drop.coverage["benchmark_weight_unknown_return"] > 0
    fill = _case(ex, 3, unknown_returns="benchmark_fill")
    assert gone in fill.keys and fill.coverage["unknown_returns_policy"] == "benchmark_fill"
    assert fill.benchmark.sum() == pytest.approx(1.0) and not np.isnan(fill.asset_returns).any()
    with pytest.raises(ValueError):
        _case(ex, 3, unknown_returns="zero")


def _snap(shares, price):
    return pd.DataFrame({"security_key": [f"cusip:{i}" for i in range(len(shares))], "balance": shares,
                         "value_usd": np.asarray(shares, dtype=float) * np.asarray(price, dtype=float)})


def test_share_count_changes_are_judged_by_what_the_price_did():
    a, b = date(2025, 3, 31), date(2025, 6, 30)
    keys = [f"cusip:{i}" for i in range(7)]
    #          plain  plain  2-for-1  +70% shares  plain  +5%   10x shares, price flat
    shares1 = [100, 100, 200, 170, 100, 105, 1000]
    price1 = [11, 9, 5.6, 10.5, 10, 10, 10]
    src = NportImpliedReturns({a: _snap([100.0] * 7, [10.0] * 7), b: _snap(shares1, price1)})
    r = src.period_returns(keys + ["cusip:zz"], a, b)
    assert r.loc["cusip:0", "ret"] == pytest.approx(0.10) and r.loc["cusip:1", "ret"] == pytest.approx(-0.10)
    assert r.loc["cusip:2", "ret"] == pytest.approx(0.12) and r.loc["cusip:2", "split_adjusted"]          # split: price halved
    # shares up 70% but the price did not fall: an issue or float change, plain return kept and labelled
    assert r.loc["cusip:3", "ret"] == pytest.approx(0.05) and r.loc["cusip:3", "status"] == "assumed"
    assert r.loc["cusip:3", "share_count_changed"] and not r.loc["cusip:3", "split_adjusted"]
    assert r.loc["cusip:5", "ret"] == pytest.approx(0.0) and r.loc["cusip:5", "status"] == "derived"       # 5%: not flagged
    assert r.loc["cusip:6", "status"] == "assumed"                                                         # 10x shares, flat price
    assert np.isnan(r.loc["cusip:zz", "ret"]) and r.loc["cusip:zz", "status"] == "unknown"
    assert set(r.loc[r["ret"].notna(), "kind"]) == {"price"}
    # the strict policy leaves those returns unknown instead
    strict = NportImpliedReturns({a: _snap([100.0] * 7, [10.0] * 7), b: _snap(shares1, price1)}, share_change="void")
    rs = strict.period_returns(keys, a, b)
    assert np.isnan(rs.loc["cusip:3", "ret"]) and "no split explains it" in rs.loc["cusip:3", "note"]
    assert rs.loc["cusip:2", "ret"] == pytest.approx(0.12)
    assert np.isnan(src.period_returns(["cusip:0"], a, date(2025, 9, 30)).loc["cusip:0", "ret"])
    with pytest.raises(ValueError):
        NportImpliedReturns({a: _snap([1.0], [1.0]), b: _snap([1.0], [1.0])},
                            share_change="guess").period_returns(["cusip:0"], a, b)


def test_flow_scaling_does_not_hide_a_split():
    a, b = date(2025, 3, 31), date(2025, 6, 30)
    keys = [f"cusip:{i}" for i in range(9)]
    s0 = np.full(9, 1000.0)
    s1 = s0 * 1.07                                 # the reference fund grew 7% through inflows
    s1[4] *= 3                                     # and one company split 3-for-1
    p1 = np.full(9, 10.0)
    p1[4] = 10.0 * 1.2 / 3
    src = NportImpliedReturns({a: pd.DataFrame({"security_key": keys, "balance": s0, "value_usd": s0 * 10.0}),
                               b: pd.DataFrame({"security_key": keys, "balance": s1, "value_usd": s1 * p1})})
    r = src.period_returns(keys, a, b)
    assert r.loc["cusip:4", "ret"] == pytest.approx(0.2) and r.loc["cusip:4", "split_adjusted"]
    assert r.loc["cusip:0", "ret"] == pytest.approx(0.0) and not r["split_adjusted"].drop("cusip:4").any()


def test_active_fund_prices_only_unchanged_positions_and_sources_chain():
    """A name only the active fund holds: its shares change by trading, so a doubling of
    shares with a falling price is NOT read as a split; the return is left unknown."""
    a, b = date(2025, 3, 31), date(2025, 6, 30)
    idx = [f"cusip:{i}" for i in range(5)]
    parent = lambda px: pd.DataFrame({"security_key": idx, "balance": 100.0, "value_usd": 100.0 * np.asarray(px)})
    fund = lambda sh, px: pd.DataFrame({"security_key": ["cusip:0", "cusip:x", "cusip:y"], "balance": sh,
                                        "value_usd": np.asarray(sh) * np.asarray(px)})
    index_fund = NportImpliedReturns({a: parent([10.0] * 5), b: parent([11.0] * 5)}, name="parent index fund")
    own = NportImpliedReturns({a: fund([50.0, 40.0, 40.0], [10.0, 20.0, 20.0]),
                               b: fund([50.0, 80.0, 41.0], [11.05, 14.0, 22.0])}, name="the fund's own filings",
                              passive=False)
    chain = ChainedReturns([index_fund, own])
    r = chain.period_returns(idx + ["cusip:x", "cusip:y"], a, b)
    assert np.isnan(r.loc["cusip:x", "ret"]) and "cannot be told from a split" in r.loc["cusip:x", "note"]
    assert r.loc["cusip:y", "ret"] == pytest.approx(0.10) and r.loc["cusip:y", "note"].startswith("the fund's own filings")
    assert r.loc["cusip:0", "ret"] == pytest.approx(0.10) and r.loc["cusip:0", "note"].startswith("parent index fund")
    assert chain.max_disagreement == pytest.approx(0.005)            # the two sources price cusip:0 half a point apart


def test_nav_units_are_inferred_from_an_independent_estimate(ex):
    """Some filings report monthly returns as fractions although the form asks for percent."""
    import copy
    rep = copy.copy(ex["fund"][ex["dates"][1]])
    truth = ex["meta"]["periods"][0]["fund_frozen_price_return"]
    as_filed = nav_period_return(rep, reference=truth)
    assert as_filed["units"] == "percent" and as_filed["status"] == "disclosed"
    rep.monthly_returns = rep.monthly_returns.assign(return_pct=rep.monthly_returns["return_pct"] / 100.0)
    blind = nav_period_return(rep)
    assert blind["units"] == "percent" and "not checked" in blind["note"] and abs(blind["nav_return"]) < 0.001
    fixed = nav_period_return(rep, reference=truth)
    assert fixed["units"] == "fraction (inferred)" and fixed["status"] == "derived"
    assert fixed["nav_return"] == pytest.approx(as_filed["nav_return"], abs=1e-6)


def test_csv_returns_adapter(tmp_path):
    f = tmp_path / "ret.csv"
    f.write_text("security_key,start,end,total_return\ncusip:A,2025-03-31,2025-06-30,0.05\n"
                 "cusip:B,2025-03-31,2025-06-30,\ncusip:A,2025-06-30,2025-09-30,0.5\n")
    r = CsvReturns(f).period_returns(["cusip:A", "cusip:B", "cusip:C"], date(2025, 3, 31), date(2025, 6, 30))
    assert r.loc["cusip:A", "ret"] == 0.05 and r.loc["cusip:A", "kind"] == "total"
    assert np.isnan(r.loc["cusip:B", "ret"]) and np.isnan(r.loc["cusip:C", "ret"])


def test_nav_return_and_flows_come_from_the_filing(ex):
    d = ex["dates"]
    nav = nav_period_return(ex["fund"][d[1]])
    truth = ex["meta"]["periods"][0]["fund_frozen_price_return"]
    assert nav["status"] == "disclosed" and nav["class_id"] == "C000000091"
    assert nav["nav_return"] == pytest.approx((1 + truth) * (1 - 0.0005) ** 3 - 1, abs=2e-4)   # fee of 5 bp a month
    assert nav_period_return(ex["fund"][d[0]])["status"] == "unknown"                           # first report: N/A
    assert net_flow_ratio(ex["fund"][d[1]]) < 0 and np.isnan(net_flow_ratio(ex["fund"][d[0]]))


# ------------------------------------------------------------ period runner
def test_keyed_sleeve_reports_what_it_left_out(ex):
    s = keyed_sleeve(ex["fund"][ex["dates"][0]])
    assert len(s) == 15 and s["weight"].sum() == pytest.approx(1.0) and s.attrs["n_unkeyed"] == 0
    assert 0.98 < s.attrs["equity_share_of_net_assets"] < 1.0                                    # the cash sweep is not equity


def test_run_period_reports_every_distribution_with_its_manifest(ex):
    case = _case(ex, 1)
    res = run_period(case, excluded_keys=ex["meta"]["excluded_keys"], n_samples=2000, seed=3)
    labels = [r["label"] for r in res.rows]
    assert labels[:2] == ["D1 uniform subsets, equal weights", "D1 uniform subsets, capped benchmark weights"]
    assert labels[2].startswith("D2 uniform weight grid") and len(labels) == 3
    for r in res.rows:
        for key in ("definition", "sampler", "seed", "n_samples", "acceptance_rate", "log10_n_feasible",
                    "n_violations", "weighting_policy", "effective_sample_size", "median", "p05", "p95",
                    "percentile", "percentile_se", "within_mandate_return_difference"):
            assert key in r, key
        assert r["n_violations"] == 0 and 0 <= r["percentile"] <= 100 and r["p05"] <= r["median"] <= r["p95"]
    assert res.rules == ["15 holdings", "6 securities excluded"]
    assert any("skipped" in n for n in res.notes)                         # no covariance -> D3 is skipped, and says so
    assert res.realised["nav_minus_frozen"] == pytest.approx(-0.0015, abs=5e-4)
    out = res.to_dict()
    assert out["caveats"] == list(CAVEATS) and out["return_kind"] == "price"
    assert "skill" in " ".join(out["caveats"]) and "15 holdings" in res.summary()

    def keys_of(o):
        if isinstance(o, dict):
            return [str(k) for k in o] + [k for v in o.values() for k in keys_of(v)]
        return [k for v in o for k in keys_of(v)] if isinstance(o, list) else []
    assert not [k for k in keys_of(out) if "manager" in k or "skill" in k or "alpha" in k]


def test_exclusions_and_weight_rules_bind(ex):
    case = _case(ex, 1)
    free = run_period(case, n_samples=1500, seed=0)
    ruled = run_period(case, excluded_keys=ex["meta"]["excluded_keys"], n_samples=1500, seed=0)
    assert ruled.rows[0]["log10_n_feasible"] < free.rows[0]["log10_n_feasible"]     # fewer feasible subsets
    # the excluded names were planted to earn more, so the rule lowers the reference median
    assert np.mean([_run_median(ex, i, True) - _run_median(ex, i, False) for i in range(5)]) < 0
    capped = run_period(case, n_samples=1000, seed=0, weight_rules=WeightRuleSet([PositionBound(0, 0.09)]), cap=0.09)
    assert capped.rows[1]["n_violations"] == 0 and "9.00%" in " ".join(capped.notes)


def _run_median(ex, i, with_rule):
    kw = dict(excluded_keys=ex["meta"]["excluded_keys"]) if with_rule else {}
    return run_period(_case(ex, i), n_samples=1500, seed=5, **kw).rows[0]["median"]


def test_fund_holding_an_excluded_name_is_flagged(ex):
    case = _case(ex, 1)
    held = case.keys[int(np.argmax(case.fund_weights))]
    res = run_period(case, excluded_keys=[held], n_samples=500, seed=0)
    assert any("holds 1 excluded securities" in n for n in res.notes)


def test_run_history_uses_only_past_returns_for_the_risk_model(ex):
    results = run_history(ex["parent"], ex["fund"], excluded_keys=ex["meta"]["excluded_keys"], n_samples=1500, seed=1)
    assert len(results) == len(ex["dates"]) - 1
    has_d3 = [any(r["label"].startswith("D3") for r in res.rows) for res in results]
    assert has_d3 == [False, False, False, False, True]                  # needs four completed periods first
    last = results[-1]
    d3 = [r for r in last.rows if r["label"].startswith("D3")]
    assert len(d3) == 3 and all(r["distribution"] == "benchmark_aware" and r["tau"] > 0 for r in d3)
    assert d3[0]["effective_sample_size"] <= d3[2]["effective_sample_size"] <= 1500    # tighter tau, fewer effective draws
    assert any("single index" in n for n in last.notes)
    table = history_table(results)
    assert len(table) == 5 and table["realised_portfolio_percentile"].between(0, 100).all()
    assert list(table["period_end"]) == [str(d) for d in ex["dates"][1:]]


def test_single_index_cov():
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(8)]
    hist = pd.DataFrame(rng.normal(0.02, 0.08, (6, 8)), columns=keys)
    hist.iloc[:, 7] = np.nan                                             # one security with no history
    b = np.full(8, 1 / 8)
    cov, info = single_index_cov(hist, keys, b)
    assert cov.shape == (8, 8) and np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    assert info["n_thin_history"] == 1 and info["n_periods"] == 6
    assert single_index_cov(hist.iloc[:3], keys, b) is None


def test_dates_must_line_up(ex):
    d = ex["dates"]
    with pytest.raises(ValueError):
        build_period_case(ex["parent"][d[0]], ex["parent"][d[1]], ex["fund"][d[1]])


# ------------------------------------------------------------------ storage
def test_ingest_stores_every_row_with_provenance_and_is_idempotent(ex):
    conn = db.open_db()
    d = ex["dates"][0]
    url = "https://www.sec.gov/Archives/edgar/data/2/000000000226000001/primary_doc.xml"
    report, snap = ingest_nport_document(conn, ex["raw"]["fund"][d], url=url, retrieved_at="2026-10-09T00:00:00Z",
                                         accession="0000000002-26-000001", filing_date="2024-05-28")
    assert snap is not None and report.series_id == "S000000002"
    assert all(v == 0 for v in db.check_provenance(conn).values())
    rows = conn.execute("SELECT holding_id FROM holdings WHERE snapshot_id = ?", (snap,)).fetchall()
    assert len(rows) == 16
    prov = db.provenance(conn, "holdings", rows[3][0])
    assert prov["source_url"] == url and prov["filing_date"] == "2024-05-28" and prov["report_date"] == str(d)
    assert prov["locator"] == "invstOrSecs/invstOrSec[4]" and prov["status"] == "disclosed"
    assert prov["extraction_method"] == "xml_parse" and prov["retrieved_at"] == "2026-10-09T00:00:00Z"
    stored = db.load_holdings(conn, "S000000002", str(d))
    assert stored["security_key"].str.startswith("cusip:").sum() == 15           # the cash sweep has no real identifier
    again, snap2 = ingest_nport_document(conn, ex["raw"]["fund"][d], url=url, accession="0000000002-26-000001")
    assert snap2 is None and len(db.snapshot_dates(conn, "S000000002")) == 1


def test_fetch_series_goes_through_the_polite_client(ex, tmp_path):
    from fairbench.ingest.archive import RawArchive
    from fairbench.ingest.edgar import filing_document_url, series_filings_url
    from fairbench.ingest.http import SecClient
    from fairbench.ingest.pipeline import fetch_series_nport

    d0, d1 = ex["dates"][:2]
    accs = {"0000000002-24-000001": d0, "0000000002-24-000002": d1}
    entries = "".join(
        f'<entry><category label="form type" scheme="https://www.sec.gov/" term="NPORT-P"/><content type="text/xml">'
        f"<accession-number>{a}</accession-number><filing-date>{d.isoformat()}</filing-date>"
        f"<filing-type>NPORT-P</filing-type></content></entry>" for a, d in accs.items())
    feed = (f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><company-info><cik>0000000002</cik>'
            f"<conformed-name>Synthetic</conformed-name><fiscal-year-end>1231</fiscal-year-end></company-info>"
            f"{entries}</feed>").encode()
    pages = {series_filings_url("S000000002", "NPORT-P", count=40, start=0): feed}
    pages.update({filing_document_url("0000000002", a): ex["raw"]["fund"][d] for a, d in accs.items()})
    calls = []

    def transport(url, headers):
        calls.append((url, headers["User-Agent"]))
        return (200, {"Content-Type": "application/xml"}, pages[url]) if url in pages else (404, {}, b"")

    client = SecClient(RawArchive(tmp_path / "raw"), user_agent="FairBench tests contact@example.org",
                       transport=transport, sleep=lambda s: None)
    conn = db.open_db()
    done = fetch_series_nport(client, conn, "S000000002")
    assert [r.report_date for r, _ in done] == [d0, d1] and all(s is not None for _, s in done)
    assert all(ua == "FairBench tests contact@example.org" for _, ua in calls) and len(calls) == 3
    assert all(v == 0 for v in db.check_provenance(conn).values())
    path = conn.execute("SELECT archive_path FROM documents LIMIT 1").fetchone()[0]
    assert open(path, "rb").read() == ex["raw"]["fund"][d0]                       # the raw response is preserved
    n = len(calls)
    assert all(s is None for _, s in fetch_series_nport(client, conn, "S000000002"))
    assert len(calls) == n + 1                                                    # only the listing is re-requested


# ------------------------------------------------- hold to the end, and averages
def test_hold_to_end_table(ex):
    t = run_hold_to_end(ex["parent"], ex["fund"], excluded_keys=ex["meta"]["excluded_keys"], n_samples=1500, seed=2)
    n_starts = len(ex["dates"]) - 1
    assert len(t) == n_starts * 4 and set(t["reference"]) == {"any k names", "k names obeying the exclusions"}
    assert set(t["weights"]) == {"equal", "benchmark_capped"} and (t["n_violations"] == 0).all()
    assert t["percentile_1q"].between(0, 100).all() and t["percentile_to_end"].between(0, 100).all()
    last = t[t["start"] == str(ex["dates"][-2])]
    assert (last["quarters_to_end"] == 1).all()
    assert last["percentile_1q"].tolist() == last["percentile_to_end"].tolist()        # one quarter left: same thing
    assert (t["fund_holds_excluded"] == 0).all() and t["fund_weight_in_universe"].to_numpy() == pytest.approx(1.0)
    ruled = t[t["reference"] == "k names obeying the exclusions"]
    assert (ruled["n_eligible"].to_numpy() < t[t["reference"] == "any k names"]["n_eligible"].to_numpy()).all()
    assert ruled["rule_conditioned_shift_1q"].mean() < 0                                 # excluded names were planted to earn more
    first = t.iloc[0]
    assert first["fund_return_1q"] == pytest.approx(ex["meta"]["periods"][0]["fund_frozen_price_return"], abs=2e-4)
    assert first["quarters_to_end"] == n_starts and np.isfinite(first["fund_return_to_end"])

    s = summarise_percentiles(t)
    assert len(s) == 4 and (s["n_periods"] == n_starts).all()
    assert s["se_if_random_1q"].iloc[0] == pytest.approx(28.87 / np.sqrt(n_starts))
    assert ((s["mean_percentile_1q"] - 50) / s["se_if_random_1q"]).to_numpy() == pytest.approx(s["z_1q"].to_numpy())


def test_path_values_drop_unknown_holdings_and_renormalise():
    from fairbench.apps.real_fund import _path_values
    W = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    R = np.array([[0.10, np.nan, 0.00], [0.20, 0.50, -0.10]])
    v = _path_values(W, R)
    # row 0: quarter 1 only asset 0 is known (+10%), asset 1 leaves; quarter 2 everything is in asset 0 (+20%)
    assert v[0] == pytest.approx(1.10 * 1.20)
    # row 1: quarter 1 only asset 2 is known (0%); quarter 2 all in asset 2 (-10%)
    assert v[1] == pytest.approx(1.00 * 0.90)
    assert _path_values(np.array([[1.0, 0.0, 0.0]]), np.array([[np.nan, 0.1, 0.1]]))[0] == pytest.approx(1.0)
