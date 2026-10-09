import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import skill_table as st


def test_percentile_bounds_and_ties():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    assert st.percentile_of(null, 0.0) == 0.0
    assert st.percentile_of(null, 10.0) == 100.0
    assert st.percentile_of(null, 2.5) == 50.0
    assert st.percentile_of(null, 2.0) == 100 * (0.25 + 0.5 * 0.25)  # one below, one tie
    assert st.percentile_of(np.ones(10), 1.0) == 50.0


def test_whole_universe_fund_matches_benchmark():
    rng = np.random.default_rng(1)
    r = rng.normal(0.02, 0.1, 60); w = rng.uniform(1, 10, 60)
    vw, ew = st.null_subset_returns(r, w, 60, 50, rng)
    bench = (r * w).sum() / w.sum()
    assert np.allclose(vw, bench)
    assert st.percentile_of(vw, bench) == 50.0
    assert np.allclose(ew, r.mean())


def test_null_mean_right():
    rng = np.random.default_rng(2)
    r = rng.normal(0.01, 0.08, 200); w = np.ones(200)
    vw, ew = st.null_subset_returns(r, w, 25, 4000, rng)
    assert abs(ew.mean() - r.mean()) < 4 * r.std() / np.sqrt(25 * 4000) * 2
    assert abs(vw.mean() - r.mean()) < 0.003
    assert (vw.std() > 0)


def test_russell_builder_price_ratios_and_split():
    rows = []
    for fid, d, key, px, sh in [("P", "t0", "a", 10.0, 100), ("P", "t1", "a", 11.0, 100),
                                ("P", "t0", "b", 50.0, 10), ("P", "t1", "b", 40.0, 10),
                                ("P", "t0", "c", 20.0, 10), ("P", "t1", "c", 2.2, 100),  # 10:1 split, +10%
                                ("P", "t0", "only0", 5.0, 1),
                                ("OWN", "t0", "a", 999.0, 1), ("OWN", "t1", "a", 1.0, 1)]:
        rows.append(dict(fund_id=fid, report_date=d, security_key=key, price_usd=px, shares=sh))
    r = st.build_price_returns(pd.DataFrame(rows), "P", "t0", "t1")
    assert abs(r["a"] - 0.1) < 1e-12 and abs(r["b"] + 0.2) < 1e-12
    assert abs(r["c"] - 0.1) < 1e-9
    assert "only0" not in r.index


def test_no_own_prices_by_construction():
    src = inspect.getsource(st.build_price_returns)
    assert "fund_id == parent_id" in src
    # the main loop takes fund returns from the universe return series, not from prices
    main_src = inspect.getsource(st.main)
    assert "r_all.loc[fm.index]" in main_src
    assert "build_price_returns(prices, uni" not in main_src and "fh_prices" not in main_src
    # prices table is restricted to the two parents
    assert "prices.fund_id.isin([SP, RU])" in main_src
