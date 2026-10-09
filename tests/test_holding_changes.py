"""Holdings-difference calculator: expected numbers are worked out by hand in the comments."""
import numpy as np
import pandas as pd
import pytest

from fairbench.portfolio.changes import (CHANGE_TYPES, DECISION_OBSERVABILITY, holding_changes,
                                         summarise_changes)

# ---------------------------------------------------------------- fixtures
# prev (total value 100_000):             shares  price  value   weight
#   X  exits                                100    100  10_000   0.10
#   I  increases (+50 shares)               100    200  20_000   0.20
#   D  decreases (-50 shares)               200    100  20_000   0.20
#   U  unchanged shares                      50    200  10_000   0.10
#   S  2-for-1 split, no trading            100    200  20_000   0.20
#   F  scaled by the flow ratio              100    200  20_000   0.20
#
# curr (total value 96_400):              shares  price  value
#   I                                       150    200  30_000   (price return  0%)
#   D                                       150     90  13_500   (price return -10%)
#   U                                        50    210  10_500   (price return +5%)
#   S                                       200    102  20_400   (pre-split price 200 -> 204, +2%)
#   F                                        95    200  19_000   (price return 0%)
#   N  new position                          30    100   3_000
#   X  is gone; its price return over the period was +10% (supplied from outside)
PREV = pd.DataFrame({
    "security_key": ["cusip:X", "cusip:I", "cusip:D", "cusip:U", "cusip:S", "cusip:F"],
    "issuer_name": ["X Corp", "I Corp", "D Corp", "U Corp", "S Corp", "F Corp"],
    "balance": [100.0, 100, 200, 50, 100, 100],
    "value_usd": [10_000.0, 20_000, 20_000, 10_000, 20_000, 20_000],
})
CURR = pd.DataFrame({
    "security_key": ["cusip:I", "cusip:D", "cusip:U", "cusip:S", "cusip:F", "cusip:N"],
    "issuer_name": ["I Corp", "D Corp", "U Corp", "S Corp", "F Corp", "N Corp"],
    "balance": [150.0, 150, 50, 200, 95, 30],
    "value_usd": [30_000.0, 13_500, 10_500, 20_400, 19_000, 3_000],
})
TOTAL_CURR = 96_400.0
EXIT_RETURN = pd.Series({"cusip:X": 0.10})


def _row(df, key):
    return df.set_index("security_key").loc[key]


@pytest.fixture(scope="module")
def ch():
    return holding_changes(PREV, CURR, price_return=EXIT_RETURN, net_flow_ratio=-0.05)


# ----------------------------------------------------------------- classification
def test_one_row_per_key_in_the_union(ch):
    assert sorted(ch["security_key"]) == sorted(set(PREV["security_key"]) | set(CURR["security_key"]))
    assert ch["security_key"].is_unique and len(ch) == 7


def test_change_types(ch):
    expected = {
        "cusip:N": "observed_new_position",
        "cusip:X": "observed_exit",
        "cusip:I": "observed_holding_increase",   # 150 / 100 = 1.5 (a listed split ratio, but the price did not halve)
        "cusip:D": "observed_holding_decrease",   # 150 / 200 = 0.75
        "cusip:U": "unchanged",
        "cusip:S": "unchanged",                   # 200 / 100 = 2 is the split, not trading
        "cusip:F": "observed_holding_decrease",   # 95 / 100 = 0.95: flow scaling still changes shares
    }
    got = ch.set_index("security_key")["change_type"].to_dict()
    assert got == expected
    assert set(got.values()) <= set(CHANGE_TYPES)


def test_split_is_detected_without_touching_filed_shares(ch):
    s = _row(ch, "cusip:S")
    assert bool(s.split_suspected) and s.split_ratio == 2
    assert s.shares_prev == 100 and s.shares_curr == 200 and s.shares_delta == 100  # as filed
    assert s.shares_ratio == pytest.approx(2.0)
    assert s.shares_ratio_adjusted == pytest.approx(1.0)
    # raw implied prices: 20_000 / 100 = 200 before, 20_400 / 200 = 102 after
    assert s.implied_price_prev == pytest.approx(200) and s.implied_price_curr == pytest.approx(102)
    # split-adjusted price return: 102 * 2 / 200 - 1 = +2% (not -49%)
    assert s.implied_price_return == pytest.approx(0.02)
    others = ch[ch["security_key"] != "cusip:S"]
    assert not others["split_suspected"].any()
    assert others["split_ratio"].isna().all()


def test_share_ratio_that_looks_like_a_split_but_price_did_not_move_is_not_a_split(ch):
    i = _row(ch, "cusip:I")
    assert i.shares_ratio == pytest.approx(1.5) and not i.split_suspected
    assert i.shares_ratio_adjusted == pytest.approx(1.5)


def test_reverse_split_is_detected():
    prev = pd.DataFrame({"security_key": ["a", "b"], "balance": [100.0, 100.0], "value_usd": [1000.0, 1000.0]})
    # 1-for-10 reverse split of a: 100 -> 10 shares, price 10 -> 101 (+1%)
    curr = pd.DataFrame({"security_key": ["a", "b"], "balance": [10.0, 100.0], "value_usd": [1010.0, 1000.0]})
    out = holding_changes(prev, curr)
    a = _row(out, "a")
    assert bool(a.split_suspected) and a.split_ratio == pytest.approx(0.1)
    assert a.change_type == "unchanged"
    assert a.implied_price_return == pytest.approx(0.01)


# ------------------------------------------------------------------ weights, prices
def test_shares_values_weights_and_prices(ch):
    d = _row(ch, "cusip:D")
    assert (d.shares_prev, d.shares_curr, d.shares_delta) == (200, 150, -50)
    assert d.shares_ratio == pytest.approx(0.75)
    assert (d.value_prev, d.value_curr) == (20_000, 13_500)
    assert d.weight_prev == pytest.approx(0.20) and d.weight_curr == pytest.approx(13_500 / TOTAL_CURR)
    assert d.weight_delta == pytest.approx(13_500 / TOTAL_CURR - 0.20)
    assert d.implied_price_prev == pytest.approx(100) and d.implied_price_curr == pytest.approx(90)
    assert d.implied_price_return == pytest.approx(-0.10)
    assert d.issuer_name == "D Corp"
    assert ch["weight_prev"].sum() == pytest.approx(1.0) and ch["weight_curr"].sum() == pytest.approx(1.0)


def test_new_and_exited_positions_have_zero_on_the_missing_side(ch):
    n, x = _row(ch, "cusip:N"), _row(ch, "cusip:X")
    assert n.shares_prev == 0 and n.value_prev == 0 and n.weight_prev == 0 and n.n_rows_prev == 0
    assert np.isnan(n.shares_ratio) and np.isnan(n.implied_price_prev) and np.isnan(n.implied_price_return)
    assert n.shares_delta == 30 and n.weight_curr == pytest.approx(3_000 / TOTAL_CURR)
    assert x.shares_curr == 0 and x.value_curr == 0 and x.weight_curr == 0 and x.n_rows_curr == 0
    assert x.shares_ratio == 0  # defined: curr / prev = 0 / 100
    assert np.isnan(x.implied_price_curr) and np.isnan(x.implied_price_return)
    assert x.issuer_name == "X Corp"  # name falls back to prev
    assert n.issuer_name == "N Corp"


# --------------------------------------------------------------------- drift
def test_drift_and_active_weight_by_hand(ch):
    # weight_prev * (1 + r): X .10*1.10  I .20*1.00  D .20*0.90  U .10*1.05  S .20*1.02  F .20*1.00
    grown = {"X": 0.11, "I": 0.20, "D": 0.18, "U": 0.105, "S": 0.204, "F": 0.20}
    norm = sum(grown.values())
    assert norm == pytest.approx(0.999)
    curr_value = {"I": 30_000, "D": 13_500, "U": 10_500, "S": 20_400, "F": 19_000, "N": 3_000, "X": 0}
    for k, g in {**grown, "N": 0.0}.items():
        row = _row(ch, f"cusip:{k}")
        drift = g / 0.999
        assert row.drift_weight == pytest.approx(drift)
        assert row.active_weight_change == pytest.approx(curr_value[k] / TOTAL_CURR - drift)
    # new position: nothing to drift, so the active change is its whole weight; exit: minus its drift
    assert _row(ch, "cusip:N").drift_weight == 0
    assert _row(ch, "cusip:N").active_weight_change == pytest.approx(3_000 / TOTAL_CURR)
    assert _row(ch, "cusip:X").active_weight_change == pytest.approx(-0.11 / 0.999)


def test_drift_weights_sum_to_one_and_active_changes_to_zero_when_all_returns_known(ch):
    assert ch["drift_weight"].notna().all()
    assert ch["drift_weight"].sum() == pytest.approx(1.0)
    assert ch["active_weight_change"].sum() == pytest.approx(0.0, abs=1e-12)
    assert ch.attrs["n_unknown_return"] == 0


def test_attrs_counts_and_turnover(ch):
    a = ch.attrs
    assert (a["n_prev"], a["n_curr"]) == (6, 6)
    assert (a["n_new"], a["n_exit"], a["n_increase"], a["n_decrease"], a["n_unchanged"]) == (1, 1, 1, 2, 2)
    assert a["n_split_suspected"] == 1 and a["n_unknown_return"] == 0 and a["n_unclassified"] == 0
    n_types = a["n_new"] + a["n_exit"] + a["n_increase"] + a["n_decrease"] + a["n_unchanged"]
    assert n_types == len(ch)
    assert a["one_way_turnover"] == pytest.approx(0.5 * ch["active_weight_change"].abs().sum())
    assert 0 < a["one_way_turnover"] < 1


def test_no_trading_means_no_active_change_and_zero_turnover():
    prev = PREV[PREV["security_key"] != "cusip:X"].reset_index(drop=True)
    # same shares (S split 2-for-1), prices move: I +10%, D -5%, U 0%, S +3% (pre-split basis), F +20%
    curr = pd.DataFrame({
        "security_key": ["cusip:I", "cusip:D", "cusip:U", "cusip:S", "cusip:F"],
        "balance": [100.0, 200, 50, 200, 100],
        "value_usd": [22_000.0, 19_000, 10_000, 20_600, 24_000],
    })
    out = holding_changes(prev, curr)  # no external returns: implied returns are used
    assert (out["change_type"] == "unchanged").all()
    assert out.set_index("security_key")["split_suspected"].to_dict() == {
        "cusip:D": False, "cusip:F": False, "cusip:I": False, "cusip:S": True, "cusip:U": False}
    assert out["active_weight_change"].abs().max() < 1e-12
    assert out.attrs["one_way_turnover"] == pytest.approx(0.0, abs=1e-12)
    assert out.attrs["n_unknown_return"] == 0
    assert out["drift_weight"].sum() == pytest.approx(1.0)
    assert (out["decision_observability"] == "unavailable").all()
    # but the raw weights did move (price effect alone), which is the point of the drift baseline
    assert out["weight_delta"].abs().max() > 0.01


def test_no_change_at_all_with_identical_frames():
    out = holding_changes(PREV, PREV)
    assert (out["change_type"] == "unchanged").all() and not out["split_suspected"].any()
    assert out["weight_delta"].abs().max() == 0 and out["active_weight_change"].abs().max() == 0
    assert out.attrs["one_way_turnover"] == 0


# -------------------------------------------------------------- unknown returns
def test_unknown_return_is_left_out_of_the_normalisation_and_counted():
    out = holding_changes(PREV, CURR)  # no price_return: X has no later price, so r is unknown
    x = _row(out, "cusip:X")
    assert np.isnan(x.drift_weight) and np.isnan(x.active_weight_change)
    assert out.attrs["n_unknown_return"] == 1
    # known positions are re-normalised without X: I .20, D .18, U .105, S .204, F .20 -> .889
    for k, g in {"I": 0.20, "D": 0.18, "U": 0.105, "S": 0.204, "F": 0.20}.items():
        assert _row(out, f"cusip:{k}").drift_weight == pytest.approx(g / 0.889)
    assert _row(out, "cusip:N").drift_weight == 0  # a new position is not "unknown"
    assert out["drift_weight"].sum() == pytest.approx(1.0)  # over the known positions
    # turnover only uses the rows where the active change is known
    known = out["active_weight_change"].dropna()
    assert len(known) == 6
    assert out.attrs["one_way_turnover"] == pytest.approx(0.5 * known.abs().sum())


def test_missing_value_gives_unknown_weight_and_is_not_filled():
    curr = CURR.copy()
    curr.loc[curr["security_key"] == "cusip:U", "value_usd"] = np.nan
    out = holding_changes(PREV, curr, price_return=EXIT_RETURN)
    u = _row(out, "cusip:U")
    assert u.shares_curr == 50 and np.isnan(u.value_curr)
    assert np.isnan(u.implied_price_curr) and np.isnan(u.implied_price_return)
    assert np.isnan(u.drift_weight) and np.isnan(u.active_weight_change)
    assert out.attrs["n_unknown_return"] == 1
    assert u.change_type == "unchanged"  # shares are known and did not move


def test_price_return_overrides_and_falls_back_to_implied():
    # D really returned -20% in the market data (the filings imply -10%); only D is supplied
    out = holding_changes(PREV, CURR, price_return=pd.Series({"cusip:D": -0.20, "cusip:X": 0.10}))
    grown = {"X": 0.11, "I": 0.20, "D": 0.20 * 0.8, "U": 0.105, "S": 0.204, "F": 0.20}
    norm = sum(grown.values())
    for k, g in grown.items():
        assert _row(out, f"cusip:{k}").drift_weight == pytest.approx(g / norm)
    # NaN entries in price_return are "not given" and fall back as well
    nan_ret = holding_changes(PREV, CURR, price_return=pd.Series({"cusip:X": 0.10, "cusip:D": np.nan}))
    assert _row(nan_ret, "cusip:D").drift_weight == pytest.approx(0.18 / 0.999)


# --------------------------------------------------------------------- flows
def test_flow_consistency(ch):
    f = ch.set_index("security_key")["flow_consistent"]
    assert bool(f["cusip:F"])  # 95 / 100 = 1 - 0.05 exactly
    assert not any(bool(f[k]) for k in ["cusip:I", "cusip:D", "cusip:U", "cusip:N", "cusip:X"])
    assert not bool(f["cusip:S"])  # split-adjusted ratio is 1.0, which a -5% flow does not explain
    # the flag explains shares only: it does not change the label
    assert _row(ch, "cusip:F").change_type == "observed_holding_decrease"


def test_flow_consistent_is_none_without_a_flow_ratio():
    out = holding_changes(PREV, CURR, price_return=EXIT_RETURN)
    assert out["flow_consistent"].isna().all()
    assert all(v is None for v in out["flow_consistent"])


def test_pro_rata_scaling_moves_shares_but_not_weights():
    # every position scaled by -10% (outflow): shares drop, weights and drift stay put
    curr = PREV.assign(balance=PREV["balance"] * 0.9, value_usd=PREV["value_usd"] * 0.9)
    out = holding_changes(PREV, curr, net_flow_ratio=-0.10)
    assert out["flow_consistent"].all()
    assert (out["change_type"] == "observed_holding_decrease").all()
    assert out["weight_delta"].abs().max() < 1e-12
    assert out["active_weight_change"].abs().max() < 1e-12


# -------------------------------------------------------- observability
def test_decision_observability_values(ch):
    assert DECISION_OBSERVABILITY == ("direct", "observed_position_change", "inferred", "unavailable")
    assert set(ch["decision_observability"]) <= {"observed_position_change", "unavailable"}
    assert not ({"direct", "inferred"} & set(ch["decision_observability"]))
    by_type = ch.groupby("change_type")["decision_observability"].agg(set).to_dict()
    assert by_type.pop("unchanged") == {"unavailable"}
    assert all(v == {"observed_position_change"} for v in by_type.values())


def test_unclassifiable_row_is_unavailable_not_guessed():
    prev = pd.DataFrame({"security_key": ["a", "b"], "balance": [np.nan, 10.0], "value_usd": [100.0, 100.0]})
    curr = pd.DataFrame({"security_key": ["a", "b"], "balance": [5.0, 10.0], "value_usd": [100.0, 100.0]})
    out = holding_changes(prev, curr)
    a = _row(out, "a")
    assert pd.isna(a.change_type) and a.decision_observability == "unavailable"
    assert out.attrs["n_unclassified"] == 1 and out.attrs["n_unchanged"] == 1


# -------------------------------------------------------------- duplicate keys
def test_duplicate_keys_are_summed_and_counted():
    # I filed on two rows (90 + 10 shares, 19_000 + 1_000 value) is the same position as the single row in PREV
    is_i = PREV["security_key"] == "cusip:I"
    lot_1 = PREV.assign(balance=PREV["balance"].where(~is_i, 90.0), value_usd=PREV["value_usd"].where(~is_i, 19_000.0))
    lot_2 = PREV[is_i].assign(balance=10.0, value_usd=1_000.0)
    two_rows = pd.concat([lot_1, lot_2], ignore_index=True)
    merged = holding_changes(two_rows, CURR)
    i = _row(merged, "cusip:I")
    assert (i.shares_prev, i.value_prev, i.n_rows_prev, i.n_rows_curr) == (100, 20_000, 2, 1)
    assert merged.attrs["n_prev"] == 6 and len(merged) == 7  # distinct keys, not rows
    single = holding_changes(PREV, CURR)
    pd.testing.assert_frame_equal(merged.drop(columns="n_rows_prev"), single.drop(columns="n_rows_prev"))

    # an extra lot on top of the original row: 100 + 10 shares, 20_000 + 1_000 value
    extra = pd.concat([PREV, PREV[PREV["security_key"] == "cusip:I"].assign(balance=10.0, value_usd=1_000.0)],
                      ignore_index=True)
    j = _row(holding_changes(extra, CURR), "cusip:I")
    assert (j.shares_prev, j.value_prev, j.n_rows_prev) == (110, 21_000, 2)
    assert j.shares_ratio == pytest.approx(150 / 110)


# ------------------------------------------------------------- inputs / edge
def test_inputs_are_not_modified_and_custom_key_works():
    p0, c0 = PREV.copy(deep=True), CURR.copy(deep=True)
    holding_changes(PREV, CURR, price_return=EXIT_RETURN)
    pd.testing.assert_frame_equal(PREV, p0)
    pd.testing.assert_frame_equal(CURR, c0)
    out = holding_changes(PREV.rename(columns={"security_key": "cusip"}), CURR.rename(columns={"security_key": "cusip"}),
                          key="cusip", price_return=EXIT_RETURN)
    pd.testing.assert_frame_equal(out, holding_changes(PREV, CURR, price_return=EXIT_RETURN), check_like=True)


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="missing columns"):
        holding_changes(PREV.drop(columns="balance"), CURR)
    with pytest.raises(ValueError, match="empty"):
        holding_changes(PREV.assign(security_key=[None] + list(PREV["security_key"][1:])), CURR)
    with pytest.raises(ValueError):
        holding_changes(PREV, CURR, price_return=pd.Series([0.1, 0.2], index=["cusip:X", "cusip:X"]))
    with pytest.raises(ValueError):
        holding_changes(PREV, CURR, split_ratios=(2, 0))


def test_share_tolerance_boundary():
    prev = pd.DataFrame({"security_key": ["a", "b"], "balance": [1000.0, 1000.0], "value_usd": [1.0, 1.0]})
    curr = pd.DataFrame({"security_key": ["a", "b"], "balance": [1004.0, 1006.0], "value_usd": [1.0, 1.0]})
    out = holding_changes(prev, curr)  # share_tol = 0.005: +0.4% is noise, +0.6% is a change
    assert _row(out, "a").change_type == "unchanged"
    assert _row(out, "b").change_type == "observed_holding_increase"
    loose = holding_changes(prev, curr, share_tol=0.01)
    assert (loose["change_type"] == "unchanged").all()


# ------------------------------------------------------------------ summarise
def test_summarise_changes_without_groups_returns_the_counts(ch):
    s = summarise_changes(ch)
    assert len(s) == 1
    row = s.iloc[0]
    for k in ("n_prev", "n_curr", "n_new", "n_exit", "n_increase", "n_decrease", "n_unchanged",
              "n_split_suspected", "n_unknown_return"):
        assert row[k] == ch.attrs[k], k
    assert row["one_way_turnover"] == pytest.approx(ch.attrs["one_way_turnover"])


def test_summarise_changes_with_groups(ch):
    groups = pd.Series({"cusip:X": "Tech", "cusip:I": "Tech", "cusip:D": "Energy", "cusip:U": "Energy",
                        "cusip:S": "Energy", "cusip:F": "Health"})  # cusip:N deliberately unlabelled
    s = summarise_changes(ch, groups).set_index("group")
    assert set(s.index) == {"Tech", "Energy", "Health", "unlabelled"}
    assert (s["change_label"] == "observed_sector_change").all()
    assert s["n_positions"].to_dict() == {"Energy": 3, "Health": 1, "Tech": 2, "unlabelled": 1}

    # Tech = X + I: weight_prev .10 + .20, weight_curr 0 + 30_000 / 96_400, drift (.11 + .20) / .999
    tech = s.loc["Tech"]
    assert tech.weight_prev == pytest.approx(0.30)
    assert tech.weight_curr == pytest.approx(30_000 / TOTAL_CURR)
    assert tech.weight_delta == pytest.approx(30_000 / TOTAL_CURR - 0.30)
    assert tech.drift_weight == pytest.approx(0.31 / 0.999)
    assert tech.active_weight_change == pytest.approx(30_000 / TOTAL_CURR - 0.31 / 0.999)
    # the unlabelled bucket is the new position
    assert s.loc["unlabelled", "weight_curr"] == pytest.approx(3_000 / TOTAL_CURR)
    assert s.loc["unlabelled", "weight_prev"] == 0
    # nothing is lost: group totals equal the position totals
    for col in ("weight_prev", "weight_curr", "drift_weight"):
        assert s[col].sum() == pytest.approx(1.0)
    assert s["active_weight_change"].sum() == pytest.approx(0.0, abs=1e-12)
    assert (s["n_unknown_drift"] == 0).all()


def test_summarise_groups_marks_unknown_instead_of_hiding_it():
    out = holding_changes(PREV, CURR)  # X's drift is unknown
    groups = pd.Series({"cusip:X": "Tech", "cusip:I": "Tech", "cusip:D": "Energy", "cusip:U": "Energy",
                        "cusip:S": "Energy", "cusip:F": "Health", "cusip:N": "Health"})
    s = summarise_changes(out, groups).set_index("group")
    assert np.isnan(s.loc["Tech", "drift_weight"]) and np.isnan(s.loc["Tech", "active_weight_change"])
    assert s.loc["Tech", "n_unknown_drift"] == 1
    assert np.isfinite(s.loc["Energy", "drift_weight"]) and s.loc["Energy", "n_unknown_drift"] == 0
    assert s.loc["Tech", "weight_prev"] == pytest.approx(0.30)  # known columns stay known
