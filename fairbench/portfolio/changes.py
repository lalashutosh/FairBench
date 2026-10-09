"""Holdings differences between two disclosed snapshots, described without claiming intent.

What this is and what it is not
-------------------------------
A public filing shows a fund's positions at two dates and nothing in between. This module
subtracts one snapshot from the other and labels what the numbers show. The limits, in plain words:

* Snapshots only. We see positions at two quarter-ends, not the trades between them.
* The filing date differs from the portfolio date. The positions are as of the report date;
  they became public later (for N-PORT, up to 60 days later, so the last month of a filing is
  often not yet public for the most recent quarter).
* A change in shares or weight mixes trading, price moves, investor flows (a fund that scales
  every position up or down after inflows or outflows) and corporate actions (splits, mergers,
  spin-offs, identifier changes), plus any rebalancing the rules required.
* The order and timing of trades, and the reason for any of them, are not public.
* Therefore every row here is an OBSERVATION about two snapshots, not a manager decision.
  `decision_observability` is "observed_position_change" or "unavailable", never "direct" or
  "inferred": those two values may only be set by other code that has a published explanation.

Contract of `holding_changes`
-----------------------------
* Inputs are holdings frames with the key column, `balance` (quantity) and `value_usd`; rows with
  the same key are summed first (counts kept in n_rows_prev / n_rows_curr). Rows must have a key:
  a null or empty key raises, and a catch-all key such as "unknown:" must be removed by the caller
  (it would be summed into one fake position). The frames passed in define the sleeve: weights are
  value_usd / sum(value_usd) within each frame.
* A key absent from a frame is a zero position at that date (shares 0, value 0, weight 0,
  n_rows 0). A key present with a missing balance or value stays NaN, never filled.
* drift_weight is the weight a position would have at the later date with NO trading, from
  weight_prev * (1 + r), re-normalised over positions whose return r is known. r is the supplied
  `price_return` for that key (a missing or NaN entry falls back to the return implied by the two
  filings), else `implied_price_return`. A position with weight_prev == 0 has drift 0. Other
  positions with unknown r get NaN, are left out of the normalising sum, and are counted in
  attrs["n_unknown_return"]. An exited position has no later price in the filings, so its r is
  unknown unless `price_return` supplies it.
* implied_price_return = implied_price_curr / implied_price_prev - 1, EXCEPT for suspected splits,
  where the later price is first multiplied by the split ratio. Without that, a 2-for-1 split with
  no trading would read as a -50% price move and show up as a large fake active change. The raw
  implied prices stay in their own columns.
* A split is suspected when shares_ratio is within `split_tol` (relative) of a listed ratio AND
  the implied price ratio is within 15% of its reciprocal. It needs value data on both dates, and
  a split combined with trading is not detected (it reads as a holding change). shares_prev is
  never modified; `shares_ratio_adjusted` is what change_type and flow_consistent use.
* change_type is None when it cannot be classified (position present on both dates but with a
  missing balance), and decision_observability is then "unavailable".
* attrs["one_way_turnover"] = 0.5 * sum(|active_weight_change|) over rows where it is known; it is
  a lower bound whenever n_unknown_return > 0. "Shares" means the filed quantity (shares, or
  principal amount for bonds); that is why the ratio of filed balances is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECISION_OBSERVABILITY = ("direct", "observed_position_change", "inferred", "unavailable")
CHANGE_TYPES = ("observed_new_position", "observed_exit", "observed_holding_increase",
                "observed_holding_decrease", "unchanged")

_DEFAULT_SPLITS = (1.5, 2, 3, 4, 5, 10, 20, 0.5, 1 / 3, 0.25, 0.2, 0.1, 0.05)
_PRICE_BAND = 0.15  # implied price ratio must be within this of 1/split_ratio
_WEIGHT_COLS = ["weight_prev", "weight_curr", "weight_delta", "drift_weight", "active_weight_change"]


# ------------------------------------------------------------------ helpers
def _div(a, b) -> np.ndarray:
    """a / b with NaN wherever b is 0 (NaN stays NaN)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan)


def _aggregate(df: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    """One row per key: summed balance and value (NaN only if every row is NaN), row count, name."""
    missing = [c for c in (key, "balance", "value_usd") if c not in df.columns]
    if missing:
        raise ValueError(f"{label} frame is missing columns {missing}")
    k = df[key]
    if k.isna().any() or (k.astype(str).str.strip() == "").any():
        raise ValueError(f"{label} frame has rows with an empty {key!r}; drop them or key them first")
    d = pd.DataFrame({
        "k": k.to_numpy(),
        "balance": pd.to_numeric(df["balance"], errors="coerce").to_numpy(),
        "value": pd.to_numeric(df["value_usd"], errors="coerce").to_numpy(),
        "issuer_name": pd.Series(df["issuer_name"].tolist() if "issuer_name" in df.columns
                                 else [None] * len(df), dtype=object).to_numpy(),
    })
    g = d.groupby("k", sort=True)
    out = g[["balance", "value"]].sum(min_count=1)
    out["n_rows"] = g.size()
    out["issuer_name"] = g["issuer_name"].first()
    out.index.name = None
    return out


def _side(agg: pd.DataFrame, idx: pd.Index):
    """(shares, value, n_rows, absent) over `idx`; a key missing from the frame is a zero position."""
    shares = agg["balance"].reindex(idx, fill_value=0.0).to_numpy(float)
    value = agg["value"].reindex(idx, fill_value=0.0).to_numpy(float)
    rows = agg["n_rows"].reindex(idx, fill_value=0).to_numpy(int)
    return shares, value, rows, rows == 0


def _weights(value: np.ndarray, absent: np.ndarray) -> np.ndarray:
    total = np.nansum(value)
    w = _div(value, total) if np.isfinite(total) else np.full(len(value), np.nan)
    return np.where(absent, 0.0, w)


def _counts(df: pd.DataFrame) -> dict:
    """Summary counts recomputed from the frame itself (so they survive filtering)."""
    ct = df["change_type"]
    unknown = df["drift_weight"].isna() & (df["weight_prev"] != 0)
    return {
        "n_prev": int((df["n_rows_prev"] > 0).sum()),
        "n_curr": int((df["n_rows_curr"] > 0).sum()),
        "n_new": int((ct == "observed_new_position").sum()),
        "n_exit": int((ct == "observed_exit").sum()),
        "n_increase": int((ct == "observed_holding_increase").sum()),
        "n_decrease": int((ct == "observed_holding_decrease").sum()),
        "n_unchanged": int((ct == "unchanged").sum()),
        "n_unclassified": int(ct.isna().sum()),
        "n_split_suspected": int(df["split_suspected"].sum()),
        "n_unknown_return": int(unknown.sum()),
        "one_way_turnover": float(0.5 * df["active_weight_change"].abs().sum()),
    }


# -------------------------------------------------------------- the calculator
def holding_changes(prev: pd.DataFrame, curr: pd.DataFrame, *, key: str = "security_key",
                    price_return: pd.Series | None = None, net_flow_ratio: float | None = None,
                    share_tol: float = 0.005,
                    split_ratios: tuple[float, ...] = _DEFAULT_SPLITS,
                    split_tol: float = 0.02) -> pd.DataFrame:
    """Observed differences between two holdings snapshots; see the module docstring.

    `price_return` is indexed by key and gives the price return of each security between the two
    dates (e.g. from market data). `net_flow_ratio` is net investor flow over the period divided by
    starting net assets (e.g. -0.05); a fund scaling every position pro rata changes shares but not
    weights, and `flow_consistent` marks the rows that such scaling alone would explain.
    The output column holding the key is always named "security_key".
    """
    ratios = np.asarray(split_ratios, dtype=float)
    if ratios.size and not (np.all(np.isfinite(ratios)) and np.all(ratios > 0)):
        raise ValueError("split_ratios must be positive and finite")
    if price_return is not None and not price_return.index.is_unique:
        raise ValueError("price_return must have a unique index")

    p, c = _aggregate(prev, key, "prev"), _aggregate(curr, key, "curr")
    idx = p.index.union(c.index)
    n = len(idx)
    sp, vp, rows_p, abs_p = _side(p, idx)
    sc, vc, rows_c, abs_c = _side(c, idx)
    wp, wc = _weights(vp, abs_p), _weights(vc, abs_c)

    ratio = _div(sc, sp)  # shares_ratio, NaN if prev is 0 or missing
    pp, pc = _div(vp, sp), _div(vc, sc)
    price_ratio = _div(pc, pp)

    # ---- split detection: nearest listed ratio, then the price must have moved the other way
    if ratios.size:
        with np.errstate(invalid="ignore"):
            dev = np.abs(ratio[:, None] / ratios[None, :] - 1.0)
        dev = np.where(np.isnan(dev), np.inf, dev)
        j = dev.argmin(axis=1)
        best = ratios[j]
        with np.errstate(invalid="ignore"):
            suspected = (dev[np.arange(n), j] <= split_tol) & (np.abs(price_ratio * best - 1.0) <= _PRICE_BAND)
    else:
        best, suspected = np.ones(n), np.zeros(n, dtype=bool)
    factor = np.where(suspected, best, 1.0)
    split_ratio = np.where(suspected, best, np.nan)
    ratio_adj = ratio / factor
    implied_ret = price_ratio * factor - 1.0

    # ---- no-trading drift
    r = implied_ret
    if price_return is not None:
        ext = price_return.reindex(idx).to_numpy(float)
        r = np.where(np.isnan(ext), implied_ret, ext)
    zero = wp == 0
    known = ~zero & np.isfinite(wp) & np.isfinite(r)
    grown = np.where(known, wp * (1.0 + np.where(known, r, 0.0)), 0.0)
    denom = grown.sum()
    if not denom > 0:
        known[:] = False
    drift = np.full(n, np.nan)
    drift[known] = grown[known] / denom
    drift[zero] = 0.0
    active = wc - drift

    # ---- labels
    zero_p, zero_c = sp == 0, sc == 0
    new, gone, neither = zero_p & ~zero_c, zero_c & ~zero_p, zero_p & zero_c
    both = ~(zero_p | zero_c) & np.isfinite(ratio_adj)
    with np.errstate(invalid="ignore"):
        moved = np.abs(ratio_adj - 1.0) > share_tol
        up = ratio_adj > 1.0
    change = np.full(n, None, dtype=object)
    change[new] = "observed_new_position"
    change[gone] = "observed_exit"
    change[neither | (both & ~moved)] = "unchanged"
    change[both & moved & up] = "observed_holding_increase"
    change[both & moved & ~up] = "observed_holding_decrease"
    observability = np.where(pd.notna(change) & (change != "unchanged"),
                             "observed_position_change", "unavailable")

    if net_flow_ratio is None:
        flow = pd.Series([None] * n, dtype=object)
    else:
        with np.errstate(invalid="ignore"):
            flow = pd.Series(np.abs(ratio_adj - (1.0 + net_flow_ratio)) <= 4 * share_tol)

    name = c["issuer_name"].reindex(idx)
    name = name.where(name.notna(), p["issuer_name"].reindex(idx))

    out = pd.DataFrame({
        "security_key": idx.to_numpy(),
        "issuer_name": pd.Series(name.tolist(), dtype=object),
        "shares_prev": sp, "shares_curr": sc, "shares_delta": sc - sp, "shares_ratio": ratio,
        "value_prev": vp, "value_curr": vc,
        "weight_prev": wp, "weight_curr": wc, "weight_delta": wc - wp,
        "implied_price_prev": pp, "implied_price_curr": pc, "implied_price_return": implied_ret,
        "drift_weight": drift, "active_weight_change": active,
        "change_type": pd.Series(change, dtype=object),
        "split_suspected": suspected, "split_ratio": split_ratio,
        "shares_ratio_adjusted": ratio_adj,
        "flow_consistent": flow,
        "decision_observability": observability,
        "n_rows_prev": rows_p, "n_rows_curr": rows_c,
    })
    out.attrs.update(_counts(out))
    return out


# --------------------------------------------------------------- summarising
def summarise_changes(changes: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    """Counts, or weights aggregated by a group label (e.g. sector).

    Without `groups`: a one-row frame of the summary counts (recomputed from the frame, so it
    works on filtered results). With `groups` (index = security key, values = label): one row per
    label with weight_prev, weight_curr, weight_delta, drift_weight, active_weight_change summed,
    plus n_positions and n_unknown_drift, and change_label "observed_sector_change". A group sum is
    NaN if any member of the group is unknown, and keys without a label go to "unlabelled".
    """
    if groups is None:
        return pd.DataFrame([_counts(changes)])
    label = changes["security_key"].map(groups).astype(object)
    label = label.where(label.notna(), "unlabelled")
    d = changes[_WEIGHT_COLS].assign(group=label.to_numpy())
    g = d.groupby("group", sort=True)
    out = g[_WEIGHT_COLS].agg(lambda s: s.sum(skipna=False))
    out.insert(0, "n_positions", g.size())
    out["n_unknown_drift"] = g["drift_weight"].apply(lambda s: int(s.isna().sum()))
    out["change_label"] = "observed_sector_change"
    return out.reset_index()
