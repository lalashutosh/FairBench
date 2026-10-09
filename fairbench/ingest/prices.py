"""Security and fund returns: the adapter interface and the two sources that need no licence.

A ``ReturnSource`` answers one question: what did each security return between two dates?
Every answer carries its kind ("total" includes dividends, "price" does not), a status
(disclosed / derived / assumed / unknown) and a note, and an unknown return is NaN, never
a filled-in number.

``NportImpliedReturns``  SEC-only. A fund's N-PORT filing reports shares and market value
    for every position, so value / shares is the closing price the fund used on its report
    date. Two snapshots of a broad index fund give a price return for every security it
    held at both dates. Limits, all carried in the output: PRICE return only (dividends are
    missing, which understates every return by roughly the dividend yield of the period);
    only at the reference fund's report dates; a security that left the reference fund
    before the second date has no return. A split of 2-for-1 or more (or a reverse split)
    is inferred from the share count and adjusted (flagged ``split_adjusted``); any other
    jump in shares of more than 8% beyond fund flows leaves the return unknown. Smaller
    share changes, such as a 5% stock dividend, are NOT detected and bias that security's
    return by up to that amount.
``CsvReturns``  returns the user supplies from their own or a licensed source, kept out of
    the repository. Columns: security_key, start, end, total_return.

``nav_period_return`` reads the fund's own disclosed monthly total returns from N-PORT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

RETURN_COLUMNS = ["ret", "kind", "status", "split_adjusted", "note"]
# Only ratios whose mechanical price effect (a halving or more, or a doubling or more) cannot
# be mistaken for an ordinary quarter. A 3-for-2 split or a 25% share issue is NOT guessed at.
SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 0.5, 1 / 3, 0.25, 0.2, 0.1, 0.05)


class ReturnSource(Protocol):
    name: str
    licence: str

    def period_returns(self, keys: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """DataFrame indexed by security_key with RETURN_COLUMNS; ``ret`` is NaN when unknown."""
        ...


def _empty(keys: Sequence[str], note: str) -> pd.DataFrame:
    df = pd.DataFrame({"ret": np.nan, "kind": None, "status": "unknown", "split_adjusted": False, "note": note},
                      index=pd.Index(list(keys), name="security_key"))
    return df.astype({"kind": object, "status": object, "note": object})


def _sleeve(h: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for an implied price: keyed, positive shares and value."""
    ok = h["security_key"].notna() & ~h["security_key"].astype(str).str.startswith("unknown:")
    return h[ok & (h["balance"] > 0) & (h["value_usd"] > 0)]


@dataclass
class NportImpliedReturns:
    """Price returns implied by N-PORT snapshots of one or more reference funds.

    snapshots: {report_date: holdings frame with security_key, balance, value_usd} of a
    reference fund (typically the parent index fund; equity sleeve only). ``add`` merges
    another fund's snapshots; where two funds price the same security on the same date the
    first one added is used and the largest relative disagreement is kept in
    ``max_price_disagreement`` as a data-quality check. A security priced only by an added
    fund gets a return only while that fund's share count is unchanged (within 8%): an
    active fund's shares move through trading, which cannot be told from a split.
    """
    snapshots: dict[date, pd.DataFrame] = field(default_factory=dict)
    name: str = "nport_implied"
    licence: str = "derived from public SEC Form N-PORT filings; redistributable"
    max_price_disagreement: float = 0.0
    share_jump: float = 0.08  # change in shares, beyond fund flows, that needs an explanation
    split_tol: float = 0.03   # how close the share ratio must be to a split ratio
    secondary: dict[date, set] = field(default_factory=dict)  # keys priced only by a fund added later

    def add(self, more: dict[date, pd.DataFrame]) -> "NportImpliedReturns":
        for d, h in more.items():
            h = _sleeve(h)
            if d not in self.snapshots:
                self.snapshots[d] = h
                continue
            have = _sleeve(self.snapshots[d])
            pa = (have.groupby("security_key")["value_usd"].sum() / have.groupby("security_key")["balance"].sum())
            pb = (h.groupby("security_key")["value_usd"].sum() / h.groupby("security_key")["balance"].sum())
            both = pa.index.intersection(pb.index)
            if len(both):
                self.max_price_disagreement = max(self.max_price_disagreement,
                                                  float((pb[both] / pa[both] - 1).abs().max()))
            extra = h[~h["security_key"].isin(pa.index)]
            self.secondary.setdefault(d, set()).update(extra["security_key"])
            self.snapshots[d] = pd.concat([have, extra], ignore_index=True)
        return self

    def period_returns(self, keys: Sequence[str], start: date, end: date) -> pd.DataFrame:
        out = _empty(keys, "")
        if start not in self.snapshots or end not in self.snapshots:
            missing = [str(d) for d in (start, end) if d not in self.snapshots]
            out["note"] = "no reference snapshot on " + ", ".join(missing)
            return out
        g0 = _sleeve(self.snapshots[start]).groupby("security_key")[["balance", "value_usd"]].sum()
        g1 = _sleeve(self.snapshots[end]).groupby("security_key")[["balance", "value_usd"]].sum()
        both = g0.index.intersection(g1.index)
        ratio = g1.loc[both, "balance"] / g0.loc[both, "balance"]
        # An index fund scales every position by the same factor when money flows in or out.
        # Dividing that factor out leaves 1 for an ordinary security and 2, 3, 0.5, ... for a split.
        # Securities priced only by a fund added later (an active fund) are a different matter:
        # their share counts change through trading, so they do not set the factor and a jump
        # in their shares is never read as a split.
        second = self.secondary.get(start, set()) | self.secondary.get(end, set())
        primary = [k for k in both if k not in second]
        scale = float(np.median(ratio[primary])) if primary else 1.0
        ratio = ratio.where(ratio.index.isin(second), ratio / scale)
        gross = (g1.loc[both, "value_usd"] / g1.loc[both, "balance"]) / (g0.loc[both, "value_usd"] / g0.loc[both, "balance"])
        for k in out.index:
            if k not in both:
                where = [str(d) for d, g in ((start, g0), (end, g1)) if k not in g.index]
                out.loc[k, "note"] = "not held by the reference fund on " + " or ".join(where)
                continue
            rk, note, split = float(ratio[k]), "N-PORT value / shares at both dates; dividends excluded", False
            growth = float(gross[k])
            if abs(rk - 1.0) > self.share_jump and k in second:
                out.loc[k, "note"] = (f"priced only by a fund that trades it and its shares changed by a factor "
                                      f"{rk:.3g}: a trade cannot be told from a split; return not derived")
                continue
            if abs(rk - 1.0) > self.share_jump:
                near = min(SPLIT_RATIOS, key=lambda s_: abs(rk / s_ - 1.0))
                if abs(rk / near - 1.0) > self.split_tol or not (0.5 <= growth * near <= 1.6):
                    # shares moved far more than flows explain and it does not look like a clean
                    # split: value / shares is not comparable across whatever happened
                    out.loc[k, "note"] = (f"shares changed by a factor {rk:.3g} beyond fund flows and no "
                                          "split explains it; return not derived")
                    continue
                growth, split = growth * near, True
                note += f"; adjusted for a {near:g}-for-1 share change inferred from the share count"
            out.loc[k, ["ret", "kind", "status", "split_adjusted", "note"]] = [growth - 1.0, "price", "derived", split, note]
        return out


@dataclass
class CsvReturns:
    """Returns from a user-supplied CSV (own or licensed data; never committed)."""
    path: str | Path
    name: str = "csv"
    licence: str = "user supplied; check the provider's terms before sharing"
    kind: str = "total"

    def period_returns(self, keys: Sequence[str], start: date, end: date) -> pd.DataFrame:
        df = pd.read_csv(self.path, dtype={"security_key": str})
        need = {"security_key", "start", "end", "total_return"}
        if not need <= set(df.columns):
            raise ValueError(f"returns CSV needs columns {sorted(need)}")
        df = df[(df["start"] == str(start)) & (df["end"] == str(end))].set_index("security_key")
        if not df.index.is_unique:
            raise ValueError("returns CSV has duplicate rows for a security and period")
        out = _empty(keys, f"not in {Path(self.path).name} for this period")
        for k in out.index.intersection(df.index):
            r = df.at[k, "total_return"]
            if pd.notna(r):
                out.loc[k, ["ret", "kind", "status", "note"]] = [float(r), self.kind, "disclosed", f"from {Path(self.path).name}"]
        return out


def nav_period_return(report, class_id: str | None = None) -> dict:
    """The fund's disclosed NAV total return over the three months ending on the report date
    of an ``NportReport``: the three monthly returns compounded. ``class_id`` defaults to
    the first class listed. NaN if any of the three months is missing."""
    mr = report.monthly_returns
    cid = class_id or (mr["class_id"].iloc[0] if len(mr) else None)
    rows = mr[mr["class_id"] == cid].sort_values("month")
    if len(rows) != 3 or rows["return_pct"].isna().any():
        return dict(class_id=cid, nav_return=float("nan"), status="unknown",
                    note="monthly total returns incomplete for this class")
    total = float(np.prod(1.0 + rows["return_pct"].to_numpy(dtype=float) / 100.0) - 1.0)
    return dict(class_id=cid, nav_return=total, status="disclosed",
                note="three monthly total returns from N-PORT, compounded; net of fees",
                period_start_month_end=str(rows["month_end"].iloc[0]), period_end=str(rows["month_end"].iloc[-1]))


def net_flow_ratio(report) -> float:
    """Net investor flow over the quarter ending on the report date, as a share of net
    assets at that date (sales + reinvestments - redemptions) / net assets. NaN if unknown."""
    f = report.flows
    if f[["sales", "redemption"]].isna().all().all() or not report.net_assets > 0:
        return float("nan")
    net = f["sales"].fillna(0).sum() + f["reinvestment"].fillna(0).sum() - f["redemption"].fillna(0).sum()
    return float(net / report.net_assets)
