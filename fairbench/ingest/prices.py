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
    only at the fund's report dates; a security that left the fund before the second date
    has no return; splits and other share-count changes are handled as the class docstring
    says, and every such case is flagged.
``ChainedReturns``  several sources in order of preference (for example the S&P 500 index
    fund, then a broader index fund for names that left it, then the fund's own filings).
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

RETURN_COLUMNS = ["ret", "kind", "status", "split_adjusted", "share_count_changed", "note"]
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
    df = pd.DataFrame({"ret": np.nan, "kind": None, "status": "unknown", "split_adjusted": False,
                       "share_count_changed": False, "note": note}, index=pd.Index(list(keys), name="security_key"))
    return df.astype({"kind": object, "status": object, "note": object})


def _sleeve(h: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for an implied price: keyed, positive shares and value."""
    ok = h["security_key"].notna() & ~h["security_key"].astype(str).str.startswith("unknown:")
    return h[ok & (h["balance"] > 0) & (h["value_usd"] > 0)]


@dataclass
class NportImpliedReturns:
    """Price returns implied by the N-PORT snapshots of ONE fund.

    snapshots: {report_date: frame with security_key, balance, value_usd} (equity sleeve).
    passive: True for an index fund. Its share counts move together with investor flows,
    so that common factor is divided out and what remains is each security's own share
    change. False for an actively managed fund, whose share counts move through trading:
    then a return is derived only while the position is unchanged (within ``share_jump``),
    because a trade cannot be told from a split.

    For a passive fund, a security whose share count changed by more than ``share_jump``
    beyond flows is handled by what the price did:
      * a split of 2-for-1 or more (or a reverse split) is recognised when the share ratio
        is near that ratio AND the price moved by about its inverse; the return is
        adjusted (``split_adjusted``, status "derived");
      * otherwise the shares changed without a matching price jump: an issue of new
        shares, a buyback or a change in the free float. With share_change="plain"
        (default) the plain price return is kept and marked status "assumed" and
        ``share_count_changed``, since a small stock split (3-for-2, 5-for-4) would look
        the same and would understate the return. With share_change="void" it is unknown.
    """
    snapshots: dict[date, pd.DataFrame] = field(default_factory=dict)
    name: str = "nport_implied"
    licence: str = "derived from public SEC Form N-PORT filings; redistributable"
    passive: bool = True
    share_change: str = "plain"
    share_jump: float = 0.08   # share change, beyond fund flows, that needs an explanation
    split_tol: float = 0.15    # how close the share ratio must be to a split ratio

    def period_returns(self, keys: Sequence[str], start: date, end: date) -> pd.DataFrame:
        if self.share_change not in ("plain", "void"):
            raise ValueError('share_change must be "plain" or "void"')
        out = _empty(keys, "")
        if start not in self.snapshots or end not in self.snapshots:
            missing = [str(d) for d in (start, end) if d not in self.snapshots]
            out["note"] = f"no {self.name} snapshot on " + ", ".join(missing)
            return out
        g0 = _sleeve(self.snapshots[start]).groupby("security_key")[["balance", "value_usd"]].sum()
        g1 = _sleeve(self.snapshots[end]).groupby("security_key")[["balance", "value_usd"]].sum()
        both = g0.index.intersection(g1.index)
        ratio = g1.loc[both, "balance"] / g0.loc[both, "balance"]
        if self.passive and len(both):
            ratio = ratio / float(np.median(ratio))
        gross = (g1.loc[both, "value_usd"] / g1.loc[both, "balance"]) / (g0.loc[both, "value_usd"] / g0.loc[both, "balance"])
        base = f"{self.name}: N-PORT value / shares at both dates; dividends excluded"
        for k in out.index:
            if k not in both:
                where = [str(d) for d, g in ((start, g0), (end, g1)) if k not in g.index]
                out.loc[k, "note"] = f"not held by {self.name} on " + " or ".join(where)
                continue
            rk, growth = float(ratio[k]), float(gross[k])
            status, split, changed, note = "derived", False, False, base
            if abs(rk - 1.0) > self.share_jump:
                if not self.passive:
                    out.loc[k, "note"] = (f"{self.name} trades this security and its shares changed by a factor "
                                          f"{rk:.3g}: a trade cannot be told from a split; return not derived")
                    continue
                near = min(SPLIT_RATIOS, key=lambda s_: abs(np.log(rk / s_)))
                if abs(rk / near - 1.0) <= self.split_tol and 0.5 <= growth * near <= 1.6:
                    growth, split = growth * near, True
                    note += f"; adjusted for a {near:g}-for-1 share change (shares x{rk:.3g}, price moved to match)"
                elif self.share_change == "plain" and 0.4 <= growth <= 2.5:
                    status, changed = "assumed", True
                    note += (f"; shares changed x{rk:.3g} beyond fund flows with no matching price jump, taken as an "
                             "issue, buyback or float change (a small stock split would not be detected)")
                else:
                    out.loc[k, "note"] = (f"shares changed by a factor {rk:.3g} beyond fund flows and no "
                                          "split explains it; return not derived")
                    continue
            out.loc[k, ["ret", "kind", "status", "split_adjusted", "share_count_changed", "note"]] = \
                [growth - 1.0, "price", status, split, changed, note]
        return out


@dataclass
class ChainedReturns:
    """Several sources in order of preference: each security takes its return from the
    first source that knows it. ``max_disagreement`` (set by each call) is the largest
    absolute difference between two sources that both know a return, a data-quality check."""
    sources: list
    name: str = "chained"
    licence: str = "see the individual sources"
    max_disagreement: float = 0.0

    def period_returns(self, keys: Sequence[str], start: date, end: date) -> pd.DataFrame:
        out = None
        for src in self.sources:
            r = src.period_returns(keys, start, end)
            if out is None:
                out = r
                continue
            both = out["ret"].notna() & r["ret"].notna()
            if both.any():
                self.max_disagreement = max(self.max_disagreement, float((out.loc[both, "ret"] - r.loc[both, "ret"]).abs().max()))
            take = out["ret"].isna() & r["ret"].notna()
            still = out["ret"].isna() & r["ret"].isna()   # nobody knows yet: keep every source's reason
            out.loc[still, "note"] = out.loc[still, "note"].astype(str) + "; " + r.loc[still, "note"].astype(str)
            out.loc[take] = r.loc[take]
        return out if out is not None else _empty(keys, "no sources")


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


def nav_period_return(report, class_id: str | None = None, reference: float | None = None) -> dict:
    """The fund's disclosed NAV total return over the three months ending on the report date
    of an ``NportReport``: the three monthly returns compounded. ``class_id`` defaults to
    the lowest class id (stable across filings). NaN if any of the three months is missing.

    UNITS. The form asks for percent (2.04 for +2.04%), but some filings report fractions
    (0.0204). The two readings differ by a factor of 100. With ``reference`` (an independent
    estimate of the same quarter's return, such as the frozen-holdings return) the reading
    closer to it is chosen and the result is marked "derived" with the inferred unit;
    without it the form's stated unit, percent, is used and the note says it was not checked."""
    mr = report.monthly_returns
    cid = class_id or (min(mr["class_id"]) if len(mr) else None)
    rows = mr[mr["class_id"] == cid].sort_values("month")
    if len(rows) != 3 or rows["return_pct"].isna().any():
        return dict(class_id=cid, nav_return=float("nan"), status="unknown", units=None,
                    note="monthly total returns incomplete for this class")
    raw = rows["return_pct"].to_numpy(dtype=float)
    as_percent = float(np.prod(1.0 + raw / 100.0) - 1.0)
    as_fraction = float(np.prod(1.0 + raw) - 1.0) if (np.abs(raw) < 1).all() else float("nan")
    out = dict(class_id=cid, period_start_month_end=str(rows["month_end"].iloc[0]), period_end=str(rows["month_end"].iloc[-1]))
    if reference is None or np.isnan(reference) or np.isnan(as_fraction):
        return dict(out, nav_return=as_percent, status="disclosed", units="percent",
                    note="three monthly total returns from N-PORT, compounded; net of fees"
                         + ("" if reference is not None else "; units not checked against an independent estimate"))
    pick_fraction = abs(as_fraction - reference) < abs(as_percent - reference)
    return dict(out, nav_return=as_fraction if pick_fraction else as_percent, status="derived" if pick_fraction else "disclosed",
                units="fraction (inferred)" if pick_fraction else "percent",
                note="three monthly total returns from N-PORT, compounded; net of fees"
                     + ("; the filing reports fractions, not percent: reading chosen by closeness to the "
                        "frozen-holdings return" if pick_fraction else ""))


def net_flow_ratio(report) -> float:
    """Net investor flow over the quarter ending on the report date, as a share of net
    assets at that date (sales + reinvestments - redemptions) / net assets. NaN if unknown."""
    f = report.flows
    if f[["sales", "redemption"]].isna().all().all() or not report.net_assets > 0:
        return float("nan")
    net = f["sales"].fillna(0).sum() + f["reinvestment"].fillna(0).sum() - f["redemption"].fillna(0).sum()
    return float(net / report.net_assets)
