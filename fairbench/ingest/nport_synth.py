"""Writer of SYNTHETIC N-PORT documents, for offline tests and examples.

``build_nport_xml`` turns plain Python inputs into a document with the element names, order and
namespaces of a real N-PORT filing (see ``fairbench.ingest.nport``). The output is NOT a filing:
it describes no real fund. Callers should use an obviously fictional registrant name, such as
"Synthetic Example Trust (not a real fund)", so that a synthetic file cannot be mistaken for data.

Contract
    ``parse_nport(build_nport_xml(...))`` reproduces the inputs: every given holdings value comes
    back unchanged (floats are written at full repr precision, never in exponent form).
    ``holdings`` uses the HOLDING_COLUMNS names. ``issuer_name`` and ``value_usd`` are required.
    A column that is absent gets the documented default (curCd USD, payoffProfile Long, assetCat
    EC, issuerCat CORP, invCountry US, isRestrictedSec N, fairValLevel 1, not loaned); a column
    that is present but null in a row omits that element, so it parses back as absent.
    ``balance``, ``units``, ``title``, ``cusip`` and the identifiers have no default and are
    written only when given. ``pct_net_assets`` is 100 * value_usd / net_assets when absent.
    ``row_index`` is ignored: positions follow row order.
Writer-only choices, none of which the parser reads: ``desc`` of an "OTHER" asset or issuer
category is the literal "Other"; a loaned row has no cash or non-cash collateral; a row with
``has_derivative_info`` gets an empty ``derivativeInfo`` element; ``ccc`` is "XXXXXXXX".
"""
from __future__ import annotations

import calendar

import xml.etree.ElementTree as ET
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from .nport import HOLDING_COLUMNS

_NS = {
    "xmlns": "http://www.sec.gov/edgar/nport",
    "xmlns:com": "http://www.sec.gov/edgar/common",
    "xmlns:ncom": "http://www.sec.gov/edgar/nportcommon",
    "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
}
# Defaults for holdings columns that are absent altogether.
_DEFAULTS = {"currency": "USD", "payoff_profile": "Long", "asset_category": "EC",
             "issuer_category": "CORP", "country": "US", "is_restricted": False,
             "fair_value_level": "1", "is_loaned": False}


def _isnull(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA


def _num(x: float) -> str:
    """Full-precision decimal text that float() reads back exactly (repr, without exponent)."""
    return format(Decimal(repr(float(x))), "f")


def _text(x) -> str:
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))  # a fair value level of 1.0 is written "1"
    return str(x)


def _add(parent: ET.Element, tag: str, text: str | None = None, **attrib: str) -> ET.Element:
    el = ET.SubElement(parent, tag, {k: v for k, v in attrib.items() if v is not None})
    if text is not None:
        el.text = text
    return el


def _opt(parent: ET.Element, tag: str, value, as_number: bool = False) -> None:
    if not _isnull(value):
        _add(parent, tag, _num(value) if as_number else _text(value))


def _yn(flag: bool) -> str:
    return "Y" if flag else "N"


def _get(row: pd.Series, col: str):
    """Row value of a column, or its documented default when the column is absent."""
    return row[col] if col in row.index else _DEFAULTS.get(col)


def _holding_element(parent: ET.Element, row: pd.Series, net_assets: float) -> None:
    el = ET.SubElement(parent, "invstOrSec")
    _opt(el, "name", row["issuer_name"])
    _opt(el, "lei", _get(row, "issuer_lei"))
    _opt(el, "title", _get(row, "title"))
    _opt(el, "cusip", _get(row, "cusip"))
    ids = [("isin", {"value": _get(row, "isin")}), ("ticker", {"value": _get(row, "ticker")}),
           ("other", {"otherDesc": _get(row, "other_id_desc"), "value": _get(row, "other_id")})]
    ids = [(tag, {k: _text(v) for k, v in a.items() if not _isnull(v)}) for tag, a in ids]
    if any(a for _, a in ids):
        ident = ET.SubElement(el, "identifiers")
        for tag, a in ids:
            if a:
                ET.SubElement(ident, tag, a)
    _opt(el, "balance", _get(row, "balance"), as_number=True)
    _opt(el, "units", _get(row, "units"))

    cur, rate = _get(row, "currency"), _get(row, "exchange_rate")
    if _isnull(rate) and cur == "USD":
        _add(el, "curCd", "USD")
    elif not (_isnull(cur) and _isnull(rate)):
        _add(el, "currencyConditional", curCd=None if _isnull(cur) else str(cur),
             exchangeRt=None if _isnull(rate) else _num(rate))
    _opt(el, "valUSD", row["value_usd"], as_number=True)
    pct = row["pct_net_assets"] if "pct_net_assets" in row.index else \
        100.0 * float(row["value_usd"]) / net_assets
    _opt(el, "pctVal", pct, as_number=True)
    _opt(el, "payoffProfile", _get(row, "payoff_profile"))
    for plain, cond, attr, col in (("assetCat", "assetConditional", "assetCat", "asset_category"),
                                   ("issuerCat", "issuerConditional", "issuerCat", "issuer_category")):
        v = _get(row, col)
        if v == "OTHER":
            _add(el, cond, desc="Other", **{attr: "OTHER"})
        else:
            _opt(el, plain, v)
    _opt(el, "invCountry", _get(row, "country"))
    restricted = _get(row, "is_restricted")
    if not _isnull(restricted):
        _add(el, "isRestrictedSec", _yn(bool(restricted)))
    _opt(el, "fairValLevel", _get(row, "fair_value_level"))
    if "has_derivative_info" in row.index and not _isnull(row["has_derivative_info"]) \
            and row["has_derivative_info"]:
        _add(el, "derivativeInfo")
    loaned = _get(row, "is_loaned")
    if not _isnull(loaned):
        lend = ET.SubElement(el, "securityLending")
        _add(lend, "isCashCollateral", "N")
        _add(lend, "isNonCashCollateral", "N")
        if loaned:
            loan_value = _get(row, "loan_value")
            _add(lend, "loanByFundCondition", isLoanByFund="Y",
                 loanVal=None if _isnull(loan_value) else _num(loan_value))
        else:
            _add(lend, "isLoanByFund", "N")


def build_nport_xml(*, registrant_name: str, registrant_cik: str, series_name: str, series_id: str,
                    class_ids: list[str], fiscal_year_end: date, report_date: date, net_assets: float,
                    total_assets: float | None = None, total_liabilities: float = 0.0,
                    holdings: pd.DataFrame,
                    monthly_returns: dict[str, tuple[float, float, float]] | None = None,
                    flows: list[dict] | None = None, submission_type: str = "NPORT-P",
                    file_number: str = "811-00000") -> bytes:
    """Write a SYNTHETIC N-PORT document (UTF-8 bytes) from plain inputs; see the module docstring.

    ``monthly_returns`` maps class id -> (rtn1, rtn2, rtn3) in PERCENT; NaN is written "N/A".
    ``flows`` is positional, entry i for month i + 1, a dict with any of the keys ``sales``,
    ``reinvestment``, ``redemption`` (a missing key is a missing attribute). ``total_assets``
    defaults to net_assets + total_liabilities.
    """
    missing = {"issuer_name", "value_usd"} - set(holdings.columns)
    if missing:
        raise ValueError(f"holdings is missing required columns: {sorted(missing)}")
    unknown = set(holdings.columns) - set(HOLDING_COLUMNS)
    if unknown:
        raise ValueError(f"holdings has columns outside HOLDING_COLUMNS: {sorted(unknown)}")
    if flows is not None and len(flows) > 3:
        raise ValueError(f"flows has one entry per month, at most 3; got {len(flows)}")
    if total_assets is None:
        total_assets = net_assets + total_liabilities

    root = ET.Element("edgarSubmission", dict(_NS))
    head = ET.SubElement(root, "headerData")
    _add(head, "submissionType", submission_type)
    _add(head, "isConfidential", "false")
    filer_info = ET.SubElement(head, "filerInfo")
    creds = ET.SubElement(ET.SubElement(filer_info, "filer"), "issuerCredentials")
    _add(creds, "cik", registrant_cik)
    _add(creds, "ccc", "XXXXXXXX")
    sci = ET.SubElement(filer_info, "seriesClassInfo")
    _add(sci, "seriesId", series_id)
    for c in class_ids:
        _add(sci, "classId", c)

    form = ET.SubElement(root, "formData")
    gen = ET.SubElement(form, "genInfo")
    _add(gen, "regName", registrant_name)
    _add(gen, "regFileNumber", file_number)
    _add(gen, "regCik", registrant_cik)
    _add(gen, "seriesName", series_name)
    _add(gen, "seriesId", series_id)
    _add(gen, "repPdEnd", fiscal_year_end.isoformat())
    _add(gen, "repPdDate", report_date.isoformat())
    _add(gen, "isFinalFiling", "N")

    fund = ET.SubElement(form, "fundInfo")
    _add(fund, "totAssets", _num(total_assets))
    _add(fund, "totLiabs", _num(total_liabilities))
    _add(fund, "netAssets", _num(net_assets))
    if monthly_returns:
        rtns = ET.SubElement(ET.SubElement(fund, "returnInfo"), "monthlyTotReturns")
        for cls, vals in monthly_returns.items():
            _add(rtns, "monthlyTotReturn", classId=cls,
                 **{f"rtn{i + 1}": "N/A" if _isnull(v) else _num(v) for i, v in enumerate(vals)})
    for i, flow in enumerate(flows or []):
        _add(fund, f"mon{i + 1}Flow",
             **{k: _num(flow[k]) for k in ("redemption", "reinvestment", "sales")
                if k in flow and not _isnull(flow[k])})

    block = ET.SubElement(form, "invstOrSecs")
    for _, row in holdings.iterrows():
        _holding_element(block, row, net_assets)

    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


# ------------------------------------------------------------ example history
def _quarter_ends(start: date, n: int) -> list[date]:
    out, y, m = [], start.year, start.month
    for _ in range(n):
        out.append(date(y, m, calendar.monthrange(y, m)[1]))
        y, m = (y + 1, m - 9) if m > 9 else (y, m + 3)
    return out


def example_history(seed: int = 0, n_assets: int = 60, k: int = 15, n_quarters: int = 6,
                    start: date = date(2024, 3, 31)) -> dict:
    """A SYNTHETIC parent index fund and ESG fund over ``n_quarters`` quarter-ends, as
    N-PORT documents, for offline tests and the worked example. Nothing here is real.

    Planted on purpose, so the pipeline's handling of each can be checked against the truth:
      * company 7 splits 2-for-1 between the 3rd and 4th dates (shares double, price halves);
      * company 11 leaves the index after the 4th date (no return is derivable afterwards);
      * every 10th company is on an exclusion list that the ESG fund obeys, and those
        companies earn an extra 3% a quarter, so the rule has a cost;
      * the ESG fund holds ``k`` names, weights capped at 12%, with sticky preferences;
      * the fund's monthly returns are its true frozen-holdings return less a 0.05% monthly fee.

    Returns {"parent": {date: xml bytes}, "fund": {date: xml bytes}, "meta": {...}}; meta has
    the series ids, the excluded security keys and, per period, the fund's true frozen price
    return and every company's true price return.
    """
    rng = np.random.default_rng(seed)
    dates = _quarter_ends(start, n_quarters)
    from .identifiers import cusip_check_digit

    cusips = [(lambda b: b + cusip_check_digit(b))(f"99{i:04d}10") for i in range(n_assets)]
    names = [f"Synthetic Co {i:03d}" for i in range(n_assets)]
    excluded = np.arange(n_assets) % 10 == 0
    shares_out = np.round(rng.lognormal(18.0, 0.9, n_assets))
    price = rng.uniform(20, 300, n_assets)
    beta = rng.uniform(0.6, 1.4, n_assets)
    pref = rng.normal(0, 1, n_assets) + np.log(shares_out * price) * 0.3
    in_index = np.ones(n_assets, dtype=bool)
    parent_scale, fund_aum = 1e-3, 5e8
    parent_xml, fund_xml, truth = {}, {}, []
    prev_fund_w, prev_price = None, None

    def frame(idx, shares, px, extra_cash=0.0):
        df = pd.DataFrame({"issuer_name": [names[i] for i in idx], "title": [f"{names[i]} common stock" for i in idx],
                           "cusip": [cusips[i] for i in idx], "balance": shares, "units": "NS",
                           "value_usd": shares * px, "asset_category": "EC", "issuer_category": "CORP"})
        if extra_cash:
            df = pd.concat([df, pd.DataFrame({"issuer_name": ["Synthetic Cash Sweep Vehicle"], "title": ["cash sweep"],
                                              "cusip": ["000000000"], "balance": [extra_cash], "units": ["NS"],
                                              "value_usd": [extra_cash], "asset_category": ["STIV"],
                                              "issuer_category": ["RF"]})], ignore_index=True)
        return df

    for t, d in enumerate(dates):
        if t > 0:
            m = rng.normal(0.02, 0.06)
            r = 0.005 + beta * m + rng.normal(0, 0.08, n_assets) + 0.03 * excluded
            true_frozen = float(prev_fund_w @ r)
            price = price * (1 + r)
            if t == 3:  # 2-for-1 split in company 7
                shares_out[7] *= 2
                price[7] /= 2
            if t == 4:
                in_index[11] = False
            fund_aum *= (1 + true_frozen) * (1 + rng.uniform(-0.03, 0.02))
            parent_scale *= 1 + rng.uniform(-0.02, 0.03)
            truth.append({"start": str(dates[t - 1]), "end": str(d), "fund_frozen_price_return": true_frozen,
                          "asset_price_returns": {f"cusip:{cusips[i]}": float(r[i]) for i in range(n_assets)}})
            monthly = 100 * ((1 + true_frozen) ** (1 / 3) - 1 - 0.0005)  # N-PORT reports percent
            fund_returns = {"C000000091": (monthly, monthly, monthly), "C000000092": (monthly - 0.02,) * 3}
        else:
            fund_returns = None
        members = np.flatnonzero(in_index)
        p_shares = np.round(parent_scale * shares_out[members])
        p_frame = frame(members, p_shares, price[members], extra_cash=2e5)
        parent_xml[d] = build_nport_xml(
            registrant_name="Synthetic Example Index Trust (not a real fund)", registrant_cik="0000000001",
            series_name="Synthetic Broad Market Index Fund (not a real fund)", series_id="S000000001",
            class_ids=["C000000001"], fiscal_year_end=date(d.year, 12, 31), report_date=d,
            net_assets=float(p_frame["value_usd"].sum()), holdings=p_frame)

        eligible = np.flatnonzero(in_index & ~excluded)
        score = pref[eligible] + rng.normal(0, 0.25, eligible.size)
        held = eligible[np.argsort(-score)[:k]]
        raw = np.sqrt(shares_out[held] * price[held])
        w = raw / raw.sum()
        for _ in range(k):  # cap at 12%, spread the excess
            over = w > 0.12
            if not over.any():
                break
            w = np.where(over, 0.12, w * (1 - 0.12 * over.sum()) / w[~over].sum())
        f_shares = np.round(w * fund_aum * 0.99 / price[held])
        f_frame = frame(held, f_shares, price[held], extra_cash=round(fund_aum * 0.01))
        fund_xml[d] = build_nport_xml(
            registrant_name="Synthetic Example ESG Trust (not a real fund)", registrant_cik="0000000002",
            series_name="Synthetic Example ESG Equity Fund (not a real fund)", series_id="S000000002",
            class_ids=["C000000091", "C000000092"], fiscal_year_end=date(d.year, 12, 31), report_date=d,
            net_assets=float(f_frame["value_usd"].sum()), holdings=f_frame, monthly_returns=fund_returns,
            flows=None if t == 0 else [{"sales": 1e6, "reinvestment": 0.0, "redemption": 2e6}] * 3)
        prev_fund_w = np.zeros(n_assets)
        vals = f_shares * price[held]
        prev_fund_w[held] = vals / vals.sum()
        prev_price = price.copy()
    meta = {"synthetic": True, "seed": seed, "dates": [str(d) for d in dates],
            "parent_series_id": "S000000001", "fund_series_id": "S000000002",
            "excluded_keys": [f"cusip:{cusips[i]}" for i in np.flatnonzero(excluded)],
            "split": {"key": f"cusip:{cusips[7]}", "ratio": 2, "between": [str(dates[2]), str(dates[3])]},
            "left_index": {"key": f"cusip:{cusips[11]}", "after": str(dates[3])}, "periods": truth}
    return {"parent": parent_xml, "fund": fund_xml, "meta": meta}
