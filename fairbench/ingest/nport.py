"""Parser for SEC Form N-PORT filings: the public primary_doc.xml of a fund's quarter-end report.

Contract
    ``parse_nport(xml)`` turns the bytes (or text) of one N-PORT document into an ``NportReport``.
    It is a pure function: no network, no files. Elements are matched on their local names, so a
    file with a namespace prefix or a slightly different namespace URI parses the same.

    A field absent from the file is None (text, bool) or NaN (numbers); nothing is guessed. The
    placeholders "N/A" and "" are read as absent. A cusip of "000000000" is kept as filed: whether
    it is a real identifier is for a later module to decide. Values keep the units they were filed
    in: ``pct_net_assets`` and ``return_pct`` are PERCENT (10.33 is +10.33%), and only
    ``NportReport.equity_sleeve`` converts to fractions.

    Nothing is dropped silently. Every ``invstOrSec`` becomes exactly one holdings row whose
    ``row_index`` is its 0-based position, so ``NportReport.locator`` points back to the source
    element. Each class in ``monthlyTotReturns`` gives three return rows, and flows / other gains
    always give three rows (months 1-3), with NaN where the filing is silent. A number that is
    present but not numeric raises ValueError instead of becoming NaN.

Not parsed: the contents of ``debtSec``, ``repurchaseAgrmt`` and ``derivativeInfo`` (only the
presence of ``derivativeInfo`` is recorded), ``monthlyReturnCats`` and the signature block.
"""
from __future__ import annotations

import calendar
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

import pandas as pd

HOLDING_COLUMNS = [
    "row_index", "issuer_name", "issuer_lei", "title", "cusip", "isin", "ticker", "other_id",
    "other_id_desc", "balance", "units", "currency", "exchange_rate", "value_usd", "pct_net_assets",
    "payoff_profile", "asset_category", "issuer_category", "country", "is_restricted",
    "fair_value_level", "is_loaned", "loan_value", "has_derivative_info",
]
_FLOAT_COLUMNS = ["balance", "exchange_rate", "value_usd", "pct_net_assets", "loan_value"]
_NAN = float("nan")


# ---------------------------------------------------------------- XML helpers (local names only)
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _kids(el: ET.Element | None, name: str) -> list[ET.Element]:
    return [] if el is None else [c for c in el if _local(c.tag) == name]


def _find(el: ET.Element | None, *path: str) -> ET.Element | None:
    for name in path:
        if el is None:
            return None
        el = next((c for c in el if _local(c.tag) == name), None)
    return el


def _clean(s: str | None) -> str | None:
    s = (s or "").strip()
    return None if s == "" or s.upper() == "N/A" else s


def _text(el: ET.Element | None, *path: str) -> str | None:
    node = _find(el, *path)
    return None if node is None else _clean(node.text)


def _attr(el: ET.Element | None, name: str) -> str | None:
    return None if el is None else _clean(el.get(name))


def _num(s: str | None, what: str) -> float:
    if s is None:
        return _NAN
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{what}: not a number: {s!r}") from None


def _yn(s: str | None, what: str) -> bool | None:
    if s is None:
        return None
    if s.upper() not in ("Y", "N"):
        raise ValueError(f"{what}: expected Y or N, got {s!r}")
    return s.upper() == "Y"


def _date(s: str | None, what: str) -> date | None:
    if s is None:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{what}: not a yyyy-mm-dd date: {s!r}") from None


def _cik10(s: str | None, what: str) -> str | None:
    if s is None:
        return None
    if not s.isdigit():
        raise ValueError(f"{what}: not a numeric CIK: {s!r}")
    return s.zfill(10)


def _month_end(report_date: date, month: int) -> date:
    """Last calendar day of month ``month`` (1..3) of the quarter that ends in report_date's month."""
    y, m = divmod(report_date.year * 12 + report_date.month - 1 - (3 - month), 12)
    return date(y, m + 1, calendar.monthrange(y, m + 1)[1])


def _frame(rows: list[dict], columns: list[str], floats: list[str],
           ints: tuple[str, ...] = ()) -> pd.DataFrame:
    """Object-dtype frame (so None stays None), then numeric columns cast; works for zero rows."""
    df = pd.DataFrame(rows, columns=columns, dtype=object)
    for c in floats:
        df[c] = df[c].astype(float)
    for c in ints:
        df[c] = df[c].astype("int64")
    return df


# ---------------------------------------------------------------- one holding
def _first_value(ident: ET.Element | None, name: str) -> ET.Element | None:
    """First identifier element of that name that carries a value."""
    return next((e for e in _kids(ident, name) if _attr(e, "value") is not None), None)


def _holding(el: ET.Element, i: int) -> dict:
    ident = _find(el, "identifiers")
    other = _first_value(ident, "other")
    cur_cond = _find(el, "currencyConditional")
    lending = _find(el, "securityLending")
    loan = _find(lending, "loanByFundCondition")
    tag = f"invstOrSec[{i + 1}]"

    def category(plain: str, cond: str, attr: str) -> str | None:
        return _text(el, plain) or _attr(_find(el, cond), attr)

    return {
        "row_index": i,
        "issuer_name": _text(el, "name"),
        "issuer_lei": _text(el, "lei"),
        "title": _text(el, "title"),
        "cusip": _text(el, "cusip"),
        "isin": _attr(_first_value(ident, "isin"), "value"),
        "ticker": _attr(_first_value(ident, "ticker"), "value"),
        "other_id": _attr(other, "value"),
        "other_id_desc": _attr(other, "otherDesc"),
        "balance": _num(_text(el, "balance"), f"{tag}/balance"),
        "units": _text(el, "units"),
        "currency": _text(el, "curCd") or _attr(cur_cond, "curCd"),
        "exchange_rate": _num(_attr(cur_cond, "exchangeRt"), f"{tag}/currencyConditional@exchangeRt"),
        "value_usd": _num(_text(el, "valUSD"), f"{tag}/valUSD"),
        "pct_net_assets": _num(_text(el, "pctVal"), f"{tag}/pctVal"),
        "payoff_profile": _text(el, "payoffProfile"),
        "asset_category": category("assetCat", "assetConditional", "assetCat"),
        "issuer_category": category("issuerCat", "issuerConditional", "issuerCat"),
        "country": _text(el, "invCountry"),
        "is_restricted": _yn(_text(el, "isRestrictedSec"), f"{tag}/isRestrictedSec"),
        "fair_value_level": _text(el, "fairValLevel"),
        "is_loaned": (_yn(_attr(loan, "isLoanByFund"), f"{tag}/loanByFundCondition@isLoanByFund")
                      if loan is not None else
                      _yn(_text(lending, "isLoanByFund"), f"{tag}/securityLending/isLoanByFund")),
        "loan_value": _num(_attr(loan, "loanVal"), f"{tag}/loanByFundCondition@loanVal"),
        "has_derivative_info": _find(el, "derivativeInfo") is not None,
    }


def _holdings_frame(root: ET.Element) -> pd.DataFrame:
    block = _find(root, "formData", "invstOrSecs")
    rows = [_holding(el, i) for i, el in enumerate(_kids(block, "invstOrSec"))]
    df = _frame(rows, HOLDING_COLUMNS, _FLOAT_COLUMNS, ["row_index"])
    df["has_derivative_info"] = df["has_derivative_info"].astype(bool)
    return df


# ---------------------------------------------------------------- the report
@dataclass(eq=False)  # DataFrame fields make generated __eq__ ambiguous
class NportReport:
    submission_type: str | None
    registrant_name: str | None
    registrant_cik: str | None  # 10 digits, zero padded
    registrant_lei: str | None
    file_number: str | None
    series_name: str | None
    series_id: str | None
    series_lei: str | None
    class_ids: list[str]
    fiscal_year_end: date | None  # repPdEnd: the fund's fiscal year end
    report_date: date  # repPdDate: the date the holdings are reported as of
    is_final_filing: bool | None
    total_assets: float
    total_liabilities: float
    net_assets: float
    misc_securities_assets: float
    cash_not_reported: float
    monthly_returns: pd.DataFrame  # class_id, month, month_end, return_pct (PERCENT)
    flows: pd.DataFrame  # month, month_end, sales, reinvestment, redemption (USD)
    other_gains: pd.DataFrame  # month, net_realized_gain, net_unrealized_appreciation
    holdings: pd.DataFrame  # HOLDING_COLUMNS

    def locator(self, row_index: int) -> str:
        """XPath-style path of a holdings row inside the source document (1-based position)."""
        if not 0 <= row_index < len(self.holdings):
            raise IndexError(f"row_index {row_index} outside 0..{len(self.holdings) - 1}")
        return f"invstOrSecs/invstOrSec[{int(row_index) + 1}]"

    def equity_sleeve(self) -> pd.DataFrame:
        """Long common-stock positions held in shares with positive balance and value.

        Keeps the holdings index (== row_index) and adds ``sleeve_weight``, the share of the sleeve's
        own value_usd as a fraction, so the weights sum to 1. Empty if no position qualifies.
        """
        h = self.holdings
        keep = ((h["asset_category"] == "EC") & (h["payoff_profile"] == "Long") & (h["units"] == "NS")
                & (h["balance"] > 0) & (h["value_usd"] > 0))
        sleeve = h[keep].copy()
        sleeve["sleeve_weight"] = sleeve["value_usd"] / sleeve["value_usd"].sum()
        return sleeve

    def implied_price(self) -> pd.Series:
        """value_usd / balance for each equity-sleeve row (USD per share), indexed like the sleeve."""
        sleeve = self.equity_sleeve()
        return (sleeve["value_usd"] / sleeve["balance"]).rename("implied_price")


def _monthly_returns(fund: ET.Element | None, report_date: date) -> pd.DataFrame:
    block = _find(fund, "returnInfo", "monthlyTotReturns")
    rows = []
    for el in _kids(block, "monthlyTotReturn"):
        cls = _attr(el, "classId")
        for m in (1, 2, 3):
            rows.append({"class_id": cls, "month": m, "month_end": _month_end(report_date, m),
                         "return_pct": _num(_attr(el, f"rtn{m}"), f"monthlyTotReturn[{cls}]@rtn{m}")})
    return _frame(rows, ["class_id", "month", "month_end", "return_pct"], ["return_pct"], ["month"])


def _flows(fund: ET.Element | None, report_date: date) -> pd.DataFrame:
    rows = []
    for m in (1, 2, 3):
        el = _find(fund, f"mon{m}Flow")
        rows.append({"month": m, "month_end": _month_end(report_date, m),
                     **{k: _num(_attr(el, a), f"mon{m}Flow@{a}")
                        for k, a in (("sales", "sales"), ("reinvestment", "reinvestment"),
                                     ("redemption", "redemption"))}})
    return _frame(rows, ["month", "month_end", "sales", "reinvestment", "redemption"],
                  ["sales", "reinvestment", "redemption"], ["month"])


def _other_gains(fund: ET.Element | None) -> pd.DataFrame:
    rows = []
    for m in (1, 2, 3):
        el = _find(fund, "returnInfo", f"othMon{m}")
        rows.append({"month": m,
                     "net_realized_gain": _num(_attr(el, "netRealizedGain"), f"othMon{m}@netRealizedGain"),
                     "net_unrealized_appreciation": _num(_attr(el, "netUnrealizedAppr"),
                                                         f"othMon{m}@netUnrealizedAppr")})
    return _frame(rows, ["month", "net_realized_gain", "net_unrealized_appreciation"],
                  ["net_realized_gain", "net_unrealized_appreciation"], ["month"])


def parse_nport(xml: bytes | str) -> NportReport:
    """Parse one N-PORT document. ValueError if it is not well-formed, not an N-PORT submission,
    has no repPdDate, or holds a malformed number, date or Y/N flag."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"not well-formed XML: {e}") from e
    if _local(root.tag) != "edgarSubmission":
        raise ValueError(f"not an N-PORT submission: root element is <{_local(root.tag)}>, "
                         "expected <edgarSubmission>")
    sub_type = _text(root, "headerData", "submissionType")
    if sub_type is not None and not sub_type.upper().startswith("NPORT"):
        raise ValueError(f"not an N-PORT submission: submissionType is {sub_type!r}")

    gen = _find(root, "formData", "genInfo")
    report_date = _date(_text(gen, "repPdDate"), "genInfo/repPdDate")
    if report_date is None:
        raise ValueError("N-PORT document has no genInfo/repPdDate (the report date)")

    fund = _find(root, "formData", "fundInfo")
    class_ids = [c for sci in _kids(_find(root, "headerData", "filerInfo"), "seriesClassInfo")
                 for el in _kids(sci, "classId") if (c := _clean(el.text)) is not None]
    return NportReport(
        submission_type=sub_type,
        registrant_name=_text(gen, "regName"),
        registrant_cik=_cik10(_text(gen, "regCik"), "genInfo/regCik"),
        registrant_lei=_text(gen, "regLei"),
        file_number=_text(gen, "regFileNumber"),
        series_name=_text(gen, "seriesName"),
        series_id=_text(gen, "seriesId"),
        series_lei=_text(gen, "seriesLei"),
        class_ids=class_ids,
        fiscal_year_end=_date(_text(gen, "repPdEnd"), "genInfo/repPdEnd"),
        report_date=report_date,
        is_final_filing=_yn(_text(gen, "isFinalFiling"), "genInfo/isFinalFiling"),
        total_assets=_num(_text(fund, "totAssets"), "fundInfo/totAssets"),
        total_liabilities=_num(_text(fund, "totLiabs"), "fundInfo/totLiabs"),
        net_assets=_num(_text(fund, "netAssets"), "fundInfo/netAssets"),
        misc_securities_assets=_num(_text(fund, "assetsAttrMiscSec"), "fundInfo/assetsAttrMiscSec"),
        cash_not_reported=_num(_text(fund, "cshNotRptdInCorD"), "fundInfo/cshNotRptdInCorD"),
        monthly_returns=_monthly_returns(fund, report_date),
        flows=_flows(fund, report_date),
        other_gains=_other_gains(fund),
        holdings=_holdings_frame(root),
    )
