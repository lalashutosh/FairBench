"""Industry codes for portfolio companies, and an exclusion list built from them.

A mandate says "no fossil fuels, no tobacco". Public filings do not say which holdings
those are: N-PORT carries no sector field and the classifications funds use are licensed.
What IS public is the SEC's own Standard Industrial Classification (SIC) code for every
company that files with it. This module links holdings to SEC registrants by company name
and turns SIC ranges into an exclusion list.

It is a PROXY, and every row it produces is labelled "assumed":
  * SIC describes a company's main line of business, not its revenue split, so a "10% of
    revenue" test cannot be reproduced; a diversified company with a tobacco or oil arm is
    not caught, and a company is either in or out;
  * the name link is by normalised company name; only unique matches are accepted, the
    method is recorded per row, and unmatched holdings are reported (never guessed);
  * the SIC code is the one on file today, applied to every past date;
  * ``EXCLUSION_SIC`` covers fossil fuel extraction, services and refining, tobacco, makers
    of beer, wine and spirits, and ordnance and missiles. Most defence contractors, casino
    operators and diversified drinks companies sit under broad codes shared with ordinary
    businesses and are NOT captured.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Iterable

import pandas as pd

from .edgar import series_filings_url

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"   # registrants with a ticker today
CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"  # every registrant name, past and present

EXCLUSION_SIC: dict[str, tuple[tuple[int, int], ...]] = {
    "fossil fuels (coal mining; oil and gas extraction and field services; petroleum refining)":
        ((1200, 1299), (1300, 1399), (2900, 2999)),
    "tobacco": ((2100, 2199),),
    "alcohol (beer, wine and spirits makers)": ((2082, 2082), (2084, 2085)),
    "weapons (ordnance; guided missiles; tanks)": ((3480, 3489), (3760, 3769), (3795, 3795)),
}

_SUFFIX = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC", "LP",
           "NV", "SA", "AG", "SE", "THE", "DE", "NEW", "CLASS", "COM", "OF"}
_LOOSE = _SUFFIX | {"HOLDINGS", "HOLDING", "HLDGS", "GROUP", "GRP", "INTERNATIONAL", "INTL", "COS", "COMPANIES",
                    "TRUST", "REIT", "AND", "A", "B", "C", "ORD", "SHS"}
_STATE = re.compile(r"\s*[/\\]\s*[A-Z]{2,3}\s*[/\\]?\s*$")   # "/DE/", " / MA": state of incorporation


def normalise_company_name(name: str, loose: bool = False) -> str:
    """Upper case, punctuation and legal suffixes removed. ``loose`` also drops words such as
    HOLDINGS and GROUP. Used only to link a holding to a registrant; never shown as data."""
    s = _STATE.sub("", str(name).upper().replace("&", " AND ")).replace("'", "")
    tokens = [t for t in re.sub(r"[^A-Z0-9]+", " ", s).split() if t not in (_LOOSE if loose else _SUFFIX)]
    return " ".join(tokens)


def parse_company_tickers(raw: bytes) -> pd.DataFrame:
    """company_tickers.json -> one row per registrant: cik (10 digits), ticker, title."""
    rows = [{"cik": f"{int(v['cik_str']):010d}", "ticker": v["ticker"], "title": v["title"]}
            for v in json.loads(raw).values()]
    return pd.DataFrame(rows).drop_duplicates("cik").reset_index(drop=True)


def parse_cik_lookup(raw: bytes) -> pd.DataFrame:
    """cik-lookup-data.txt ("NAME:CIK:" per line, former names included) -> cik, title."""
    rows = []
    for line in raw.decode("latin-1").splitlines():
        parts = line.rstrip(":").rsplit(":", 1)
        if len(parts) == 2 and parts[1].strip().isdigit():
            rows.append((f"{int(parts[1]):010d}", parts[0].strip()))
    return pd.DataFrame(rows, columns=["cik", "title"]).drop_duplicates()


def _name_forms(name: str) -> list[tuple[str, str]]:
    """Progressively looser forms of a company name, each with the label a match on it gets.
    "compact" ignores spacing ("W.W. Grainger" / "WW Grainger", "ExxonMobil" / "Exxon Mobil");
    "reordered" also ignores word order, for the SEC's surname-first style ("SCHWAB CHARLES",
    "HORTON D R") and needs at least eight letters."""
    exact, loose = normalise_company_name(name), normalise_company_name(name, loose=True)
    compact = loose.replace(" ", "")
    forms = [("exact", exact), ("loose", loose), ("compact", "~" + compact)]
    if len(compact) >= 8:
        forms.append(("reordered", "#" + "".join(sorted(compact))))
    return [(how, f) for how, f in forms if f.strip("~#")]


def match_issuers(names: Iterable[str], companies: pd.DataFrame) -> pd.DataFrame:
    """Link issuer names to registrants. A name is linked only when exactly ONE registrant
    shares a form of it (see ``_name_forms``; stricter forms are tried first and the form
    that matched is recorded). Otherwise cik is None and match is "unmatched", or
    "ambiguous" when several registrants share the name. Nothing is guessed."""
    index: dict[str, set[str]] = {}
    for cik, title in zip(companies["cik"], companies["title"]):
        for _, form in _name_forms(title):
            index.setdefault(form, set()).add(cik)
    titles = dict(zip(companies["cik"], companies["title"]))
    rows = []
    for name in dict.fromkeys(names):
        cik, how, candidates = None, "unmatched", ""
        for label, form in _name_forms(name):
            hit = index.get(form, set())
            if len(hit) == 1:
                cik, how, candidates = next(iter(hit)), label, ""
                break
            if len(hit) > 1 and not candidates:
                how, candidates = "ambiguous", "|".join(sorted(hit))
        rows.append({"issuer_name": name, "cik": cik, "registrant_title": titles.get(cik), "match": how,
                     "candidates": candidates})
    return pd.DataFrame(rows)


def resolve_ambiguous(matches: pd.DataFrame, sic_of: dict[str, str | None], max_candidates: int = 6) -> pd.DataFrame:
    """For names shared by several registrants (a parent and its subsidiaries, old and new
    shells): if the candidates that HAVE an SIC code all fall in the same exclusion category
    (or all in none), the name is linked to the first of them and marked as resolved that
    way. If they disagree, or none has a code, it stays unlinked."""
    out = matches.copy()
    for i, row in out[out["match"] == "ambiguous"].iterrows():
        cands = [c for c in str(row["candidates"]).split("|") if c][:max_candidates]
        coded = [c for c in cands if sic_of.get(c)]
        reasons = {exclusion_reason(sic_of[c]) for c in coded}
        if coded and len(reasons) == 1:
            out.loc[i, ["cik", "match"]] = [coded[0], f"ambiguous name, {len(coded)} registrants with the same classification"]
    return out


def parse_assigned_sic(atom: bytes) -> dict:
    """SIC code from the company-info block of an EDGAR company listing (Atom)."""
    root = ET.fromstring(atom)
    out = {"sic": None, "sic_description": None}
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "assigned-sic" and el.text:
            out["sic"] = el.text.strip()
        elif tag == "assigned-sic-desc" and el.text:
            out["sic_description"] = el.text.strip()
    return out


def fetch_sic(client, ciks: Iterable[str]) -> pd.DataFrame:
    """One small request per registrant through the polite client; cached in its archive."""
    rows = []
    for cik in dict.fromkeys(c for c in ciks if c):
        url = series_filings_url(cik, form="", count=1)
        try:
            rows.append({"cik": cik, "sic_source_url": url, **parse_assigned_sic(client.read(url, revalidate=False))})
        except Exception as e:  # one registrant failing must not lose the rest; it stays unknown
            rows.append({"cik": cik, "sic_source_url": url, "sic": None, "sic_description": f"not retrieved: {e}"})
    return pd.DataFrame(rows)


def exclusion_reason(sic: str | None) -> str | None:
    if sic is None or not str(sic).isdigit():
        return None
    code = int(sic)
    return next((why for why, ranges in EXCLUSION_SIC.items() if any(lo <= code <= hi for lo, hi in ranges)), None)


def sic_exclusions(holdings: pd.DataFrame, matches: pd.DataFrame, sic: pd.DataFrame) -> pd.DataFrame:
    """holdings (security_key, issuer_name) + name links + SIC codes -> one row per security
    with its SIC code, and ``excluded`` / ``reason`` under ``EXCLUSION_SIC``. Status is
    "assumed" for every row: this is a proxy for the mandate's own screen."""
    t = holdings[["security_key", "issuer_name"]].drop_duplicates("security_key")
    t = t.merge(matches, on="issuer_name", how="left").merge(sic, on="cik", how="left")
    t["reason"] = [exclusion_reason(s) for s in t["sic"]]
    t["excluded"] = t["reason"].notna()
    t["status"] = "assumed"
    t["source"] = "SEC SIC code of the registrant, linked by company name"
    return t
