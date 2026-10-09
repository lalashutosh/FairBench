"""Security identifier normaliser: CUSIP / ISIN / LEI checks and one stable key per security.

Contract
--------
* Identifiers are accepted only if they pass their published check-digit rule
  (CUSIP: modulus-10 double-add-double; ISIN: ISO 6166 Luhn; LEI: ISO 17442 / ISO 7064
  MOD 97-10). A syntactically invalid identifier is never used as a key and never repaired.
* Placeholders ("000000000", "N/A", None, NaN) are treated as missing (`clean_identifier`).
  Missing stays missing: nothing is filled with a guess.
* `security_key` picks ONE (scheme, value) per row by a fixed precedence:
  valid CUSIP > valid ISIN (US/CA ISINs are reduced to their embedded CUSIP) >
  valid LEI + title > issuer name + title > unknown. Name keys are the weakest scheme and
  are reported as such, not promoted.
* One security can legitimately appear on several rows of one snapshot (several lots,
  sleeves or derivative legs). This module reports such duplicates; it never merges rows.
* CUSIPs change after corporate actions. This module does not link an old CUSIP to a new
  one; `build_identifier_history` only lists, per key, when it was seen and under which issuer
  names, which is the raw material for spotting such changes.
* `derived_isin` is a derivation (country prefix + CUSIP + check digit), never a filed value.
  It assumes the country prefix given to `add_security_keys` (default "US").
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

_CUSIP_CHARS = {**{str(d): d for d in range(10)},
                **{chr(ord("A") + i): 10 + i for i in range(26)},
                "*": 36, "@": 37, "#": 38}
_PLACEHOLDERS = frozenset({"", "N/A", "NA", "NONE", "NAN", "NULL"})
_ALL_ZERO = re.compile(r"0+")
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_LEI = re.compile(r"[0-9A-Z]{20}")
_COUNTRY = re.compile(r"[A-Z]{2}")
_CUSIP_ISIN_COUNTRIES = ("US", "CA")  # ISINs whose national number is a CUSIP
KEY_SCHEMES = ("cusip", "isin", "lei_title", "name", "unknown")


# ----------------------------------------------------------------- cleaning
def clean_identifier(value: object) -> str | None:
    """Strip + upper-case; None for None/NaN/"", "N/A", "NA", "NONE" and all-zero strings."""
    if not isinstance(value, str):
        if value is None or pd.isna(value):
            return None
        value = str(value)
    s = value.strip().upper()
    if s in _PLACEHOLDERS or _ALL_ZERO.fullmatch(s):
        return None
    return s


def normalise_text(value: object) -> str:
    """Upper-case, drop '.' and apostrophes, other punctuation to a space, collapse whitespace."""
    if not isinstance(value, str):
        if value is None or pd.isna(value):
            return ""
        value = str(value)
    s = re.sub(r"[.'’`]", "", value.upper())
    s = re.sub(r"[^\w\s]|_", " ", s)
    return " ".join(s.split())


# -------------------------------------------------------------------- CUSIP
def cusip_check_digit(cusip8: str) -> str:
    """Check digit for the first 8 characters of a CUSIP (modulus-10 double-add-double)."""
    s = str(cusip8).strip().upper()
    if len(s) != 8 or any(ch not in _CUSIP_CHARS for ch in s):
        raise ValueError(f"CUSIP base must be 8 characters from 0-9, A-Z, *, @, #: {cusip8!r}")
    vals = np.array([_CUSIP_CHARS[ch] for ch in s])
    vals[1::2] *= 2  # positions 2, 4, 6, 8 (1-based)
    total = int((vals // 10 + vals % 10).sum())
    return str((10 - total % 10) % 10)


def is_valid_cusip(s: object) -> bool:
    """9 characters, correct check digit, not a placeholder (all zeros)."""
    c = clean_identifier(s)
    return (c is not None and len(c) == 9 and all(ch in _CUSIP_CHARS for ch in c[:8])
            and c[8] == cusip_check_digit(c[:8]))


# --------------------------------------------------------------------- ISIN
def _alnum_digits(s: str) -> np.ndarray:
    """Letters -> two-digit numbers (A=10 .. Z=35), digits kept; one int per decimal digit."""
    return np.array([int(d) for d in "".join(str(int(ch, 36)) for ch in s)])


def isin_check_digit(isin11: str) -> str:
    """Check digit for the first 11 characters of an ISIN (ISO 6166, Luhn mod 10)."""
    s = str(isin11).strip().upper()
    if len(s) != 11 or not (s.isascii() and s.isalnum()):
        raise ValueError(f"ISIN base must be 11 alphanumeric characters: {isin11!r}")
    d = _alnum_digits(s)
    d[::-2] *= 2  # rightmost body digit and every second digit to its left
    total = int((d // 10 + d % 10).sum())
    return str((10 - total % 10) % 10)


def is_valid_isin(s: object) -> bool:
    i = clean_identifier(s)
    return i is not None and _ISIN.fullmatch(i) is not None and i[11] == isin_check_digit(i[:11])


def cusip_to_isin(cusip: str, country: str = "US") -> str:
    """ISIN = country prefix + 9-character CUSIP + check digit. Raises on an invalid CUSIP."""
    c = clean_identifier(cusip)
    cc = clean_identifier(country)
    if c is None or not is_valid_cusip(c):
        raise ValueError(f"not a valid CUSIP: {cusip!r}")
    if cc is None or not _COUNTRY.fullmatch(cc):
        raise ValueError(f"country must be a 2-letter code: {country!r}")
    body = cc + c
    return body + isin_check_digit(body)


def isin_to_cusip(isin: str) -> str | None:
    """The CUSIP embedded in a valid US/CA ISIN; None for other prefixes or an invalid result."""
    i = clean_identifier(isin)
    if i is None or not is_valid_isin(i) or i[:2] not in _CUSIP_ISIN_COUNTRIES:
        return None
    c = i[2:11]
    return c if is_valid_cusip(c) else None


# ---------------------------------------------------------------------- LEI
def is_valid_lei(s: object) -> bool:
    """20 alphanumerics whose letters-as-numbers integer is 1 mod 97 (ISO 17442)."""
    v = clean_identifier(s)
    return (v is not None and _LEI.fullmatch(v) is not None
            and int("".join(str(int(ch, 36)) for ch in v)) % 97 == 1)


# ------------------------------------------------------------- security key
def security_key(cusip: object = None, isin: object = None, lei: object = None,
                 title: object = None, issuer_name: object = None) -> tuple[str, str]:
    """(scheme, value) by precedence; see the module docstring. Invalid identifiers are skipped."""
    c = clean_identifier(cusip)
    if c is not None and is_valid_cusip(c):
        return "cusip", c
    i = clean_identifier(isin)
    if i is not None and is_valid_isin(i):
        embedded = isin_to_cusip(i)
        return ("cusip", embedded) if embedded is not None else ("isin", i)
    t = normalise_text(title)
    lei_c = clean_identifier(lei)
    if lei_c is not None and is_valid_lei(lei_c) and t:
        return "lei_title", f"{lei_c}|{t}"
    n = normalise_text(issuer_name)
    if n or t:
        return "name", f"{n}|{t}"
    return "unknown", ""


def _column(h: pd.DataFrame, name: str) -> list:
    return h[name].tolist() if name in h.columns else [None] * len(h)


def _derive_isin(cusip_ok: bool, isin_filed: str | None, cusip: object, country: object) -> str | None:
    if isin_filed is not None or not cusip_ok:
        return isin_filed
    cc = clean_identifier(country)
    if cc is None or not _COUNTRY.fullmatch(cc):
        return None  # unknown country: do not guess a prefix
    return cusip_to_isin(cusip, cc)


def add_security_keys(holdings: pd.DataFrame, *, country: str | pd.Series = "US") -> pd.DataFrame:
    """Copy of `holdings` plus key_scheme, key_value, security_key ("scheme:value"),
    cusip_valid, isin_valid, derived_isin. Filed columns are never overwritten.

    `derived_isin` is the filed ISIN (cleaned; None if a placeholder) unless the CUSIP is valid
    and the ISIN is missing, in which case it is `cusip_to_isin(cusip, country)`. `country` is
    one 2-letter prefix for every row, or a per-row Series aligned to `holdings` (rows whose
    country is unknown then get no derived ISIN).
    """
    h = holdings.copy()
    n = len(h)
    if isinstance(country, str):
        if not _COUNTRY.fullmatch(country.strip().upper()):
            raise ValueError(f"country must be a 2-letter code: {country!r}")
        countries = [country] * n
    else:
        countries = (country if country.index.equals(h.index) else country.reindex(h.index)).tolist()

    cols = [_column(h, c) for c in ("cusip", "isin", "issuer_lei", "title", "issuer_name")]
    keys = [security_key(*row) for row in zip(*cols)]
    cusip_ok = [is_valid_cusip(c) for c in cols[0]]
    isin_filed = [clean_identifier(i) for i in cols[1]]

    h["key_scheme"] = [s for s, _ in keys]
    h["key_value"] = [v for _, v in keys]
    h["security_key"] = [f"{s}:{v}" for s, v in keys]
    h["cusip_valid"] = np.array(cusip_ok, dtype=bool)
    h["isin_valid"] = np.array([is_valid_isin(i) for i in isin_filed], dtype=bool)
    h["derived_isin"] = pd.Series(
        [_derive_isin(ok, i, c, co) for ok, i, c, co in zip(cusip_ok, isin_filed, cols[0], countries)],
        index=h.index, dtype=object)
    return h


def identifier_report(holdings: pd.DataFrame) -> dict:
    """Identifier quality of one snapshot.

    n_by_scheme            rows per key scheme (all five schemes listed, zeros included)
    n_invalid_cusip        rows with a filed CUSIP that fails validation (placeholders are
                           "missing", counted in n_missing_cusip, not here)
    n_duplicate_keys       distinct keys on more than one row; n_duplicate_rows = surplus rows.
                           Rows with scheme "unknown" are excluded (they share no identity).
    cusip_value_share      gross |value_usd| on cusip-keyed rows / gross |value_usd| of all rows
                           (gross, so negative derivative values cannot push it outside [0, 1]);
                           NaN if value_usd is absent or sums to zero.
    """
    h = add_security_keys(holdings)
    n = len(h)
    counts = h["key_scheme"].value_counts()
    filed = np.array([clean_identifier(c) is not None for c in _column(h, "cusip")], dtype=bool)
    known = h.loc[h["key_scheme"] != "unknown", "security_key"].value_counts()
    dup = known[known > 1]

    share = float("nan")
    if "value_usd" in h.columns:
        v = pd.to_numeric(h["value_usd"], errors="coerce").abs()
        total = float(v.sum())
        if total > 0:
            share = float(v[h["key_scheme"] == "cusip"].sum()) / total
    return {
        "n_rows": n,
        "n_by_scheme": {s: int(counts.get(s, 0)) for s in KEY_SCHEMES},
        "n_invalid_cusip": int((filed & ~h["cusip_valid"].to_numpy()).sum()),
        "n_missing_cusip": int((~filed).sum()),
        "n_duplicate_keys": int(len(dup)),
        "n_duplicate_rows": int((dup - 1).sum()),
        "cusip_value_share": share,
    }


# ------------------------------------------------------------------ history
HISTORY_COLUMNS = ["security_key", "scheme", "value", "first_seen", "last_seen", "n_snapshots", "names"]


def build_identifier_history(snapshots: dict[date, pd.DataFrame]) -> pd.DataFrame:
    """One row per security key across snapshots: when it was first/last seen, in how many
    snapshots, and the distinct issuer names filed under it (joined by " | ", in order of first
    appearance). Rows with an "unknown" key are left out. A CUSIP change shows up as one key
    ending and another starting under the same name; this function does not link them.
    """
    parts = []
    for d in sorted(snapshots):
        h = add_security_keys(snapshots[d])
        parts.append(pd.DataFrame({
            "security_key": h["security_key"].to_numpy(),
            "scheme": h["key_scheme"].to_numpy(),
            "value": h["key_value"].to_numpy(),
            "issuer_name": pd.Series(_column(h, "issuer_name"), dtype=object).to_numpy(),
        }).assign(snapshot=d))
    if not parts:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    long = pd.concat(parts, ignore_index=True)
    long = long[long["scheme"] != "unknown"]
    long["issuer_name"] = long["issuer_name"].map(
        lambda x: None if x is None or pd.isna(x) or not str(x).strip() else str(x).strip())

    out = long.groupby("security_key", sort=False).agg(
        scheme=("scheme", "first"), value=("value", "first"),
        first_seen=("snapshot", "min"), last_seen=("snapshot", "max"),
        n_snapshots=("snapshot", "nunique"),
        names=("issuer_name", lambda s: " | ".join(dict.fromkeys(s.dropna()))),
    ).reset_index()
    out = out.sort_values(["first_seen", "security_key"], kind="stable").reset_index(drop=True)
    return out[HISTORY_COLUMNS]
