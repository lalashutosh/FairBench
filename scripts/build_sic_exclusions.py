"""Build an exclusion list for a universe from SEC industry (SIC) codes. LIVE: contacts the SEC.

    export FAIRBENCH_SEC_USER_AGENT="Your Project Name contact@your-domain.org"
    .venv/bin/python scripts/build_sic_exclusions.py --parent-dir data/nport/S000004310

Writes data/sic_classification.csv (every security ever in the universe, its SEC registrant,
SIC code and how the link was made) and data/exclusions_sic.csv (the excluded ones, for
scripts/real_fund_attribution.py --exclude-file). A PROXY for a mandate's own screen; see
fairbench/ingest/sic.py for what it does and does not catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any folder, installed or not

import argparse

import pandas as pd

from fairbench.apps.real_fund import keyed_sleeve
from fairbench.ingest.archive import RawArchive
from fairbench.ingest.http import SecClient
from fairbench.ingest.nport import parse_nport
from fairbench.ingest.sic import (CIK_LOOKUP_URL, COMPANY_TICKERS_URL, fetch_sic, match_issuers, parse_cik_lookup,
                                  parse_company_tickers, resolve_ambiguous, sic_exclusions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent-dir", required=True), ap.add_argument("--data", default="data")
    ap.add_argument("--max-rps", type=float, default=5.0)
    args = ap.parse_args()
    root = Path(args.data)
    frames = []
    for f in sorted(Path(args.parent_dir).glob("*.xml")):
        s = keyed_sleeve(parse_nport(f.read_bytes()))
        frames.append(s.assign(last_date=f.stem))
    hold = pd.concat(frames).sort_values("last_date").drop_duplicates("security_key", keep="last")
    client = SecClient(RawArchive(root / "raw"), max_rps=args.max_rps)
    companies = parse_company_tickers(client.read(COMPANY_TICKERS_URL, revalidate=False))
    matches = match_issuers(hold["issuer_name"], companies)
    left = matches.loc[matches["cik"].isna(), "issuer_name"]
    if len(left):  # names that no longer have a ticker (acquired, renamed): the full historical name list
        everyone = parse_cik_lookup(client.read(CIK_LOOKUP_URL, revalidate=False))
        second = match_issuers(left, everyone)
        second["match"] = second["match"].where(second["cik"].isna(), second["match"] + " (historical name list)")
        matches = pd.concat([matches[matches["cik"].notna()], second], ignore_index=True)
    amb = [c for cs in matches.loc[matches["match"] == "ambiguous", "candidates"] for c in str(cs).split("|")[:6] if c]
    sic = fetch_sic(client, list(matches["cik"].dropna()) + amb)
    matches = resolve_ambiguous(matches, dict(zip(sic["cik"], sic["sic"])))
    table = sic_exclusions(hold, matches, sic).merge(hold[["security_key", "weight", "last_date"]], on="security_key")
    table.to_csv(root / "sic_classification.csv", index=False)
    excl = table[table["excluded"]]
    excl.to_csv(root / "exclusions_sic.csv", index=False)
    print(f"{len(table)} securities ever in the universe; name link: " + table["match"].value_counts().to_dict().__str__())
    print(f"SIC code found for {table['sic'].notna().sum()}; excluded {len(excl)}; requests made {client.requests_made}")
    print(excl.groupby("reason").size().to_string())
    print("wrote", root / "sic_classification.csv", "and", root / "exclusions_sic.csv")


if __name__ == "__main__":
    main()
