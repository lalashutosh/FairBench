"""Rank a fund's disclosed portfolio among the portfolios its rules allowed, quarter by quarter.

    .venv/bin/python scripts/real_fund_attribution.py --example            # synthetic, offline
    .venv/bin/python scripts/real_fund_attribution.py \
        --fund-dir data/nport/S000000856 --parent-dir data/nport/S000004310 --name parnassus_core_equity

Inputs are directories of N-PORT XML files, one per report date (scripts/real_fund_ingest.py
writes them). Optional:
  --exclude-file  a CSV with a column security_key (and, for the record, reason, status,
                  source) listing securities the mandate rules out
                  (scripts/build_sic_exclusions.py writes one from SEC industry codes)
  --price-dir     N-PORT files of a broader index fund (repeatable), used only to price
                  securities the parent fund no longer holds

Outputs in results/:
  real_fund_<name>.json          every quarter, every reference distribution, with its
                                 definition, sampler, seed, acceptance rate, estimated number
                                 of feasible portfolios and coverage
  real_fund_<name>.csv           one row per quarter
  real_fund_<name>_hold.csv      each quarter's portfolio held unchanged for one quarter and
                                 until the last report date, with and without the exclusions
  real_fund_<name>_summary.json  average percentiles and the yardstick for reading them

Read the result as a descriptive rule-versus-choice comparison. It is not a causal
decomposition and a percentile is not evidence of skill; the JSON carries the caveats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fairbench.apps.real_fund import history_table, run_history, run_hold_to_end, summarise_percentiles
from fairbench.ingest.nport import parse_nport

ROOT = Path(__file__).resolve().parents[1]


def load_reports(directory: Path) -> dict:
    reports = {}
    for f in sorted(directory.glob("*.xml")):
        r = parse_nport(f.read_bytes())
        reports[r.report_date] = r
    if not reports:
        raise SystemExit(f"no N-PORT XML files in {directory}")
    return reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", action="store_true", help="run the synthetic example in examples/real_data_example")
    ap.add_argument("--fund-dir"), ap.add_argument("--parent-dir"), ap.add_argument("--exclude-file")
    ap.add_argument("--price-dir", action="append", default=[], help="extra index fund N-PORT directory (repeatable)")
    ap.add_argument("--name", default=None), ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unknown-returns", default="drop", choices=["drop", "benchmark_fill"])
    args = ap.parse_args()

    excluded, synthetic = [], False
    if args.example:
        ex = ROOT / "examples" / "real_data_example"
        meta = json.loads((ex / "meta.json").read_text())
        fund_dir, parent_dir = ex / "nport" / meta["fund_series_id"], ex / "nport" / meta["parent_series_id"]
        excluded, synthetic, name = meta["excluded_keys"], True, args.name or "example"
    else:
        if not (args.fund_dir and args.parent_dir):
            raise SystemExit("give --example, or both --fund-dir and --parent-dir")
        fund_dir, parent_dir, name = Path(args.fund_dir), Path(args.parent_dir), args.name or Path(args.fund_dir).name
        if args.exclude_file:
            excluded = pd.read_csv(args.exclude_file, dtype=str)["security_key"].dropna().tolist()

    fund, parent = load_reports(fund_dir), load_reports(parent_dir)
    common = sorted(set(fund) & set(parent))
    print(f"fund reports: {len(fund)}, parent reports: {len(parent)}, common report dates: {len(common)}")
    if len(common) < 2:
        only_f, only_p = sorted(set(fund) - set(parent)), sorted(set(parent) - set(fund))
        raise SystemExit("need two common report dates. The two funds report on different fiscal quarter-ends: "
                         f"fund-only {only_f[:4]}, parent-only {only_p[:4]}. Pick a parent fund on the same cycle.")
    extra = [load_reports(Path(d)) for d in args.price_dir]
    results = run_history(parent, fund, excluded_keys=excluded, n_samples=args.samples, seed=args.seed,
                          unknown_returns=args.unknown_returns, price_reports=extra)
    for res in results:
        print("\n" + res.summary())
    table = history_table(results)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"real_fund_{name}.json").write_text(json.dumps(
        {"synthetic": synthetic, "seed": args.seed, "n_samples": args.samples, "excluded_keys": excluded,
         "periods": [r.to_dict() for r in results]}, indent=1, default=float) + "\n")
    table.to_csv(out / f"real_fund_{name}.csv", index=False)
    print("\n" + ("SYNTHETIC EXAMPLE. " if synthetic else "") + "Realised portfolio percentile by period "
          "(reference: uniform subsets, capped benchmark weights):")
    print(table[["period_end", "realised_frozen_return", "median", "p05", "p95",
                 "realised_portfolio_percentile", "percentile_se"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    hold = run_hold_to_end(parent, fund, price_reports=extra, excluded_keys=excluded, n_samples=args.samples,
                           seed=args.seed)
    hold.to_csv(out / f"real_fund_{name}_hold.csv", index=False)
    summary = summarise_percentiles(hold)
    (out / f"real_fund_{name}_summary.json").write_text(json.dumps(
        {"synthetic": synthetic, "fund": results[0].case.fund_name, "universe_proxy": results[0].case.parent_name,
         "first_start": hold["start"].min(), "last_end": hold["end"].max(), "n_excluded_keys": len(excluded),
         "return_kind": results[0].case.return_kind, "by_reference": summary.to_dict(orient="records"),
         "mean_rule_conditioned_shift_1q": (float(hold["rule_conditioned_shift_1q"].mean())
                                            if "rule_conditioned_shift_1q" in hold else None),
         "reading": "50 = the typical portfolio allowed by the same rules. One-quarter percentiles do not overlap; "
                    "if the picks were no better than random, their average would be 50 give or take se_if_random_1q. "
                    "Hold-to-end windows overlap, so their average has no such yardstick. Not evidence of skill."},
        indent=1, default=float) + "\n")
    print("\nAverage percentile of the fund's portfolio (50 = typical rule-abiding portfolio):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if "rule_conditioned_shift_1q" in hold:
        sh = hold[hold["reference"] != "any k names"].groupby("weights")["rule_conditioned_shift_1q"].mean()
        print("Average one-quarter effect of the exclusions on a typical portfolio: "
              + ", ".join(f"{w} weights {100 * v:+.2f}%" for w, v in sh.items()))
    print(f"\nwrote results/real_fund_{name}.json, .csv, _hold.csv and _summary.json")


if __name__ == "__main__":
    main()
