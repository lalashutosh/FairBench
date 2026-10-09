"""The quantum estimator on the same real-fund data the classical pipeline uses.

    .venv/bin/python scripts/real_fund_quantum.py --example
    .venv/bin/python scripts/real_fund_quantum.py \
        --fund-dir data/nport/S000000856 --parent-dir data/nport/S000004310 \
        --price-dir data/nport/S000004347 --exclude-file data/exclusions_sic.csv --name parnassus_core_equity

For every quarter it builds the same case as scripts/real_fund_attribution.py and reports,
for the equal-weight version of the question (the one the oracle can express):
  * the fund's percentile and what an IDEAL quantum computer would need to estimate it to
    +-1 point (oracle queries), against the classical samples needed for the same precision;
  * an exact check on a small real sub-universe (the largest index stocks): exact answer,
    state-vector Grover simulation, classical sampling;
and once, for the last quarter: how the query counts change with the precision asked for,
and the logical size of the circuit for the full universe.

Everything quantum here is a noiseless SIMULATION and counts oracle queries, not time. It
supports no claim of a speedup on any existing machine; see QUANTUM_CORE.md.
Outputs: results/real_fund_<name>_quantum.csv and .json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any folder, installed or not

import argparse
import json

import numpy as np
import pandas as pd

from fairbench.apps.real_fund import build_period_case
from fairbench.apps.real_fund_quantum import exact_sleeve, oracle_resources, quantum_percentile
from fairbench.ingest.nport import parse_nport

ROOT = Path(__file__).resolve().parents[1]


def load(directory: Path) -> dict:
    return {r.report_date: r for r in (parse_nport(f.read_bytes()) for f in sorted(directory.glob("*.xml")))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", action="store_true"), ap.add_argument("--fund-dir"), ap.add_argument("--parent-dir")
    ap.add_argument("--price-dir", action="append", default=[]), ap.add_argument("--exclude-file")
    ap.add_argument("--name", default=None), ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--mc", type=int, default=100_000), ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    excluded, synthetic = [], False
    if args.example:
        ex = ROOT / "examples" / "real_data_example"
        meta = json.loads((ex / "meta.json").read_text())
        fund, parent = load(ex / "nport" / meta["fund_series_id"]), load(ex / "nport" / meta["parent_series_id"])
        excluded, synthetic, name, extra = meta["excluded_keys"], True, args.name or "example", []
    else:
        fund, parent, name = load(Path(args.fund_dir)), load(Path(args.parent_dir)), args.name or Path(args.fund_dir).name
        extra = [load(Path(d)) for d in args.price_dir]
        if args.exclude_file:
            excluded = pd.read_csv(args.exclude_file, dtype=str)["security_key"].dropna().tolist()

    dates = sorted(set(fund) & set(parent))
    rows, sleeves, case = [], [], None
    for t, t1 in zip(dates[:-1], dates[1:]):
        case = build_period_case(parent[t], parent[t1], fund[t], fund[t1],
                                 extra_price_reports=[(e.get(t), e.get(t1)) for e in extra])
        q = quantum_percentile(case, excluded_keys=excluded, eps=args.eps, n_mc=args.mc, seed=args.seed)
        s = exact_sleeve(case, seed=args.seed)
        rows.append({"period_end": str(t1), **{k: v for k, v in q.items() if k != "device"}})
        if s is not None:
            sleeves.append({"period_end": str(t1), **s})
        print(f"{t1}  percentile {q['percentile']:5.1f}   quantum {q['quantum_queries_median']:6.0f} queries vs classical "
              f"{q['classical_samples_same_precision']:6.0f} samples ({q['query_ratio']:.1f}x)"
              + (f"   sleeve n={s['n']} k={s['k']}: exact {s['exact']:.1f}, quantum {s['quantum']:.1f}" if s else ""))
    table = pd.DataFrame(rows)
    scaling = []
    for eps in (0.02, 0.01, 0.005, 0.002, 0.001):   # last quarter: cost against the precision asked for
        q = quantum_percentile(case, excluded_keys=excluded, eps=eps, n_mc=args.mc, seed=args.seed, reps=15)
        scaling.append({"eps_points": 100 * eps, "quantum_queries": q["quantum_queries_median"],
                        "classical_samples": q["classical_samples_same_precision"], "ratio": q["query_ratio"]})
    res = oracle_resources(case, excluded_keys=excluded)
    sl = pd.DataFrame(sleeves)
    out = ROOT / "results"
    table.to_csv(out / f"real_fund_{name}_quantum.csv", index=False)
    summary = {
        "synthetic": synthetic, "weights": "equal", "eps_points": 100 * args.eps, "n_periods": len(table),
        "mean_percentile": float(table["percentile"].mean()),
        "median_quantum_queries": float(table["quantum_queries_median"].median()),
        "median_classical_samples": float(table["classical_samples_same_precision"].median()),
        "median_query_ratio": float(table["query_ratio"].median()),
        "largest_quantum_error_points": float(table["quantum_worst_error"].max()),
        "scaling_last_period": scaling, "resources_last_period": res,
        "sleeve": {"n_periods": len(sl), "mean_abs_error_quantum": float((sl["quantum"] - sl["exact"]).abs().mean()) if len(sl) else None,
                   "mean_abs_error_classical_same_queries": float((sl["classical_same_queries"] - sl["exact"]).abs().mean()) if len(sl) else None,
                   "share_exact_inside_quantum_ci": float(np.mean([a <= e <= b for e, (a, b) in zip(sl["exact"], sl["quantum_ci"])])) if len(sl) else None,
                   "rows": sleeves},
        "reading": "Noiseless simulation; one query = one oracle call = one classical sample checked. Query counts, not "
                   "time: each query on a fault-tolerant machine is far slower than a classical check. No advantage on "
                   "current hardware.",
    }
    (out / f"real_fund_{name}_quantum.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"\nFor +-{100 * args.eps:g} point: median {summary['median_quantum_queries']:.0f} quantum queries against "
          f"{summary['median_classical_samples']:.0f} classical samples ({summary['median_query_ratio']:.1f}x fewer).")
    print("Precision asked for (points) -> quantum queries / classical samples / ratio, last quarter:")
    for s_ in scaling:
        print(f"  +-{s_['eps_points']:<5g} {s_['quantum_queries']:>9.0f} {s_['classical_samples']:>11.0f} {s_['ratio']:>6.1f}x")
    print(f"Full-universe circuit: {res['logical_qubits']} logical qubits, {res['t_gates_per_step']:.2e} T gates per step.")
    if len(sl):
        print(f"Exact check on the {int(sl['n'].median())} largest stocks, {len(sl)} quarters: quantum off by "
              f"{summary['sleeve']['mean_abs_error_quantum']:.2f} points on average, classical with the same number of "
              f"queries by {summary['sleeve']['mean_abs_error_classical_same_queries']:.2f}.")
    print(f"wrote results/real_fund_{name}_quantum.csv and .json")


if __name__ == "__main__":
    main()
