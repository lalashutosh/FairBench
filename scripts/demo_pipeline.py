"""The whole pipeline in one command, for the recorded demo. No network, a few seconds.

    .venv/bin/python scripts/demo_pipeline.py            # add --pause 3 to hold each stage on screen

    1  READ, CHECK, FORMULAS   a policy text becomes rules, checked against the region, written as formulas
                               (run live here, on the FICTIONAL example mandate and a synthetic universe)
    2  DATA                    what was downloaded from the SEC for the real fund
    3  ESTIMATE, classical     the real fund's rank among portfolios its rules allow
    4  ESTIMATE, quantum       the same quarters through the quantum estimator (noiseless simulation)
    5  REPORT                  where the figure and the write-up are

Stages 2 to 5 read the committed files in results/ (written by scripts/real_fund_attribution.py and
scripts/real_fund_quantum.py), so every number shown is a number in the repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any folder, installed or not

import argparse
import json
import time

import numpy as np
import pandas as pd

from fairbench.ingest.documents import parse_text
from fairbench.instances import named_universe
from fairbench.mandates.compile import select
from fairbench.mandates.dsl import approve
from fairbench.mandates.extract import constraints_from_spec
from fairbench.mandates.formulas import FORMULAS, formula_sheet
from fairbench.mandates.regions import applicable_rules, check_mandate
from fairbench.rules import load_spec

ROOT = Path(__file__).resolve().parents[1]
RULED = "k names obeying the exclusions"


def stage_mandate(k: int = 20) -> dict:
    """Stage 1 on the fictional example mandate (offline: the hand-written rule spec)."""
    text_path = ROOT / "examples" / "mandate_example.txt"
    u = named_universe(150, seed=0)
    spec = load_spec(text_path.with_suffix(".rules.json"))
    cons = constraints_from_spec(spec, parse_text(text_path.read_text()), u, fund_id="example")
    approve(cons, reviewer="reference spec (hand-written)")
    findings = check_mandate(cons, applicable_rules("EU", fund_name=spec.get("fund_name", "")))
    sheet = formula_sheet(cons, u, k, fund_name=spec.get("fund_name", ""), regional_findings=findings, promote_soft=True)
    sel = select(cons, promote_soft=True)
    share = sheet[sheet.index("**", sheet.index("## What the enforced rules leave")) + 2:]
    example = next(c for c in sel.enforced if c.compile_target["kind"] == "sector_cap")
    return dict(fund=spec.get("fund_name", ""), n_rules=len(cons), n_rejected=sum(c.status == "unknown" for c in cons),
                n_enforced=len(sel.enforced), verdicts=pd.Series([f["verdict"] for f in findings]).value_counts().to_dict(),
                example_quote=example.evidence_text, example_formula=FORMULAS["sector_cap"][0],
                valid_share=share[:share.index("**")], sheet=sheet)


def run(name: str = "parnassus_core_equity") -> dict:
    """Everything the demo shows, as data (also used by the test)."""
    res = ROOT / "results"
    fund = json.loads((res / f"real_fund_{name}.json").read_text())
    summ = json.loads((res / f"real_fund_{name}_summary.json").read_text())
    hold = pd.read_csv(res / f"real_fund_{name}_hold.csv")
    quantum = json.loads((res / f"real_fund_{name}_quantum.json").read_text())
    periods = fund["periods"]
    h = hold[(hold["reference"] == RULED) & (hold["weights"] == "benchmark_capped")]
    free = hold[(hold["reference"] == "any k names") & (hold["weights"] == "benchmark_capped")]
    grow = lambda r: float(100 * (np.prod(1 + np.asarray(r, dtype=float)) - 1))
    row = next(r for r in summ["by_reference"] if r["reference"] == RULED and r["weights"] == "benchmark_capped")
    return dict(
        mandate=stage_mandate(),
        data=dict(fund=summ["fund"], universe=summ["universe_proxy"], start=periods[0]["period_start"],
                  end=periods[-1]["period_end"], quarters=len(periods), k=int(round(h["k"].median())),
                  n_universe=int(np.median([p["coverage"]["n_universe"] for p in periods])), n_excluded=summ["n_excluded_keys"]),
        classical=dict(fund=grow([p["realised"]["frozen_return"] for p in periods]),
                       index=grow([p["realised"]["benchmark_proxy_return"] for p in periods]),
                       typical=grow(h["median_1q"]), typical_without_rules=grow(free["median_1q"]),
                       mean_rank=row["mean_percentile_1q"], luck=row["se_if_random_1q"], samples=fund["n_samples"]),
        quantum=quantum, figure=f"results/real_fund_{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="parnassus_core_equity"), ap.add_argument("--pause", type=float, default=0.0)
    args = ap.parse_args()
    d = run(args.name)
    m, da, c, q = d["mandate"], d["data"], d["classical"], d["quantum"]
    sl, rs = q["sleeve"], q["resources_last_period"]

    def stage(title: str, lines: list[str]) -> None:
        print(f"\n{title}\n" + "-" * len(title))
        print("\n".join("  " + line for line in lines))
        time.sleep(args.pause)

    print("FairBench: is it the rules or the manager?")
    stage("1  READ, CHECK, FORMULAS   (fictional example mandate, synthetic universe)", [
        f"{m['fund']}: {m['n_rules']} rules read from the policy text, {m['n_rejected']} rejected by the quote and number checks",
        "against the EU's requirements: " + ", ".join(f"{n} {v}" for v, n in m["verdicts"].items()),
        f"one rule:    \"{m['example_quote']}\"",
        f"as a formula: {m['example_formula']}",
        f"{m['n_enforced']} rules enforced; they leave {m['valid_share']} of random portfolios valid",
    ])
    stage("2  DATA   (real, from SEC filings)", [
        f"{da['fund']}: its holdings at {da['quarters'] + 1} quarter-ends, {da['start']} to {da['end']}, about {da['k']} stocks",
        f"pool: about {da['n_universe']} companies from {da['universe']}; {da['n_excluded']} excluded by the rules",
        "prices worked out from the filings themselves (value / shares); no dividends",
    ])
    stage("3  ESTIMATE, classical   (the product today)", [
        f"{c['samples']:,} random rule-abiding portfolios per quarter, each held for the quarter",
        f"price growth:  index fund {c['index']:+.0f}%   the fund {c['fund']:+.0f}%   typical portfolio under the rules {c['typical']:+.0f}%",
        f"the rules cost about {abs(c['typical'] - c['typical_without_rules']):.0f} points over the period",
        f"the fund's average rank: {c['mean_rank']:.0f} of 100   (luck alone: 50, give or take {c['luck']:.0f})",
    ])
    stage("4  ESTIMATE, quantum   (same inputs; noiseless simulation; counts checks, not time)", [
        f"same answer: on the {int(sl['rows'][0]['n'])} largest stocks, where the exact rank is known, off by "
        f"{sl['mean_abs_error_quantum']:.2f} points (classical with the same effort: {sl['mean_abs_error_classical_same_queries']:.2f})",
        f"to pin the rank to +-{q['eps_points']:g} point: {q['median_quantum_queries']:,.0f} quantum queries against "
        f"{q['median_classical_samples']:,.0f} classical samples",
        "precision asked for -> queries / samples:  " + "   ".join(
            f"+-{r['eps_points']:g}: {r['quantum_queries']:,.0f} / {r['classical_samples']:,.0f}" for r in q["scaling_last_period"][1::2]),
        f"the real circuit: about {rs['logical_qubits']:,} error-corrected qubits, {rs['t_gates_per_step']:.0e} T gates per step: "
        "beyond today's machines",
    ])
    stage("5  REPORT", [f"figure:   {d['figure']}", "write-up: REAL_FUND_RESULTS.md",
                        "proxy rules, price returns without dividends, one fund; a rank is not proof of skill"])


if __name__ == "__main__":
    main()
