"""Mandate demo: from a fund's policy text to its benchmark.

  1. read a mandate written in plain language (default: examples/mandate_example.txt,
     a fictional fund);
  2. turn it into a rule spec. With --live Claude does this (needs the ``ai`` extra and
     ANTHROPIC_API_KEY); without it the hand-written reference spec next to the mandate
     (``<mandate>.rules.json``) is used, so the demo runs offline;
  3. compile the spec into numeric constraints for the universe and print what every rule
     became, including the rules that could not be expressed;
  4. run the attribution with those constraints.

SYNTHETIC DATA with a planted ground truth: ``named_universe`` (named sectors, involvement
flags, countries, extra fields), Gaussian returns with a planted rally in carbon-heavy names (``--premium`` extra
annual return per standard deviation of the carbon figure), and a rule-abiding fund planted
at a known rank (``--rank``). No real fund or mandate is analysed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from fairbench.apps.attribution import (attribute, performance, plot_attribution, portfolio_returns,
                                        rejection_sampler, weights_matrix)
from fairbench.data import synthetic_returns
from fairbench.instances import named_universe
from fairbench.rules import compile_spec, feasible_fraction, load_spec, save_spec

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def spec_diff(a: dict, b: dict) -> list[str]:
    """Rules present in one spec and not the other, ignoring quotes, notes and order."""
    def key(r):
        return json.dumps({k: (sorted(v) if isinstance(v, list) else v) for k, v in r.items()
                           if k not in ("source", "note", "reason")}, sort_keys=True)

    ka, kb = [key(r) for r in a["rules"]], [key(r) for r in b["rules"]]
    return [f"only in extracted: {k}" for k in ka if k not in kb] + \
           [f"only in reference: {k}" for k in kb if k not in ka]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mandate", type=Path, default=ROOT / "examples" / "mandate_example.txt")
    ap.add_argument("--spec", type=Path, default=None, help="rule spec JSON (default: <mandate>.rules.json)")
    ap.add_argument("--live", action="store_true", help="extract the spec with Claude (one API call)")
    ap.add_argument("--n", type=int, default=150, help="universe size")
    ap.add_argument("--k", type=int, default=20, help="holdings of the planted fund")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--premium", type=float, default=0.08)
    ap.add_argument("--rank", type=float, default=0.70)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RES.mkdir(exist_ok=True)
    t0 = time.time()

    u = named_universe(a.n, seed=a.seed)
    text = a.mandate.read_text()
    ref_path = a.spec or a.mandate.with_suffix(".rules.json")
    print(f"MANDATE  {a.mandate.name}: {len(text.split())} words; universe: {u.n} assets, "
          f"{len(set(u.sector))} sectors, flags {sorted(u.flags)}, categories {sorted(u.categories)}, "
          f"fields {u.numeric_names}")

    # ---- 1. text -> rule spec
    if a.live:
        from fairbench.mandate import extract_rules
        ex = extract_rules(text, u)
        spec = ex.spec
        save_spec(spec, RES / "mandate_rules_live.json")
        print(f"EXTRACT  {ex.model}: {len(spec['rules'])} rules, {ex.input_tokens} input / "
              f"{ex.output_tokens} output tokens; saved results/mandate_rules_live.json")
        for i in ex.unverified:
            print(f"  WARNING: rule {i} ({spec['rules'][i]['kind']}) quotes text that is not in the mandate")
        if ref_path.exists():
            diff = spec_diff(spec, load_spec(ref_path))
            print("  matches the hand-written reference spec" if not diff else
                  "  differs from the hand-written reference spec:\n    " + "\n    ".join(diff))
        source = f"extracted by {ex.model}"
    else:
        spec = load_spec(ref_path)
        source = f"hand-written reference {ref_path.name}"
        print(f"EXTRACT  skipped (no --live): using the {source}")

    # ---- 2. rule spec -> numeric constraints
    compiled = compile_spec(spec, u, k=a.k)
    cs = compiled.constraint_set
    print("COMPILE  " + compiled.report())
    frac, lo, hi = feasible_fraction(u, cs, seed=a.seed)
    left = cs.allowed_indices(u.n).size
    print(f"  {u.n - left} of {u.n} assets excluded; {frac:.2%} of random {a.k}-name portfolios from the "
          f"remaining {left} satisfy every rule (95% CI {lo:.2%} to {hi:.2%})")
    if frac == 0:
        raise SystemExit("  the rule set looks infeasible on this universe; stopping before attribution")

    # ---- 3. attribution with the compiled rules
    z = (u.carbon - u.carbon.mean()) / u.carbon.std()
    u.returns = synthetic_returns(u, periods=int(round(252 * a.years)), mu_shift=a.premium * z, seed=a.seed + 1)
    pool = rejection_sampler(u, cs, 1000, seed=a.seed + 2).samples
    perf = performance(portfolio_returns(weights_matrix(pool, "equal", u), u.returns.to_numpy()))
    fund = pool[np.argsort(perf)[int(a.rank * (len(pool) - 1))]]
    assert all(ok for _, ok in compiled.check_fund(fund, u))
    note = (f"Synthetic data, planted ground truth: carbon-heavy names earn +{a.premium:.0%}/yr per s.d.; "
            f"fund planted at rank {a.rank:.0%}. Rules compiled from mandate text ({source}).")
    res = attribute(u, cs, fund, n_samples=a.samples, seed=a.seed)
    print("ATTRIBUTE  " + note)
    print(res.summary())
    if compiled.unmapped:
        print(f"  not captured by this benchmark: {len(compiled.unmapped)} mandate rule(s) "
              "(see [unmapped] above)")
    plt.close(plot_attribution(res, RES / "mandate_attribution.png", note=note))
    out = dict(synthetic=True, mandate=a.mandate.name, spec_source=source, k=a.k,
               rules=[dict(kind=r.rule["kind"], status=r.status, detail=r.detail, source=r.rule["source"])
                      for r in compiled.rules],
               warnings=compiled.warnings, excluded_assets=int(u.n - left),
               feasible_fraction=dict(estimate=frac, low=lo, high=hi),
               planted_premium=a.premium, planted_rank=a.rank, seed=a.seed, attribution=res.to_dict())
    (RES / "mandate_demo.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"  saved results/mandate_attribution.png and results/mandate_demo.json  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
