"""Analyse hardware / simulator records from scripts/hw_q50.py and draw the slide figure.

    python scripts/hw_q50_analyze.py results/hw/hw_q50_q50_*.json [results/hw/hw_q50_noisy_*.json ...]

Experiment A: share of shots that are valid k-holding portfolios (vs chance C(n,k)/2^n and ideal 1),
and total-variation distance of the valid shots from uniform.
Experiment B: noise-aware maximum-likelihood fit of the Grover data,
    P(good | m) = f^(2m+1) sin^2((2m+1) theta) + (1 - f^(2m+1)) p_mix,   p_mix = |marked| / 2^n,
giving the estimated rank sin^2(theta) (the fund's percentile; every Dicke state is a valid
portfolio here, so a_F = 1) with a parametric-bootstrap 95% interval, and the per-step fidelity f.
Compared with the naive m=0 estimate (= plain sampling of the same shots).
Writes results/hw/hw_q50_summary.json and results/hw/hw_q50.png (slide-size fonts).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hw"


def valid_stats(r: dict) -> dict:
    c, n, k = r["counts"], r["n"], r["k"]
    tot = sum(c.values())
    valid = {s: v for s, v in c.items() if s.count("1") == k}
    nv = sum(valid.values())
    C = math.comb(n, k)
    tv = 0.5 * sum(abs(valid.get(s, 0) / nv - 1 / C) for s in _weight_k(n, k)) if nv else float("nan")
    return dict(id=r["id"], n=n, k=k, valid=nv / tot, chance=r["chance_valid"], tv_uniform=tv,
                two_qubit=r["transpiled"]["two_qubit"], shots=tot)


def _weight_k(n: int, k: int):
    from itertools import combinations
    for p in combinations(range(n), k):
        yield "".join("1" if (n - 1 - j) in p else "0" for j in range(n))


def _loglik(theta, f, ms, hits, shots, pmix):
    L = 0.0
    for m, h, N in zip(ms, hits, shots):
        d = f ** (2 * m + 1)
        p = d * np.sin((2 * m + 1) * theta) ** 2 + (1 - d) * pmix
        p = np.clip(p, 1e-9, 1 - 1e-9)
        L = L + h * np.log(p) + (N - h) * np.log(1 - p)
    return L


def fit(ms, hits, shots, pmix):
    th = np.linspace(1e-3, math.pi / 2 - 1e-3, 1200)[:, None]
    ff = np.linspace(0.05, 1.0, 400)[None, :]
    L = _loglik(th, ff, ms, hits, shots, pmix)
    i, j = np.unravel_index(np.argmax(L), L.shape)
    return float(th[i, 0]), float(ff[0, j])


def qae_fit(recs: list[dict], n_boot: int = 300, seed: int = 0) -> dict:
    recs = sorted(recs, key=lambda r: r["m"])
    ms = [r["m"] for r in recs]
    shots = [sum(r["counts"].values()) for r in recs]
    hits = [sum(r["counts"].get(s, 0) for s in r["marked"]) for r in recs]
    pmix = len(recs[0]["marked"]) / 2 ** recs[0]["n"]
    th, f = fit(ms, hits, shots, pmix)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        d = [f ** (2 * m + 1) for m in ms]
        p = [di * math.sin((2 * m + 1) * th) ** 2 + (1 - di) * pmix for di, m in zip(d, ms)]
        h = rng.binomial(shots, np.clip(p, 0, 1))
        boots.append(math.sin(fit(ms, h, shots, pmix)[0]) ** 2)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    m0 = recs[0]
    naive = hits[0] / shots[0]
    valid0 = sum(v for s, v in m0["counts"].items() if s.count("1") == m0["k"])
    return dict(variant=m0["variant"], a_exact=m0["a_exact"], a_hat=math.sin(th) ** 2, ci95=[float(lo), float(hi)],
                fidelity_per_step=f, survival_per_grover_step=f * f, naive_m0=naive, naive_m0_postselected=hits[0] / max(valid0, 1),
                ms=ms, p_good=[h / N for h, N in zip(hits, shots)], p_ideal=[r["p_good_ideal"] for r in recs],
                two_qubit=[r["transpiled"]["two_qubit"] for r in recs], theta=th, pmix=pmix)


def main(paths):
    fig_sel = {}
    for flag in ("--dicke", "--qae", "--qae2"):
        if flag in paths:
            i = paths.index(flag)
            fig_sel[flag[2:]] = paths[i + 1]
            del paths[i:i + 2]
    paths = list(dict.fromkeys(paths + list(fig_sel.values())))
    if not paths:
        paths = sorted(str(p) for p in OUT.glob("hw_q50_*.json") if "summary" not in p.name)
    summary = {}
    for p in paths:
        rec = json.loads(Path(p).read_text())
        if not all("counts" in r for r in rec["experiments"]):
            print(f"skip {p}: no counts yet (job {rec['meta'].get('job_id')})")
            continue
        label = rec["meta"]["backend"] + (f"_ems{rec['meta']['ems']}" if rec["meta"].get("ems") else "") + (
            f"_p2={rec['meta']['p2_sim']}" if rec["meta"].get("p2_sim") else "")
        A = [valid_stats(r) for r in rec["experiments"] if r["kind"] == "dicke"]
        B = {}
        for var in ("rank", "single"):
            rs = [r for r in rec["experiments"] if r["kind"] == "qae" and r["variant"] == var]
            if rs:
                B[var] = qae_fit(rs)
        label = label + "@" + rec["meta"]["utc"]
        summary[label] = dict(file=str(p), meta=rec["meta"], dicke=A, qae=B)
        print(f"\n== {label}  ({rec['meta'].get('backend_name')}, {rec['meta']['utc']})")
        for a in A:
            print(f"  Dicke n={a['n']} k={a['k']}: valid {a['valid']:.1%} (chance {a['chance']:.1%}), "
                  f"TV-to-uniform {a['tv_uniform']:.3f}, 2q gates {a['two_qubit']}")
        for v, b in B.items():
            print(f"  QAE[{v}]: rank estimate {b['a_hat']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] "
                  f"(exact {b['a_exact']:.3f}); signal surviving per Grover step {b['survival_per_grover_step']:.2f}; "
                  f"naive m=0 {b['naive_m0']:.3f}; P(good) {['%.2f' % x for x in b['p_good']]} "
                  f"vs ideal {['%.2f' % x for x in b['p_ideal']]}")
    if not summary:
        return
    (OUT / "hw_q50_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pick = lambda f: next((k for k, v in summary.items() if v["file"] == f), None)
    plot(summary, pick(fig_sel.get("dicke")), pick(fig_sel.get("qae")), pick(fig_sel.get("qae2")))


def plot(summary: dict, dk: str | None = None, qk: str | None = None, qk2: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 22, "axes.titlesize": 26, "axes.labelsize": 24, "legend.fontsize": 18})
    pref = next((k for k in summary if k.startswith("q50")), next(iter(summary)))
    dk = dk or next((k for k in summary if summary[k]["dicke"]), pref)
    qk = qk or next((k for k in summary if summary[k]["qae"]), pref)
    S = dict(summary[dk], qae=summary[qk]["qae"], meta=summary[qk]["meta"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19.2, 10.8))
    # left: valid-portfolio share
    lab = [f"n={d['n']}\nk={d['k']}" for d in S["dicke"]]
    x = np.arange(len(lab))
    a1.bar(x - 0.2, [d["valid"] for d in S["dicke"]], 0.4, color="#1f78b4", label="measured")
    a1.bar(x + 0.2, [d["chance"] for d in S["dicke"]], 0.4, color="#bbbbbb", label="random bitstrings")
    a1.axhline(1.0, color="k", ls="--", lw=2, label="ideal")
    a1.set_xticks(x, lab)
    a1.set_ylim(0, 1.3)
    a1.set_ylabel("share of shots that are valid portfolios")
    a1.set_title("Constraint-preserving circuit")
    a1.legend(loc="upper center", ncol=3, fontsize=16)
    # right: Grover signal (rank variant)
    b = S["qae"].get("rank") or next(iter(S["qae"].values()))
    mm = np.linspace(0, max(b["ms"]), 200)
    a2.plot(mm, np.sin((2 * mm + 1) * math.asin(math.sqrt(b["a_exact"]))) ** 2, "k--", lw=2, label="ideal")
    d = b["fidelity_per_step"] ** (2 * mm + 1)
    a2.plot(mm, d * np.sin((2 * mm + 1) * b["theta"]) ** 2 + (1 - d) * b["pmix"], color="#1f78b4", lw=3,
            label=f"noise-aware fit ({b['survival_per_grover_step']:.0%} signal/step)")
    a2.plot(b["ms"], b["p_good"], "o", color="#1f78b4", ms=16, label="measured")
    if qk2:
        b2 = summary[qk2]["qae"].get("rank")
        lab2 = "measured + readout mitigation" if summary[qk2]["meta"].get("ems") else "measured (run 2)"
        a2.plot(b2["ms"], b2["p_good"], "s", color="#d95f02", ms=14, label=lab2)
    a2.axhline(b["pmix"], color="#999999", ls=":", lw=2, label="random-output level")
    a2.set_xticks(b["ms"])
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("Grover steps m")
    a2.set_ylabel("P(portfolio worse than fund)")
    a2.set_title(f"Amplitude estimation: rank {b['a_hat']:.0%} (exact {b['a_exact']:.0%})")
    a2.legend(loc="upper right", fontsize=15)
    name = "VTT Q50" if S["meta"].get("backend") == "q50" else S["meta"].get("backend_name", pref)
    fig.suptitle(f"FairBench quantum core on {name}", fontsize=30, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "hw_q50.png", dpi=100)
    print(f"\nsaved {OUT / 'hw_q50.png'} (from {pref})")


if __name__ == "__main__":
    main(sys.argv[1:])
