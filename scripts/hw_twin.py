"""Digital twin of the VTT Q50 runs: calibrate a noise model to the measured data, then sweep the
two-qubit error rate down to show what hardware amplitude estimation needs.

Inputs (from scripts/hw_q50.py runs on Q50):
  * readout:  results/hw/hw_q50_q50_20261009T135044Z.json  (per-qubit 0->1 and 1->0 flip rates)
  * Grover:   results/hw/hw_q50_q50_20261009T135154Z.json  (toy rank QAE, pinned to qubits 0-3)
  * Dicke:    results/hw/hw_q50_q50_20261009T141216Z.json  (valid-portfolio share, n = 4 pinned)
Model: Aer simulator, square-grid connectivity (stand-in for Q50's lattice; it routes ~15% fewer CZ
than the real chip, which the fitted error absorbs), depolarising error p2 on CZ and p2/10 on
single-qubit gates, measured per-qubit readout errors. ONE parameter (p2) is fitted by maximum
likelihood to all measured points.
Sweep: p2 from the fitted value down to 1e-5 for m = 0..8 Grover steps: P(good) curves, signal
surviving per step, and the noise-aware rank estimate (same fit as scripts/hw_q50_analyze.py).
Extrapolation: per-step survival s = (1 - p2)^G for the real problem's gate count G per step
(from results/qae_noise.json / transpile counts) -> error rate needed for a given number of steps.

    python scripts/hw_twin.py              # ~2-5 min on a laptop or one LUMI CPU node
Writes results/hw/hw_twin.json and results/hw/hw_twin.png (slide-size fonts).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hw_q50 as H                    # noqa: E402  (circuits, toy instance)
import hw_q50_analyze as A            # noqa: E402  (noise-aware fit)

OUT = ROOT / "results" / "hw"
F_RO = OUT / "hw_q50_q50_20261009T135044Z.json"
F_QAE = OUT / "hw_q50_q50_20261009T135154Z.json"
F_DICKE = OUT / "hw_q50_q50_20261009T141216Z.json"
BASIS = ["cz", "rx", "ry", "rz"]
SHOTS = 4000
P2_FIT_GRID = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.04]
P2_SWEEP = [None, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]     # None = fitted value
M_SWEEP = list(range(9))


def readout_from_measurement():
    d = json.loads(F_RO.read_text())
    flips = {}
    for r in d["experiments"]:
        if r["kind"] != "readout":
            continue
        c, e = r["counts"], r["expected"]
        tot = sum(c.values())
        flips[r["id"]] = [sum(v for s, v in c.items() if s[3 - q] != e[3 - q]) / tot for q in range(4)]
    p01 = flips["ro_0000"]                      # prepared 0, read 1
    p10 = flips["ro_1111"]                      # prepared 1, read 0
    return p01, p10


def noise_model(p2: float, p01, p10, scale_readout: float = 1.0):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    a01, a10 = float(np.mean(p01)), float(np.mean(p10))
    for q in range(56):
        e01 = (p01[q] if q < 4 else a01) * scale_readout
        e10 = (p10[q] if q < 4 else a10) * scale_readout
        nm.add_readout_error(ReadoutError([[1 - e01, e01], [e10, 1 - e10]]), [q])
    return nm


_TC = {}


def transpiled(qc, key):
    """Route on a 7x8 grid with the logical qubits on the first row (like the pinned chain 0-3)."""
    if key not in _TC:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        _TC[key] = transpile(qc, coupling_map=CouplingMap.from_grid(7, 8), basis_gates=BASIS,
                             initial_layout=list(range(qc.num_qubits)), optimization_level=3,
                             seed_transpiler=7)
    return _TC[key]


def run(circs: dict, p2: float, p01, p10, shots=SHOTS, seed=11):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(noise_model=noise_model(p2, p01, p10), seed_simulator=seed)
    keys = list(circs)
    res = sim.run([transpiled(circs[k], k) for k in keys], shots=shots).result()
    return {k: {s.replace(" ", ""): int(v) for s, v in res.get_counts(i).items()} for i, k in enumerate(keys)}


def p_good(counts, marked):
    tot = sum(counts.values())
    return sum(counts.get(s, 0) for s in marked) / tot


def valid_share(counts, k):
    tot = sum(counts.values())
    return sum(v for s, v in counts.items() if s.count("1") == k) / tot


def measured_targets():
    """(key, kind, hits, shots, extra) for every measured point used in the fit."""
    t = []
    q = json.loads(F_QAE.read_text())
    for r in q["experiments"]:
        c, tot = r["counts"], sum(r["counts"].values())
        t.append((r["id"], "good", sum(c.get(s, 0) for s in r["marked"]), tot, r["marked"]))
    d = json.loads(F_DICKE.read_text())
    for r in d["experiments"]:
        if r["n"] == 4:                                   # the pinned case (others were re-routed)
            c, tot = r["counts"], sum(r["counts"].values())
            t.append((r["id"], "valid", sum(v for s, v in c.items() if s.count("1") == r["k"]), tot, r["k"]))
    return t


def circuits_for_targets():
    worse = H.worse_than_fund()
    circs = {}
    for label, marked in (("rank", worse), ("single", [worse[0]])):
        for m in H.M_LIST:
            circs[f"qae_{label}_m{m}"] = H.grover_circuit(marked, m)
    qc = H.dicke_state(4, 2)
    qc.measure_all()
    circs["dicke_n4_k2"] = qc
    return circs


def calibrate(p01, p10):
    targets = measured_targets()
    circs = circuits_for_targets()
    rows = []
    for p2 in P2_FIT_GRID:
        cnt = run(circs, p2, p01, p10)
        ll = 0.0
        for key, kind, h, n, extra in targets:
            p = p_good(cnt[key], extra) if kind == "good" else valid_share(cnt[key], extra)
            p = min(max(p, 1e-4), 1 - 1e-4)
            ll += h * math.log(p) + (n - h) * math.log(1 - p)
        rows.append((p2, ll, cnt))
        print(f"  calibrate p2={p2:.3f}  loglik={ll:.1f}")
    best = max(rows, key=lambda r: r[1])
    # refine with a parabola through the best three grid points (in log p2)
    i = [r[0] for r in rows].index(best[0])
    p2_hat = best[0]
    if 0 < i < len(rows) - 1:
        x = np.log([rows[i - 1][0], rows[i][0], rows[i + 1][0]])
        y = [rows[i - 1][1], rows[i][1], rows[i + 1][1]]
        a, b, _ = np.polyfit(x, y, 2)
        if a < 0:
            p2_hat = float(np.exp(-b / (2 * a)))
    model = {key: (p_good(best[2][key], extra) if kind == "good" else valid_share(best[2][key], extra))
             for key, kind, h, n, extra in targets}
    meas = {key: h / n for key, kind, h, n, extra in targets}
    return p2_hat, dict(grid=[(r[0], r[1]) for r in rows], model_at_grid_best=model, measured=meas,
                        grid_best=best[0])


def sweep(p2_values, p01, p10):
    worse = H.worse_than_fund()
    a_exact = len(worse) / 6
    th = math.asin(math.sqrt(a_exact))
    out = []
    for p2 in p2_values:
        circs = {f"m{m}": H.grover_circuit(worse, m) for m in M_SWEEP}
        cnt = run(circs, p2, p01, p10)
        pg = [p_good(cnt[f"m{m}"], [H.bitstring(p, 4) for p in worse]) for m in M_SWEEP]
        recs = [dict(m=m, n=4, k=2, variant="rank", a_exact=a_exact, p_good_ideal=math.sin((2 * m + 1) * th) ** 2,
                     marked=[H.bitstring(p, 4) for p in worse], counts=cnt[f"m{m}"],
                     transpiled=dict(two_qubit=transpiled(circs[f"m{m}"], f"m{m}").count_ops().get("cz", 0)))
                for m in M_SWEEP]
        f = A.qae_fit(recs, n_boot=100)
        out.append(dict(p2=p2, p_good=pg, rank_hat=f["a_hat"], ci95=f["ci95"],
                        survival_per_step=f["survival_per_grover_step"],
                        cz_per_step=recs[1]["transpiled"]["two_qubit"] - recs[0]["transpiled"]["two_qubit"]))
        print(f"  sweep p2={p2:.1e}: rank {f['a_hat']:.3f} [{f['ci95'][0]:.3f},{f['ci95'][1]:.3f}] "
              f"(exact {a_exact:.3f}), signal/step {f['survival_per_grover_step']:.2f}")
    return out, a_exact


def requirements():
    """Error rate needed so that the signal survives M Grover steps at half strength:
    (1 - p2)^(G M) >= 1/2  ->  p2 <= ln 2 / (G M), G = two-qubit gates per step."""
    # gates per Grover step: toy measured on Q50 (this run); n=16 from QA-4 (results/qae_noise_transpile.csv)
    G = {"toy (n=4, measured on Q50)": 150, "n=16 assets (transpiled)": 16785,
         "n=16 on heavy-hex routing": 43481}
    rows = []
    for name, g in G.items():
        for M in (10, 100, 1000):
            rows.append(dict(problem=name, G=g, steps=M, p2_needed=math.log(2) / (g * M)))
    return rows


def plot(cal, sw, a_exact, p2_hat, p01, p10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 22, "axes.titlesize": 26, "axes.labelsize": 24, "legend.fontsize": 16})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19.2, 10.8))
    # left: twin vs measured (rank Grover points + Dicke valid share)
    q = json.loads(F_QAE.read_text())
    ms = [r["m"] for r in q["experiments"] if r["variant"] == "rank"]
    meas = [cal["measured"][f"qae_rank_m{m}"] for m in ms]
    twin = sw[0]["p_good"][: len(ms)]
    ideal = [math.sin((2 * m + 1) * math.asin(math.sqrt(a_exact))) ** 2 for m in ms]
    x = np.arange(len(ms))
    a1.plot(ms, ideal, "k--", lw=2, marker="o", ms=8, label="ideal")
    a1.plot(ms, meas, "o", color="#1f78b4", ms=18, label="measured on Q50")
    a1.plot(ms, twin, "s-", color="#d95f02", ms=14, lw=3, label=f"twin (fitted CZ error {p2_hat:.1%})")
    a1.set_xticks(ms)
    a1.set_ylim(0, 1.25)
    a1.set_xlabel("Grover steps m")
    a1.set_ylabel("P(portfolio worse than fund)")
    a1.set_title("Digital twin reproduces Q50")
    a1.legend(loc="upper center", ncol=1, fontsize=16)
    # right: sweep (subset of error rates for legibility; all rates are in hw_twin.json)
    show = [s for s in sw if s is sw[0] or any(abs(s["p2"] - v) < 1e-12 for v in (3e-3, 1e-3, 1e-4))]
    cols = ["#bbbbbb", "#9ecae1", "#3182bd", "#08306b"]
    th = math.asin(math.sqrt(a_exact))
    a2.plot(M_SWEEP, [math.sin((2 * m + 1) * th) ** 2 for m in M_SWEEP], "k--", lw=2, marker="o", ms=7,
            label="ideal")
    for c, s in zip(cols, show):
        lab = f"today: {s['p2']:.1%} (fitted)" if s is sw[0] else f"{s['p2']:.0e}"
        a2.plot(M_SWEEP, s["p_good"], "-o", color=c, lw=4, ms=10, label=lab)
    a2.set_ylim(0, 1.25)
    a2.set_xticks(M_SWEEP)
    a2.set_xlabel("Grover steps m")
    a2.set_title("Simulated at lower CZ error rates")
    a2.legend(loc="upper center", ncol=3, fontsize=15, title="two-qubit error", title_fontsize=15)
    fig.suptitle("Amplitude estimation: measured on Q50, simulated below", fontsize=28,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "hw_twin.png", dpi=100)
    print(f"saved {OUT / 'hw_twin.png'}")


def main():
    p01, p10 = readout_from_measurement()
    print("readout 0->1:", [round(x, 3) for x in p01], " 1->0:", [round(x, 3) for x in p10])
    print("calibrating ...")
    p2_hat, cal = calibrate(p01, p10)
    print(f"fitted two-qubit (CZ) depolarising error: {p2_hat:.4f}")
    print("measured vs twin:", {k: (round(cal['measured'][k], 3), round(v, 3)) for k, v in cal["model_at_grid_best"].items()})
    print("sweeping ...")
    sw, a_exact = sweep([p2_hat if p is None else p for p in P2_SWEEP], p01, p10)
    req = requirements()
    for r in req:
        print(f"  needs p2 <= {r['p2_needed']:.1e} for {r['steps']} steps at half signal: {r['problem']}")
    rec = dict(readout_p01=p01, readout_p10=p10, p2_fitted=p2_hat, calibration=cal, sweep=sw,
               a_exact=a_exact, requirements=req,
               notes=["Toy 4-asset instance; noise model = depolarising CZ + 1q + measured readout on a grid "
                      "coupling map; one fitted parameter. Not a prediction for specific future hardware.",
                      "Requirement rows use (1-p2)^(G*M) >= 1/2; gate counts from results/qae_noise.json (QA-4)."])
    (OUT / "hw_twin.json").write_text(json.dumps(rec, indent=1, default=float))
    plot(cal, sw, a_exact, p2_hat, p01, p10)


if __name__ == "__main__":
    main()
