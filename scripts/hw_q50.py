"""Hardware run on VTT Q50 via LUMI (or a local simulator): the FairBench quantum core in practice.

Self-contained: needs only qiskit (+ qiskit-aer for local runs, + iqm-client/qiskit-iqm and optionally
fiqci-ems for Q50 -- all provided on LUMI by the event module ``fiqci-vtt-qiskit-QxF``). It does not
import the rest of fairbench, so the LUMI module environment is enough.

Experiment A  Dicke state |D^n_k> (equal superposition of every portfolio with exactly k holdings)
              for (n, k) in DICKE_CASES. Measures how many shots are valid k-holding portfolios and how
              uniform they are. Random bitstrings would be valid only C(n,k)/2^n of the time.

Experiment B  Toy amplitude estimation of a fund's rank: 4 assets, k = 2 -> 6 portfolios. The fund
              holds assets A and D; 2 of the 6 portfolios returned less, so the exact rank
              (percentile) is 2/6. Circuits: Dicke prep, then m Grover steps Q = A S0 A^dag S_chi,
              then measure, for m in M_LIST. P(good) = sin^2((2m+1) theta) ideally, sin^2 theta = 2/6.
              For k = 2 the phase oracle on the Dicke subspace is one CZ per marked portfolio (two held
              assets identify the portfolio). Also a single-marked-portfolio variant (sin^2 theta = 1/6).
              Fit offline with scripts/hw_q50_analyze.py (noise-aware maximum likelihood).

Usage (local test):   python scripts/hw_q50.py --backend aer
                      python scripts/hw_q50.py --backend noisy --p2 0.01
Usage (LUMI, Q50):    sbatch scripts/run_q50.sh      (see that file; the module sets Q50_CORTEX_URL)
                      python scripts/hw_q50.py --backend q50 [--ems 1]
Results: results/hw/hw_q50_<backend>_<UTC timestamp>.json (job ids are written BEFORE waiting, so a
crashed session can recover results with backend.retrieve_job(job_id)).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from itertools import combinations
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hw"

DICKE_CASES = [(4, 2), (6, 2), (6, 3), (8, 3)]
M_LIST = [0, 1, 2, 3]
ASSETS = ["A", "B", "C", "D"]
GROWTH = [1.12, 0.95, 1.30, 1.05]          # gross return of each asset over the window (toy)
FUND = (0, 3)                                # the fund holds A and D
TWO_Q = {"cz", "cx", "ecr", "iswap", "rzz", "move"}


# ------------------------------------------------------------------ Dicke state (copy of fairbench.quantum.dicke)
def _scs(qc: QuantumCircuit, m: int, l_max: int) -> None:
    q = lambda j: j - 1
    theta = 2 * math.acos(math.sqrt(1 / m))
    qc.cx(q(m - 1), q(m))
    qc.cry(theta, q(m), q(m - 1))
    qc.cx(q(m - 1), q(m))
    for l in range(2, l_max + 1):
        theta = 2 * math.acos(math.sqrt(l / m))
        qc.cx(q(m - l), q(m))
        qc.append(RYGate(theta).control(2, annotated=False), [q(m), q(m - l + 1), q(m - l)])
        qc.cx(q(m - l), q(m))


def dicke_state(n: int, k: int) -> QuantumCircuit:
    """Baertschi & Eidenbenz 2019 split-and-cyclic-shift; qubit i = asset i."""
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    for j in range(n - k, n):
        qc.x(j)
    if k in (0, n):
        return qc
    for m in range(n, 1, -1):
        _scs(qc, m, min(k, m - 1))
    return qc


# ------------------------------------------------------------------ toy rank problem
def portfolios(n: int = 4, k: int = 2) -> list[tuple[int, ...]]:
    return list(combinations(range(n), k))


def worse_than_fund() -> list[tuple[int, int]]:
    """Portfolios whose (equal-weight, buy-and-hold) return is strictly below the fund's."""
    ret = lambda p: sum(GROWTH[i] for i in p) / len(p)
    return [p for p in portfolios() if ret(p) < ret(FUND)]


def bitstring(p: tuple[int, ...], n: int) -> str:
    """Qiskit count key (qubit 0 rightmost) for the portfolio p."""
    return "".join("1" if (n - 1 - j) in p else "0" for j in range(n))


def grover_circuit(marked: list[tuple[int, int]], m: int, n: int = 4, k: int = 2) -> QuantumCircuit:
    """Dicke prep then m x [S_chi ; A^dag ; S0 ; A], measured. k must be 2 (CZ oracle)."""
    assert k == 2
    A = dicke_state(n, k)
    qc = QuantumCircuit(n)
    qc.compose(A, inplace=True)
    for _ in range(m):
        for i, j in marked:                       # S_chi: phase -1 on each marked portfolio
            qc.cz(i, j)
        qc.compose(A.inverse(), inplace=True)     # S0 = I - 2|0><0| (up to global phase)
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.compose(A, inplace=True)
    qc.measure_all()
    qc.name = f"grover_m{m}_marked{len(marked)}"
    return qc


# ------------------------------------------------------------------ backends
def get_backend(name: str, p2: float, ems: int | None):
    if name == "q50":
        from iqm.qiskit_iqm import IQMProvider
        url = os.getenv("Q50_CORTEX_URL")
        if not url:
            raise SystemExit("Q50_CORTEX_URL not set: load the event module "
                             "(module use /appl/local/quantum/modulefiles/hackathon; "
                             "module --ignore_cache load fiqci-vtt-qiskit-QxF)")
        backend = IQMProvider(url, quantum_computer="q50").get_backend()
        if ems:
            from fiqci.ems import FiQCIBackend
            backend = FiQCIBackend(backend, mitigation_level=ems)
        return backend, None
    from qiskit_aer import AerSimulator
    if name == "aer":
        return AerSimulator(), ["cz", "rx", "ry", "rz"]
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["rx", "ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.03, 0.97]]))
    return AerSimulator(noise_model=nm), ["cz", "rx", "ry", "rz"]


def gate_stats(tc: QuantumCircuit) -> dict:
    ops = tc.count_ops()
    return dict(two_qubit=int(sum(v for g, v in ops.items() if g in TWO_Q)), depth=int(tc.depth()),
                n_qubits_used=len({tc.find_bit(q).index for inst in tc.data for q in inst.qubits}),
                ops={g: int(v) for g, v in ops.items()})


# ------------------------------------------------------------------ main
def build_experiments(which: str) -> list[dict]:
    exps = []
    if which in ("A", "all"):
        for n, k in DICKE_CASES:
            qc = dicke_state(n, k)
            qc.measure_all()
            qc.name = f"dicke_n{n}_k{k}"
            exps.append(dict(id=qc.name, kind="dicke", n=n, k=k, circuit=qc,
                             chance_valid=math.comb(n, k) / 2 ** n))
    if which in ("B", "all"):
        worse = worse_than_fund()
        for label, marked in (("rank", worse), ("single", [worse[0]])):
            a = len(marked) / math.comb(4, 2)
            th = math.asin(math.sqrt(a))
            for m in M_LIST:
                qc = grover_circuit(marked, m)
                exps.append(dict(id=f"qae_{label}_m{m}", kind="qae", variant=label, n=4, k=2, m=m,
                                 circuit=qc, marked=[bitstring(p, 4) for p in marked],
                                 a_exact=a, p_good_ideal=math.sin((2 * m + 1) * th) ** 2))
    return exps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["aer", "noisy", "q50"], default="aer")
    ap.add_argument("--exp", choices=["A", "B", "all"], default="all")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--p2", type=float, default=0.01, help="two-qubit depolarising error (noisy sim)")
    ap.add_argument("--ems", type=int, default=None, help="FiQCI-EMS mitigation level (q50 only)")
    ap.add_argument("--opt", type=int, default=3, help="transpiler optimization level")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT / f"hw_q50_{a.backend}{'_ems' + str(a.ems) if a.ems else ''}_{stamp}.json"

    backend, basis = get_backend(a.backend, a.p2, a.ems)
    exps = build_experiments(a.exp)
    tcs = [transpile(e["circuit"], backend=None if basis else backend, basis_gates=basis,
                     optimization_level=a.opt, seed_transpiler=a.seed) for e in exps]
    import qiskit
    meta = dict(backend=a.backend, backend_name=str(getattr(backend, "name", backend)), shots=a.shots,
                ems=a.ems, p2_sim=a.p2 if a.backend == "noisy" else None, utc=stamp,
                qiskit=qiskit.__version__, assets=ASSETS, growth=GROWTH, fund=[ASSETS[i] for i in FUND],
                note="Toy instance for hardware characterisation; not an advantage claim.")
    rec = dict(meta=meta, experiments=[])
    for e, tc in zip(exps, tcs):
        r = {k: v for k, v in e.items() if k != "circuit"}
        r["transpiled"] = gate_stats(tc)
        rec["experiments"].append(r)
        print(f"{e['id']:<22} 2q={r['transpiled']['two_qubit']:>4} depth={r['transpiled']['depth']:>4}")

    job = backend.run(tcs, shots=a.shots)
    job_id = job.job_id() if hasattr(job, "job_id") else None
    rec["meta"]["job_id"] = job_id
    path.write_text(json.dumps(rec, indent=1))           # job id saved before waiting
    print(f"submitted job {job_id}; waiting ... (record: {path})")

    result = job.result()
    for i, r in enumerate(rec["experiments"]):
        r["counts"] = {k.replace(" ", ""): int(v) for k, v in result.get_counts(i).items()}
    path.write_text(json.dumps(rec, indent=1))
    print(f"saved {path}")
    for r in rec["experiments"]:
        c, tot = r["counts"], sum(r["counts"].values())
        if r["kind"] == "dicke":
            valid = sum(v for s, v in c.items() if s.count("1") == r["k"]) / tot
            print(f"  {r['id']:<22} valid={valid:.3f}  (chance {r['chance_valid']:.3f}, ideal 1.000)")
        else:
            good = sum(c.get(s, 0) for s in r["marked"]) / tot
            print(f"  {r['id']:<22} P(good)={good:.3f}  (ideal {r['p_good_ideal']:.3f})")


if __name__ == "__main__":
    main()
