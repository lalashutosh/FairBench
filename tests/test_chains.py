import math

import numpy as np
import pytest
from scipy.stats import chi2

from fairbench.chains import (_basis, feasible_mask, independent_uniform_proposal, matrix_proposal,
                              mixing_time_tv, multi_swap_proposal, proposal_matrix_independent,
                              proposal_matrix_multi_swap, proposal_matrix_swap, relaxation_time,
                              run_chain, spectral_gap, swap_proposal, transition_matrix,
                              proposal_matrix_ctrw, move_stats, swap_component_labels,
                              proposal_cost, min_trotter_steps)
from fairbench.baselines import enumerate_feasible
from fairbench.instances import island_family, island_instance, describe_instance


@pytest.fixture(scope="module")
def isl():
    u, cs = island_instance()
    return u, cs, feasible_mask(u, cs)


def _Qs(n, k):
    return {"swap": proposal_matrix_swap(n, k), "m3": proposal_matrix_multi_swap(n, k, 3),
            "ind": proposal_matrix_independent(n, k)}


def test_mask_matches_enumerate(isl):
    u, cs, mask = isl
    F = enumerate_feasible(u, cs)
    assert np.array_equal(_basis(u.n, 5)[mask], F)


def test_stationary_uniform_and_symmetric(isl):
    u, cs, mask = isl
    M = mask.sum()
    for name, Q in _Qs(u.n, 5).items():
        assert np.allclose(Q, Q.T) and np.allclose(Q.sum(1), 1)
        P = transition_matrix(Q, mask)
        assert P.shape == (M, M) and np.allclose(P, P.T)
        pi = np.full(M, 1 / M)
        assert np.allclose(pi @ P, pi)


def test_independent_gap_analytic(isl):
    u, cs, mask = isl
    C, M = math.comb(u.n, 5), int(mask.sum())
    P = transition_matrix(proposal_matrix_independent(u.n, 5), mask)
    assert spectral_gap(P) == pytest.approx(M / C, rel=1e-8)
    assert relaxation_time(P) == pytest.approx(C / M, rel=1e-8)


def test_swap_gap_zero_on_island(isl):
    u, cs, mask = isl
    assert describe_instance(u, cs)["n_components"] >= 2
    assert spectral_gap(transition_matrix(proposal_matrix_swap(u.n, 5), mask)) == 0.0
    assert mixing_time_tv(transition_matrix(proposal_matrix_swap(u.n, 5), mask), max_steps=50) == math.inf


def test_multi_swap_m1_equals_swap():
    assert np.allclose(proposal_matrix_multi_swap(8, 3, 1), proposal_matrix_swap(8, 3))
    assert np.allclose(proposal_matrix_multi_swap(8, 3, 2), proposal_matrix_swap(8, 3) @ proposal_matrix_swap(8, 3))


def test_multi_swap_proposal_matches_matrix():
    n, k, m = 6, 2, 2
    Q = proposal_matrix_multi_swap(n, k, m)
    B = _basis(n, k)
    rng = np.random.default_rng(0)
    prop = multi_swap_proposal(m)
    N = 20000
    cnt = np.zeros(len(B))
    idx = {b.tobytes(): i for i, b in enumerate(B)}
    for _ in range(N):
        cnt[idx[prop(B[0], rng).tobytes()]] += 1
    exp = Q[0] * N
    ok = exp > 0
    assert (cnt[~ok] == 0).all()
    stat = ((cnt[ok] - exp[ok]) ** 2 / exp[ok]).sum()
    assert stat < chi2.ppf(0.9999, ok.sum() - 1)


def test_run_chain_independent_and_counts(isl):
    u, cs, mask = isl
    F = enumerate_feasible(u, cs)
    r = run_chain(independent_uniform_proposal(u.n, 5), F[0], 20000, cs, u, seed=1)
    assert r.states.shape == (20000, u.n) and r.n_proposal_weight_violations == 0
    assert cs.check_batch(r.states, u).all()
    assert len({s.tobytes() for s in r.states}) == len(F)           # visits all
    # acceptance ~ M/C
    assert r.n_accepted / r.n_proposals == pytest.approx(len(F) / math.comb(u.n, 5), abs=0.01)
    r2 = run_chain(swap_proposal(), F[0], 300, cs, u, seed=2, record_every=10)
    assert len(r2.states) == 30 and r2.n_accepted <= 300


def test_swap_chain_trapped(isl):
    u, cs, mask = isl
    F = enumerate_feasible(u, cs)
    r = run_chain(swap_proposal(), F[0], 5000, cs, u, seed=3)
    assert len({s.tobytes() for s in r.states}) < len(F)


def test_weight_violation_counted(isl):
    u, cs, mask = isl
    F = enumerate_feasible(u, cs)
    r = run_chain(lambda x, rng: np.ones_like(x), F[0], 10, cs, u, seed=0)
    assert r.n_proposal_weight_violations == 10 and r.n_accepted == 0


def test_matrix_proposal_chi2():
    n, k = 6, 2
    Q = proposal_matrix_multi_swap(n, k, 2)
    B = _basis(n, k)
    prop = matrix_proposal(Q, B)
    rng = np.random.default_rng(5)
    N = 20000
    i = 4
    idx = {b.tobytes(): j for j, b in enumerate(B)}
    cnt = np.zeros(len(B))
    for _ in range(N):
        cnt[idx[prop(B[i], rng).tobytes()]] += 1
    exp = Q[i] * N
    ok = exp > 0
    assert (cnt[~ok] == 0).all()
    assert ((cnt[ok] - exp[ok]) ** 2 / exp[ok]).sum() < chi2.ppf(0.9999, ok.sum() - 1)


def test_mixing_time_independent():
    u, cs = island_instance()
    mask = feasible_mask(u, cs)
    P = transition_matrix(proposal_matrix_independent(u.n, 5), mask)
    t = mixing_time_tv(P, 0.25)
    assert 1 <= t < 1000


def test_island_family_small():
    u, cs = island_family(12, seed=0)
    d = describe_instance(u, cs)
    assert d["n"] == 12 and d["k"] == 5 and d["M"] >= 10
    assert d["n_components"] >= 2 and d["component_sizes"][1] >= 3
    mask = feasible_mask(u, cs)
    assert spectral_gap(transition_matrix(proposal_matrix_swap(12, 5), mask)) == 0.0


# ------------------------------------------------------------- new: gap / ctrw / mh_min
def _loop_P(Q, mask):
    """Brute-force MH kernel, uniform target, explicit loops."""
    F = np.flatnonzero(mask)
    M = len(F)
    P = np.zeros((M, M))
    for a, x in enumerate(F):
        for b, y in enumerate(F):
            if a != b:
                P[a, b] = Q[x, y] * min(1.0, Q[y, x] / Q[x, y]) if Q[x, y] > 0 else 0.0
        P[a, a] = 1.0 - P[a].sum()
    return P


def test_bruteforce_loop_P_and_gap():
    from fairbench.instances import p0_instance
    u, cs = p0_instance()
    mask = feasible_mask(u, cs)
    for Q in (proposal_matrix_multi_swap(12, 4, 2), proposal_matrix_ctrw(12, 4, 0.3)):
        P = transition_matrix(Q, mask)
        Pl = _loop_P(Q, mask)
        assert np.allclose(P, Pl, atol=1e-12)
        ev = np.linalg.eigvals(Pl)
        ev = ev[np.argsort(-np.abs(ev))]
        assert abs(ev[0] - 1) < 1e-9
        assert spectral_gap(P) == pytest.approx(1 - abs(ev[1]), abs=1e-9)


def test_periodic_gap_zero():
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert spectral_gap(P) == 0.0
    P3 = np.roll(np.eye(3), 1, axis=1)  # 3-cycle, complex eigenvalues of modulus 1
    assert spectral_gap(P3) == 0.0


def test_spectral_gap_complex_eigenvalues():
    # non-reversible doubly stochastic: lazy rotation on a 4-cycle
    R = np.roll(np.eye(4), 1, axis=1)
    P = 0.5 * np.eye(4) + 0.3 * R + 0.2 * R.T
    ev = np.linalg.eigvals(P)
    assert np.abs(ev.imag).max() > 0.05
    others = np.sort(np.abs(ev))[::-1][1:]
    assert spectral_gap(P) == pytest.approx(1 - others[0], abs=1e-12)
    # a complex pair whose real part is small but modulus is large must dominate
    Pr = 0.05 * np.eye(3) + 0.9 * np.roll(np.eye(3), 1, axis=1) + 0.05 * np.roll(np.eye(3), 2, axis=1)
    lam = np.linalg.eigvals(Pr)
    assert spectral_gap(Pr) == pytest.approx(1 - np.sort(np.abs(lam))[-2], abs=1e-12)
    assert spectral_gap(Pr) < 1 - np.sort(lam.real)[-2]  # real-part sorting would overstate


def test_spectral_gap_reversible_nonuniform():
    # birth-death chain (reversible, non-uniform pi): gap from symmetrised similarity
    P = np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])
    ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    assert spectral_gap(P) == pytest.approx(1 - ev[1], abs=1e-12)


def test_ctrw_untilted_symmetric():
    from scipy.linalg import expm
    for topo in ("complete", "ring"):
        Q = proposal_matrix_ctrw(7, 3, 0.9, topology=topo)
        assert np.allclose(Q, Q.T) and np.allclose(Q.sum(1), 1) and Q.min() >= 0
    S = proposal_matrix_swap(7, 3) * (3 * 4)
    L = S - np.diag(S.sum(1))
    assert np.allclose(proposal_matrix_ctrw(7, 3, 0.9), expm(0.9 * L), atol=1e-12)
    assert np.allclose(proposal_matrix_ctrw(7, 3, 0.0), np.eye(35))


def test_ctrw_tilted_matches_expm_and_mh_min_uniform():
    from scipy.linalg import expm
    n, k = 7, 3
    rng = np.random.default_rng(1)
    d = rng.normal(size=35)
    B = _basis(n, k).astype(int)
    A = (B @ (1 - B).T == 1).astype(float)
    Dm = 2.0 * (d[None, :] - d[:, None])
    for rate, R in (("metropolis", np.minimum(1, np.exp(-Dm))), ("barker", 1 / (1 + np.exp(Dm)))):
        L = A * R
        np.fill_diagonal(L, 0)
        L -= np.diag(L.sum(1))
        Q = proposal_matrix_ctrw(n, k, 0.6, d, beta=2.0, rate=rate)
        assert np.allclose(Q, expm(0.6 * L), atol=1e-12)
        assert not np.allclose(Q, Q.T)
        pi = np.exp(-2.0 * d)
        pi /= pi.sum()
        assert np.allclose(pi @ Q, pi, atol=1e-10)  # Q itself targets exp(-beta d)
    mask = rng.random(35) < 0.6
    P = transition_matrix(Q, mask, mode="mh_min")
    M = mask.sum()
    assert np.allclose(P.sum(1), 1) and P.min() >= -1e-15 and np.allclose(P, P.T)
    assert np.allclose(np.full(M, 1 / M) @ P, np.full(M, 1 / M))
    assert np.allclose(P, _loop_P(Q, mask), atol=1e-12)


def test_transition_matrix_raises_on_asymmetric():
    Q = proposal_matrix_ctrw(6, 2, 0.5, np.arange(15.0), beta=1.0)
    mask = np.ones(15, bool)
    with pytest.raises(ValueError):
        transition_matrix(Q, mask)
    with pytest.raises(ValueError):
        transition_matrix(Q, mask, mode="nope")
    Qs = proposal_matrix_swap(6, 2)
    assert np.allclose(transition_matrix(Qs, mask), transition_matrix(Qs, mask, mode="mh_min"))


def test_move_stats_island():
    u, cs = island_family(12, seed=0)
    mask = feasible_mask(u, cs)
    lab = swap_component_labels(12, 5, mask)
    assert len(np.unique(lab)) == describe_instance(u, cs)["n_components"]
    st = move_stats(proposal_matrix_swap(12, 5), mask, lab)
    assert st["self_proposal"] == 0.0 and st["between_island"] == 0.0 and 0 < st["move_rate"] < 1
    st = move_stats(proposal_matrix_independent(12, 5), mask, lab)
    C, M = math.comb(12, 5), int(mask.sum())
    assert st["self_proposal"] == pytest.approx(1 / C)
    assert st["move_rate"] == pytest.approx((M - 1) / C)
    assert st["between_island"] > 0


def test_proposal_cost():
    assert proposal_cost("per_step") == 1 and proposal_cost("swap") == 1
    assert proposal_cost("multi_swap", m=4) == 4
    assert proposal_cost("ctrw", t=0.5, n=16, k=5) == 0.5 * 55
    assert proposal_cost("qemcmc", n_edges=16, n_trotter=3) == 3 * 31
    assert proposal_cost("qemcmc", n_edges=16, t=1.0, eps=0.01, H_norm=4.0) == 80 * 31
    with pytest.raises(ValueError):
        proposal_cost("bogus")


def test_min_trotter_steps_small():
    r = min_trotter_steps(1.0, 0.02, 6, 3)
    assert 1 <= r <= 64
    r2 = min_trotter_steps(1.0, 0.002, 6, 3)
    assert r2 >= r
