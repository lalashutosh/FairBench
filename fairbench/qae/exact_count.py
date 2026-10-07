"""Classical exact-counting dynamic program for the attribution percentile (QA-7).

Counts weight-k selections satisfying Cardinality, SectorCap (per sector), Exclusion, CountBound,
and the WEIGHTED rules (MinESG, CarbonCap, AvgBound, LinearThreshold), the latter on INTEGER-
QUANTISED weights (``bins`` levels per rule, w_i = rint(s (v_i - lo)), s = (bins-1)/(hi-lo), as in
``fairbench.ft.oracle.quantise`` but with one fixed scale), by a DP over SECTORS:

  state  = (count so far, CountBound counts, saturating integer sum of each weighted rule)
  step   = for each allowed subset of <= cap assets of the sector, shift the state array (numpy slices).

Weighted sums are SATURATING at the integer threshold T (weights >= 0, rule is  sum w x >= T), so each
weighted axis has T+1 cells (T ~ k * bins / 2): the state is ~ (k+1) * prod_r (T_r + 1) = O(k^(R+1) bins^R)
for R weighted rules.  That exponent is the point: it is what limits the DP.

Brackets: thresholds as in oracle.quantise modes -- "superset" (T - k/2, F subset F_q), "subset"
(T + k/2, F_q subset F), "mid" (plain rounding).  Because every |rounding error| <= 1/2 per weight,
F_subset <= F_true <= F_superset (counts), hence for p = |G|/|F|:
    G_subset / F_superset  <=  p  <=  G_superset / F_subset.
Counts are float64 (relative error ~1e-16 per add; counts up to C(200,20) ~ 1e27 do not fit int64).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..constraints import (CarbonCap, Cardinality, ConstraintSet, CountBound, AvgBound, Exclusion,
                           LinearThreshold, MinESG, SectorCap)
from ..data import Universe
from ..ft.oracle import _mode_threshold, _weighted_specs

MAX_STATES = 60_000_000        # dense float64 cells per array (arr + new + one temp ~ 1.44 GB)
MAX_OPS = 1.5e11               # element-ops budget per run (sum over sectors of subsets x state cells)
WEIGHTED = (MinESG, CarbonCap, AvgBound, LinearThreshold)
SUPPORTED = (Cardinality, SectorCap, Exclusion, CountBound) + WEIGHTED


class StateTooLarge(RuntimeError):
    def __init__(self, msg, states=None, ops=None):
        super().__init__(msg)
        self.states, self.ops = states, ops


@dataclass
class _Axis:
    size: int
    saturate: bool            # True: sum saturates at size-1; False: overflow dropped
    accept: tuple             # (lo, hi) inclusive index range of accepted final states


@dataclass
class DPResult:
    count: float              # count under ALL rules
    count_wo_last: float      # count ignoring the last weighted rule (the F count when the last is the fund rule)
    peak_states: int          # dense cells of one state array
    ops: float                # element-ops performed
    runtime: float
    mode: str
    bins: int
    weights: list = field(default_factory=list)      # per weighted rule: (int weights (n,), T)
    state_shape: tuple = ()


def make_rules(u: Universe, cs: ConstraintSet, bins: int, mode: str, extra_linear: LinearThreshold | None = None):
    """Quantised weighted rules: list of (w (n,) int64 >= 0, T int) with rule  w.x >= T  (any x of weight k)."""
    k = cs.cardinality
    cs_all = ConstraintSet(list(cs.constraints) + ([extra_linear] if extra_linear is not None else []))
    allowed = cs_all.allowed_indices(u.n)
    out = []
    for sp in _weighted_specs(u, cs_all, k):
        v = sp.v
        lo, hi = float(v[allowed].min()), float(v[allowed].max())
        s = (bins - 1) / (hi - lo) if hi > lo else 1.0
        w = np.rint(s * (np.clip(v, lo, hi) - lo)).astype(np.int64)
        t0 = sp.t0(s, lo, k)
        out.append((w, _mode_threshold(mode, t0, k, sp.strict)))
    return out


def count_dp(u: Universe, cs: ConstraintSet, extra_linear: LinearThreshold | None = None, bins: int = 256,
             mode: str = "mid", max_states: int = MAX_STATES, max_ops: float = MAX_OPS) -> DPResult:
    """Exact count of weight-k selections satisfying ``cs`` (+ ``extra_linear``) under the QUANTISED weighted
    rules (``bins`` levels, threshold ``mode`` in superset|subset|mid).  Raises StateTooLarge if the dense
    state would exceed ``max_states`` cells or ``max_ops`` element-ops.  ``count_wo_last`` is the count with
    the LAST weighted rule dropped (use with ``extra_linear`` to get |F| in the same pass)."""
    t_start = time.perf_counter()
    k = cs.cardinality
    if k is None:
        raise ValueError("needs a Cardinality constraint")
    cs_all = ConstraintSet(list(cs.constraints) + ([extra_linear] if extra_linear is not None else []))
    for c in cs_all.constraints:
        if not isinstance(c, SUPPORTED):
            raise NotImplementedError(f"count_dp does not support {type(c).__name__}")
    n = u.n
    allowed = cs_all.allowed_indices(n)
    rules = make_rules(u, cs, bins, mode, extra_linear)
    R = len(rules)
    # ---- axes: count | CountBound... | weighted...
    axes = [_Axis(k + 1, False, (k, k))]
    cbs = []
    for c in cs_all.constraints:
        if isinstance(c, CountBound):
            mx = c.max_count
            mn = max(c.min_count, 0)
            in_idx = np.zeros(n, bool); in_idx[c.indices] = True
            if mn > k or (mx is not None and mx < mn):
                return DPResult(0.0, 0.0, 1, 0.0, time.perf_counter() - t_start, mode, bins)
            if mx is not None and mx < k:
                axes.append(_Axis(mx + 1, False, (mn, mx)))
            elif mn > 0:
                axes.append(_Axis(mn + 1, True, (mn, mn)))
            else:
                continue
            cbs.append(in_idx)
    ncb = len(cbs)
    wsum_max = []
    for w, T in rules:
        wmax_k = int(np.sort(w[allowed])[-k:].sum()) if allowed.size >= k else 0
        if T > wmax_k:                                    # unreachable
            return DPResult(0.0, 0.0, 1, 0.0, time.perf_counter() - t_start, mode, bins, rules)
        Tc = max(T, 0)
        axes.append(_Axis(Tc + 1, True, (Tc, Tc)))
    shape = tuple(a.size for a in axes)
    states = int(np.prod(shape, dtype=object))
    # ---- sectors
    caps = {}
    for c in cs_all.constraints:
        if isinstance(c, SectorCap):
            caps[c.sector] = min(caps.get(c.sector, k), c.max_count)
    sec_assets = {}
    for i in allowed:
        sec_assets.setdefault(u.sector[i], []).append(int(i))
    sec_subsets = []
    for s, items in sec_assets.items():
        cap = min(caps.get(s, k), k, len(items))
        rows = []
        for r in range(cap + 1):
            comb = (np.array(list(itertools.combinations(items, r)), dtype=np.int64) if r else np.zeros((1, 0), np.int64))
            sh = np.zeros((len(comb), 1 + ncb + R), np.int64)
            sh[:, 0] = r
            for b, m in enumerate(cbs):
                sh[:, 1 + b] = m[comb].sum(1) if r else 0
            for q, (w, _T) in enumerate(rules):
                sh[:, 1 + ncb + q] = w[comb].sum(1) if r else 0
            rows.append(sh)
        sh = np.concatenate(rows)
        # CountBound axes that were skipped (trivial) are not in `axes`; cbs only holds active ones.
        uq, mult = np.unique(sh, axis=0, return_counts=True)
        sec_subsets.append((uq, mult.astype(np.float64)))
    ops = float(sum(len(uq) for uq, _ in sec_subsets)) * states
    if states > max_states:
        raise StateTooLarge(f"state {shape} = {states:.3g} cells > {max_states:.3g}", states, ops)
    if ops > max_ops:
        raise StateTooLarge(f"ops {ops:.3g} > {max_ops:.3g}", states, ops)
    arr = np.zeros(shape)
    arr[(0,) * len(shape)] = 1.0
    for uq, mult in sec_subsets:
        new = np.zeros(shape)
        for row, m in zip(uq, mult):
            _shift_add(new, arr, row, axes, m)
        arr = new
    final = arr[k]
    sl = tuple(slice(a.accept[0], a.accept[1] + 1) for a in axes[1:])
    count = float(final[sl].sum())
    if R:
        sl2 = sl[:-1] + (slice(None),)
        count_wo = float(final[sl2].sum())
    else:
        count_wo = count
    return DPResult(count, count_wo, states, ops, time.perf_counter() - t_start, mode, bins, rules, shape)


def _shift_add(new, arr, row, axes, mult):
    """new[...] += mult * arr shifted by row[a] along axis a (axis 0 dropped on overflow, saturating axes fold)."""
    pieces = [[]]                 # list of (src_slice, dst_slice, sum_axis or None) per axis, product
    per_axis = []
    for a, ax in enumerate(axes):
        d = int(row[a]); S = ax.size
        opts = []
        if ax.saturate:
            T = S - 1
            if d <= T:
                opts.append((slice(0, T - d), slice(d, T), False))
                opts.append((slice(T - d, T + 1), slice(T, T + 1), True))
            else:
                opts.append((slice(0, T + 1), slice(T, T + 1), True))
        else:
            if d < S:
                opts.append((slice(0, S - d), slice(d, S), False))
        if not opts:
            return
        per_axis.append(opts)
    for combo in itertools.product(*per_axis):
        src = tuple(c[0] for c in combo); dst = tuple(c[1] for c in combo)
        sa = tuple(i for i, c in enumerate(combo) if c[2])
        blk = arr[src]
        if blk.size == 0:
            continue
        if sa:
            blk = blk.sum(axis=sa, keepdims=True)
        if mult == 1.0:
            new[dst] += blk
        else:
            new[dst] += mult * blk


def percentile_dp(u: Universe, cs: ConstraintSet, fund: LinearThreshold, bins: int = 256,
                  **kw):
    """Attribution percentile p = |F and fund-rule| / |F| by DP.  ``fund`` is the LinearThreshold
    'return < s_fund' (weights g, threshold k(1+s_fund), '<').  Returns ((p_lo, p_hi, p_point), runtime,
    peak_states) with p_lo = G_subset/F_superset <= p_true <= p_hi = G_superset/F_subset (guaranteed)."""
    t0 = time.perf_counter()
    r = {m: count_dp(u, cs, fund, bins, m, **kw) for m in ("subset", "mid", "superset")}
    p_lo = r["subset"].count / r["superset"].count_wo_last if r["superset"].count_wo_last > 0 else 0.0
    p_hi = r["superset"].count / r["subset"].count_wo_last if r["subset"].count_wo_last > 0 else 1.0
    p_pt = r["mid"].count / r["mid"].count_wo_last if r["mid"].count_wo_last > 0 else float("nan")
    peak = max(x.peak_states for x in r.values())
    return (min(p_lo, 1.0), min(p_hi, 1.0), p_pt), time.perf_counter() - t0, peak
