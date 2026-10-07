"""Slide-ready pitch figure for the QAE results. Reads existing result files only.

Left panel: percentile |error| vs total queries (n24_k6): median + IQR band, classical vs IQAE (20 shots/round).
Right panel: queries for 95%-of-runs |error| <= 0.01 (tol 0.01, percentile only), vs feasible fraction P_F,
with the example mandate (n=150, P_F 0.55%, full variant, eps 0.01) as stars.
"""
import csv
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT_PNG = os.path.join(RES, "qae_pitch_figure.png")
OUT_SVG = os.path.join(RES, "qae_pitch_figure.svg")

C_REJ = "#4d4d4d"    # dark grey: classical rejection
C_MCMC = "#d95f02"   # orange: swap-MCMC
C_QAE = "#1f78b4"    # blue: quantum AE (IQAE, 20 shots/round)
QAE_KEY = "iqae_s20"
QAE_LABEL = "IQAE (20 shots/round)"

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "font.family": "DejaVu Sans",
})


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def load_csv(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def interp_q(pts, target, key):
    """Replicates scripts/qae_scaling.py::interp_q (monotone min, log-log crossing).
    pts: list of dicts with 'q' and key. Returns None if the curve never reaches target."""
    import numpy as np
    x = np.array([d["q"] for d in pts], dtype=float)
    y = np.array([d[key] for d in pts], dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    y = np.maximum(np.minimum.accumulate(y), 1e-12)
    if y[-1] > target:
        return None
    if y[0] <= target:
        return float(x[0])
    j = int(np.argmax(y <= target))
    lx = np.interp(np.log(target), [np.log(y[j]), np.log(y[j - 1])], [np.log(x[j]), np.log(x[j - 1])])
    return float(np.exp(lx))


# ---------------------------------------------------------------- data
scal = load_json("qae_scaling.json")
scal_rows = load_csv("qae_scaling.csv")
mand_rows = load_csv("qae_mandate.csv")

# Left panel: instance n24_k6, part 'a'
def series(case, method):
    pts = []
    for r in scal_rows:
        if r["part"] == "a" and r["case"] == case and r["method"] == method:
            pts.append((float(r["q"]), float(r["e50"]), float(r["e25"]), float(r["e75"])))
    pts.sort()
    return pts

left = {"rej": series("n24_k6", "classical"), QAE_KEY: series("n24_k6", QAE_KEY)}
slopes_a = scal["a"]["n24_k6"]["slopes"]
missing_left = [k for k, v in left.items() if not v]

# Right panel: q95 cost at tol 0.01 for each tight case, from the csv 'b' error curves
cases = sorted(scal["b"]["cases"], key=lambda c: -c["P_F"])
TOL = 0.01
METHODS = (("rej", "classical", C_REJ), ("mcmc", "mcmc", C_MCMC), ("qae", QAE_KEY, C_QAE))
right = {k: [] for k, _, _ in METHODS}      # (P_F, q95) for reached points
right_none = {k: [] for k, _, _ in METHODS}  # P_F where q95 never reached within budget
check = []
for c in cases:
    tag = f"esg{c['esg_q']}_carb{c['carbon_q']}"
    pf = c["P_F"]
    for k, m, _ in METHODS:
        pts = [{"q": float(r["q"]), "e95": float(r["e95"])}
               for r in scal_rows if r["part"] == "b" and r["case"] == tag and r["method"] == m]
        q95 = interp_q(pts, TOL, "e95") if pts else None
        if q95 is None:
            right_none[k].append(pf)
        else:
            right[k].append((pf, q95))
        if m in scal["b"]["cases"][0].get("q95", {}).get("0.01", {}):
            ref = c["q95"]["0.01"].get(m)
            if ref is not None and q95 is not None:
                check.append((tag, m, round(q95, 1), round(ref, 1)))

# Example mandate: full variant at eps 0.01
ex = [r for r in mand_rows if r["variant"] == "full" and abs(float(r["eps"]) - 0.01) < 1e-12][0]
pf_ex = float(ex["a_F"])
ex_rej = float(ex["rejection_proposals"])
ex_mcmc = float(ex["mcmc_queries"])
ex_mcmc_extrap = ex["mcmc_extrapolated"] == "True"
ex_qae_matched = float(ex["quantum_matched_interp"])
ex_qae_nominal = float(ex["quantum_nominal"])
ex_rat_lo, ex_rat_hi = float(ex["ratio_rej_over_q_min"]), float(ex["ratio_rej_over_q_max"])

# ---------------------------------------------------------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=150, facecolor="white")
fig.subplots_adjust(left=0.07, right=0.975, top=0.80, bottom=0.25, wspace=0.28)
fig.suptitle("Amplitude estimation: quadratically fewer queries (noiseless simulation)",
             fontsize=20, fontweight="bold", y=0.955, color="#111111")

# Left: precision
axL.set_title("Precision (n=24, k=6 instance)", loc="left", pad=10, color="#111111", fontsize=15)
for key, col, label in (("rej", C_REJ, "Classical rejection (i.i.d.)"),
                        (QAE_KEY, C_QAE, QAE_LABEL)):
    pts = left[key]
    if not pts:
        continue
    q = [p[0] for p in pts]
    axL.fill_between(q, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.16, linewidth=0)
    axL.plot(q, [p[1] for p in pts], "-o", color=col, lw=2.2, ms=5.5, label=label)

axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlabel("Total oracle queries")
axL.set_ylabel("Percentile error |error| (median, IQR band)")
style_axes(axL)


_rp = left["rej"]
_ri = int(0.55 * (len(_rp) - 1))
axL.annotate(f"∝ 1/√N   (fit {slopes_a['classical']:+.2f})", xy=(_rp[_ri][0], _rp[_ri][1]),
             xytext=(_rp[_ri][0] * 1.35, _rp[_ri][1] * 2.2), color=C_REJ, fontsize=15,
             fontweight="bold", arrowprops=dict(arrowstyle="-", color=C_REJ, lw=1.0))
_qp = left[QAE_KEY]
_qi = int(0.6 * (len(_qp) - 1))
axL.annotate(f"∝ 1/N   (fit {slopes_a[QAE_KEY]:+.2f})", xy=(_qp[_qi][0], _qp[_qi][1]),
             xytext=(250, 2.2e-4), color=C_QAE, fontsize=15, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=C_QAE, lw=1.0))
axL.legend(loc="upper right", frameon=False)

# Right: tight mandates, q95 cost
axR.set_title("Tight mandates: queries for ±1 pp (percentile)", loc="left", pad=10, color="#111111", fontsize=15)
axR.set_xscale("log")
axR.set_yscale("log")
axR.set_xlim(0.5, 0.0006)
axR.set_ylim(8e2, 6e7)

for k, _, col in METHODS:
    pts = right[k]
    if k == "mcmc":
        continue  # MCMC q95 is a budget artefact; shown as the trapped annotation instead
    if pts:
        mk = {"rej": "o", "mcmc": "s", "qae": "D"}[k]
        axR.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=2.2,
                 marker=mk, ms=8, zorder=3)

# Did not reach within budget: hollow triangles at the top of the axes
ytop = 1.2e7
for k, _, col in METHODS:
    if k == "mcmc":
        continue
    for pf in right_none[k]:
        axR.plot([pf], [ytop], "^", ms=13, mfc="white", mec=col, mew=2.4, zorder=5, clip_on=False)

# Swap-MCMC: fragmented feasible set (annotation only; no q95 curve)
_frag = [c for c in cases if c["P_F"] < 0.003]
_e90 = [c["mcmc_final"]["pct_e90"] for c in _frag]
_p = [c["P_F"] for c in _frag]
axR.plot([max(_p), min(_p)], [1.2e7, 1.2e7], "-", color=C_MCMC, lw=1.6, zorder=4)
for x in _p:
    axR.plot([x, x], [1.2e7, 1.0e7], "-", color=C_MCMC, lw=1.6, zorder=4)
axR.text(0.99, 0.875, f"swap-MCMC trapped on fragmented feasible set\n(90% error {min(_e90):.2f}\u2013{max(_e90):.2f})",
         transform=axR.transAxes, ha="right", va="bottom", color=C_MCMC, fontsize=12,
         fontweight="bold", linespacing=1.2)

# Example mandate
axR.plot([pf_ex], [ex_rej], "*", ms=22, color=C_REJ, mec="white", mew=1.0, zorder=6)
# MCMC: hollow star = extrapolated
axR.plot([pf_ex], [ex_mcmc], "*", ms=22, mfc="white", mec=C_MCMC, mew=2.6, zorder=6)
# Quantum: vertical range bar from matched-q95 cost (lower) to nominal-target cost (upper)
axR.plot([pf_ex, pf_ex], [ex_qae_matched, ex_qae_nominal], "-", color=C_QAE, lw=3.0, zorder=5,
         solid_capstyle="butt")
axR.plot([pf_ex], [ex_qae_matched], "_", ms=20, mew=3, color=C_QAE, zorder=6)
axR.plot([pf_ex], [ex_qae_nominal], "_", ms=20, mew=3, color=C_QAE, zorder=6)
axR.plot([pf_ex], [ex_qae_matched], "*", ms=16, color=C_QAE, mec="white", mew=1.0, zorder=7)

note = ("\u2605 Example mandate n=150, P_F 0.55%\n"
        "hypothetical oracle (2 rules not encodable)\n"
        f"rejection / quantum {ex_rat_lo:.0f}\u2013{ex_rat_hi:.0f}\u00d7 (matched q95 to nominal target)")
fig.text(0.775, 0.115, note, fontsize=11, color="#111111", ha="center", va="center", linespacing=1.3)

axR.set_xlabel("Feasible fraction $P_F$ (tighter rules to the right)")
axR.set_ylabel("Queries for ±1 pp (95% of runs)")
style_axes(axR)

handles = [Line2D([0], [0], color=C_REJ, lw=2.2, marker="o", ms=8, label="Classical rejection"),
           Line2D([0], [0], color=C_QAE, lw=2.2, marker="D", ms=8, label=QAE_LABEL)]
axR.add_artist(axR.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=12))
extra = [Line2D([0], [0], color=C_MCMC, marker="*", ms=14, mfc="white", mew=2.2, lw=0,
                label="MCMC (extrapolated)"),
         Line2D([0], [0], color=C_QAE, lw=3.0, label="quantum: matched q95 to nominal target")]
axR.legend(handles=extra, loc="lower right", frameon=False, fontsize=11)

# Footer
footer = ("Noiseless simulation, synthetic data; query = one oracle call. Percentile shown; full attribution "
          "(percentile + medians) gains 2–5x over rejection and only on tight rules (vs MCMC 0.8–1.8x). "
          "Not a wall-clock speedup: no advantage on noisy hardware; fault-tolerant wall-clock slower at analyst precision.")
fig.text(0.5, 0.03, "\n".join(textwrap.wrap(footer, 165)), fontsize=11, color="#444444",
         ha="center", va="center")

fig.savefig(OUT_PNG, dpi=150, facecolor="white")
fig.savefig(OUT_SVG, facecolor="white")

# ---------------------------------------------------------------- report
print("left series:", {k: len(v) for k, v in left.items()}, "missing:", missing_left)
print("left slopes (n24_k6):", slopes_a)
print("q95@0.01 check vs json (tag, method, recomputed, json):", check)
for k, _, _ in METHODS:
    print(f"right {k}: reached", [(round(p, 5), round(q)) for p, q in right[k]],
          "| not reached at P_F", [round(p, 5) for p in right_none[k]])
print("example mandate: P_F=%.5f rej=%.0f mcmc=%.0f (extrap=%s) qae matched=%.0f nominal=%.0f ratio=%.2f-%.2f"
      % (pf_ex, ex_rej, ex_mcmc, ex_mcmc_extrap, ex_qae_matched, ex_qae_nominal, ex_rat_lo, ex_rat_hi))
print("saved", OUT_PNG, OUT_SVG)
