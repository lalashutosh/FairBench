"""Slide-ready pitch figure for the QAE results. Reads existing result files only.

Left panel: percentile |error| vs total queries (n24_k6, median + IQR), classical vs quantum (MLAE-exp).
Right panel: queries to +-1 pp (percentile-only, tol 0.01) vs feasible fraction P_F, with the example mandate.
"""
import csv
import json
import math
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

C_REJ = "#4d4d4d"  # dark grey: classical rejection
C_MCMC = "#d95f02"  # orange: swap-MCMC
C_QAE = "#1f78b4"  # blue: quantum AE

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


# ---------------------------------------------------------------- data
scal = load_json("qae_scaling.json")
scal_rows = load_csv("qae_scaling.csv")
mand_rows = load_csv("qae_mandate.csv")
mand_json = load_json("qae_mandate.json")
present = {"qae_scaling.json": True, "qae_scaling.csv": True,
           "qae_mandate.json": True, "qae_mandate.csv": True,
           "qae_noise.json": os.path.exists(os.path.join(RES, "qae_noise.json")),
           "qae_breakeven.json": os.path.exists(os.path.join(RES, "qae_breakeven.json"))}

# Left panel series: instance n24_k6, part 'a'
def series(case, method):
    pts = []
    for r in scal_rows:
        if r["part"] == "a" and r["case"] == case and r["method"] == method:
            pts.append((float(r["q"]), float(r["e50"]), float(r["e25"]), float(r["e75"])))
    pts.sort()
    return pts

left = {
    "rej": series("n24_k6", "classical"),
    "qae": series("n24_k6", "mlae_exp"),
}
slopes_a = scal["a"]["n24_k6"]["slopes"]
missing_left = [k for k, v in left.items() if not v]

# Right panel series: tol 0.01, percentile-only, q50 per case
cases = scal["b"]["cases"]
cases = sorted(cases, key=lambda c: -c["P_F"])
right = {"rej": [], "mcmc": [], "qae": []}
mcmc_trapped = []  # (P_F, q or None)
for c in cases:
    pf = c["P_F"]
    q50 = c["q50"]["0.01"]
    if q50.get("classical") is not None:
        right["rej"].append((pf, q50["classical"]))
    if q50.get("mlae_exp") is not None:
        right["qae"].append((pf, q50["mlae_exp"]))
    m = q50.get("mcmc")
    e90 = c.get("mcmc_final", {}).get("pct_e90")
    trapped = (m is None) or (e90 is not None and e90 > 0.05)
    if m is not None:
        right["mcmc"].append((pf, m))
    if trapped:
        mcmc_trapped.append((pf, m))
missing_right = [k for k, v in right.items() if not v]

# Example mandate: 'full' variant at eps 0.01
ex = [r for r in mand_rows if r["variant"] == "full" and abs(float(r["eps"]) - 0.01) < 1e-12][0]
pf_ex = float(ex["a_F"])
ex_rej = float(ex["rejection_proposals"])
ex_mcmc = float(ex["mcmc_queries"])
ex_qae = float(ex["quantum_matched_interp"])
ex_qae_nom = float(ex["quantum_nominal"])
ex_mcmc_extrap = ex["mcmc_extrapolated"] == "True"
ex_rat_lo, ex_rat_hi = float(ex["ratio_rej_over_q_min"]), float(ex["ratio_rej_over_q_max"])

# ---------------------------------------------------------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=150, facecolor="white")
fig.subplots_adjust(left=0.07, right=0.975, top=0.80, bottom=0.22, wspace=0.28)
fig.suptitle("Amplitude estimation: quadratically fewer queries (noiseless simulation)",
             fontsize=20, fontweight="bold", y=0.955, color="#111111")

# Left: precision
axL.set_title("Precision (n=24, k=6 instance)", loc="left", pad=10, color="#111111")
for key, col, label in (("rej", C_REJ, "Classical rejection (i.i.d.)"),
                        ("qae", C_QAE, "Quantum AE (MLAE)")):
    pts = left[key]
    if not pts:
        continue
    q = [p[0] for p in pts]
    med = [p[1] for p in pts]
    lo = [p[2] for p in pts]
    hi = [p[3] for p in pts]
    axL.fill_between(q, lo, hi, color=col, alpha=0.16, linewidth=0)
    axL.plot(q, med, "-o", color=col, lw=2.2, ms=5.5, label=label)

axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlabel("Total oracle queries")
axL.set_ylabel("Percentile error |error| (median, IQR band)")
style_axes(axL)

# Slope annotations placed on the fitted curves
def place_slope(ax, pts, col, text, frac, dy):
    if not pts:
        return
    i = int(frac * (len(pts) - 1))
    qx, qy = pts[i][0], pts[i][1]
    ax.annotate(text, xy=(qx, qy), xytext=(qx * 1.35, qy * dy), color=col,
                fontsize=15, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=1.0))

place_slope(axL, left["rej"], C_REJ,
            f"∝ 1/√N   (fit {slopes_a['classical']:+.2f})", 0.55, 2.2)
_qp = left["qae"]
_qi = int(0.6 * (len(_qp) - 1))
axL.annotate(f"\u221d 1/N   (fit {slopes_a['mlae_exp']:+.2f})", xy=(_qp[_qi][0], _qp[_qi][1]),
             xytext=(250, 2.2e-4), color=C_QAE, fontsize=15, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=C_QAE, lw=1.0))

axL.legend(loc="upper right", frameon=False)

# Right: tight mandates
axR.set_title("Tight mandates: queries for ±1 pp", loc="left", pad=10,
              color="#111111")
axR.set_xscale("log")
axR.set_yscale("log")
axR.set_xlim(0.5, 0.0006)
axR.set_ylim(8e2, 3e6)

def plot_line(pts, col, marker, label, ls="-"):
    if not pts:
        return
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    axR.plot(x, y, ls, color=col, lw=2.2, marker=marker, ms=8, label=label)

plot_line(right["rej"], C_REJ, "o", "Classical rejection")
plot_line(right["mcmc"], C_MCMC, "s", "Swap-MCMC")
plot_line(right["qae"], C_QAE, "D", "Quantum AE (MLAE)")

# Trapped MCMC: hollow markers (no label on the plot; explained in the legend)
yhi = axR.get_ylim()[1]
for pf, m in mcmc_trapped:
    if m is not None:
        axR.plot([pf], [m], "o", ms=13, mfc="white", mec=C_MCMC, mew=2.4, zorder=5)
    else:
        # not reached within the budget: hollow triangle pinned near the top of the axes
        axR.plot([pf], [yhi * 0.5], "^", ms=13, mfc="white", mec=C_MCMC, mew=2.4, zorder=5)

# Example mandate: three stars, one short leader to a note in empty space
axR.plot([pf_ex], [ex_rej], "*", ms=22, color=C_REJ, mec="white", mew=1.0, zorder=6)
axR.plot([pf_ex], [ex_mcmc], "*", ms=22, color=C_MCMC, mec="white", mew=1.0, zorder=6)
axR.plot([pf_ex], [ex_qae], "*", ms=22, color=C_QAE, mec="white", mew=1.0, zorder=6)
axR.annotate(f"Example mandate (P_F 0.55%)\nrejection / quantum {ex_rat_lo:.0f}\u2013{ex_rat_hi:.0f}\u00d7",
             xy=(pf_ex, ex_rej), xytext=(0.52, 0.60), textcoords="axes fraction",
             ha="right", va="center", color="#111111", fontsize=12,
             arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8, shrinkA=2, shrinkB=12))

axR.set_xlabel("Feasible fraction $P_F$ (tighter rules to the right)")
axR.set_ylabel("Queries needed for ±1 pp (log)")
style_axes(axR)

handles = [Line2D([0], [0], color=C_REJ, lw=2.2, marker="o", ms=8, label="Classical rejection"),
           Line2D([0], [0], color=C_MCMC, lw=2.2, marker="s", ms=8, label="Swap-MCMC"),
           Line2D([0], [0], color=C_QAE, lw=2.2, marker="D", ms=8, label="Quantum AE (MLAE)")]
axR.add_artist(axR.legend(handles=handles, loc="upper left", frameon=False, fontsize=12))
hollow = [Line2D([0], [0], color=C_MCMC, marker="o", ms=10, mfc="white", mew=2.2, lw=0,
                 label="MCMC trapped on fragmented feasible set"),
          Line2D([0], [0], color=C_MCMC, marker="^", ms=10, mfc="white", mew=2.2, lw=0,
                 label="MCMC did not reach \u00b11 pp in budget")]
axR.add_artist(axR.legend(handles=hollow, loc="lower right", frameon=False, fontsize=11))

# Footer
footer = ("Noiseless simulation, synthetic data; query = one oracle call. Percentile shown; for the full attribution "
          "quantum wins only on tight rules (P_F \u2272 0.7%). Not a wall-clock speedup: no advantage on noisy hardware; "
          "fault-tolerant wall-clock slower at analyst precision.")
fig.text(0.5, 0.03, "\n".join(textwrap.wrap(footer, 150)), fontsize=11, color="#444444", ha="center", va="center")

fig.savefig(OUT_PNG, dpi=150, facecolor="white")
fig.savefig(OUT_SVG, facecolor="white")

# ---------------------------------------------------------------- report
print("present:", present)
print("left series:", {k: len(v) for k, v in left.items()}, "missing:", missing_left)
print("left slopes (n24_k6):", slopes_a)
print("right series:", {k: len(v) for k, v in right.items()}, "missing:", missing_right)
print("right points rej:", right["rej"])
print("right points qae:", right["qae"])
print("right points mcmc:", right["mcmc"])
print("mcmc trapped:", mcmc_trapped)
print("example mandate: P_F=%.5f rej=%.0f mcmc=%.0f (extrap=%s) qae_matched=%.0f qae_nominal=%.0f ratio=%.2f-%.2f"
      % (pf_ex, ex_rej, ex_mcmc, ex_mcmc_extrap, ex_qae, ex_qae_nom, ex_rat_lo, ex_rat_hi))
print("saved", OUT_PNG, OUT_SVG)
