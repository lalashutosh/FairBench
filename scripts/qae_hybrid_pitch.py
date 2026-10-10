"""Slide-ready figure for the hybrid classical + quantum attribution estimator. Reads results/qae_hybrid.json only.

Queries for the WHOLE attribution (percentile + null median + benchmark median, tol 0.01, typical run = e50) vs
the feasible fraction P_F: classical rejection, swap-MCMC, hybrid (classical warm start + amplitude estimation).
Cases where one rank step 1/|F| exceeds the tolerance (floor_1F) are left out, as in QUANTUM_CORE.md.
Outputs results/qae_hybrid_pitch.{png,svg} at 16:9.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")

C_HYB = "#2a78d6"    # blue: hybrid classical + quantum
C_REJ = "#eb6834"    # orange: classical rejection
C_MCMC = "#1baf7a"   # aqua: swap-MCMC
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#e4e4e4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.labelsize": 24,
                     "xtick.labelsize": 20, "ytick.labelsize": 20})


def main():
    with open(os.path.join(RES, "qae_hybrid.json")) as f:
        d = json.load(f)
    cases = [c for c in d["cases"] if not c["floor_1F"]]
    pf = [100 * c["P_F"] for c in cases]
    rej = [c["e50"]["whole"]["rejection"] for c in cases]
    hyb = [c["e50"]["whole"]["quantum_hybrid"] for c in cases]
    mc = [(100 * c["P_F"], c["e50"]["whole"]["mcmc"]) for c in cases if c["e50"]["whole"]["mcmc"]]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.plot(pf, rej, color=C_REJ, lw=3, marker="o", ms=11, mec="white", mew=2)
    ax.plot(*zip(*mc), color=C_MCMC, lw=3, marker="s", ms=10, mec="white", mew=2, ls=(0, (5, 2)))
    ax.plot(pf, hyb, color=C_HYB, lw=4, marker="o", ms=13, mec="white", mew=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # tighter rules to the right
    ax.xaxis.set_major_locator(FixedLocator([36, 10, 3, 1, 0.3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Share of portfolios that pass the rules  (tighter rules →)", color=INK, labelpad=12)
    ax.set_ylabel("Queries needed", color=INK, labelpad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, which="major", color=GRID, lw=1)
    ax.set_axisbelow(True)

    # direct labels at the right end (identity never by color alone)
    ax.annotate("Classical sampling", (pf[-1], rej[-1]), xytext=(14, 0), textcoords="offset points",
                va="center", color=INK, fontsize=22)
    ax.annotate("Classical MCMC", mc[-1], xytext=(14, 0), textcoords="offset points",
                va="center", color=INK, fontsize=22)
    ax.annotate("Hybrid:\nclassical + quantum", (pf[-1], hyb[-1]), xytext=(14, 0), textcoords="offset points",
                va="center", color=INK, fontsize=22, fontweight="bold")

    # ratio callouts: where classical wins, and the tightest case
    r0, r1 = rej[0] / hyb[0], rej[-1] / hyb[-1]
    # double-headed arrow spanning the gap at the tightest case, with the callout attached to it
    xg = pf[-1] * 1.18                      # just left of the last points (x axis runs loose -> tight)
    ax.annotate("", xy=(xg, hyb[-1] * 1.12), xytext=(xg, rej[-1] / 1.12),
                arrowprops=dict(arrowstyle="<->", color=C_HYB, lw=3, shrinkA=0, shrinkB=0))
    ax.annotate(f"{r1:.0f}× fewer queries\nthan classical sampling", (xg, hyb[-1] / 4.5),
                xytext=(0, 0), textcoords="offset points", ha="right", va="center", color=C_HYB,
                fontsize=28, fontweight="bold")
    ax.annotate(f"loose rules:\nclassical wins ({1 / r0:.1f}×)", (pf[0], rej[0]), xytext=(0, -30),
                textcoords="offset points", ha="left", va="top", color=MUTED, fontsize=19)

    ax.set_xlim(55, pf[-1] / 1.15)
    ax.set_ylim(1.5e3, 1.5e6)
    fig.text(0.015, 0.02, "Whole attribution (fund rank + two medians) to ±1 point, typical run. Noiseless simulation, "
             "query counts, not wall-clock time.", color=MUTED, fontsize=16)
    fig.subplots_adjust(left=0.13, right=0.74, top=0.95, bottom=0.17)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RES, f"qae_hybrid_pitch.{ext}"), facecolor="white")
    print(f"ratios rejection/hybrid: {[round(r / h, 2) for r, h in zip(rej, hyb)]}")


if __name__ == "__main__":
    main()
