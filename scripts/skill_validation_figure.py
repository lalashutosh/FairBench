"""Slide figure: every quarterly percentile from results/skill_table/skill_table_quarters.csv.

Shows that FairBench flags a fund that really behaves differently (the equal-weighted control lands
in the extreme tails half the time) while the ESG funds stay inside the bulk of what their rules
allowed. Reads the CSV only; writes results/skill_table/skill_validation.{png,svg} (16:9).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "skill_table"
EXTREME = 2.5          # percentile tails counted as "flagged": <= 2.5 or >= 97.5
COL = {"active ESG": "#0072B2", "passive ESG ETF": "#E69F00", "passive non-ESG control": "#555555"}
ORDER = ["active ESG", "passive ESG ETF", "passive non-ESG control"]
SHORT = {"American Century Large Cap Eq.": "American Century", "Nuveen Large Cap Responsible": "Nuveen Responsible",
         "Xtrackers S&P 500 Scored & Screened": "Xtrackers S&P 500 Screened",
         "iShares Paris-Aligned Climate MSCI USA": "iShares Paris-Aligned",
         "iShares MSCI USA Equal Weighted": "Equal-weighted index (control)"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 20})

    q = pd.read_csv(RES / "skill_table_quarters.csv")
    rows = []
    for g in ORDER:
        for f in sorted(q.loc[q.group == g, "fund_name"].unique()):
            rows.append((g, f))
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    rng = np.random.default_rng(0)
    y, yt, yl, gap = 0.0, [], [], 0.0
    prev = None
    for g, f in rows:
        if prev is not None and g != prev:
            y += 0.7
        prev = g
        v = q.loc[q.fund_name == f, "pct_vw"].to_numpy()
        ext = int(((v <= EXTREME) | (v >= 100 - EXTREME)).sum())
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=110, color=COL[g], alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(103, y, f"{ext}/{len(v)}", va="center", fontsize=20,
                color="#b2182b" if ext else "#666666", fontweight="bold" if ext else "normal")
        yt.append(y)
        yl.append(SHORT.get(f, f))
        y += 1
    ax.axvspan(0, EXTREME, color="#fbb4ae", alpha=0.6, zorder=1)
    ax.axvspan(100 - EXTREME, 100, color="#fbb4ae", alpha=0.6, zorder=1)
    ax.axvline(50, color="black", lw=1.5, zorder=2)
    ax.set_yticks(yt, yl)
    ax.invert_yaxis()
    ax.set_xlim(-2, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Quarterly percentile among random portfolios from the fund's own universe")
    ax.text(103, -0.9, "flagged\nquarters", ha="left", va="bottom", fontsize=17, color="#b2182b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[g], markersize=13, label=g) for g in ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#fbb4ae", label=f"extreme tails (≤{EXTREME:g} / ≥{100 - EXTREME:g})"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, -0.24), ncol=4, fontsize=17, frameon=False)
    fig.text(0.01, 0.975, "When a fund really differs, FairBench flags it.", fontsize=30, fontweight="bold",
             va="top")
    fig.text(0.01, 0.915, "The ESG funds stayed within what their rules allowed: no extreme quarters.", fontsize=22,
             va="top", color="#333333")
    fig.text(0.01, 0.015, "Real SEC N-PORT holdings 2019-2026, frozen for one quarter; value-weighted price returns; "
             "20,000 random portfolios per quarter. Descriptive, not causal.", fontsize=14, color="#555555")
    fig.subplots_adjust(left=0.24, right=0.93, top=0.8, bottom=0.23)
    for ext in ("png", "svg"):
        fig.savefig(RES / f"skill_validation.{ext}", dpi=100, facecolor="white")
    print(f"saved {RES / 'skill_validation.png'}")


if __name__ == "__main__":
    sys.exit(main())
