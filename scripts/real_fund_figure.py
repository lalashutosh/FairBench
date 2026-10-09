"""One slide-sized figure from a real-fund run: growth against the index and the typical
rule-abiding portfolio, and the fund's percentile quarter by quarter.

    .venv/bin/python scripts/real_fund_figure.py --name parnassus_core_equity

Reads results/real_fund_<name>.json, _hold.csv and _summary.json (written by
scripts/real_fund_attribution.py) and writes results/real_fund_<name>.png. Every number
drawn comes from those files.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, validated as a set


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True), ap.add_argument("--short-name", default=None)
    args = ap.parse_args()
    res = ROOT / "results"
    run = json.loads((res / f"real_fund_{args.name}.json").read_text())
    summary = json.loads((res / f"real_fund_{args.name}_summary.json").read_text())
    hold = pd.read_csv(res / f"real_fund_{args.name}_hold.csv")
    ruled = "k names obeying the exclusions" if (hold["reference"] != "any k names").any() else "any k names"
    h = hold[(hold["reference"] == ruled) & (hold["weights"] == "benchmark_capped")].reset_index(drop=True)
    free = hold[(hold["reference"] == "any k names") & (hold["weights"] == "benchmark_capped")].reset_index(drop=True)
    periods = run["periods"]
    fund_name = args.short_name or summary["fund"]
    ends = [date.fromisoformat(p["period_end"]) for p in periods]
    days = [date.fromisoformat(periods[0]["period_start"])] + ends

    def grow(r):
        return np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(r, dtype=float))])

    g_fund = grow([p["realised"]["frozen_return"] for p in periods])
    g_index = grow([p["realised"]["benchmark_proxy_return"] for p in periods])
    g_typ = grow(h["median_1q"])
    cost = (np.prod(1 + h["median_1q"]) - np.prod(1 + free["median_1q"])) * 100
    pct = h["percentile_1q"].to_numpy()
    row = next(r for r in summary["by_reference"] if r["reference"] == ruled and r["weights"] == "benchmark_capped")
    k = int(round(h["k"].median()))

    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 6.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.24})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0, labelsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # left: one axis, everything indexed to 1 at the first date
    series = [(g_index, ORANGE, "S&P 500 index fund"), (g_fund, BLUE, fund_name),
              (g_typ, AQUA, f"typical {k}-stock portfolio\nallowed by the same rules")]
    for g, colour, label in series:
        ax1.plot(days, g, color=colour, lw=2, solid_capstyle="round")
        ax1.scatter([days[-1]], [g[-1]], s=42, color=colour, zorder=3, edgecolors=SURFACE, linewidths=2, clip_on=False)
    ends_y = np.array([g[-1] for g, _, _ in series])
    order = np.argsort(-ends_y)
    label_y = ends_y.copy()
    gap = 0.085 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    for a, b in zip(order[:-1], order[1:]):                         # keep the end labels from colliding
        label_y[b] = min(label_y[b], label_y[a] - gap)
    for i, (g, colour, label) in enumerate(series):
        ax1.annotate(f"{label}\n{100 * (g[-1] - 1):+.0f}%", xy=(days[-1], g[-1]), xytext=(12, 0),
                     textcoords=("offset points", "data"), xycoords="data", va="center", ha="left", fontsize=10.5,
                     color=INK, annotation_clip=False,
                     bbox=dict(boxstyle="square,pad=0", fc="none", ec="none")).set_position((12, label_y[i]))
        ax1.plot([], [], color=colour, lw=2, label=label.replace("\n", " "))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * (v - 1):+.0f}%"))
    ax1.set_xlim(days[0], days[-1])
    titles = [(0.055, f"Price growth, {days[0]:%b %Y} to {days[-1]:%b %Y}",
               f"Excluding the screened companies changed the typical portfolio's result by {cost:+.0f} points")]

    # right: one series, position against the 50 line carries the meaning
    ax2.axhline(50, color=AXIS, lw=1.2)
    ax2.vlines(ends, 50, pct, color=BLUE, lw=1.6, alpha=0.55)
    ax2.scatter(ends, pct, s=46, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2)
    for i in (int(np.argmax(pct)), int(np.argmin(pct))):
        ax2.annotate(f"{pct[i]:.0f}", (ends[i], pct[i]), xytext=(0, 9 if pct[i] > 50 else -9), textcoords="offset points",
                     ha="center", va="bottom" if pct[i] > 50 else "top", fontsize=10, color=INK)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0", "25", "typical  50", "75", "100"])
    ax2.set_xlim(days[0], date(ends[-1].year, 12, 31))
    titles.append((0.57, f"Where the fund's portfolio ranked each quarter: average {row['mean_percentile_1q']:.0f}",
                   f"Percentile among {k}-stock portfolios allowed by the same rules.\n"
                   f"Random picks would average 50, give or take {row['se_if_random_1q']:.0f}."))
    for x, title, sub in titles:
        fig.text(x, 0.955, title, fontsize=14, color=INK, ha="left", va="top")
        fig.text(x, 0.908, sub, fontsize=10.5, color=INK2, ha="left", va="top", linespacing=1.35)

    n_excl = summary["n_excluded_keys"]
    note = (f"Source: SEC Form N-PORT filings of the fund and of {summary['universe_proxy']}. Portfolios are held unchanged "
            f"for one quarter. Price returns derived from the filings, dividends excluded. Rules applied: number of holdings"
            + (f" and {n_excl} companies excluded by SEC industry code (a stand-in for the fund's own screen)" if n_excl else "")
            + ". Descriptive comparison; a percentile is not evidence of skill.")
    import textwrap
    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(note, 190)), fontsize=8.5, color=INK2, ha="left", va="bottom")
    ax1.set_position([0.055, 0.14, 0.33, 0.68])                      # room on the right for the end labels
    ax2.set_position([0.57, 0.14, 0.415, 0.68])
    out = res / f"real_fund_{args.name}.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
