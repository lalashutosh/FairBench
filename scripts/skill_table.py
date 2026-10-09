"""Rules-adjusted skill table for real ESG funds from public SEC N-PORT CSV exports.

Per fund and calendar quarter: freeze the fund's latest snapshot, match it to the universe
parent's holdings, and rank its next-quarter price return against N random k-subsets of the
universe (value-weighted by parent cap weights, and equal-weighted).  Returns come ONLY from the
universe parent's implied prices, never from a fund's own implied prices.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "results" / "real_data"
OUT = ROOT / "results" / "skill_table"
SP, RU = "S000004310", "S000004347"
N_NULL = 20000
MAX_LAG, MIN_COV = 92, 0.7
SD_UNIF = 100 / np.sqrt(12)  # 28.87

FUNDS = [  # id, short name, group, universe
    ("S000000856", "Parnassus Core Equity", "active ESG", SP),
    ("S000008719", "Calvert Equity", "active ESG", RU),
    ("S000005371", "Nuveen Large Cap Responsible", "active ESG", RU),
    ("S000006203", "American Century Large Cap Eq.", "active ESG", RU),
    ("S000069051", "SPDR S&P 500 ESG ETF", "passive ESG ETF", SP),
    ("S000065220", "Xtrackers S&P 500 Scored & Screened", "passive ESG ETF", SP),
    ("S000055381", "iShares ESG Aware MSCI USA", "passive ESG ETF", RU),
    ("S000074757", "iShares Paris-Aligned Climate MSCI USA", "passive ESG ETF", RU),
    ("S000028709", "iShares MSCI USA Equal Weighted", "passive non-ESG control", RU),
]
UNI_NAME = {SP: "S&P 500 proxy", RU: "Russell 1000 proxy"}


def percentile_of(null: np.ndarray, x: float, tol: float = 1e-12) -> float:
    """100 x (share of null strictly below x + half of ties); ties within tol."""
    null = np.asarray(null, float)
    below = np.mean(null < x - tol)
    tie = np.mean(np.abs(null - x) <= tol)
    return float(100 * (below + 0.5 * tie))


SPLIT_FACTORS = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 50], float)


def split_adjust(pr: np.ndarray, sr: np.ndarray):
    """Implied price = market value / shares is NOT split adjusted: a 10:1 split shows up as a -90% 'return'.
    Flag a forward split of factor f when the price ratio is near 1/f (pr*f in [0.75, 1.30]) AND the share count
    held by the parent rose by roughly f (sr >= 0.45 f + 0.5; an ETF's stock share count is noisy because index
    weights move, hence the loose test).  Returns (adjusted price ratio, flag)."""
    pr = np.asarray(pr, float); sr = np.asarray(sr, float)
    f = SPLIT_FACTORS[None, :]
    hit = (pr[:, None] * f >= 0.75) & (pr[:, None] * f <= 1.30) & (sr[:, None] >= 0.45 * f + 0.5)
    # choose the largest factor that fits
    idx = np.where(hit.any(1), hit.shape[1] - 1 - hit[:, ::-1].argmax(1), 0)
    ok = hit.any(1)
    return np.where(ok, pr * SPLIT_FACTORS[idx], pr), ok


def build_price_returns(prices: pd.DataFrame, parent_id: str, t: str, t1: str, adjust: bool = True) -> pd.Series:
    """Quarterly price return (p1/p0 - 1) per security_key for names priced at both dates,
    using ONLY the given parent's own implied prices."""
    p = prices[prices.fund_id == parent_id]
    a = p[p.report_date == t].drop_duplicates("security_key").set_index("security_key")
    b = p[p.report_date == t1].drop_duplicates("security_key").set_index("security_key")
    j = pd.concat([a.price_usd.rename("p0"), b.price_usd.rename("p1"), a.shares.rename("s0"), b.shares.rename("s1")],
                  axis=1, join="inner").dropna(subset=["p0", "p1"])
    j = j[(j.p0 > 0) & (j.p1 > 0)]
    ratio = (j.p1 / j.p0).to_numpy()
    if adjust:
        sr = (j.s1 / j.s0)
        # corroborate with the other index parent's share-count ratio (ETF stock shares are noisy): use the larger
        o = prices[(prices.fund_id != parent_id) & prices.report_date.isin([t, t1])]
        o0 = o[o.report_date == t].drop_duplicates("security_key").set_index("security_key").shares
        o1 = o[o.report_date == t1].drop_duplicates("security_key").set_index("security_key").shares
        osr = (o1 / o0).reindex(j.index)
        sr = np.fmax(sr.to_numpy(), osr.to_numpy())
        sr = np.where(np.isfinite(sr) & (sr > 0), sr, 1.0)
        ratio, _ = split_adjust(ratio, sr)
    return pd.Series(ratio - 1, index=j.index, name="ret")


def null_subset_returns(r: np.ndarray, w: np.ndarray, k: int, n: int, rng, batch: int = 1000):
    """Returns (value-weighted, equal-weighted) returns of n random k-subsets of the universe.
    Value-weighted subsets use weights w renormalised within the subset."""
    m = len(r)
    vw = np.empty(n); ew = np.empty(n)
    for s in range(0, n, batch):
        b = min(batch, n - s)
        if k >= m:
            idx = np.tile(np.arange(m), (b, 1))
        else:
            idx = np.argpartition(rng.random((b, m)), k - 1, axis=1)[:, :k]
        rr = r[idx]; ww = w[idx]
        vw[s:s + b] = (rr * ww).sum(1) / ww.sum(1)
        ew[s:s + b] = rr.mean(1)
    return vw, ew


def main(n_null: int = N_NULL, seed: int = 20260101):
    t0 = time.time()
    hold = pd.read_csv(RD / "real_quarterly_holdings.csv",
                       usecols=["fund_id", "report_date", "security_key", "market_value_usd"])
    hold = hold.dropna(subset=["security_key", "market_value_usd"])
    hold = hold.groupby(["fund_id", "report_date", "security_key"], as_index=False).market_value_usd.sum()
    prices = pd.read_csv(RD / "real_quarterly_implied_prices.csv",
                         usecols=["fund_id", "report_date", "security_key", "price_usd", "shares"])
    prices = prices[prices.fund_id.isin([SP, RU])]
    spret = pd.read_csv(RD / "real_snp500_proxy_quarterly_returns.csv")
    spret = spret.dropna(subset=["price_return"])

    quarters = sorted(prices[prices.fund_id == SP].report_date.unique())
    pairs = list(zip(quarters[:-1], quarters[1:]))
    # universe returns
    ret = {}; nsplit_log = []
    for t, t1 in pairs:
        s = spret[(spret.from_date == t) & (spret.to_date == t1)].drop_duplicates("security_key").set_index("security_key").price_return
        # CSV returns are raw price ratios (not split adjusted): fix them with the parent's share counts
        raw = build_price_returns(prices, SP, t, t1, adjust=False)
        adj = build_price_returns(prices, SP, t, t1, adjust=True)
        s = s.copy()
        common = s.index.intersection(adj.index)
        s.loc[common] = np.where(np.abs(adj.loc[common] - raw.loc[common]) > 1e-9, adj.loc[common], s.loc[common])
        nsplit_log.append((t, int((np.abs(adj - raw) > 1e-9).sum())))
        ret[(SP, t)] = s
        ret[(RU, t)] = build_price_returns(prices, RU, t, t1)
    # sanity: Russell builder vs S&P proxy on overlapping names
    chk = []
    for t, t1 in pairs:
        a, b = ret[(SP, t)], ret[(RU, t)]
        j = pd.concat([a.rename("sp"), b.rename("ru")], axis=1, join="inner")
        chk.append(dict(from_date=t, n_overlap=len(j), max_abs_diff=float((j.sp - j.ru).abs().max()),
                        share_diff_gt_1pct=float(((j.sp - j.ru).abs() > 0.01).mean())))
    chk = pd.DataFrame(chk)
    print("Russell vs S&P overlap check: median share |diff|>1pct = %.4f, worst max = %.4f"
          % (chk.share_diff_gt_1pct.median(), chk.max_abs_diff.max()))

    print("split-adjusted S&P names per quarter:", [(a, n) for a, n in nsplit_log if n])
    names = dict((f[0], f[1]) for f in FUNDS)
    rows = []
    for fid, name, group, uni in FUNDS:
        fh = hold[hold.fund_id == fid]
        dates = np.array(sorted(fh.report_date.unique()))
        rng = np.random.default_rng(seed + int(fid[-5:]))
        for t, t1 in pairs:
            avail = dates[dates <= t]
            if len(avail) == 0:
                continue
            snap = str(avail[-1])
            lag = (pd.Timestamp(t) - pd.Timestamp(snap)).days
            if lag > MAX_LAG:
                continue
            par = hold[(hold.fund_id == uni) & (hold.report_date == t)].set_index("security_key").market_value_usd
            r_all = ret[(uni, t)]
            U = par.index.intersection(r_all.index)
            U = U[par.loc[U] > 0]
            r = r_all.loc[U].to_numpy(); w = par.loc[U].to_numpy()
            f = fh[fh.report_date == snap].set_index("security_key").market_value_usd
            f = f[f > 0]
            cov = float(f.loc[f.index.intersection(U)].sum() / f.sum())
            if cov < MIN_COV:
                continue
            fm = f.loc[f.index.intersection(U)]
            k = len(fm)
            fw = fm.to_numpy() / fm.sum()
            rf = r_all.loc[fm.index].to_numpy()
            fund_vw = float((fw * rf).sum()); fund_ew = float(rf.mean())
            nvw, new = null_subset_returns(r, w, k, n_null, rng)
            bench = float((r * w).sum() / w.sum())
            rows.append(dict(
                fund_id=fid, fund_name=name, group=group, universe=UNI_NAME[uni], quarter_start=t, quarter_end=t1,
                report_date_used=snap, lag_days=lag, k=k, universe_size=len(U), coverage=cov,
                fund_return_vw=fund_vw, fund_return_ew=fund_ew,
                null_median_vw=float(np.median(nvw)), null_median_ew=float(np.median(new)),
                pct_vw=percentile_of(nvw, fund_vw), pct_ew=percentile_of(new, fund_ew),
                benchmark_return=bench,
                within_universe_diff_vw=fund_vw - float(np.median(nvw)),
                within_universe_diff_ew=fund_ew - float(np.median(new)),
                selection_shift_vw=float(np.median(nvw)) - bench,
                selection_shift_ew=float(np.median(new)) - bench))
        print(f"{name}: {sum(1 for x in rows if x['fund_id']==fid)} quarters  ({time.time()-t0:.0f}s)", flush=True)
    q = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    q.to_csv(OUT / "skill_table_quarters.csv", index=False)

    summ = []
    for fid, name, group, uni in FUNDS:
        g = q[q.fund_id == fid]
        n = len(g)
        if n == 0:
            continue
        se = SD_UNIF / np.sqrt(n)
        d = dict(fund_id=fid, fund=name, group=group, universe=UNI_NAME[uni], n_quarters=n,
                 first_quarter=g.quarter_start.min(), last_quarter=g.quarter_end.max(),
                 mean_pct_vw=g.pct_vw.mean(), mean_pct_ew=g.pct_ew.mean(),
                 z_vw=(g.pct_vw.mean() - 50) / se, z_ew=(g.pct_ew.mean() - 50) / se,
                 luck_band_lo=50 - 1.96 * se, luck_band_hi=50 + 1.96 * se,
                 share_above_50_vw=float((g.pct_vw > 50).mean()), share_above_50_ew=float((g.pct_ew > 50).mean()),
                 mean_within_diff_vw_pp=100 * g.within_universe_diff_vw.mean(),
                 mean_within_diff_ew_pp=100 * g.within_universe_diff_ew.mean(),
                 mean_selection_shift_vw_pp=100 * g.selection_shift_vw.mean(),
                 mean_coverage=g.coverage.mean(), min_coverage=g.coverage.min(),
                 mean_lag_days=g.lag_days.mean(), max_lag_days=int(g.lag_days.max()), mean_k=g.k.mean())
        summ.append(d)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "skill_table_summary.csv", index=False)
    gs = []
    for grp, g in q.groupby("group"):
        n = len(g)
        gs.append(dict(group=grp, n_funds=g.fund_id.nunique(), n_fund_quarters=n,
                       mean_pct_vw=g.pct_vw.mean(), mean_pct_ew=g.pct_ew.mean(),
                       mean_of_fund_means_vw=float(s[s.group == grp].mean_pct_vw.mean()),
                       share_above_50_vw=float((g.pct_vw > 50).mean()),
                       mean_within_diff_vw_pp=100 * g.within_universe_diff_vw.mean()))
    gdf = pd.DataFrame(gs)
    gdf.to_csv(OUT / "skill_table_groups.csv", index=False)

    out = dict(
        method=dict(
            summary=("Each fund-quarter: freeze the fund's latest snapshot <= quarter start (lag <= 92 d), match to the "
                     "universe parent's holdings at quarter start, coverage >= 0.7, k = matched names. Fund return = "
                     "frozen weights (renormalised over matched names) x universe price returns. Null = %d random "
                     "k-subsets of the universe, value-weighted (parent cap weights renormalised in subset) and "
                     "equal-weighted. Percentile = 100 x (share strictly below + half of ties)." % n_null),
            universes={"S&P 500 proxy": "iShares Core S&P 500 ETF S000004310 (returns from real_snp500_proxy_quarterly_returns.csv)",
                       "Russell 1000 proxy": "iShares Russell 1000 ETF S000004347 (returns built from its own implied prices)"},
            returns_source="universe parent's prices only; a fund's own implied prices are never used",
            luck_band="50 +- 1.96 x 28.87/sqrt(N): quarterly percentiles ~Uniform(0,100) under no skill; ignores autocorrelation",
            n_null=n_null, seed=seed,
            russell_vs_sp_check=dict(median_share_diff_gt_1pct=float(chk.share_diff_gt_1pct.median()),
                                     worst_max_abs_diff=float(chk.max_abs_diff.max()))),
        summary=json.loads(s.to_json(orient="records")),
        groups=json.loads(gdf.to_json(orient="records")),
        caveats=["Descriptive, not causal.",
                 "Frozen holdings ignore trading within the quarter, fees and dividends.",
                 "Price returns only (no dividends).",
                 "Universe proxies come from index funds' own N-PORT filings, not the official indices.",
                 "Off-cycle snapshots (fiscal year-ends other than calendar quarters) lag up to ~2 months.",
                 "A percentile is not proof of skill; the luck band ignores autocorrelation and multiple funds.",
                 "Parnassus' own implied prices are unreliable and are not used."])
    (OUT / "skill_table.json").write_text(json.dumps(out, indent=1))
    plot(s)
    print(s[["fund", "group", "n_quarters", "mean_pct_vw", "mean_pct_ew", "z_vw", "luck_band_lo", "luck_band_hi",
             "share_above_50_vw", "mean_coverage", "mean_lag_days"]].round(2).to_string())
    print(gdf.round(2).to_string())
    print("runtime %.0fs" % (time.time() - t0))


def plot(s: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    col = {"active ESG": "#0072B2", "passive ESG ETF": "#E69F00", "passive non-ESG control": "#999999"}
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    s = s.reset_index(drop=True)
    ys = []; y = 0; last = None
    for _, r in s.iterrows():
        if last is not None and r.group != last:
            y += 0.6
        ys.append(y); y += 1; last = r.group
    ys = np.array(ys)
    for (_, r), yy in zip(s.iterrows(), ys):
        ax.barh(yy + 0, r.luck_band_hi - r.luck_band_lo, left=r.luck_band_lo, height=0.9, color="#D9D9D9", zorder=1)
        ax.barh(yy, r.mean_pct_vw - 50, left=50, height=0.5, color=col[r.group], zorder=2)
        right = r.mean_pct_vw >= 50
        ax.text(max(r.mean_pct_vw, r.luck_band_hi) + 1.0, yy, f"{r.mean_pct_vw:.1f}   (N={r.n_quarters})",
                va="center", ha="left", fontsize=20, zorder=3)
    ax.axvline(50, color="black", lw=2, zorder=4)
    ax.set_yticks(ys); ax.set_yticklabels(s.fund, fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(25, 105)
    ax.set_xlabel("Mean quarterly percentile (50 = typical random portfolio)", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_title("Real ESG funds vs random portfolios from their own universe\n"
                 "(value-weighted, next-quarter price return, frozen holdings)", fontsize=22, loc="left")
    handles = [Patch(color=c, label=g) for g, c in col.items()] + [Patch(color="#D9D9D9", label="95% luck band")]
    ax.legend(handles=handles, loc="center right", fontsize=20, frameon=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "skill_table.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
