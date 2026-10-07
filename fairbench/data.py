"""Asset universe: dataclass, CSV loader, synthetic generator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Universe:
    tickers: list[str]
    mu: np.ndarray  # (n,) expected returns
    cov: np.ndarray  # (n, n)
    sector: list[str]
    esg_score: np.ndarray  # (n,)
    carbon: np.ndarray  # (n,)
    returns: pd.DataFrame | None = None  # historical returns (T, n), for attribution

    @property
    def n(self) -> int:
        return len(self.tickers)


def load_universe(csv_path: str | Path, cov_path: str | Path | None = None,
                  default_var: float = 0.04, returns_path: str | Path | None = None) -> Universe:
    """Load a universe from CSV.

    Schema (one row per asset, header required):
        ticker, mu, sector, esg_score, carbon [, sigma]
    Covariance:
      * if ``cov_path`` is given: an n x n CSV with ticker header row and ticker
        index column (``pd.read_csv(index_col=0)``), reordered to match tickers;
      * else if a ``sigma`` column exists: diag(sigma**2);
      * else: identity scaled by ``default_var`` (0.04, i.e. 20% vol).
    ``returns`` is None unless ``returns_path`` is given (see ``load_returns``).
    """
    df = pd.read_csv(csv_path)
    missing = {"ticker", "mu", "sector", "esg_score", "carbon"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    tickers = df["ticker"].astype(str).tolist()
    n = len(tickers)
    if cov_path is not None:
        c = pd.read_csv(cov_path, index_col=0)
        c.index = c.index.astype(str)
        c.columns = c.columns.astype(str)
        cov = c.loc[tickers, tickers].to_numpy(dtype=float)
    elif "sigma" in df.columns:
        cov = np.diag(df["sigma"].to_numpy(dtype=float) ** 2)
    else:
        cov = np.eye(n) * default_var
    return Universe(
        tickers=tickers,
        mu=df["mu"].to_numpy(dtype=float),
        cov=cov,
        sector=df["sector"].astype(str).tolist(),
        esg_score=df["esg_score"].to_numpy(dtype=float),
        carbon=df["carbon"].to_numpy(dtype=float),
        returns=None if returns_path is None else load_returns(returns_path, tickers),
    )


def load_returns(csv_path: str | Path, tickers: list[str]) -> pd.DataFrame:
    """Load historical simple returns: wide CSV, first column = period/date index,
    one column per ticker. Columns are reordered to ``tickers`` (extra columns are
    dropped); missing tickers or NaNs raise ValueError."""
    r = pd.read_csv(csv_path, index_col=0)
    r.columns = r.columns.astype(str)
    missing = [t for t in tickers if t not in r.columns]
    if missing:
        raise ValueError(f"returns CSV missing tickers: {missing[:10]}")
    r = r[list(tickers)].astype(float)
    if r.isna().any().any():
        raise ValueError("returns CSV contains NaNs for the requested tickers")
    return r


def synthetic_returns(u: Universe, periods: int = 252, periods_per_year: int = 252,
                      mu_shift: np.ndarray | None = None, seed: int | None = 0) -> pd.DataFrame:
    """SYNTHETIC (periods, n) simple returns: i.i.d. Gaussian with annual mean
    ``u.mu + mu_shift`` and annual covariance ``u.cov``, scaled to the period.
    ``mu_shift`` (n,) plants a return tilt (e.g. an ESG drag) for attribution demos."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(u.mu, dtype=float) + (0.0 if mu_shift is None else np.asarray(mu_shift, dtype=float))
    r = rng.multivariate_normal(mean / periods_per_year, np.asarray(u.cov) / periods_per_year,
                                size=periods, method="cholesky")
    return pd.DataFrame(r, columns=list(u.tickers), index=pd.RangeIndex(periods, name="period"))


def synthetic_universe(n: int, n_sectors: int, seed: int | None = 0) -> Universe:
    """Deterministic synthetic universe. Sectors assigned round-robin
    ('S0','S1',...); ESG in [0,100], carbon in [10,500], mu ~ N(0.08,0.05),
    cov = F F^T / d + diag idiosyncratic (PSD). ``returns`` is None."""
    rng = np.random.default_rng(seed)
    d = max(2, min(n, 3))
    F = rng.normal(0, 0.1, size=(n, d))
    cov = F @ F.T + np.diag(rng.uniform(0.01, 0.04, size=n))
    return Universe(
        tickers=[f"A{i:03d}" for i in range(n)],
        mu=rng.normal(0.08, 0.05, size=n),
        cov=cov,
        sector=[f"S{i % n_sectors}" for i in range(n)],
        esg_score=rng.uniform(0, 100, size=n),
        carbon=rng.uniform(10, 500, size=n),
        returns=None,
    )
