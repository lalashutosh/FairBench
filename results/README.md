# FairBench result directories

`real_data/` is the canonical model-input directory. It is generated from the
SEC N-PORT archive in `data/fairbench.sqlite` and contains only disclosed or
explicitly derived real-data panels.

The other files directly under `results/` include historical synthetic demos,
regression fixtures, and individual experiment outputs. They are retained for
tests and reproducibility, but must not be presented as scraped market data.

`skill_table/` holds the nine-fund table built from `real_data/` by
`scripts/skill_table.py`, and its figures. `hw/` holds the raw counts, job ids
and fits of the VTT Q50 hardware run and its digital twin (`docs/QUANTUM_CORE.md`, section 13).

Use `real_data/real_data_validation.json` before running a model. The validator
checks provenance, duplicates, positive derived prices, reproducible returns,
and zero/synthetic-value contamination. External Yahoo close prices are a
review-only cross-check and never overwrite the SEC-derived panel.

Real-data quantum runs currently in this directory:

- `real_fund_parnassus_core_equity_quantum.json/.csv`: Parnassus vs the S&P 500 proxy, 27 intervals, ±1 percentage-point target. This is the run the main README and `docs/REAL_FUND_RESULTS.md` quote: a median of 1,900 quantum queries against 5,366 classical samples, median ratio 2.4×.
- `real_fund_parnassus_core_equity_real_systempy_quantum.json/.csv`: Parnassus vs the S&P 500 proxy, 27 intervals, ±2 percentage-point target. At this looser precision the median ratio is 0.84×, so classical sampling needs slightly fewer queries. The two files do not conflict: the quantum gain grows as the precision asked for tightens.
- `real_fund_calvert_equity_real_systempy_quantum.json/.csv`: Calvert vs the Russell 1000 proxy, 27 intervals, ±5 percentage-point target; no mandate-specific exclusion proxy was applied.

These are noiseless analytical simulations, not claims about a present-day
quantum-hardware speedup.
