import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from demo_qae import run  # noqa: E402


def test_demo_qae():
    out = run()
    for key in ("results", "details", "n", "k", "planted_rank", "synthetic"):
        assert key in out
    r = out["results"]
    for m in ("exact", "classical", "quantum", "classical_matched"):
        for f in ("percentile", "null_median", "benchmark_median", "constraint_effect",
                  "manager_effect", "queries"):
            assert f in r[m]
    e, q = r["exact"], r["quantum"]
    assert abs(e["percentile"] - q["percentile"]) / 100 < 0.03
    for f in ("constraint_effect", "manager_effect"):
        assert abs(e[f] - q[f]) < 0.02 + 0.3 * abs(e[f])
        if abs(e[f]) > 0.03:
            assert np.sign(e[f]) == np.sign(q[f])
    assert q["queries"] > 0


def test_demo_qae_tight():
    out = run(tight=True)
    assert 0.002 <= out["details"]["P_F"] <= 0.007
    r = out["results"]
    assert abs(r["exact"]["percentile"] - r["quantum"]["percentile"]) / 100 < 0.03
    assert r["quantum"]["queries"] > 0
