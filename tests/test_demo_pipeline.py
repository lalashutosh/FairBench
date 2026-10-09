"""The one-command demo runs offline from committed results and shows real numbers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from demo_pipeline import run  # noqa: E402


def test_demo_pipeline_reads_only_committed_files():
    d = run()
    m, c, q = d["mandate"], d["classical"], d["quantum"]
    assert m["n_rules"] == 18 and m["n_rejected"] == 0 and 0 < m["n_enforced"] <= 18
    assert "%" in m["valid_share"] and "Σ" in m["example_formula"]
    assert d["data"]["quarters"] == 27 and d["data"]["fund"].startswith("Parnassus")
    assert c["index"] > c["fund"] > c["typical"] and 40 < c["mean_rank"] < 70
    assert q["median_quantum_queries"] < q["median_classical_samples"]
    assert q["resources_last_period"]["logical_qubits"] > 500
    assert (Path(__file__).resolve().parent.parent / d["figure"]).exists()
