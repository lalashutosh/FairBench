"""Write the SYNTHETIC worked example: N-PORT documents for a fictional index fund and a
fictional ESG fund over six quarter-ends, plus the planted truth.

    .venv/bin/python scripts/build_example_dataset.py

Output: examples/real_data_example/nport/<series id>/<report date>.xml and meta.json.
Nothing in it is real; every name says so. It exists so the whole pipeline (parse, store
with provenance, build the universe, derive returns, sample references, rank the fund) can
be run and tested offline, and checked against values that are known because they were planted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fairbench.ingest.nport_synth import example_history

OUT = Path(__file__).resolve().parents[1] / "examples" / "real_data_example"


def main(seed: int = 0) -> None:
    h = example_history(seed=seed)
    meta = h["meta"]
    files = {}
    for role in ("parent", "fund"):
        series = meta[f"{role}_series_id"]
        d = OUT / "nport" / series
        d.mkdir(parents=True, exist_ok=True)
        for day, raw in h[role].items():
            (d / f"{day}.xml").write_bytes(raw)
            files[f"nport/{series}/{day}.xml"] = hashlib.sha256(raw).hexdigest()
    meta["sha256"] = files
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(files)} synthetic N-PORT documents and meta.json to {OUT}")


if __name__ == "__main__":
    main()
