"""Verify run receipts and regenerate the paired held-out estimates from checkpoints."""

import argparse
from hashlib import sha256
from io import BytesIO
import re

import numpy as np

from analyze_bread import ENDPOINTS, PERIODS, ROOT, active_markers, digest, load, paired_summary, require
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected run path")
    run = ROOT / "runs_TRE_ONLY" / args.run
    receipt = load(run / "COMPLETE.json")
    reg, summary = run / "registration.json", run / "summary.json"
    require(digest(reg) == receipt["registration"] and digest(summary) == receipt["summary"],
            "Whole-wheat run receipt differs")
    meta, results = load(reg), load(summary)
    require(all(digest(Path(p)) == sha for p, sha in meta["sources"].items()), "A registered source changed")
    require(meta["flow"] == results["flow"] and len(results["direct"]) == (0 if meta["smoke"] else 15),
            "Incomplete flow or association family")
    repeats, folds = ((0,), (0,)) if meta["smoke"] else (range(2), range(5))
    expected = {f"{period}__{endpoint}__{marker}__{part}" for period in PERIODS for endpoint in ENDPOINTS
                for marker in active_markers(period) for part in ("none", "exchange", "tss")}
    for repeat in repeats:
        totals = {}
        for fold in folds:
            part = run / f"r{repeat}_f{fold}.npz"
            blob = part.read_bytes()
            require(load(part.with_suffix(".json")) == dict(registration=receipt["registration"], repeat=repeat,
                    fold=fold, sha=sha256(blob).hexdigest()), "Checkpoint receipt differs")
            with np.load(BytesIO(blob), allow_pickle=False) as data:
                require(set(data.files) == expected, "Checkpoint is missing held-out comparisons")
                for key in data.files:
                    if key not in totals:
                        totals[key] = np.zeros_like(data[key])
                    totals[key] += data[key]
        require(paired_summary(totals, repeat) == results["prediction"][f"r{repeat}"],
                "Reaggregated held-out estimates differ")
    print(f"WHOLEWHEAT_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


if __name__ == "__main__":
    main()
