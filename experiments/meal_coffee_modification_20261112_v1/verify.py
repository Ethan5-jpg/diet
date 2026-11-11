"""Verify checkpoint receipts and exactly regenerate held-out comparisons."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

import numpy as np

from analyze_coffee import ENDPOINTS, MARKERS, PERIODS, ROOT, STRATA, digest, load, predictions, require


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected run path")
    run = ROOT / "runs_TRE_ONLY" / args.run
    receipt = load(run / "COMPLETE.json")
    reg, summary = run / "registration.json", run / "summary.json"
    require(digest(reg) == receipt["registration"] and digest(summary) == receipt["summary"],
            "Coffee result receipt differs")
    meta, results = load(reg), load(summary)
    require(all(digest(Path(p)) == sha for p, sha in meta["sources"].items()), "A registered source changed")
    require(meta["flow"] == results["flow"] and len(results["direct"]) == (0 if meta["smoke"] else 20),
            "Incomplete flow or association family")
    repeats, folds = ((0,), (0,)) if meta["smoke"] else (range(2), range(5))
    expected = {f"{s}__{period}__{ep}__{marker}__{part}" for s in STRATA for period in PERIODS for ep in ENDPOINTS
                for marker in MARKERS for part in ("none", "coffee", "tss")}
    for repeat in repeats:
        totals = {}
        for fold in folds:
            part = run / f"r{repeat}_f{fold}.npz"
            blob = part.read_bytes()
            require(load(part.with_suffix(".json")) == dict(registration=receipt["registration"], repeat=repeat,
                    fold=fold, sha=sha256(blob).hexdigest()), "Checkpoint receipt differs")
            with np.load(BytesIO(blob), allow_pickle=False) as data:
                require(set(data.files) == expected, "Checkpoint is missing a loss comparison")
                for key in data.files:
                    if key not in totals:
                        totals[key] = np.zeros_like(data[key])
                    totals[key] += data[key]
        require(predictions(totals, repeat) == results["prediction"][f"r{repeat}"],
                "Reaggregated prediction statistics differ")
    print(f"COFFEE_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


if __name__ == "__main__":
    main()
