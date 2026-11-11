"""Read back saved losses and reproduce the paired household intervals."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

import numpy as np

from analyze_timing import ENDPOINTS, MARKERS, ROOT, STRATA, digest, load, pooled_summary, require


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected run path")
    run = ROOT / "runs_TRE_ONLY" / args.run
    manifest = load(run / "COMPLETE.json")
    reg, summary = run / "registration.json", run / "summary.json"
    require(digest(reg) == manifest["registration"] and digest(summary) == manifest["summary"],
            "Timing analysis receipt differs")
    meta, result = load(reg), load(summary)
    require(all(digest(Path(p)) == sha for p, sha in meta["sources"].items()), "Registered input differs")
    require(meta["flow"] == result["flow"] and len(result["direct"]) == (0 if meta["smoke"] else 20),
            "Incomplete flow or association family")
    repeats, folds = ((0,), (0,)) if meta["smoke"] else (range(2), range(5))
    expected = {f"{s}__{m}__{c}__{e}__{v}" for s in STRATA for m in MARKERS for c in ("near", "far")
                for e in ENDPOINTS for v in ("none", "timing", "tss")}
    for repeat in repeats:
        totals = {}
        for fold in folds:
            part = run / f"r{repeat}_f{fold}.npz"
            blob = part.read_bytes()
            require(load(part.with_suffix(".json")) == dict(registration=manifest["registration"], repeat=repeat,
                    fold=fold, sha=sha256(blob).hexdigest()), "Checkpoint receipt differs")
            with np.load(BytesIO(blob), allow_pickle=False) as data:
                require(set(data.files) == expected, "Incomplete held-out comparison")
                for key in data.files:
                    if key not in totals:
                        totals[key] = np.zeros_like(data[key])
                    totals[key] += data[key]
        require(pooled_summary(totals, repeat) == result["prediction"][f"r{repeat}"],
                "Reaggregated held-out estimates differ")
    print(f"HANDOFF_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


if __name__ == "__main__":
    main()
