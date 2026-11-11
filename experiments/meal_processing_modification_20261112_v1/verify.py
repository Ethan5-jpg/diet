"""Verify completed model checkpoints and exactly rebuild prediction summaries."""

import argparse
from io import BytesIO
from pathlib import Path
import re

import numpy as np

from analyze_processing import ENDPOINTS, MARKERS, ROOT, STRATA, digest, load, require, summarize_stratum


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected run path")
    run = ROOT / "runs_TRE_ONLY" / args.run
    manifest = load(run / "COMPLETE.json")
    reg_path, summary_path = run / "registration.json", run / "summary.json"
    require(digest(reg_path) == manifest["registration"] and digest(summary_path) == manifest["summary"],
            "Model receipt differs")
    reg, report = load(reg_path), load(summary_path)
    require(all(digest(Path(p)) == sha for p, sha in reg["sources"].items()), "Registered inputs differ")
    repeats, folds = ((0,), (0,)) if reg["smoke"] else (range(2), range(5))
    require(len(report["direct"]) == (0 if reg["smoke"] else len(STRATA) * len(MARKERS) * len(ENDPOINTS)),
            "Association test family incomplete")
    expected = {(s, d, e, m, l) for s in STRATA for d in ("pooled", "early_to_late", "late_to_early")
                for e in ENDPOINTS for m in MARKERS for l in ("none", "processing", "tss")}
    for repeat in repeats:
        all_parts = {}
        for fold in folds:
            file = run / f"r{repeat}_f{fold}.npz"
            data = file.read_bytes()
            require(load(file.with_suffix(".json")) == dict(registration=manifest["registration"],
                    repeat=repeat, fold=fold, sha=digest(file)), "Fold receipt differs")
            with np.load(BytesIO(data), allow_pickle=False) as saved:
                require({tuple(key.split("__")) for key in saved.files} == expected, "Fold loss table incomplete")
                for key in saved.files:
                    if key not in all_parts:
                        all_parts[key] = np.zeros_like(saved[key])
                    all_parts[key] += saved[key]
        stratum_losses = {s: {} for s in STRATA}
        for key, value in all_parts.items():
            s, direction, endpoint, marker, loss = key.split("__")
            stratum_losses[s].setdefault(f"{direction}__{endpoint}__{marker}", {})[loss] = value
        reproduced = {s: summarize_stratum(stratum_losses[s], repeat) for s in STRATA}
        require(reproduced == report["prediction"][f"r{repeat}"], "Archived predictions do not aggregate")
    print(f"PROCESSING_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


if __name__ == "__main__":
    main()
