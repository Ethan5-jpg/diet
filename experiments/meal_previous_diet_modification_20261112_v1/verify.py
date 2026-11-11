"""Verify fold files, training snapshot and exactly reaggregate validation losses."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

import numpy as np

from analyze_previous import ENDPOINTS, PERIODS, ROOT, SEED, check_sources, digest, load, require, summarize


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected run path")
    run = ROOT / "runs_TRE_ONLY" / args.run
    receipt = load(run / "COMPLETE.json")
    reg, summary = run / "registration.json", run / "summary.json"
    require(digest(reg) == receipt["registration"] and digest(summary) == receipt["summary"], "Run receipt differs")
    meta, results = load(reg), load(summary)
    check_sources(meta["sources"])
    require(meta["support"] == results["support"] and len(results["direct"]) == (0 if meta["smoke"] else len(ENDPOINTS)),
            "Incomplete support or clustered association family")
    repeats, folds = ((0,), (0,)) if meta["smoke"] else (range(2), range(5))
    keys = {f"{ep}__{part}" for ep in ENDPOINTS for part in ("none", "joint", "direction", "tss")}
    expected = {f"{period}__{k}" for period in PERIODS for k in keys}
    for repeat in repeats:
        totals = {period: {} for period in PERIODS}
        for fold in folds:
            part = run / f"r{repeat}_f{fold}.npz"
            data = part.read_bytes()
            require(load(part.with_suffix(".json")) == dict(registration=receipt["registration"], repeat=repeat,
                    fold=fold, sha=sha256(data).hexdigest()), "Fold receipt differs")
            with np.load(BytesIO(data), allow_pickle=False) as saved:
                require(set(saved.files) == expected, "Incomplete fold loss comparisons")
                for name in saved.files:
                    period, key = name.split("__", maxsplit=1)
                    if key not in totals[period]:
                        totals[period][key] = np.zeros_like(saved[name])
                    totals[period][key] += saved[name]
        require({p: summarize(v, SEED + repeat) for p, v in totals.items()} == results["prediction"][f"r{repeat}"],
                "Discovery loss aggregation differs")
    if not meta["smoke"] and meta["support"]["additional"]["households"] >= 10:
        expected_artifacts = {"frozen_history_TRE_ONLY.npz", "FROZEN.json", "additional_losses_TRE_ONLY.npz"}
        require(set(receipt["artifacts"]) == expected_artifacts
                and all(digest(run / name) == value for name, value in receipt["artifacts"].items()),
                "Additional-household receipt differs")
        require(load(run / "FROZEN.json") == dict(registration=receipt["registration"],
                frozen=receipt["artifacts"]["frozen_history_TRE_ONLY.npz"], new_outcomes_read=False),
                "Training snapshot differs")
        with np.load(BytesIO((run / "additional_losses_TRE_ONLY.npz").read_bytes()), allow_pickle=False) as saved:
            require(set(saved.files) == keys and summarize(saved, SEED + 2) == results["additional"],
                    "Additional household losses differ")
    else:
        require(not receipt["artifacts"] and not results["additional"], "Unexpected additional household results")
    print(f"PREVIOUS_MEAL_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


if __name__ == "__main__":
    main()
