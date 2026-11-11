"""Verify frozen coefficients and paired household loss receipts."""

import argparse
from hashlib import sha256
from io import BytesIO
import re

import numpy as np

from evaluate import ENDPOINTS, ROOT, SEED, check_sources, digest, load, require, summarize


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True)
    args = cli.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Unexpected run path")
    run = ROOT / "runs_TRE_ONLY" / args.run
    receipt = load(run / "COMPLETE.json")
    reg, summary = run / "registration.json", run / "summary.json"
    require(digest(reg) == receipt["registration"] and digest(summary) == receipt["summary"],
            "Offset run receipt differs")
    meta, results = load(reg), load(summary)
    check_sources(meta["sources"])
    check_sources(receipt["extra_sources"])
    require(meta["flow"] == results["flow"], "Offset cohort differs")
    repeats, folds = ((0,), (0,)) if meta["smoke"] else (range(2), range(5))
    for repeat in repeats:
        totals = {}
        for fold in folds:
            part = run / f"r{repeat}_f{fold}.npz"
            data = part.read_bytes()
            require(load(part.with_suffix(".json")) == dict(registration=receipt["registration"], repeat=repeat,
                    fold=fold, sha=sha256(data).hexdigest()), "Offset fold receipt differs")
            with np.load(BytesIO(data), allow_pickle=False) as saved:
                require(set(saved.files) == set(ENDPOINTS), "Missing offset comparison")
                for endpoint in ENDPOINTS:
                    if endpoint not in totals:
                        totals[endpoint] = np.zeros_like(saved[endpoint])
                    totals[endpoint] += saved[endpoint]
        require(summarize(totals, SEED + repeat) == results["prediction"][f"r{repeat}"],
                "Offset fold reaggregation differs")
    if meta["smoke"]:
        require(not receipt["artifacts"] and not results["additional"], "Smoke opened additional outcomes")
    else:
        expected = {"shared_coefficients_TRE_ONLY.npz", "FROZEN.json", "additional_losses_TRE_ONLY.npz"}
        require(set(receipt["artifacts"]) == expected
                and all(digest(run / k) == v for k, v in receipt["artifacts"].items()), "Validation payload differs")
        require(load(run / "FROZEN.json") == dict(registration=receipt["registration"],
                weights=receipt["artifacts"]["shared_coefficients_TRE_ONLY.npz"], additional_outcomes_read=False),
                "Frozen shared interaction receipt differs")
        path = run / "additional_losses_TRE_ONLY.npz"
        with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as saved:
            require(set(saved.files) == set(ENDPOINTS)
                    and summarize(saved, SEED + 2) == results["additional"]["prediction"],
                    "Additional household reaggregation differs")
    print(f"SHARED_OFFSET_READBACK_PASS folds={len(repeats) * len(folds)}", flush=True)


if __name__ == "__main__":
    main()
