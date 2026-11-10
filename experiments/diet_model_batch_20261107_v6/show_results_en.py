"""Display saved aggregate results in ASCII; no fitting or private-array reads."""

import argparse
import json
import math
import re
from pathlib import Path


def render(result, run_id):
    if result.get("status") != "COMPLETE":
        raise ValueError("Run is incomplete; no scores displayed")
    scores = {}
    for row in result["scores"]:
        key = (row["model"], row["repeat"])
        if key in scores or key[1] not in (0, 1) or not math.isfinite(row["R2"]):
            raise ValueError("Invalid or duplicated aggregate score")
        scores[key] = row
    names = sorted({name for name, _ in scores})
    if any((name, repeat) not in scores for name in names for repeat in (0, 1)):
        raise ValueError("Missing paired repeat score")
    baseline = [scores[("v4_hgb_oral", repeat)]["R2"] for repeat in (0, 1)]
    lines = [
        "V6 aggregate results | run=" + run_id,
        "Status: COMPLETE | repeats: two household allocations of the same cohort",
        "Total-model dR2 below uses v4_hgb_oral as the comparator.",
        "%-23s %11s %11s %12s %12s" % ("model", "R2_0", "R2_1", "dR2_0", "dR2_1"),
    ]
    for name in names:
        a, b = [scores[(name, repeat)]["R2"] for repeat in (0, 1)]
        lines.append(
            "%-23s %11.6f %11.6f %+12.6f %+12.6f"
            % (name, a, b, a - baseline[0], b - baseline[1])
        )
    interactions = [
        row
        for row in result["comparisons"]
        if row["comparison"] == "explicit_interaction"
    ]
    lines += [
        "",
        "Matched interaction dR2: comparator=kernel_main",
        "%-23s %12s %12s %10s %10s"
        % ("model", "dR2_0", "dR2_1", "better0/5", "better1/5"),
    ]
    for name in sorted({row["model"] for row in interactions}):
        paired = [
            next(
                row
                for row in interactions
                if row["model"] == name and row["repeat"] == repeat
            )
            for repeat in (0, 1)
        ]
        if any(row["comparator"] != "kernel_main" for row in paired):
            raise ValueError("Unexpected interaction comparator")
        lines.append(
            "%-23s %+12.6f %+12.6f %10d %10d"
            % (
                name,
                paired[0]["delta_R2"],
                paired[1]["delta_R2"],
                paired[0]["improved_folds"],
                paired[1]["improved_folds"],
            )
        )
    candidates = [name for name in names if not name.startswith("v4_")]
    best = max(
        candidates, key=lambda name: min(scores[(name, r)]["R2"] for r in (0, 1))
    )
    lines += [
        "",
        "Best by the lower of two repeat R2 values: " + best,
        "Discovery results; the two repeats are not independent replication.",
    ]
    return ("\n".join(lines) + "\n").encode("ascii", "backslashreplace").decode()


def main(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
        raise ValueError("Invalid run-id")
    path = Path(__file__).resolve().parent / "runs" / args.run_id / "summary.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    print(render(result, args.run_id), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="console_20261107")
    main(parser.parse_args())
