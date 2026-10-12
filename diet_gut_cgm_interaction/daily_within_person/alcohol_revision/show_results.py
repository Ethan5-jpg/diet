#!/usr/bin/env python3
"""Show only a verified completed revision, one or all three summary screens."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily.common import sha256, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--screen", type=int, choices=[1, 2, 3])
    args = p.parse_args()
    try:
        out = Path(args.run_dir).expanduser().resolve()
        m = json.loads((out / "manifest.json").read_text())
        require(m["version"] == "2026-10-11.alcohol.1", "Unsupported revision version")
        require(m["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS", "NO_ESTIMABLE_PRIMARY_MODELS"], "Run is not verified complete")
        expected = {"SCREEN_1.txt", "SCREEN_2.txt", "SCREEN_3.txt", "tables/10_interactions.csv", "private/daily_panel.csv"}
        require(expected.issubset(m["output_sha256"]), "Incomplete output fingerprint registry")
        for name, digest in m["output_sha256"].items():
            require(sha256(out / name) == digest, "Saved output changed: " + name)
        for n in ([args.screen] if args.screen else [1, 2, 3]):
            print((out / ("SCREEN_{}.txt".format(n))).read_text())
        print("RESULT_DIR=" + str(out))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print("DISPLAY_REFUSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
