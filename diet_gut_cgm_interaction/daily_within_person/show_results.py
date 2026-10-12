#!/usr/bin/env python3
"""Display a specific completed run; verify summary/table hashes before showing."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--screen", type=int, choices=[1, 2, 3])
    args = p.parse_args()
    from daily.common import require, sha256
    try:
        out = Path(args.run_dir).expanduser().resolve()
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        require(m["status"] in ["BUILD_COMPLETE", "COMPLETE", "COMPLETE_WITH_LIMITATIONS", "NO_ESTIMABLE_PRIMARY_MODELS"],
                "This run has no verified completed results: " + m["status"])
        for name, digest in m["output_sha256"].items():
            require((out / name).is_file() and sha256(out / name) == digest, "Saved output hash mismatch: " + name)
        for n in ([args.screen] if args.screen else [1, 2, 3]):
            print((out / ("SCREEN_{}.txt".format(n))).read_text(encoding="utf-8"))
        print("RESULT_DIR=" + str(out))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print("DISPLAY_REFUSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
