#!/usr/bin/env python3
"""Display the latest summary or one core taxon page using the standard library."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="A specific outputs/step19b_* directory")
    parser.add_argument("--core", help="Taxon ID from summary, e.g. T023; use all for all core pages")
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    if args.run:
        folder = Path(args.run).resolve()
    else:
        folders = sorted((package / "outputs").glob("step19b_*"))
        if not folders:
            print("No run found. First execute: bash run_step19b.sh")
            return 1
        folder = folders[-1]
    summary = folder / "19b_SCREENSHOT_SUMMARY.txt"
    if not summary.is_file():
        print("Summary is not available: " + str(folder))
        return 1
    if not args.core:
        print(summary.read_text(encoding="utf-8"))
        return 0
    manifest = folder / "manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "PASS":
        print(summary.read_text(encoding="utf-8"))
        print("Core pages require a completed PASS run.")
        return 1
    pages = sorted(folder.glob("19b_CORE_T*_SCREENSHOT.txt"))
    matched = pages if args.core == "all" else [p for p in pages if p.name == "19b_CORE_" + args.core + "_SCREENSHOT.txt"]
    if not matched:
        print("Available core IDs: " + ", ".join(p.name.split("_")[2] for p in pages))
        return 1
    for page in matched:
        print(page.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
