#!/usr/bin/env python3
"""Print one or all compact screens from an explicitly selected completed run."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", help="Default: most recent successful real-input run recorded by this package")
p.add_argument("--screen", choices=["1", "2", "3", "all"], default="all")
args = p.parse_args()
if args.run_dir:
    root = Path(args.run_dir).expanduser().resolve()
else:
    pointer = Path(__file__).resolve().parent / "outputs" / "LATEST_REAL.txt"
    if not pointer.is_file():
        p.error("No successful real-input run recorded. Use --run-dir for an explicit run, including a demo.")
    root = Path(pointer.read_text(encoding="utf-8").strip())
name = "SCREENSHOT_SUMMARY.txt" if args.screen == "all" else "SCREEN_" + args.screen + ".txt"
file = root / name
if not file.is_file():
    p.error("Missing summary: " + str(file))
print(file.read_text(encoding="utf-8"))
