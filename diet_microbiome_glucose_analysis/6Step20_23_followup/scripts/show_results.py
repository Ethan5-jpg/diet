#!/usr/bin/env python3
"""Display concise saved summaries; never trigger another analysis."""
import argparse
import json
from pathlib import Path

package = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, help="Exact followup output directory")
p.add_argument("--section", choices=["20", "21", "22", "23", "all"], default="all")
a = p.parse_args()
last = package / "outputs/LAST_RUN.json"
if a.run:
    run = a.run.expanduser().resolve()
elif last.is_file():
    run = Path(json.loads(last.read_text(encoding="utf-8"))["run_dir"])
else:
    p.error("No run pointer; first run bash run_followup.sh or pass --run")
target = run / ("SCREENSHOT_SUMMARY.txt" if a.section == "all" else "STEP" + a.section + "_SUMMARY.txt")
print(target.read_text(encoding="utf-8"))
