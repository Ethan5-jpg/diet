#!/usr/bin/env python3
"""Display aggregate results only, suitable for terminal screenshots."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    import pandas as pd
    run = Path(args.run_dir)
    summary = run / "SUMMARY.md"
    if not summary.is_file():
        raise SystemExit("SUMMARY.md missing; check --run-dir")
    print(summary.read_text(encoding="utf-8"))
    status = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["status"]
    incomplete = [k for k, v in status.items() if v["status"] in ["RUNNING", "FAILED", "INTERRUPTED"]]
    if incomplete:
        print("\n以下阶段未完成或失败：{}。不展示可能残留的结果表，请查看状态和日志。".format(", ".join(incomplete)))
        return
    items = [
        ("CGM 窗口排除原因", "02_cgm_window_reasons.csv", None),
        ("NOVA4 个体内模型", "03_within_person_models.csv",
         ["model", "status", "people", "meals", "estimate", "ci_low", "ci_high", "p", "reason"]),
        ("菌群交互", "03_microbiome_interactions.csv",
         ["model", "status", "people", "meals", "estimate", "ci_low", "ci_high", "p", "q_planned_taxa", "reason"])
    ]
    for label, name, cols in items:
        path = run / "tables" / name
        if not path.is_file():
            continue
        d = pd.read_csv(path).fillna("")
        if cols:
            d = d[[c for c in cols if c in d]]
        print("\n" + label + "\n")
        print(d.to_string(index=False, float_format=lambda x: "{:.4g}".format(x)))


if __name__ == "__main__":
    main()
