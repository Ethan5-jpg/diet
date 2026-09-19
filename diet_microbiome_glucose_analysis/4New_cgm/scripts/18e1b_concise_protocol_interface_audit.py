#!/usr/bin/env python3
"""
Step 18e1b — Concise Paper-20 protocol mediation interface audit.

Read-only. Prints only the few implementation details needed to build Step18e2.
"""

from pathlib import Path
import re

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

TARGETS = {
    "PAPER20_PRIMARY": DG / "4New_cgm/scripts/18b_paper20_primary_mediation.py",
    "OLD_PROTOCOL": DG / "scripts/15f2_protocol_window_abxppi_mediation_sensitivity.py",
    "CANONICAL_ENGINE": DG / "scripts/10_diet_microbiome_cgm_mediation.py",
}

PATTERNS = {
    "PAPER20_PRIMARY": [
        r"FROZEN_FULL_PATH_SET",
        r"BOOTSTRAP",
        r"COHORT_MODE",
        r"mean_daily_energy_kcal",
        r"alcohol_intake_g_day",
        r"score.*map",
        r"source.*map",
        r"run_one_path",
        r"mediation_FDR05_primary",
        r"bridge_direction_consistent",
    ],
    "OLD_PROTOCOL": [
        r"protocol_window",
        r"antibiotic_use_protocol_window",
        r"ppi_use_protocol_window",
        r"covariate",
        r"run_one_path",
        r"bootstrap",
        r"FDR",
        r"consistent",
    ],
    "CANONICAL_ENGINE": [
        r"def run_one_path",
        r"def bootstrap_mediation",
        r"def bh_fdr",
        r"FDR_BH_within_score_outcome",
        r"mediation_FDR05_primary",
        r"candidate_mediator_consistent",
        r"bridge_direction_consistent",
    ],
}

def compact(line):
    return " ".join(line.strip().split())

def main():
    print("=== STEP 18e1b CONCISE INTERFACE AUDIT ===")
    print("READ_ONLY=True")
    print("MODELS_FIT=False")
    print()

    for label, path in TARGETS.items():
        print(f"--- {label} ---")
        print(f"PATH={path}")
        if not path.is_file():
            print("EXISTS=False\n")
            continue

        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        printed = set()

        for pat in PATTERNS[label]:
            rx = re.compile(pat, re.I)
            hits = [i for i, line in enumerate(lines, 1) if rx.search(line)]
            if not hits:
                print(f"{pat}: NOT_FOUND")
                continue

            # Print at most first 3 unique matching lines per pattern.
            shown = 0
            for ln in hits:
                key = (ln, compact(lines[ln-1]))
                if key in printed:
                    continue
                printed.add(key)
                print(f"{ln}: {compact(lines[ln-1])}")
                shown += 1
                if shown >= 3:
                    break
        print()

    print("=== WHAT I NEED FROM THIS OUTPUT ===")
    print("Send the complete output of this concise audit; it should be short.")
    print("No model rerun is performed.")

if __name__ == "__main__":
    main()
