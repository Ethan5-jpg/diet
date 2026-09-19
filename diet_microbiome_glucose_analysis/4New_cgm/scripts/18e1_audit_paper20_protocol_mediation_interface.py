#!/usr/bin/env python3
"""
Step 18e1 — Read-only interface audit before Paper-20 protocol-window ABX/PPI mediation.

Purpose:
Extract only the implementation details needed to extend the completed Step18b
Paper-20 primary mediation to protocol-window antibiotic/PPI adjustment without
changing any other statistical logic.

READ ONLY. Does not fit models or modify source files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"
NEW = DG / "4New_cgm" / "scripts"

TARGETS = [
    ("paper20_primary", NEW / "18b_paper20_primary_mediation.py"),
    ("old_protocol", DG / "scripts" / "15f2_protocol_window_abxppi_mediation_sensitivity.py"),
    ("canonical_engine", DG / "scripts" / "10_diet_microbiome_cgm_mediation.py"),
]

KEY_PATTERNS = [
    r"covariate",
    r"antibi",
    r"\bppi\b",
    r"protocol",
    r"score.*map",
    r"source.*map",
    r"extra_covariate",
    r"mean_daily_energy",
    r"alcohol",
    r"primary_cgm_analysis_eligible",
    r"cohort",
    r"run_one_path",
    r"bootstrap",
    r"FDR",
    r"bh_fdr",
    r"mediation_FDR05_primary",
    r"bridge_direction_consistent",
    r"highest_priority",
    r"sample_sha",
    r"OUTPUT_DIR",
    r"OUT_",
]

def read_lines(path):
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()

def function_ranges(text):
    tree = ast.parse(text)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node.lineno, getattr(node, "end_lineno", node.lineno)))
    return out

def interesting_global_assignments(text):
    tree = ast.parse(text)
    rows = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
            elif isinstance(node.target, ast.Name):
                names.append(node.target.id)
            for n in names:
                if re.search(
                    r"(ROOT|DIR|FILE|PATH|COHORT|COVARI|MICROBIOME|SCORE|OUTCOME|BOOTSTRAP|SEED|FDR|MODEL|WORKER)",
                    n,
                    re.I,
                ):
                    seg = ast.get_source_segment(text, node) or ""
                    rows.append((n, node.lineno, " ".join(seg.split())))
    return rows

def print_context(lines, hit_lines, radius=2, max_blocks=80):
    hit_lines = sorted(set(hit_lines))
    blocks = []
    if not hit_lines:
        print("NONE")
        return
    start = max(1, hit_lines[0] - radius)
    end = min(len(lines), hit_lines[0] + radius)
    for h in hit_lines[1:]:
        hs = max(1, h - radius)
        he = min(len(lines), h + radius)
        if hs <= end + 1:
            end = max(end, he)
        else:
            blocks.append((start, end))
            start, end = hs, he
    blocks.append((start, end))
    for i, (a, b) in enumerate(blocks[:max_blocks], 1):
        print(f"### block {i}: lines {a}-{b}")
        for ln in range(a, b + 1):
            print(f"{ln:04d}: {lines[ln-1]}")
        print()
    if len(blocks) > max_blocks:
        print(f"... {len(blocks)-max_blocks} more blocks omitted")

def main():
    print("=== STEP 18e1 PAPER-20 PROTOCOL MEDIATION INTERFACE AUDIT ===")
    print("READ_ONLY=True")
    print("MODELS_FIT=False")
    print("SOURCE_FILES_MODIFIED=False")
    print()

    for label, path in TARGETS:
        print("=" * 100)
        print(f"TARGET={label}")
        print(f"PATH={path}")
        print(f"EXISTS={path.is_file()}")
        if not path.is_file():
            print()
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        print(f"LINES={len(lines)}")

        print("\n--- TOP-LEVEL FUNCTIONS ---")
        for name, a, b in function_ranges(text):
            print(f"{name} [{a}-{b}]")

        print("\n--- IMPORTANT GLOBAL ASSIGNMENTS ---")
        globals_ = interesting_global_assignments(text)
        if globals_:
            for name, ln, src in globals_:
                print(f"{ln:04d} {name}: {src}")
        else:
            print("NONE")

        print("\n--- TARGETED SOURCE CONTEXT ---")
        hit_lines = []
        compiled = [re.compile(p, re.I) for p in KEY_PATTERNS]
        for i, line in enumerate(lines, 1):
            if any(rx.search(line) for rx in compiled):
                hit_lines.append(i)
        print_context(lines, hit_lines, radius=2, max_blocks=100)

    print("=" * 100)
    print("AUDIT COMPLETE")
    print(
        "NEXT=Use this output to create Step18e2 that changes ONLY the "
        "protocol-window antibiotic/PPI covariates while preserving Step18b "
        "path set, score mapping, cohort logic, bootstrap, FDR, and direction rules."
    )

if __name__ == "__main__":
    main()
