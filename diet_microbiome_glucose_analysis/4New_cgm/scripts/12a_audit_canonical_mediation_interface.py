#!/usr/bin/env python3
"""Step 12a — READ-ONLY audit of the canonical old mediation implementation.

Purpose
-------
Before adapting the already-validated old mediation pipeline to the new
73-path MAGE/TBR70 bridge set, inspect the CURRENT SERVER copy of:

    diet_microbiome_glucose_analysis/scripts/
    10_diet_microbiome_cgm_mediation.py

This does NOT import or execute that script.  It only reads/parses source code
and prints the parts needed to build a faithful adapter:
- global path/config constants
- bootstrap count / seed
- candidate-loading logic
- covariate logic
- OLS / bootstrap helper signatures
- FDR-family logic
- output schemas / filenames

Why
---
The project handoff preserves the statistical method (Model3, a*b,
1000 bootstrap, within Diet x CGM family FDR), but explicitly says the server
canonical Step10 contains later eligibility/no-statsmodels fixes and should not
be replaced by historical copies.  This audit lets the next script reuse the
real current implementation rather than guessing small but important details.

Dependencies: Python standard library only.
"""

from __future__ import annotations

from pathlib import Path
import ast
import re

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
TARGET = (
    ROOT
    / "diet_microbiome_glucose_analysis"
    / "scripts"
    / "10_diet_microbiome_cgm_mediation.py"
)
OUT = (
    ROOT
    / "diet_microbiome_glucose_analysis"
    / "4New_cgm"
    / "outputs"
    / "step12a_canonical_mediation_interface_audit.txt"
)

KEYWORDS = (
    "candidate",
    "bootstrap",
    "mediat",
    "indirect",
    "fdr",
    "covari",
    "model3",
    "strict",
    "participant",
    "species",
    "diet",
    "cgm",
    "output",
    "seed",
    "alcohol",
    "bmi",
    "device",
)

INTERESTING_GLOBAL_NAMES = {
    "ROOT", "BASE", "OUT", "OUTPUT", "OUTPUT_DIR", "REPORTS", "MODELS",
    "CANDIDATE", "CANDIDATES", "CANDIDATE_PATH", "BRIDGE", "BRIDGE_PATH",
    "BOOTSTRAP", "N_BOOT", "N_BOOTSTRAP", "BOOTSTRAPS", "SEED", "RANDOM_SEED",
    "COVARIATES", "BASE_COVARIATES", "MODEL3_COVARIATES",
    "DIET", "DIET_PATH", "MICROBIOME", "MICROBIOME_PATH",
    "CGM", "CGM_PATH", "COVARIATE", "COVARIATE_PATH",
}


def literal_repr(node):
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<unparseable>"


def function_signature(node: ast.FunctionDef) -> str:
    try:
        return ast.unparse(node.args)
    except Exception:
        return "(signature unavailable)"


def assignment_name(node):
    if isinstance(node, ast.Name):
        return node.id
    return None


def main() -> int:
    if not TARGET.is_file():
        raise FileNotFoundError(
            "Canonical mediation script not found:\n"
            f"{TARGET}\n"
            "Do not substitute a historical backup."
        )

    text = TARGET.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(TARGET))

    globals_found = []
    functions = []

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.Assign):
                names = [assignment_name(t) for t in node.targets]
                value = node.value
            else:
                names = [assignment_name(node.target)]
                value = node.value
            for name in names:
                if not name:
                    continue
                upper = name.upper()
                interesting = (
                    name in INTERESTING_GLOBAL_NAMES
                    or any(k.upper() in upper for k in (
                        "BOOT", "SEED", "COVAR", "CANDID",
                        "OUTPUT", "PATH", "MODEL", "STRICT"
                    ))
                )
                if interesting:
                    globals_found.append((name, literal_repr(value)))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                (
                    node.name,
                    function_signature(node),
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                )
            )

    # Keyword source hits with a compact +/-2 line window.
    hit_lines = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(k in low for k in KEYWORDS):
            hit_lines.append(i)

    # Merge nearby hits into source windows, capped to manageable size.
    windows = []
    for ln in hit_lines:
        lo = max(1, ln - 2)
        hi = min(len(lines), ln + 2)
        if windows and lo <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))

    # Prefer windows containing the most important terms.
    def score_window(w):
        blob = "\n".join(lines[w[0]-1:w[1]]).lower()
        weights = {
            "bootstrap": 5,
            "candidate": 5,
            "indirect": 5,
            "fdr": 5,
            "covari": 4,
            "mediat": 4,
            "model3": 3,
            "output": 2,
        }
        return sum(v for k, v in weights.items() if k in blob)

    windows = sorted(windows, key=lambda w: (-score_window(w), w[0]))[:30]
    windows = sorted(windows)

    report = []
    report.append("=== STEP 12a CANONICAL MEDIATION INTERFACE AUDIT ===")
    report.append("READ_ONLY=True")
    report.append("CANONICAL_SCRIPT_EXECUTED=False")
    report.append(f"TARGET={TARGET}")
    report.append(f"LINES={len(lines)}")
    report.append("")

    report.append("--- INTERESTING GLOBAL ASSIGNMENTS ---")
    if globals_found:
        for name, value in globals_found:
            report.append(f"{name} = {value}")
    else:
        report.append("NONE_FOUND_BY_AST_FILTER")
    report.append("")

    report.append("--- TOP-LEVEL FUNCTION SIGNATURES ---")
    for name, sig, lo, hi in functions:
        report.append(f"{name}{sig}  [lines {lo}-{hi}]")
    report.append("")

    report.append("--- IMPORTANT SOURCE EXCERPTS ---")
    if not windows:
        report.append("NO_KEYWORD_WINDOWS_FOUND")
    else:
        for lo, hi in windows:
            report.append(f"\n### lines {lo}-{hi}")
            for n in range(lo, hi + 1):
                report.append(f"{n:04d}: {lines[n-1]}")

    # Extra exact pattern searches useful for adaptation.
    report.append("")
    report.append("--- EXACT PATTERN LOCATIONS ---")
    patterns = {
        "candidate_csv_reads": r"read_csv\([^\n]*candidate|candidate[^\n]*read_csv",
        "bootstrap_loops": r"for\s+.*\s+in\s+range\([^\n]*boot|bootstrap",
        "bh_fdr": r"multipletests|benjamini|bh|fdr",
        "indirect_product": r"\ba\s*\*\s*b\b|indirect",
        "covariate_audit": r"covariate.*audit|audit.*covariate",
    }
    for label, pat in patterns.items():
        matches = []
        rx = re.compile(pat, flags=re.I)
        for i, line in enumerate(lines, start=1):
            if rx.search(line):
                matches.append(i)
        report.append(f"{label}: {matches[:50]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nAUDIT_REPORT={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
