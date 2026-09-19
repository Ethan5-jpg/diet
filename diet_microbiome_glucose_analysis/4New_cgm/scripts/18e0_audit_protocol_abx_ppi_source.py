#!/usr/bin/env python3
"""
Step 18e0 — Locate and audit the already-validated protocol-window
antibiotic/PPI covariate source before Paper-20 mediation sensitivity.

READ ONLY:
- does not fit models
- does not modify covariates
- does not redefine medication exposure
- does not fall back to broad baseline J01/A02BC as the canonical protocol rule

Goal:
Reuse the project's existing author-like "during dietary logging period"
antibiotic/PPI definition if it already exists, rather than silently rebuilding
or broadening it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

PREFERRED_ABX = (
    "antibiotic_use_during_logging",
    "antibiotic_during_logging",
    "antibiotic_use_protocol_window",
    "antibiotic_protocol_window",
)
PREFERRED_PPI = (
    "ppi_use_during_logging",
    "ppi_during_logging",
    "ppi_use_protocol_window",
    "ppi_protocol_window",
)

GENERIC_ABX = (
    "antibiotic_use",
    "antibiotics_use",
    "recent_antibiotic_use",
)
GENERIC_PPI = (
    "ppi_use",
    "proton_pump_inhibitor_use",
    "proton_pump_inhibitors_use",
)

BROAD_BASELINE_ABX = (
    "antibiotic_atc_j01_positive",
    "antibiotic_j01_positive",
)
BROAD_BASELINE_PPI = (
    "ppi_atc_a02bc_positive",
    "ppi_a02bc_positive",
)

FILENAME_TERMS = (
    "antibi",
    "ppi",
    "medicat",
    "protocol",
    "window",
    "sensitivity",
    "covariate",
    "master",
    "15f",
    "12a",
    "12b",
)


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def first_present(columns, names):
    cols = {str(c).lower(): str(c) for c in columns}
    for x in names:
        if x.lower() in cols:
            return cols[x.lower()]
    return None


def read_header(path: Path):
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def candidate_score(path: Path, cols):
    name = path.name.lower()
    abx_pref = first_present(cols, PREFERRED_ABX)
    ppi_pref = first_present(cols, PREFERRED_PPI)
    abx_gen = first_present(cols, GENERIC_ABX)
    ppi_gen = first_present(cols, GENERIC_PPI)
    abx_broad = first_present(cols, BROAD_BASELINE_ABX)
    ppi_broad = first_present(cols, BROAD_BASELINE_PPI)

    score = 0
    reason = []

    if abx_pref and ppi_pref:
        score += 100
        reason.append("explicit_during_logging_columns")
    elif abx_gen and ppi_gen:
        score += 20
        reason.append("generic_abx_ppi_columns")

    if any(t in name for t in ("protocol", "window")):
        score += 25
        reason.append("protocol_window_filename")
    if any(t in name for t in ("sensitivity", "master", "12a", "15f")):
        score += 8
        reason.append("sensitivity_master_filename")

    if abx_broad and ppi_broad:
        score -= 60
        reason.append("broad_baseline_atc_flags")

    return score, ";".join(reason), {
        "abx_preferred": abx_pref or "",
        "ppi_preferred": ppi_pref or "",
        "abx_generic": abx_gen or "",
        "ppi_generic": ppi_gen or "",
        "abx_broad": abx_broad or "",
        "ppi_broad": ppi_broad or "",
    }


def summarize_flag(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce")
    return {
        "n": int(len(x)),
        "nonmissing": int(x.notna().sum()),
        "missing": int(x.isna().sum()),
        "zero": int(x.eq(0).sum()),
        "one": int(x.eq(1).sum()),
        "other": int((x.notna() & ~x.isin([0, 1])).sum()),
    }


def find_code_refs():
    rows = []
    script_roots = [
        DG / "scripts",
        DG / "4New_cgm" / "scripts",
    ]
    for root in script_roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.py")):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            low = text.lower()
            if "antibi" not in low or "ppi" not in low:
                continue
            protocolish = any(x in low for x in (
                "protocol", "logging", "window", "during_logging"
            ))
            lines = []
            for i, line in enumerate(text.splitlines(), 1):
                ll = line.lower()
                if (
                    ("antibi" in ll or "ppi" in ll)
                    and any(k in ll for k in (
                        "protocol", "window", "logging",
                        "master", "covariate", "source",
                    ))
                ):
                    lines.append(f"{i}:{line.strip()}")
            rows.append({
                "script": str(p),
                "protocolish": protocolish,
                "matched_lines": " || ".join(lines[:12]),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search-root",
        default=str(ROOT),
        help="Root for targeted filename-based CSV search.",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=80,
    )
    args = ap.parse_args()

    search_root = Path(args.search_root).expanduser().resolve()

    # Targeted filename scan only: do not open unrelated CSVs.
    csvs = []
    for p in search_root.rglob("*.csv"):
        low = p.name.lower()
        if any(t in low for t in FILENAME_TERMS):
            csvs.append(p)

    rows = []
    for p in csvs[:10000]:
        cols = read_header(p)
        if not cols:
            continue
        score, reason, found = candidate_score(p, cols)
        if score <= 0 and not any(found.values()):
            continue
        rows.append({
            "score": score,
            "path": str(p),
            "n_columns": len(cols),
            "reason": reason,
            **found,
        })

    cand = pd.DataFrame(rows)
    if not cand.empty:
        cand = cand.sort_values(
            ["score", "path"], ascending=[False, True]
        ).reset_index(drop=True)

    print("=== STEP 18e0 PROTOCOL ABX/PPI SOURCE AUDIT ===")
    print("READ_ONLY=True")
    print("MODELS_FIT=False")
    print("COVARIATES_MODIFIED=False")
    print("BROAD_BASELINE_ATC_FALLBACK_ALLOWED=False")
    print(f"SEARCH_ROOT={search_root}")
    print()

    if cand.empty:
        print("CANDIDATE_FILES_FOUND=0")
        print("CANONICAL_PROTOCOL_SOURCE_AUTOSELECTED=False")
        print("ACTION=Need locate/rebuild validated protocol-window medication master before mediation.")
    else:
        print(f"CANDIDATE_FILES_FOUND={len(cand)}")
        print("\n--- TOP CANDIDATE FILES ---")
        print(cand.head(args.max_candidates).to_string(index=False))

        explicit = cand.loc[
            cand["abx_preferred"].astype(str).ne("")
            & cand["ppi_preferred"].astype(str).ne("")
        ].copy()

        selected = None
        if len(explicit) == 1:
            selected = Path(explicit.iloc[0]["path"])
        elif len(explicit) > 1:
            top_score = explicit["score"].max()
            top = explicit.loc[explicit["score"].eq(top_score)]
            if len(top) == 1:
                selected = Path(top.iloc[0]["path"])

        print()
        print(f"EXPLICIT_PROTOCOL_CANDIDATES={len(explicit)}")
        print(f"CANONICAL_PROTOCOL_SOURCE_AUTOSELECTED={selected is not None}")

        if selected is not None:
            cols = read_header(selected)
            abx = first_present(cols, PREFERRED_ABX)
            ppi = first_present(cols, PREFERRED_PPI)
            pid = first_present(cols, ("participant_id", "participantid", "user_id"))

            print(f"SELECTED_PROTOCOL_SOURCE={selected}")
            print(f"SELECTED_ANTIBIOTIC_COLUMN={abx}")
            print(f"SELECTED_PPI_COLUMN={ppi}")
            print(f"SELECTED_PARTICIPANT_ID_COLUMN={pid or 'MISSING'}")

            if pid:
                usecols = [pid, abx, ppi]
                d = pd.read_csv(selected, usecols=usecols, low_memory=False)
                d[pid] = norm_id(d[pid])
                dup = int(d[pid].duplicated().sum())
                print(f"ROWS={len(d)}")
                print(f"UNIQUE_PARTICIPANTS={d[pid].nunique()}")
                print(f"DUPLICATE_PARTICIPANT_IDS={dup}")

                a = summarize_flag(d[abx])
                q = summarize_flag(d[ppi])
                print(
                    "ANTIBIOTIC_COUNTS="
                    f"one:{a['one']} zero:{a['zero']} missing:{a['missing']} other:{a['other']}"
                )
                print(
                    "PPI_COUNTS="
                    f"one:{q['one']} zero:{q['zero']} missing:{q['missing']} other:{q['other']}"
                )

                if dup == 0 and a["other"] == 0 and q["other"] == 0:
                    print("SOURCE_STRUCTURE_QC=True")
                else:
                    print("SOURCE_STRUCTURE_QC=False")
            else:
                print("SOURCE_STRUCTURE_QC=False")
        else:
            print(
                "ACTION=Do not run Step18e mediation yet. "
                "Choose the previously validated protocol-window source explicitly."
            )

    refs = find_code_refs()
    print("\n--- EXISTING CODE REFERENCES ---")
    if refs.empty:
        print("NONE")
    else:
        print(refs.to_string(index=False))

    # Find prior protocol/medication QC artifacts by filename only.
    qcs = []
    for p in search_root.rglob("*.csv"):
        low = p.name.lower()
        if (
            ("15f3" in low or "protocol" in low)
            and any(x in low for x in ("med", "ppi", "missing", "qc", "covariate"))
        ):
            qcs.append(str(p))
    print("\n--- PRIOR PROTOCOL/MEDICATION QC FILES ---")
    if qcs:
        for p in sorted(qcs)[:100]:
            print(p)
    else:
        print("NONE")

    print("\nDECISION RULE:")
    print(
        "- Prefer the already-validated explicit during-logging/protocol-window flags.\n"
        "- Generic antibiotic_use/ppi_use is NOT accepted automatically unless provenance proves it is the protocol-window definition.\n"
        "- Broad baseline ATC positivity is not the canonical Step18e sensitivity."
    )


if __name__ == "__main__":
    main()
