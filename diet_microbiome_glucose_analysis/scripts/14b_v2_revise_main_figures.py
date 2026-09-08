#!/usr/bin/env python3
"""Step 14b-v2 — revise the three manuscript-facing figures after visual review.

READ-ONLY. No model is refit.

Revises only:
    B1 primary context summary
    B2 robustness summary
    B3 AMED-vs-hPDI 12-core indirect effects

The original Step14B outputs are left untouched.

Inputs
------
outputs/final/
    Table14A_mediation_overview.csv
    Table14B_robustness_overview.csv
    Table14C_12_core_species.csv

Outputs
-------
outputs/final/figures/
    Figure14B1_primary_context_summary_v2.png/.svg
    Figure14B2_robustness_counts_v2.png/.svg
    Figure14B3_core_AMED_vs_hPDI_indirect_v2.png/.svg
    Step14B_v2_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "diet_microbiome_glucose_analysis"
FINAL_DIR = PROJECT / "outputs" / "final"
FIG_DIR = FINAL_DIR / "figures"

TABLE_A = FINAL_DIR / "Table14A_mediation_overview.csv"
TABLE_B = FINAL_DIR / "Table14B_robustness_overview.csv"
TABLE_C = FINAL_DIR / "Table14C_12_core_species.csv"

SUMMARY = FIG_DIR / "Step14B_v2_summary.txt"


def read_required(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    return pd.read_csv(path, low_memory=False)


def require_cols(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def save_both(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def short_species(label: str) -> str:
    s = str(label)
    replacements = {
        "Bifidobacterium_": "B. ",
        "Blautia_": "Blautia ",
        "Enterocloster_": "E. ",
        "Clostridium_sp_": "Clostridium sp. ",
        "Lachnospiraceae_unclassified_": "Lachnospiraceae ",
    }
    for old, new in replacements.items():
        if s.startswith(old):
            s = new + s[len(old):]
            break
    return s.replace("_", " ")


def spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(
        x.loc[ok].rank(method="average").corr(
            y.loc[ok].rank(method="average"),
            method="pearson",
        )
    )


def identity_limits(x: pd.Series, y: pd.Series):
    vals = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")]
    ).dropna()
    if len(vals) == 0:
        return -1.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.08 if hi != lo else (abs(lo) * 0.1 or 0.1)
    return lo - pad, hi + pad


def figure_b1(a: pd.DataFrame):
    require_cols(
        a,
        [
            "diet_score",
            "cgm_outcome",
            "tested_paths",
            "FDR_significant_paths",
            "direction_consistent_paths",
        ],
        "Table14A",
    )

    labels = [
        f"{r.diet_score}\n{r.cgm_outcome}"
        for r in a.itertuples(index=False)
    ]
    x = np.arange(len(a), dtype=float)
    width = 0.24

    fig = plt.figure(figsize=(9.2, 5.4))
    ax = fig.add_subplot(111)

    bars1 = ax.bar(x - width, a["tested_paths"], width, label="Tested")
    bars2 = ax.bar(x, a["FDR_significant_paths"], width, label="FDR significant")
    bars3 = ax.bar(
        x + width,
        a["direction_consistent_paths"],
        width,
        label="Direction-consistent",
    )

    for bars in [bars1, bars2, bars3]:
        ax.bar_label(bars, padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of candidate paths")
    ax.set_title("Primary mediation-style results by diet–CGM context")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return save_both(fig, "Figure14B1_primary_context_summary_v2")


def figure_b2(b: pd.DataFrame):
    require_cols(
        b,
        [
            "analysis",
            "FDR_significant_paths",
            "direction_consistent_paths",
            "primary_significant_retained",
            "primary_direction_consistent_retained",
        ],
        "Table14B",
    )

    order = [
        "Primary",
        "Strict diabetes/A10",
        "Broad baseline antibiotic/PPI",
        "Protocol-window antibiotic/PPI",
    ]
    b = (
        b.set_index("analysis")
        .loc[order]
        .reset_index()
    )

    display = {
        "Primary": "Primary",
        "Strict diabetes/A10": "Strict diabetes/A10",
        "Broad baseline antibiotic/PPI": "Broad baseline ABX/PPI",
        "Protocol-window antibiotic/PPI": "Protocol-window ABX/PPI",
    }

    y = np.arange(len(b), dtype=float)

    fig = plt.figure(figsize=(9.4, 5.2))
    ax = fig.add_subplot(111)

    ax.scatter(
        b["FDR_significant_paths"],
        y,
        s=65,
        label="FDR significant",
    )
    ax.scatter(
        b["direction_consistent_paths"],
        y,
        s=65,
        label="Direction-consistent",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([display[x] for x in b["analysis"]])
    ax.invert_yaxis()
    ax.set_xlim(24.5, 30.5)
    ax.set_xticks(range(25, 31))
    ax.set_xlabel("Number of paths")
    ax.set_title("Robustness of mediation-style results")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    for i, row in b.iterrows():
        if row["analysis"] == "Primary":
            note = "reference"
        else:
            note = (
                f"retained {int(row['primary_significant_retained'])}/29; "
                f"{int(row['primary_direction_consistent_retained'])}/26"
            )
        ax.text(
            30.45,
            i,
            note,
            ha="right",
            va="center",
            fontsize=8.5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return save_both(fig, "Figure14B2_robustness_counts_v2")


def figure_b3(c: pd.DataFrame):
    require_cols(
        c,
        ["species_label", "AMED_indirect", "hPDI_indirect"],
        "Table14C",
    )

    x = pd.to_numeric(c["AMED_indirect"], errors="coerce")
    y = pd.to_numeric(c["hPDI_indirect"], errors="coerce")
    ok = x.notna() & y.notna()
    cc = c.loc[ok].reset_index(drop=True)
    x = x.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)

    if len(cc) != 12:
        raise RuntimeError(f"Expected 12 core species, found {len(cc)}")

    fig = plt.figure(figsize=(11.2, 6.8))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.69)

    ax.scatter(x, y, s=55)

    lo, hi = identity_limits(x, y)
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    for i, (xx, yy) in enumerate(zip(x, y), start=1):
        ax.annotate(
            str(i),
            (xx, yy),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("AMED indirect effect")
    ax.set_ylabel("hPDI indirect effect")
    ax.set_title("Shared 12-core indirect effects: AMED vs hPDI")

    rho = spearman(x, y)
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    key_lines = []
    for i, label in enumerate(cc["species_label"], start=1):
        key_lines.append(f"{i}. {short_species(label)}")
    fig.text(
        0.71,
        0.90,
        "\n".join(key_lines),
        ha="left",
        va="top",
        fontsize=8.6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return save_both(fig, "Figure14B3_core_AMED_vs_hPDI_indirect_v2")


def main() -> int:
    a = read_required(TABLE_A, "Table14A")
    b = read_required(TABLE_B, "Table14B")
    c = read_required(TABLE_C, "Table14C")

    outputs = []
    for name, fn in [
        ("B1", lambda: figure_b1(a)),
        ("B2", lambda: figure_b2(b)),
        ("B3", lambda: figure_b3(c)),
    ]:
        png, svg = fn()
        outputs.append((name, png, svg))

    lines = [
        "=== STEP 14B-v2 REVISED MANUSCRIPT FIGURES ===",
        "NO_MODELS_REFIT=True",
        "FIGURES_REVISED=3",
        "",
    ]
    for name, png, svg in outputs:
        lines.append(f"{name}_PNG={png}")
        lines.append(f"{name}_SVG={svg}")
    lines += [
        "",
        "Review decisions implemented:",
        "- B1: added value labels; retained grouped-bar design.",
        "- B2: replaced crowded bars with horizontal dot summary.",
        "- B3: replaced overlapping species labels with numbered points + side key.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
