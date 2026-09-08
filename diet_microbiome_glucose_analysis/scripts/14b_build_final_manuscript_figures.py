#!/usr/bin/env python3
"""Step 14b — build final manuscript figures from finalized outputs.

READ-ONLY: no model is refit and no canonical result is modified.

Inputs
------
outputs/final/
    Table14A_mediation_overview.csv
    Table14B_robustness_overview.csv
    Table14C_12_core_species.csv
    Table14D_primary_31_paths.csv

outputs/models/
    10_mediation_paths_model3.csv
    10_strict_mediation_paths_model3.csv
    12b_antibiotic_ppi_mediation_paths_model3.csv
    12e_protocol_window_mediation_paths_model3.csv

Outputs
-------
outputs/final/figures/
    Figure14B1_primary_context_summary.png/.svg
    Figure14B2_robustness_counts.png/.svg
    Figure14B3_core_AMED_vs_hPDI_indirect.png/.svg
    Figure14B4_primary_vs_strict_indirect.png/.svg
    Figure14B5_primary_vs_broad_abxppi_indirect.png/.svg
    Figure14B6_primary_vs_protocol_abxppi_indirect.png/.svg
    Step14B_figure_manifest.csv
    Step14B_figure_summary.txt

Design rules
------------
- matplotlib only; no seaborn.
- one figure per plot (no subplots).
- no custom color palette; matplotlib defaults are used.
- PNG is saved at 300 dpi, with SVG for vector editing.
- Exact 31-path match is QC only, not a robustness claim.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise RuntimeError(
        "matplotlib is required for Step 14b. "
        "Use the existing pheno environment's matplotlib installation."
    ) from e


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "diet_microbiome_glucose_analysis"
FINAL_DIR = PROJECT / "outputs" / "final"
MODEL_DIR = PROJECT / "outputs" / "models"
FIG_DIR = FINAL_DIR / "figures"

TABLE_A = FINAL_DIR / "Table14A_mediation_overview.csv"
TABLE_B = FINAL_DIR / "Table14B_robustness_overview.csv"
TABLE_C = FINAL_DIR / "Table14C_12_core_species.csv"
TABLE_D = FINAL_DIR / "Table14D_primary_31_paths.csv"

PRIMARY = MODEL_DIR / "10_mediation_paths_model3.csv"
STRICT = MODEL_DIR / "10_strict_mediation_paths_model3.csv"
BROAD = MODEL_DIR / "12b_antibiotic_ppi_mediation_paths_model3.csv"
PROTOCOL = MODEL_DIR / "12e_protocol_window_mediation_paths_model3.csv"

MANIFEST = FIG_DIR / "Step14B_figure_manifest.csv"
SUMMARY = FIG_DIR / "Step14B_figure_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]


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
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    vals = pd.concat([xv, yv]).dropna()
    if len(vals) == 0:
        return -1, 1
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        pad = abs(lo) * 0.1 if lo != 0 else 0.1
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def figure_context_summary(a: pd.DataFrame):
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
    xlabels = [
        f"{r.diet_score}\n{r.cgm_outcome}"
        for r in a.itertuples(index=False)
    ]
    x = np.arange(len(a), dtype=float)
    width = 0.24

    fig = plt.figure(figsize=(9.2, 5.4))
    ax = fig.add_subplot(111)
    ax.bar(x - width, a["tested_paths"], width, label="Tested paths")
    ax.bar(x, a["FDR_significant_paths"], width, label="FDR significant")
    ax.bar(
        x + width,
        a["direction_consistent_paths"],
        width,
        label="Direction-consistent",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Number of mediation paths")
    ax.set_title("Primary mediation-style results by diet–CGM context")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_both(fig, "Figure14B1_primary_context_summary")


def figure_robustness_counts(b: pd.DataFrame):
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

    label_map = {
        "Primary": "Primary",
        "Strict diabetes/A10": "Strict\nDM/A10",
        "Broad baseline antibiotic/PPI": "Broad\nABX/PPI",
        "Protocol-window antibiotic/PPI": "Protocol\nABX/PPI",
    }
    xlabels = [label_map.get(str(x), str(x)) for x in b["analysis"]]
    x = np.arange(len(b), dtype=float)
    width = 0.33

    fig = plt.figure(figsize=(9.0, 5.4))
    ax = fig.add_subplot(111)
    ax.bar(
        x - width / 2,
        b["FDR_significant_paths"],
        width,
        label="FDR significant",
    )
    ax.bar(
        x + width / 2,
        b["direction_consistent_paths"],
        width,
        label="Direction-consistent",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, max(31, int(b["FDR_significant_paths"].max()) + 3))
    ax.set_ylabel("Number of paths")
    ax.set_title("Robustness of mediation-style results")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Put retained-primary counts above bars as a compact QC annotation.
    for i, row in b.reset_index(drop=True).iterrows():
        if row["analysis"] != "Primary":
            ax.text(
                i,
                max(
                    row["FDR_significant_paths"],
                    row["direction_consistent_paths"],
                ) + 0.8,
                f"retained {int(row['primary_significant_retained'])}/29; "
                f"{int(row['primary_direction_consistent_retained'])}/26",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    return save_both(fig, "Figure14B2_robustness_counts")


def figure_core_indirect(c: pd.DataFrame):
    require_cols(
        c,
        ["species_label", "AMED_indirect", "hPDI_indirect"],
        "Table14C",
    )
    x = pd.to_numeric(c["AMED_indirect"], errors="coerce")
    y = pd.to_numeric(c["hPDI_indirect"], errors="coerce")
    ok = x.notna() & y.notna()
    x, y = x.loc[ok], y.loc[ok]
    cc = c.loc[ok].reset_index(drop=True)

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=38)

    lo, hi = identity_limits(x, y)
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("AMED indirect effect")
    ax.set_ylabel("hPDI indirect effect")
    ax.set_title("Shared 12-core indirect effects: AMED vs hPDI")

    for i, row in cc.iterrows():
        ax.annotate(
            short_species(row["species_label"]),
            (x.iloc[i], y.iloc[i]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.2,
        )

    rho = spearman(x, y)
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_both(fig, "Figure14B3_core_AMED_vs_hPDI_indirect")


def aligned_effects(reference: pd.DataFrame, variant: pd.DataFrame):
    for label, d in [("reference", reference), ("variant", variant)]:
        require_cols(d, KEY + ["indirect_effect"], label)
        if d.duplicated(KEY).any():
            raise RuntimeError(f"{label} has duplicate exact paths")

    m = reference[KEY + ["indirect_effect"]].merge(
        variant[KEY + ["indirect_effect"]],
        on=KEY,
        suffixes=("_primary", "_variant"),
        how="inner",
        validate="one_to_one",
    )
    if len(m) != 31:
        raise RuntimeError(
            f"Expected 31 aligned fixed paths; found {len(m)}"
        )
    return m


def figure_primary_vs_variant(
    primary: pd.DataFrame,
    variant: pd.DataFrame,
    title: str,
    stem: str,
):
    m = aligned_effects(primary, variant)
    x = pd.to_numeric(m["indirect_effect_primary"], errors="coerce")
    y = pd.to_numeric(m["indirect_effect_variant"], errors="coerce")
    ok = x.notna() & y.notna()
    x, y = x.loc[ok], y.loc[ok]

    fig = plt.figure(figsize=(6.7, 6.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=32)

    lo, hi = identity_limits(x, y)
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Primary indirect effect")
    ax.set_ylabel("Sensitivity indirect effect")
    ax.set_title(title)

    rho = spearman(x, y)
    sign_flips = int(
        (
            np.sign(x.to_numpy())
            != np.sign(y.to_numpy())
        ).sum()
    )
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.3f}\nSign flips = {sign_flips}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_both(fig, stem)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    a = read_required(TABLE_A, "Table14A")
    b = read_required(TABLE_B, "Table14B")
    c = read_required(TABLE_C, "Table14C")
    _ = read_required(TABLE_D, "Table14D")

    primary = read_required(PRIMARY, "Primary mediation")
    strict = read_required(STRICT, "Strict mediation")
    broad = read_required(BROAD, "Broad ABX/PPI mediation")
    protocol = read_required(PROTOCOL, "Protocol ABX/PPI mediation")

    outputs = []

    for figure_name, maker in [
        (
            "Figure14B1_primary_context_summary",
            lambda: figure_context_summary(a),
        ),
        (
            "Figure14B2_robustness_counts",
            lambda: figure_robustness_counts(b),
        ),
        (
            "Figure14B3_core_AMED_vs_hPDI_indirect",
            lambda: figure_core_indirect(c),
        ),
        (
            "Figure14B4_primary_vs_strict_indirect",
            lambda: figure_primary_vs_variant(
                primary,
                strict,
                "Indirect-effect stability: Primary vs strict DM/A10",
                "Figure14B4_primary_vs_strict_indirect",
            ),
        ),
        (
            "Figure14B5_primary_vs_broad_abxppi_indirect",
            lambda: figure_primary_vs_variant(
                primary,
                broad,
                "Indirect-effect stability: Primary vs broad ABX/PPI",
                "Figure14B5_primary_vs_broad_abxppi_indirect",
            ),
        ),
        (
            "Figure14B6_primary_vs_protocol_abxppi_indirect",
            lambda: figure_primary_vs_variant(
                primary,
                protocol,
                "Indirect-effect stability: Primary vs protocol-window ABX/PPI",
                "Figure14B6_primary_vs_protocol_abxppi_indirect",
            ),
        ),
    ]:
        png, svg = maker()
        outputs.append(
            {
                "figure": figure_name,
                "png": str(png),
                "svg": str(svg),
            }
        )

    manifest = pd.DataFrame(outputs)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")

    lines = [
        "=== STEP 14B FINAL FIGURE BUILD ===",
        "NO_MODELS_REFIT=True",
        f"FIGURES_CREATED={len(manifest)}",
        "",
        manifest.to_string(index=False),
        "",
        "NOTES:",
        "- One chart per figure; no subplots.",
        "- PNG saved at 300 dpi; SVG saved for vector editing.",
        "- Exact 31-path match is pipeline QC only.",
        "- Mediation-style results are not causal mediation.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
