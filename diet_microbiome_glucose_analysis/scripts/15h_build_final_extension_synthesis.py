#!/usr/bin/env python3
"""Step 15h — Final synthesis for the new-diet extension.

This script REFITS NO MODELS.

It summarizes the completed new-diet extension into manuscript-ready tables,
figures, and concise Results/Discussion drafts.

Outputs are written to:
diet_microbiome_glucose_analysis/outputs/new_diet_extension/final/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
DG = ROOT / "diet_microbiome_glucose_analysis"

NEW = DG / "outputs" / "new_diet_extension"
MED = NEW / "mediation"
INT = NEW / "integration"
REPORT = NEW / "reports"
FINAL = NEW / "final"
FINAL.mkdir(parents=True, exist_ok=True)

THREEWAY = MED / "15f3_threeway_path_robustness.csv"
CONTEXT_ROBUST = MED / "15f3_threeway_context_robustness.csv"
PRIMARY_MED = MED / "15f_primary_mediation_paths.csv"
HIGH_PATHS = INT / "15g_high_robust_new_diet_paths.csv"
HIGH_SPECIES = INT / "15g_high_robust_new_diet_species.csv"
OLD12_OVERLAP = INT / "15g_old12_vs_new_species_overlap.csv"
POLARITY8 = INT / "15g1_nova4_glucose_cv_old12_polarity_check_FIXED.csv"
CGM_M3 = REPORT / "15d3_bridge_eligible_diet_cgm_contexts.csv"

T1 = FINAL / "Table15H1_extension_overview.csv"
T2 = FINAL / "Table15H2_high_robust_paths.csv"
T3 = FINAL / "Table15H3_high_robust_species.csv"
T4 = FINAL / "Table15H4_old12_nova4_shared8.csv"
T5 = FINAL / "Table15H5_context_robustness.csv"

F1_PNG = FINAL / "Figure15H1_shared8_polarity_panel.png"
F1_SVG = FINAL / "Figure15H1_shared8_polarity_panel.svg"
F2_PNG = FINAL / "Figure15H2_robustness_flow.png"
F2_SVG = FINAL / "Figure15H2_robustness_flow.svg"
F3_PNG = FINAL / "Figure15H3_high_robust_context_counts.png"
F3_SVG = FINAL / "Figure15H3_high_robust_context_counts.svg"

RESULTS_MD = FINAL / "15h_results_draft.md"
DISCUSSION_MD = FINAL / "15h_discussion_draft.md"
SUMMARY_TXT = FINAL / "15h_final_extension_summary.txt"


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str).str.strip().str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def short_species(x: str) -> str:
    x = str(x)
    if "|s__" in x:
        x = x.rsplit("|s__", 1)[-1]
    x = x.replace("_", " ")
    replacements = {
        "Bifidobacterium adolescentis": "B. adolescentis",
        "Bifidobacterium bifidum": "B. bifidum",
        "Bifidobacterium catenulatum": "B. catenulatum",
        "Bifidobacterium longum": "B. longum",
        "Clostridium sp AF15 49": "Clostridium sp. AF15-49",
        "Lachnospiraceae unclassified SGB4882": "Lachnospiraceae SGB4882",
        "GGB9758 SGB15368": "GGB9758 SGB15368",
    }
    return replacements.get(x, x)


def savefig(fig, png: Path, svg: Path):
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def build_overview(three: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "analysis_stage": "Bridge candidates",
            "all_new_exposures": 45,
            "NOVA4": 20,
            "Carbohydrate_pct": 25,
            "interpretation": "Frozen robust Diet→microbiome→CGM overlap paths",
        },
        {
            "analysis_stage": "Primary mediation-style direction-consistent",
            "all_new_exposures": 30,
            "NOVA4": 14,
            "Carbohydrate_pct": 16,
            "interpretation": "Primary cross-sectional indirect-association result set",
        },
        {
            "analysis_stage": "Retained in strict diabetes/A10 sensitivity",
            "all_new_exposures": int(three["primary_consistent_retained_strict"].sum()),
            "NOVA4": int(
                three.loc[three["diet_score"].eq("NOVA4"),
                          "primary_consistent_retained_strict"].sum()
            ),
            "Carbohydrate_pct": int(
                three.loc[three["diet_score"].eq("Carbohydrate_pct"),
                          "primary_consistent_retained_strict"].sum()
            ),
            "interpretation": "Primary-consistent paths retained under strict cohort",
        },
        {
            "analysis_stage": "Retained in protocol-window ABX/PPI sensitivity",
            "all_new_exposures": int(three["primary_consistent_retained_protocol"].sum()),
            "NOVA4": int(
                three.loc[three["diet_score"].eq("NOVA4"),
                          "primary_consistent_retained_protocol"].sum()
            ),
            "Carbohydrate_pct": int(
                three.loc[three["diet_score"].eq("Carbohydrate_pct"),
                          "primary_consistent_retained_protocol"].sum()
            ),
            "interpretation": "Primary-consistent paths retained with protocol-window medications",
        },
        {
            "analysis_stage": "Retained in BOTH sensitivities",
            "all_new_exposures": int(three["primary_consistent_retained_both"].sum()),
            "NOVA4": int(high["diet_score"].eq("NOVA4").sum()),
            "Carbohydrate_pct": int(high["diet_score"].eq("Carbohydrate_pct").sum()),
            "interpretation": "Highest-robustness subset; does not replace primary set",
        },
    ])


def figure_shared8(p8: pd.DataFrame):
    d = p8.copy().sort_values("species_label")
    d["display_species"] = d["species_label"].map(short_species)

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.axis("off")

    headers = [
        ("Species", 0.02),
        ("Old healthy-diet class", 0.34),
        ("NOVA4→taxon β", 0.64),
        ("Taxon→glucose CV β", 0.79),
        ("Mirror", 0.95),
    ]
    for text, x in headers:
        ax.text(x, 0.96, text, transform=ax.transAxes,
                fontsize=11, fontweight="bold")

    y0, dy = 0.87, 0.10
    for i, (_, r) in enumerate(d.iterrows()):
        y = y0 - i * dy
        cls = str(r["old_direction_class"])
        if "depleted_relative__higher_glucose_cv_taxon" in cls:
            old_text = "healthy diet ↓ relative; higher-CV taxon"
        elif "enriched_relative__lower_glucose_cv" in cls:
            old_text = "healthy diet ↑ relative; lower-CV taxon"
        else:
            old_text = cls

        ax.text(0.02, y, r["display_species"], transform=ax.transAxes, fontsize=10)
        ax.text(0.34, y, old_text, transform=ax.transAxes, fontsize=9)
        ax.text(0.67, y, f"{float(r['model2_beta_diet']):+.3f}",
                transform=ax.transAxes, fontsize=10, ha="center")
        ax.text(0.83, y, f"{float(r['model2_beta_cgm']):+.3f}",
                transform=ax.transAxes, fontsize=10, ha="center")
        mirror = "Yes" if bool(r["healthy_vs_NOVA4_polarity_concordant"]) else "No"
        ax.text(0.95, y, mirror, transform=ax.transAxes, fontsize=10, ha="center")

    ax.text(
        0.02, 0.035,
        "All 8 shared taxa show mirror diet→taxon polarity and concordant taxon→glucose-CV direction.",
        transform=ax.transAxes, fontsize=10
    )
    ax.set_title(
        "Shared old-core taxa: healthy-diet vs NOVA4 mirror-polarity pattern",
        fontsize=14, pad=12
    )
    fig.tight_layout()
    savefig(fig, F1_PNG, F1_SVG)


def figure_flow():
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axis("off")

    items = [
        ("45", "robust bridge paths"),
        ("30", "primary direction-consistent\nmediation-style paths"),
        ("23", "retained in strict\ndiabetes/A10 sensitivity"),
        ("28", "retained with protocol-window\nantibiotic/PPI adjustment"),
        ("22", "retained in BOTH\nsensitivities"),
    ]
    xs = np.linspace(0.08, 0.92, len(items))

    for i, ((n, label), x) in enumerate(zip(items, xs)):
        ax.text(x, 0.62, n, transform=ax.transAxes,
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(x, 0.46, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
        if i < len(items) - 1:
            ax.annotate(
                "",
                xy=(xs[i+1] - 0.055, 0.62),
                xytext=(x + 0.055, 0.62),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops={"arrowstyle": "->"},
            )

    ax.text(
        0.5, 0.12,
        "Primary result set remains 30 paths; 22 paths are the highest-robustness subset.",
        transform=ax.transAxes, ha="center", fontsize=10
    )
    ax.set_title("New-diet extension robustness flow", fontsize=14, pad=12)
    fig.tight_layout()
    savefig(fig, F2_PNG, F2_SVG)


def figure_context_counts(high: pd.DataFrame):
    ctx = (
        high.groupby(["diet_score", "cgm_outcome"], as_index=False)
        .size()
        .rename(columns={"size": "N"})
    )
    order = [
        ("NOVA4", "glucose_cv"),
        ("NOVA4", "mean_glucose"),
        ("NOVA4", "time_above_140"),
        ("Carbohydrate_pct", "mean_glucose"),
        ("Carbohydrate_pct", "glucose_cv"),
        ("Carbohydrate_pct", "time_above_140"),
    ]
    lookup = {(r["diet_score"], r["cgm_outcome"]): int(r["N"])
              for _, r in ctx.iterrows()}
    vals = [lookup.get(k, 0) for k in order]
    labels = [
        "NOVA4\nGlucose CV",
        "NOVA4\nMean glucose",
        "NOVA4\nTime >140",
        "Carbohydrate %\nMean glucose",
        "Carbohydrate %\nGlucose CV",
        "Carbohydrate %\nTime >140",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(labels, vals)
    ax.set_ylabel("High-robustness paths")
    ax.set_title("High-robustness paths by exposure and CGM phenotype")
    ax.set_ylim(0, max(vals) + 2 if vals else 2)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.15,
            str(v),
            ha="center", va="bottom", fontsize=10
        )

    fig.tight_layout()
    savefig(fig, F3_PNG, F3_SVG)


def write_markdown(high: pd.DataFrame):
    nova_high = int(high["diet_score"].eq("NOVA4").sum())
    carb_high = int(high["diet_score"].eq("Carbohydrate_pct").sum())
    unique_high = int(high["species"].nunique())

    results = f"""# New-diet extension: Results draft

## New dietary exposures and downstream gating

Three additional dietary exposures were evaluated: modified EAT-Lancet-13 (EAT13), NOVA4 ultra-processed-food energy share, and carbohydrate energy percentage. EAT13 showed microbiome associations but did not retain a stable Model2 primary Diet–CGM gate and therefore did not advance to bridge or mediation-style analyses. NOVA4 was retained as a primary extension, whereas carbohydrate percentage remained an exploratory macronutrient-composition exposure.

## Bridge and mediation-style results

Across NOVA4 and carbohydrate percentage, 45 robust Diet→microbiome→CGM bridge paths were identified. In the primary cross-sectional mediation-style analysis, 30 of these paths were FDR-significant and direction-consistent. Of the 30 primary-consistent paths, 23 were retained under the strict known-nondiabetes/A10-negative sensitivity analysis and 28 were retained after adjustment for protocol-window antibiotic and proton-pump-inhibitor use. Twenty-two primary-consistent paths were retained in both sensitivity analyses; these involved {unique_high} unique exact species.

NOVA4 contributed {nova_high} of the 22 highest-robustness paths. Ten involved glucose variability, one involved mean glucose, and one involved time above 140 mg/dL. Carbohydrate percentage contributed {carb_high} highest-robustness paths, all involving mean glucose.

## Integration with the original AMED/hPDI core

Eight of the original 12 recurrent AMED/hPDI glucose-variability core species were recovered among the highest-robustness NOVA4 paths. No original core species was recovered among the highest-robustness carbohydrate-percentage paths. For all eight shared NOVA4–old-core species, the NOVA4 diet→taxon association was the mirror of the old healthy-diet direction, and the taxon→glucose-variability direction was concordant across analyses (8/8 for both checks).

The shared taxa were Bifidobacterium adolescentis, Bifidobacterium bifidum, Bifidobacterium catenulatum, Bifidobacterium longum, Blautia massiliensis, Clostridium sp. AF15-49, GGB9758 SGB15368, and Lachnospiraceae SGB4882.

## Statistical interpretation

The mediation-style analyses are cross-sectional indirect-association analyses. They should not be interpreted as evidence of causal or temporal mediation. Microbiome effects are based on CLR-Z transformed relative composition and therefore do not imply absolute taxon abundance changes.
"""
    RESULTS_MD.write_text(results, encoding="utf-8")

    discussion = """# New-diet extension: Discussion draft

## Main interpretation

The strongest new result is the extension of the original healthy-diet–microbiome–glucose-variability axis to ultra-processed-food exposure. NOVA4 did not merely identify a separate set of microbiome associations: eight of the original 12 recurrent AMED/hPDI glucose-variability core taxa were recovered among the highest-robustness NOVA4 paths. For each of these eight taxa, the direction of the NOVA4→taxon association was opposite to the direction expected from the healthier-diet analyses, while the taxon→glucose-variability association retained the same orientation.

This pattern is consistent with a shared, polarity-opposed microbiome compositional axis linking healthier dietary patterns and ultra-processed-food exposure with glucose variability. The result is strengthened by the fact that carbohydrate percentage did not reproduce the old 12-core species, despite showing strong exploratory associations with CGM phenotypes.

## NOVA4

NOVA4 showed the most coherent primary extension. Its Diet→CGM associations were retained across mean glucose, glucose variability, and time above 140 mg/dL, and the downstream bridge/mediation-style signal was concentrated in glucose variability. Ten NOVA4 glucose-variability paths remained direction-consistent in both the strict diabetes/A10 and protocol-window medication sensitivities.

## Carbohydrate percentage

Carbohydrate percentage showed strong exploratory direct associations with all three CGM phenotypes, but its most robust indirect-association signal was concentrated in mean glucose. Its glucose-variability mediation-style paths were sensitive to the strict diabetes/A10 cohort definition, and none of the original AMED/hPDI 12-core species appeared among its highest-robustness paths.

## EAT13

EAT13 showed microbiome-wide associations but did not retain a stable primary Model2 Diet→CGM gate. It was therefore not advanced into bridge or mediation-style analyses. This negative downstream gating result should be retained rather than overridden by later sensitivity-only significance.

## Limitations

The analyses are observational and cross-sectional with respect to the mediation-style models, so causal or temporal mediation cannot be inferred. CLR-Z transformation represents relative compositional changes rather than absolute abundance. The strongest biological interpretation should therefore focus on reproducible direction patterns, species recurrence, and robustness across prespecified sensitivities rather than on causal microbial mechanisms.
"""
    DISCUSSION_MD.write_text(discussion, encoding="utf-8")


def main() -> int:
    for p, label in [
        (THREEWAY, "Step15f3 three-way robustness"),
        (CONTEXT_ROBUST, "Step15f3 context robustness"),
        (PRIMARY_MED, "Step15f primary mediation"),
        (HIGH_PATHS, "Step15g high-robust paths"),
        (HIGH_SPECIES, "Step15g high-robust species"),
        (OLD12_OVERLAP, "Step15g old12 overlap"),
        (POLARITY8, "Step15g1 polarity verification"),
        (CGM_M3, "Step15d3 bridge-eligible contexts"),
    ]:
        require(p, label)

    three = pd.read_csv(THREEWAY, low_memory=False)
    context = pd.read_csv(CONTEXT_ROBUST, low_memory=False)
    primary = pd.read_csv(PRIMARY_MED, low_memory=False)
    high = pd.read_csv(HIGH_PATHS, low_memory=False)
    high_species = pd.read_csv(HIGH_SPECIES, low_memory=False)
    old_overlap = pd.read_csv(OLD12_OVERLAP, low_memory=False)
    p8 = pd.read_csv(POLARITY8, low_memory=False)

    # Hard final QC.
    if len(primary) != 45:
        raise RuntimeError(f"Expected 45 tested paths, found {len(primary)}")
    if int(truthy(primary["candidate_mediator_consistent"]).sum()) != 30:
        raise RuntimeError("Expected 30 primary direction-consistent paths")
    if int(three["primary_consistent_retained_both"].sum()) != 22:
        raise RuntimeError("Expected 22 paths retained in both sensitivities")
    if len(high) != 22:
        raise RuntimeError(f"Expected 22 high-robust paths, found {len(high)}")
    if high["species"].nunique() != 19:
        raise RuntimeError(
            f"Expected 19 unique high-robust species, found {high['species'].nunique()}"
        )
    if len(p8) != 8:
        raise RuntimeError(f"Expected 8 shared taxa, found {len(p8)}")
    if int(truthy(p8["healthy_vs_NOVA4_polarity_concordant"]).sum()) != 8:
        raise RuntimeError("Expected 8/8 healthy-vs-NOVA4 polarity concordance")
    if int(truthy(p8["old_vs_new_species_CGM_direction_concordant"]).sum()) != 8:
        raise RuntimeError("Expected 8/8 species-CGM direction concordance")

    overview = build_overview(three, high)
    overview.to_csv(T1, index=False)
    high.to_csv(T2, index=False)
    high_species.to_csv(T3, index=False)
    p8.to_csv(T4, index=False)
    context.to_csv(T5, index=False)

    figure_shared8(p8)
    figure_flow()
    figure_context_counts(high)
    write_markdown(high)

    old_shared_any = int(
        truthy(old_overlap["present_in_new_high_robust"]).sum()
    )
    old_shared_nova = int(
        truthy(old_overlap["present_in_nova4_high_robust"]).sum()
    )
    old_shared_carb = int(
        truthy(old_overlap["present_in_carbohydrate_high_robust"]).sum()
    )

    summary_lines = [
        "=== STEP 15h FINAL NEW-DIET EXTENSION SYNTHESIS ===",
        "NO_MODELS_REFIT=True",
        "STATISTICAL_EXPLORATION_FROZEN=True",
        "",
        "PRIMARY_MEDIATION_STYLE_PATHS=30",
        f"HIGH_ROBUST_PATHS_BOTH_SENSITIVITIES={len(high)}",
        f"HIGH_ROBUST_UNIQUE_EXACT_SPECIES={high['species'].nunique()}",
        f"NOVA4_HIGH_ROBUST_PATHS={int(high['diet_score'].eq('NOVA4').sum())}",
        f"CARBOHYDRATE_HIGH_ROBUST_PATHS={int(high['diet_score'].eq('Carbohydrate_pct').sum())}",
        "",
        f"OLD12_SHARED_ANY_NEW_HIGH_ROBUST={old_shared_any}/12",
        f"OLD12_SHARED_NOVA4_HIGH_ROBUST={old_shared_nova}/12",
        f"OLD12_SHARED_CARBOHYDRATE_HIGH_ROBUST={old_shared_carb}/12",
        "OLD12_NOVA4_GLUCOSE_CV_POLARITY_CONCORDANCE=8/8",
        "OLD12_NOVA4_SPECIES_CGM_DIRECTION_CONCORDANCE=8/8",
        "",
        "MAIN_INTERPRETATION:",
        "- NOVA4 extends the original AMED/hPDI glucose-variability microbiome axis.",
        "- The 8 shared taxa show mirror healthy-diet vs NOVA4 diet→taxon directions.",
        "- Carbohydrate_pct shows a distinct exploratory signal, concentrated in mean glucose.",
        "- EAT13 did not pass the stable primary Diet→CGM gate and was not advanced.",
        "- Mediation-style analyses are cross-sectional and non-causal.",
        "- CLR-Z effects are compositional/relative, not absolute abundance changes.",
        "",
        f"TABLE1={T1}",
        f"TABLE2={T2}",
        f"TABLE3={T3}",
        f"TABLE4={T4}",
        f"TABLE5={T5}",
        f"FIGURE1={F1_PNG}",
        f"FIGURE2={F2_PNG}",
        f"FIGURE3={F3_PNG}",
        f"RESULTS_DRAFT={RESULTS_MD}",
        f"DISCUSSION_DRAFT={DISCUSSION_MD}",
    ]

    SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
