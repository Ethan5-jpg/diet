#!/usr/bin/env python3
"""Step 13b — project-direction QC and biological grouping of the 12-core.

This is a READ-ONLY interpretation/QC step. No model is refit.

Why this step exists
--------------------
Step 13a showed an important structure:
  * all 12 recurrent AMED/hPDI × glucose_cv core taxa have negative indirect effects;
  * however, the component signs split into two mirrored patterns:
      (+ a, - b) versus (- a, + b).

Before writing biological discussion, this script:
  1) verifies the mediation a/b signs against the carried-forward Step09/Model3
     bridge betas wherever available;
  2) classifies each species into a project direction class;
  3) explicitly interprets CLR signs as log-ratio / relative signals, NOT
     absolute abundance changes;
  4) replaces the overly strong "opposite to literature" wording with a more
     conservative biological-evidence category.

Inputs
------
outputs/models/10_mediation_paths_model3.csv
outputs/reports/13a_12_core_species_biological_annotation.csv

Outputs
-------
outputs/reports/
    13b_12_core_direction_qc_long.csv
    13b_12_core_direction_classes.csv
    13b_12_core_direction_classes.md
    13b_12_core_direction_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
PROJECT = ROOT / "diet_microbiome_glucose_analysis"
MODEL_DIR = PROJECT / "outputs" / "models"
REPORT_DIR = PROJECT / "outputs" / "reports"

PRIMARY = MODEL_DIR / "10_mediation_paths_model3.csv"
ANNOT = REPORT_DIR / "13a_12_core_species_biological_annotation.csv"

OUT_LONG = REPORT_DIR / "13b_12_core_direction_qc_long.csv"
OUT_CLASS = REPORT_DIR / "13b_12_core_direction_classes.csv"
OUT_MD = REPORT_DIR / "13b_12_core_direction_classes.md"
OUT_SUMMARY = REPORT_DIR / "13b_12_core_direction_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]


# Conservative interpretation categories.  These do not overwrite or change any
# statistical result.  They only govern discussion language.
BIO_GROUP = {
    "Bifidobacterium_adolescentis": {
        "category": "literature_tension_not_direct_contradiction",
        "discussion": (
            "Project CLR pattern is inverse to the common bifidogenic/favorable-metabolic "
            "orientation in the literature. Because CLR is a relative log-ratio signal and "
            "external studies often use prebiotic interventions, disease status, or other "
            "glycemic endpoints, treat this as context-dependent tension rather than a contradiction."
        ),
    },
    "Bifidobacterium_bifidum": {
        "category": "literature_tension_not_direct_contradiction",
        "discussion": (
            "Project CLR pattern is inverse to common prebiotic/probiotic expectations. "
            "Do not label the taxon harmful; the project estimate is relative and conditional, "
            "whereas external evidence is strain- and endpoint-specific."
        ),
    },
    "Bifidobacterium_catenulatum": {
        "category": "literature_tension_not_direct_contradiction",
        "discussion": (
            "Known fiber responsiveness and cross-feeding plausibility do not map directly onto "
            "the project's relative CLR direction or conditional glucose-CV coefficient. "
            "Treat the sign difference as a context-dependent finding."
        ),
    },
    "Bifidobacterium_longum": {
        "category": "literature_tension_not_direct_contradiction",
        "discussion": (
            "External prospective/probiotic evidence generally points toward favorable glycemic "
            "associations, whereas the project shows the mirrored CLR pattern. This is important "
            "discussion material but is not proof of a biological reversal."
        ),
    },
    "Blautia_massiliensis": {
        "category": "mixed_context_dependent",
        "discussion": (
            "Blautia/B. massiliensis literature is metabolically mixed. The project pattern should "
            "be reported without assigning a universally beneficial or adverse label."
        ),
    },
    "Enterocloster_bolteae": {
        "category": "directly_plausible_adverse_orientation",
        "discussion": (
            "Project pattern—lower relative abundance with higher healthy-diet scores and positive "
            "association with glucose variability—is qualitatively consistent with human reports "
            "linking E. bolteae to T2D or poorer insulin sensitivity."
        ),
    },
    "GGB3118_SGB4130": {
        "category": "novel_low_evidence",
        "discussion": (
            "Direct species-level biological evidence is insufficient. Treat as a robust novel "
            "marker rather than infer mechanism from neighboring taxa."
        ),
    },
    "Clostridium_sp_AF15_49": {
        "category": "novel_low_evidence",
        "discussion": (
            "Species-specific diet/glucose biology remains sparse; retain as exploratory core marker."
        ),
    },
    "Lachnospiraceae_unclassified_SGB4882": {
        "category": "suggestive_concordance_low_evidence",
        "discussion": (
            "Project healthy-diet-enriched/lower-CV pattern is plausible for a Lachnospiraceae SGB, "
            "but family-level SCFA biology must not be promoted to species-level mechanism."
        ),
    },
    "GGB9634_SGB15093": {
        "category": "novel_low_evidence",
        "discussion": (
            "Taxonomy and function remain poorly characterized; interpret as a novel marker."
        ),
    },
    "GGB9707_SGB15229": {
        "category": "suggestive_concordance_low_evidence",
        "discussion": (
            "Project healthy-diet-enriched/lower-CV orientation agrees with limited external "
            "pro-health SGB rankings/intervention signals, but species-level mechanism is unknown."
        ),
    },
    "GGB9758_SGB15368": {
        "category": "suggestive_but_mixed_low_evidence",
        "discussion": (
            "Project healthy-diet-enriched/lower-CV orientation has some external support, but "
            "disease associations are mixed and species-level function is unresolved."
        ),
    },
}


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "t"})
    )


def sign_num(x) -> int:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v) or v == 0:
        return 0
    return 1 if v > 0 else -1


def sign_chr(x) -> str:
    s = sign_num(x)
    return "+" if s > 0 else "-" if s < 0 else "0/NA"


def label_from_row(row: pd.Series) -> str:
    if "species_label" in row.index and pd.notna(row["species_label"]):
        return str(row["species_label"]).strip()
    species = str(row["species"])
    return species.split("|s__")[-1] if "|s__" in species else species


def direction_class(a, b, indirect) -> str:
    sa, sb, si = sign_num(a), sign_num(b), sign_num(indirect)
    if (sa, sb, si) == (1, -1, -1):
        return "healthy_diet_enriched_relative__lower_glucose_cv"
    if (sa, sb, si) == (-1, 1, -1):
        return "healthy_diet_depleted_relative__higher_glucose_cv_taxon"
    if si == -1:
        return "other_negative_indirect_pattern"
    if si == 1:
        return "positive_indirect_pattern"
    return "unclassified"


def same_sign(x, y):
    sx, sy = sign_num(x), sign_num(y)
    if sx == 0 or sy == 0:
        return np.nan
    return bool(sx == sy)



def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    """Dependency-free markdown table; avoids pandas.to_markdown/tabulate."""
    if df is None or df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                s = ""
            else:
                s = str(v)
            s = s.replace("|", r"\|").replace("\n", " ")
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    if not PRIMARY.is_file():
        raise FileNotFoundError(PRIMARY)
    if not ANNOT.is_file():
        raise FileNotFoundError(
            f"Step13a annotation missing: {ANNOT}\nRun Step13a first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    p = pd.read_csv(PRIMARY, low_memory=False)
    p["candidate_mediator_consistent"] = truthy(
        p["candidate_mediator_consistent"]
    )
    p["species_label_derived"] = p.apply(label_from_row, axis=1)

    cv = p.loc[
        p["cgm_outcome"].eq("glucose_cv")
        & p["diet_score"].isin(["AMED", "hPDI"])
        & p["candidate_mediator_consistent"]
    ].copy()
    core = sorted(
        set(cv.loc[cv["diet_score"].eq("AMED"), "species"])
        & set(cv.loc[cv["diet_score"].eq("hPDI"), "species"])
    )
    if len(core) != 12:
        raise RuntimeError(f"Expected 12 core taxa; found {len(core)}")

    long = p.loc[
        p["species"].isin(core)
        & p["cgm_outcome"].eq("glucose_cv")
        & p["diet_score"].isin(["AMED", "hPDI"])
    ].copy()

    if len(long) != 24:
        raise RuntimeError(f"Expected 24 core path rows; found {len(long)}")

    for col in [
        "a_diet_to_microbiome",
        "b_microbiome_to_cgm",
        "indirect_effect",
        "bmi_beta_diet",
        "bmi_beta_cgm",
        "bmi_beta_product",
        "primary_beta_diet",
        "primary_beta_cgm",
        "primary_beta_product",
    ]:
        if col in long.columns:
            long[col] = pd.to_numeric(long[col], errors="coerce")

    long["a_sign"] = long["a_diet_to_microbiome"].map(sign_chr)
    long["b_sign"] = long["b_microbiome_to_cgm"].map(sign_chr)
    long["indirect_sign"] = long["indirect_effect"].map(sign_chr)
    long["direction_class"] = long.apply(
        lambda r: direction_class(
            r["a_diet_to_microbiome"],
            r["b_microbiome_to_cgm"],
            r["indirect_effect"],
        ),
        axis=1,
    )

    # Sanity-check the mediation signs against the carried-forward bridge/model3
    # coefficients.  This is QC only: a/b models use a different complete-case
    # set/conditioning structure, so exact coefficient equality is not expected.
    if "bmi_beta_diet" in long.columns:
        long["a_same_sign_as_step09_model3_diet_beta"] = long.apply(
            lambda r: same_sign(
                r["a_diet_to_microbiome"], r["bmi_beta_diet"]
            ),
            axis=1,
        )
    else:
        long["a_same_sign_as_step09_model3_diet_beta"] = np.nan

    if "bmi_beta_cgm" in long.columns:
        long["b_same_sign_as_step09_model3_cgm_beta"] = long.apply(
            lambda r: same_sign(
                r["b_microbiome_to_cgm"], r["bmi_beta_cgm"]
            ),
            axis=1,
        )
    else:
        long["b_same_sign_as_step09_model3_cgm_beta"] = np.nan

    if "bmi_beta_product" in long.columns:
        long["indirect_same_sign_as_step09_model3_product"] = long.apply(
            lambda r: same_sign(
                r["indirect_effect"], r["bmi_beta_product"]
            ),
            axis=1,
        )
    else:
        long["indirect_same_sign_as_step09_model3_product"] = np.nan

    long.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")

    annot = pd.read_csv(ANNOT, low_memory=False)
    annot["species"] = annot["species"].astype(str)
    annot["species_label"] = annot["species_label"].astype(str)

    rows = []
    for species in core:
        g = long.loc[long["species"].eq(species)].copy()
        if len(g) != 2:
            raise RuntimeError(f"Core species does not have two paths: {species}")

        amed = g.loc[g["diet_score"].eq("AMED")].iloc[0]
        hpdi = g.loc[g["diet_score"].eq("hPDI")].iloc[0]
        label = str(amed["species_label_derived"])

        if (
            direction_class(
                amed["a_diet_to_microbiome"],
                amed["b_microbiome_to_cgm"],
                amed["indirect_effect"],
            )
            != direction_class(
                hpdi["a_diet_to_microbiome"],
                hpdi["b_microbiome_to_cgm"],
                hpdi["indirect_effect"],
            )
        ):
            cross_class = "AMED_hPDI_component_pattern_differs"
        else:
            cross_class = direction_class(
                amed["a_diet_to_microbiome"],
                amed["b_microbiome_to_cgm"],
                amed["indirect_effect"],
            )

        bio = BIO_GROUP.get(
            label,
            {
                "category": "manual_review_needed",
                "discussion": "No curated Step13b category.",
            },
        )

        row = {
            "species": species,
            "species_label": label,
            "project_direction_class": cross_class,
            "AMED_a_sign": sign_chr(amed["a_diet_to_microbiome"]),
            "AMED_b_sign": sign_chr(amed["b_microbiome_to_cgm"]),
            "AMED_indirect_sign": sign_chr(amed["indirect_effect"]),
            "hPDI_a_sign": sign_chr(hpdi["a_diet_to_microbiome"]),
            "hPDI_b_sign": sign_chr(hpdi["b_microbiome_to_cgm"]),
            "hPDI_indirect_sign": sign_chr(hpdi["indirect_effect"]),
            "biological_evidence_category": bio["category"],
            "recommended_discussion_language": bio["discussion"],
            "clr_interpretation": (
                "a is an association with species CLR-Z (log abundance relative "
                "to the within-sample geometric mean), not an absolute abundance change."
            ),
        }

        for qc_col in [
            "a_same_sign_as_step09_model3_diet_beta",
            "b_same_sign_as_step09_model3_cgm_beta",
            "indirect_same_sign_as_step09_model3_product",
        ]:
            vals = g[qc_col].dropna()
            row[f"both_paths_{qc_col}"] = (
                bool(vals.astype(bool).all()) if len(vals) == 2 else np.nan
            )

        # Bring through Step13a evidence-strength label.
        hit = annot.loc[annot["species"].eq(species)]
        row["literature_evidence_strength"] = (
            str(hit.iloc[0]["literature_evidence_strength"])
            if len(hit)
            else ""
        )
        rows.append(row)

    classes = pd.DataFrame(rows)
    classes.to_csv(OUT_CLASS, index=False, encoding="utf-8-sig")

    count_table = (
        classes["project_direction_class"]
        .value_counts()
        .rename_axis("project_direction_class")
        .reset_index(name="n_species")
    )

    # Strong checks from what Step13a output visually suggested.
    all_negative_indirect = bool(
        (
            classes["AMED_indirect_sign"].eq("-")
            & classes["hPDI_indirect_sign"].eq("-")
        ).all()
    )
    same_component_class = bool(
        ~classes["project_direction_class"]
        .eq("AMED_hPDI_component_pattern_differs")
        .any()
    )

    def qc_fraction(col):
        s = pd.to_numeric(long[col].astype(object), errors="coerce")
        # bool columns do not coerce cleanly in all pandas versions; handle explicitly.
        vals = long[col].dropna()
        if not len(vals):
            return np.nan
        return float(vals.astype(bool).mean())

    a_qc = qc_fraction("a_same_sign_as_step09_model3_diet_beta")
    b_qc = qc_fraction("b_same_sign_as_step09_model3_cgm_beta")
    prod_qc = qc_fraction("indirect_same_sign_as_step09_model3_product")

    md = [
        "# Step 13b — 12-core direction classes and biological interpretation",
        "",
        "## Critical interpretation",
        "",
        "All species coefficients are based on **CLR-Z microbiome features**. A positive or negative "
        "`a` coefficient therefore means a higher or lower **log-ratio relative to the sample's "
        "geometric-mean community abundance**, not a proven absolute increase or decrease in cell count.",
        "",
        "The recurrent core is specific to **AMED/hPDI × glucose_cv** and is not a core across all CGM outcomes.",
        "",
        "## Project direction classes",
        "",
        dataframe_to_markdown_simple(count_table),
        "",
        f"- All 12 have negative indirect effects for both AMED and hPDI: **{all_negative_indirect}**",
        f"- AMED and hPDI use the same component-sign class for each species: **{same_component_class}**",
        "",
        "### Class 1: healthy-diet enriched (relative) / lower glucose-CV taxon",
        "",
        "Pattern: `a > 0`, `b < 0`, `a×b < 0`.",
        "",
    ]

    c1 = classes.loc[
        classes["project_direction_class"].eq(
            "healthy_diet_enriched_relative__lower_glucose_cv"
        )
    ]
    for _, r in c1.iterrows():
        md.append(
            f"- **{r['species_label']}** — {r['biological_evidence_category']}: "
            f"{r['recommended_discussion_language']}"
        )

    md += [
        "",
        "### Class 2: healthy-diet depleted (relative) / higher glucose-CV taxon",
        "",
        "Pattern: `a < 0`, `b > 0`, `a×b < 0`.",
        "",
    ]

    c2 = classes.loc[
        classes["project_direction_class"].eq(
            "healthy_diet_depleted_relative__higher_glucose_cv_taxon"
        )
    ]
    for _, r in c2.iterrows():
        md.append(
            f"- **{r['species_label']}** — {r['biological_evidence_category']}: "
            f"{r['recommended_discussion_language']}"
        )

    md += [
        "",
        "## Direction QC against carried-forward Step09/Model3 bridge coefficients",
        "",
        f"- mediation `a` sign agrees with Step09 Model3 diet beta in {a_qc:.1%} of available core paths",
        f"- mediation `b` sign agrees with Step09 Model3 microbiome→CGM beta in {b_qc:.1%} of available core paths",
        f"- mediation indirect sign agrees with Step09 Model3 beta-product sign in {prod_qc:.1%} of available core paths",
        "",
        "These are QC concordance measures, not an expectation of coefficient equality: the mediation "
        "`b` model additionally conditions on the diet score, and complete-case samples can differ.",
        "",
        "## Recommended discussion principle",
        "",
        "Do not use generic labels such as 'good bacteria' or 'bad bacteria'. The defensible result is "
        "the reproducible project-specific direction of the diet–microbiome–glucose-CV path, followed "
        "by species-specific external evidence and explicit acknowledgement of compositional CLR interpretation.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    summary_lines = [
        "=== STEP 13b 12-CORE DIRECTION SUMMARY ===",
        f"CORE_N={len(classes)}",
        f"ALL_NEGATIVE_INDIRECT_BOTH_DIETS={all_negative_indirect}",
        f"SAME_COMPONENT_CLASS_AMED_HPDI={same_component_class}",
        "",
        count_table.to_string(index=False),
        "",
        f"A_SIGN_QC_VS_STEP09_MODEL3={a_qc:.6f}",
        f"B_SIGN_QC_VS_STEP09_MODEL3={b_qc:.6f}",
        f"INDIRECT_SIGN_QC_VS_STEP09_MODEL3_PRODUCT={prod_qc:.6f}",
        "",
        "--- SPECIES CLASSES ---",
        classes[
            [
                "species_label",
                "project_direction_class",
                "biological_evidence_category",
            ]
        ].to_string(index=False),
        "",
        f"LONG_QC={OUT_LONG}",
        f"CLASSES={OUT_CLASS}",
        f"REPORT={OUT_MD}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
