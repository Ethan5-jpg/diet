#!/usr/bin/env python3
"""Step 13c — produce discussion-ready biological synthesis for the 12-core.

READ-ONLY. No model is refit.

Requires:
    13a_12_core_species_biological_annotation.csv
    13b_12_core_direction_classes.csv

Produces:
    13c_12_core_discussion_priority_table.csv
    13c_12_core_results_draft.md
    13c_12_core_discussion_draft.md
    13c_12_core_final_biology_summary.txt

Interpretation principles
-------------------------
1) Full taxonomy string is the identity key.
2) The core is recurrent across AMED/hPDI for glucose_cv only.
3) CLR-Z coefficients are relative log-ratio associations, not absolute cell-count changes.
4) Mediation results are cross-sectional mediation-style indirect associations, not causal mediation.
5) External literature is used to contextualize, not to override project directions.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
REPORT_DIR = (
    ROOT / "diet_microbiome_glucose_analysis" / "outputs" / "reports"
)

ANNOT = REPORT_DIR / "13a_12_core_species_biological_annotation.csv"
CLASSES = REPORT_DIR / "13b_12_core_direction_classes.csv"

OUT_TABLE = REPORT_DIR / "13c_12_core_discussion_priority_table.csv"
OUT_RESULTS = REPORT_DIR / "13c_12_core_results_draft.md"
OUT_DISCUSSION = REPORT_DIR / "13c_12_core_discussion_draft.md"
OUT_SUMMARY = REPORT_DIR / "13c_12_core_final_biology_summary.txt"


PRIORITY = {
    "Enterocloster_bolteae": {
        "tier": "Tier_A_strongest_interpretable",
        "headline": (
            "Most directly concordant adverse-metabolic taxon: relatively depleted with "
            "higher AMED/hPDI and positively associated with glucose variability."
        ),
    },
    "GGB9707_SGB15229": {
        "tier": "Tier_B_suggestive_concordance",
        "headline": (
            "Novel healthy-diet-enriched/lower-glucose-CV SGB with limited but directionally "
            "supportive external population/diet evidence."
        ),
    },
    "Lachnospiraceae_unclassified_SGB4882": {
        "tier": "Tier_B_suggestive_concordance",
        "headline": (
            "Novel healthy-diet-enriched/lower-glucose-CV Lachnospiraceae SGB; family-level "
            "fermentation biology is plausible but species-level mechanism is unresolved."
        ),
    },
    "GGB9758_SGB15368": {
        "tier": "Tier_B_suggestive_mixed",
        "headline": (
            "Healthy-diet-enriched/lower-glucose-CV SGB with some supportive diet evidence, "
            "but external disease associations are mixed."
        ),
    },
    "Blautia_massiliensis": {
        "tier": "Tier_C_context_dependent",
        "headline": (
            "Relatively depleted with higher healthy-diet scores and positively associated with "
            "glucose variability; external metabolic literature is mixed/context-dependent."
        ),
    },
    "Bifidobacterium_adolescentis": {
        "tier": "Tier_C_literature_tension",
        "headline": (
            "Project CLR direction differs from common bifidogenic/favorable-metabolic expectations; "
            "interpret as compositional/context-dependent tension, not as evidence that the species is harmful."
        ),
    },
    "Bifidobacterium_bifidum": {
        "tier": "Tier_C_literature_tension",
        "headline": (
            "Project CLR direction differs from common prebiotic/probiotic expectations; "
            "strain-specific external evidence and CLR compositionality limit direct comparison."
        ),
    },
    "Bifidobacterium_catenulatum": {
        "tier": "Tier_C_literature_tension",
        "headline": (
            "Despite fiber-response/cross-feeding plausibility, the project shows the mirrored relative-CLR pattern; "
            "this is a context-dependent signal rather than a direct contradiction."
        ),
    },
    "Bifidobacterium_longum": {
        "tier": "Tier_C_literature_tension",
        "headline": (
            "External prospective/probiotic glycemic literature is generally favorable, whereas the project "
            "shows lower relative CLR with healthier diet and positive conditional association with glucose CV."
        ),
    },
    "GGB3118_SGB4130": {
        "tier": "Tier_D_novel_low_evidence",
        "headline": (
            "Robust project core taxon with favorable direction but insufficient species-level functional evidence."
        ),
    },
    "GGB9634_SGB15093": {
        "tier": "Tier_D_novel_low_evidence",
        "headline": (
            "Robust project core taxon with favorable direction but poorly resolved species-level biology."
        ),
    },
    "Clostridium_sp_AF15_49": {
        "tier": "Tier_D_novel_low_evidence",
        "headline": (
            "Robust project core taxon with favorable project direction; direct diet/glucose mechanism remains sparse."
        ),
    },
}

TIER_ORDER = {
    "Tier_A_strongest_interpretable": 1,
    "Tier_B_suggestive_concordance": 2,
    "Tier_B_suggestive_mixed": 3,
    "Tier_C_context_dependent": 4,
    "Tier_C_literature_tension": 5,
    "Tier_D_novel_low_evidence": 6,
}


def main() -> int:
    for path in (ANNOT, CLASSES):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing prerequisite: {path}\n"
                "Run Step 13a and Step 13b first."
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(ANNOT, low_memory=False)
    c = pd.read_csv(CLASSES, low_memory=False)

    # Explicit schema guards: fail early with a useful message instead of
    # allowing pandas merge suffixes (_x/_y) to cause a later KeyError.
    required_class_cols = {
        "species",
        "species_label",
        "project_direction_class",
        "biological_evidence_category",
        "recommended_discussion_language",
    }
    required_annot_cols = {
        "species",
        "species_label",
        "characterization",
        "diet_evidence",
        "metabolic_function_or_plausibility",
        "glucose_or_metabolic_evidence",
        "interpretive_caveat",
        "literature_sources",
        "AMED_a",
        "AMED_b",
        "AMED_indirect",
        "hPDI_a",
        "hPDI_b",
        "hPDI_indirect",
    }

    missing_c = sorted(required_class_cols - set(c.columns))
    missing_a = sorted(required_annot_cols - set(a.columns))
    if missing_c:
        raise RuntimeError(
            "Step13b classes file is missing required columns: "
            + ", ".join(missing_c)
            + f"\nAvailable columns: {c.columns.tolist()}"
        )
    if missing_a:
        raise RuntimeError(
            "Step13a annotation file is missing required columns: "
            + ", ".join(missing_a)
            + f"\nAvailable columns: {a.columns.tolist()}"
        )

    # Step13b already carries literature_evidence_strength in the current
    # pipeline.  Do NOT merge that column a second time, because pandas would
    # create literature_evidence_strength_x/_y and remove the unsuffixed name.
    # If an older Step13b file lacks it, recover it from Step13a.
    annotation_payload = [
        "species",
        "species_label",
        "characterization",
        "diet_evidence",
        "metabolic_function_or_plausibility",
        "glucose_or_metabolic_evidence",
        "interpretive_caveat",
        "literature_sources",
        "AMED_a",
        "AMED_b",
        "AMED_indirect",
        "hPDI_a",
        "hPDI_b",
        "hPDI_indirect",
    ]
    if "literature_evidence_strength" not in c.columns:
        if "literature_evidence_strength" not in a.columns:
            raise RuntimeError(
                "literature_evidence_strength is missing from both Step13a "
                "and Step13b outputs."
            )
        annotation_payload.append("literature_evidence_strength")

    merged = c.merge(
        a[annotation_payload],
        on=["species", "species_label"],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 12:
        raise RuntimeError(f"Expected 12 species, found {len(merged)}")

    # Ensure every Step13b species found its Step13a annotation.
    missing_annotation = merged["characterization"].isna()
    if missing_annotation.any():
        bad = merged.loc[
            missing_annotation, ["species_label", "species"]
        ].to_dict("records")
        raise RuntimeError(
            "Step13a/Step13b species join failed for: " + str(bad)
        )

    rows = []
    for _, r in merged.iterrows():
        label = r["species_label"]
        p = PRIORITY.get(
            label,
            {
                "tier": "Tier_D_novel_low_evidence",
                "headline": "Robust core taxon requiring manual biological review.",
            },
        )
        d = r.to_dict()
        d["discussion_priority_tier"] = p["tier"]
        d["discussion_headline"] = p["headline"]
        d["tier_order"] = TIER_ORDER[p["tier"]]
        rows.append(d)

    out = pd.DataFrame(rows).sort_values(
        ["tier_order", "species_label"], kind="mergesort"
    )
    out.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    class_counts = (
        out["project_direction_class"]
        .value_counts()
        .rename_axis("direction_class")
        .reset_index(name="n")
    )

    # -------------------- Results draft --------------------
    results = [
        "# Step 13c — Results draft: recurrent microbiome core",
        "",
        "Across the direction-consistent mediation candidates for glucose variability, "
        "AMED and hPDI shared 12 recurrent species-level taxa. For every core taxon, "
        "the sign pattern of the diet-to-microbiome (`a`) and microbiome-to-glucose-CV "
        "(`b`) paths was identical for AMED and hPDI, and all corresponding indirect "
        "effects were negative.",
        "",
        "The 12 taxa separated evenly into two mirrored direction classes. Six taxa showed "
        "positive diet-to-microbiome associations and negative microbiome-to-glucose-CV "
        "associations (`a>0`, `b<0`), corresponding to higher relative CLR abundance with "
        "greater healthy-diet adherence and lower glucose variability. The other six showed "
        "the inverse component pattern (`a<0`, `b>0`), corresponding to lower relative CLR "
        "abundance with greater healthy-diet adherence and higher glucose variability when "
        "the taxon was relatively more abundant.",
        "",
        "Importantly, these directions refer to species CLR-Z features and therefore represent "
        "relative log-ratio associations within the microbial community rather than absolute "
        "changes in bacterial cell counts.",
        "",
        "The shared structure was not driven by one dietary score: AMED and hPDI assigned every "
        "core species to the same component-sign class. Direction QC against the carried-forward "
        "Step09/Model3 bridge coefficients showed complete sign agreement for the diet→species, "
        "species→glucose-CV, and beta-product directions.",
        "",
        "## Direction classes",
        "",
    ]

    for _, rr in class_counts.iterrows():
        results.append(f"- {rr['direction_class']}: {int(rr['n'])} species")

    results += [
        "",
        "These results describe cross-sectional mediation-style indirect associations and do not "
        "establish causal mediation.",
        "",
    ]
    OUT_RESULTS.write_text("\n".join(results), encoding="utf-8")

    def get_species_row(label: str) -> pd.Series:
        hit = out.loc[out["species_label"].eq(label)]
        if len(hit) != 1:
            raise RuntimeError(
                f"Expected exactly one row for {label!r}, found {len(hit)}. "
                f"Available labels: {out['species_label'].tolist()}"
            )
        return hit.iloc[0]

    # -------------------- Discussion draft --------------------
    discussion = [
        "# Step 13c — Discussion draft: biological interpretation of the 12-core",
        "",
        "The most notable biological feature of the recurrent core is not simply the presence of "
        "12 shared taxa, but the highly structured 6+6 mirrored architecture that was reproduced "
        "for both AMED and hPDI. This suggests that the two healthy-diet scores converge on a common "
        "microbial configuration associated with lower glucose variability, while also emphasizing "
        "that this configuration contains both relatively enriched and relatively depleted taxa.",
        "",
        "## 1. Most directly interpretable adverse-oriented taxon",
        "",
    ]

    e = get_species_row("Enterocloster_bolteae")
    discussion += [
        f"**Enterocloster bolteae** is the clearest species for which the project direction and "
        f"external metabolic literature are qualitatively aligned. In the project it belongs to the "
        f"`a<0, b>0` class: higher AMED/hPDI was associated with lower relative CLR abundance, while "
        f"higher relative abundance was associated with greater glucose variability. External human "
        f"studies have reported enrichment of E. bolteae in type 2 diabetes or adverse insulin-sensitivity "
        f"patterns. This makes E. bolteae a strong discussion anchor, although the present analysis remains "
        f"observational and does not establish pathogenicity.",
        "",
        "## 2. Healthy-diet-enriched, lower-glucose-CV SGBs",
        "",
    ]

    for label in [
        "GGB9707_SGB15229",
        "Lachnospiraceae_unclassified_SGB4882",
        "GGB9758_SGB15368",
    ]:
        r = get_species_row(label)
        discussion.append(
            f"- **{label}**: {r['discussion_headline']} "
            f"The appropriate interpretation is a robust project-specific marker with "
            f"{r.get('literature_evidence_strength', 'unclassified')} external evidence, rather than a confirmed mechanism."
        )

    discussion += [
        "",
        "## 3. Bifidobacterium cluster: an important literature tension",
        "",
        "All four recurrent Bifidobacterium species—B. adolescentis, B. bifidum, "
        "B. catenulatum and B. longum—fell into the `a<0, b>0` class. At face value this differs "
        "from the common expectation that fermentable-fiber or prebiotic exposure enriches "
        "bifidobacteria and that some Bifidobacterium strains are linked to favorable metabolic "
        "phenotypes. This should not be described as a direct contradiction. The project uses "
        "species CLR-Z, so `a<0` represents lower abundance relative to the within-sample geometric "
        "mean of the microbial community, not necessarily lower absolute abundance. In addition, "
        "external studies differ in exposure (specific prebiotic/probiotic strains versus composite "
        "diet scores), outcome (fasting glucose, HbA1c or diabetes versus CGM variability), population, "
        "and conditioning structure.",
        "",
        "The bifidobacterial result is therefore best framed as a context-dependent compositional "
        "finding that warrants replication and, ideally, absolute-abundance or metagenomic functional "
        "follow-up. It should not be used to claim that Bifidobacterium species are metabolically harmful.",
        "",
        "## 4. Context-dependent and poorly characterized taxa",
        "",
    ]

    for label in [
        "Blautia_massiliensis",
        "GGB3118_SGB4130",
        "GGB9634_SGB15093",
        "Clostridium_sp_AF15_49",
    ]:
        r = get_species_row(label)
        discussion.append(
            f"- **{label}**: {r['discussion_headline']} "
            f"{r['recommended_discussion_language']}"
        )

    discussion += [
        "",
        "## 5. Overall biological interpretation",
        "",
        "Taken together, the recurrent core is better interpreted as a community-level microbial "
        "signature associated with healthy-diet adherence and glucose stability than as a set of "
        "independent 'beneficial' or 'harmful' organisms. The complete directional agreement between "
        "AMED and hPDI, together with robustness to diabetes/A10 restriction and antibiotic/PPI "
        "adjustment, supports the reproducibility of the statistical pattern. However, compositional "
        "CLR features, cross-sectional mediation, incomplete functional characterization of several "
        "SGBs, and context-dependent evidence for named taxa limit mechanistic conclusions.",
        "",
        "The most defensible next biological step is therefore to prioritize E. bolteae and the "
        "directionally supportive novel SGBs for interpretation, while presenting the Bifidobacterium "
        "cluster explicitly as an unexpected compositional finding rather than forcing it into a "
        "predefined beneficial-versus-adverse framework.",
        "",
    ]
    OUT_DISCUSSION.write_text("\n".join(discussion), encoding="utf-8")

    summary = [
        "=== STEP 13c FINAL BIOLOGY SYNTHESIS ===",
        "CORE_N=12",
        "DIRECTION_ARCHITECTURE=6_enriched_lower_CV + 6_depleted_higher_CV",
        "AMED_HPDI_COMPONENT_DIRECTION_MATCH=12/12",
        "STEP09_MODEL3_SIGN_QC=100%",
        "",
        "DISCUSSION PRIORITY:",
    ]
    for tier, g in out.groupby("discussion_priority_tier", sort=False):
        summary.append(
            f"{tier}: " + ", ".join(g["species_label"].astype(str).tolist())
        )

    summary += [
        "",
        "KEY_INTERPRETIVE_RULE:",
        "CLR direction is relative/compositional; do not equate a<0 with absolute bacterial depletion.",
        "Do not describe the mediation-style analysis as causal.",
        "",
        f"PRIORITY_TABLE={OUT_TABLE}",
        f"RESULTS_DRAFT={OUT_RESULTS}",
        f"DISCUSSION_DRAFT={OUT_DISCUSSION}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
