#!/usr/bin/env python3
"""Step 13a — build 12-core species biological annotation table.

This script does NOT refit any model. It reads the existing canonical mediation
outputs, derives the recurrent AMED/hPDI -> glucose_cv core exactly as Step10b
did, attaches path directions/effect sizes, and adds a conservative literature
annotation layer.

Important interpretation rules
------------------------------
- `species` (full taxonomy string) remains the identity key.
- The 12-core means recurrent across AMED and hPDI for glucose_cv, NOT across all
  three CGM outcomes.
- Biological annotations are hypothesis/context, not proof of mechanism.
- Uncharacterized SGBs are explicitly labeled as such; family/genus knowledge is
  not silently promoted to species-level evidence.
- "External expected pattern" is only a literature-oriented prior used to flag
  concordance/discordance. It never changes the statistical result.

Inputs
------
diet_microbiome_glucose_analysis/outputs/models/
    10_mediation_paths_model3.csv
Optional robustness files:
    10_strict_mediation_paths_model3.csv
    12b_antibiotic_ppi_mediation_paths_model3.csv
    12e_protocol_window_mediation_paths_model3.csv

Outputs
-------
diet_microbiome_glucose_analysis/outputs/reports/
    13a_12_core_project_effects_long.csv
    13a_12_core_species_biological_annotation.csv
    13a_12_core_species_biological_annotation.md
    13a_12_core_annotation_summary.txt
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
STRICT = MODEL_DIR / "10_strict_mediation_paths_model3.csv"
BROAD = MODEL_DIR / "12b_antibiotic_ppi_mediation_paths_model3.csv"
PROTOCOL = MODEL_DIR / "12e_protocol_window_mediation_paths_model3.csv"

OUT_LONG = REPORT_DIR / "13a_12_core_project_effects_long.csv"
OUT_ANNOT = REPORT_DIR / "13a_12_core_species_biological_annotation.csv"
OUT_MD = REPORT_DIR / "13a_12_core_species_biological_annotation.md"
OUT_SUMMARY = REPORT_DIR / "13a_12_core_annotation_summary.txt"

KEY = ["diet_score", "cgm_outcome", "species"]

# Conservative literature layer.  Evidence is deliberately species-specific where
# possible. For unnamed SGBs, "very_low" means that direct species-level biology
# is not established enough to support mechanistic claims.
LIT = {
    "Bifidobacterium_adolescentis": {
        "characterization": "Named, cultured human gut Bifidobacterium species; common adult commensal.",
        "diet_evidence": (
            "Human prebiotic/fiber interventions repeatedly increase B. adolescentis; "
            "it is responsive to inulin/FOS/GOS and other fermentable carbohydrate substrates."
        ),
        "metabolic_function": (
            "Saccharolytic bifidobacterial metabolism with acetate/lactate production; "
            "cross-feeding may support butyrate-producing organisms."
        ),
        "glucose_evidence": (
            "Mouse T2D work found several B. adolescentis strains improved glycemic phenotypes, "
            "with higher SCFA-producing flora and lower inflammation; direct human species-level "
            "glycemic intervention evidence remains limited."
        ),
        "caveat": "Effects are strain-dependent; mouse probiotic results must not be treated as human causal evidence.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "moderate",
        "sources": "PMID:35745208; PMID:35782954; PMCID:PMC7285360",
    },
    "Bifidobacterium_bifidum": {
        "characterization": "Named Bifidobacterium species and established human commensal/probiotic taxon.",
        "diet_evidence": (
            "Human prebiotic/fiber supplementation can increase B. bifidum together with other bifidobacteria."
        ),
        "metabolic_function": (
            "Carbohydrate fermentation and acetate/lactate production; bifidobacterial ecosystem effects "
            "may include cross-feeding and barrier/inflammatory modulation."
        ),
        "glucose_evidence": (
            "Recent human B. bifidum BGN4 RCT in adults with excess adiposity reported lower fasting insulin, "
            "zonulin and TNF-alpha but not a clear glucose effect; an earlier T2D study improved GI symptoms "
            "without changing HbA1c."
        ),
        "caveat": "Human metabolic evidence is strain-specific and does not establish that endogenous B. bifidum lowers CGM variability.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "moderate",
        "sources": "PMID:42116047; PMID:34665938; PMID:35782954",
    },
    "Bifidobacterium_catenulatum": {
        "characterization": "Named Bifidobacterium species; human gut commensal with carbohydrate-fermenting capacity.",
        "diet_evidence": (
            "B. catenulatum increases in human prebiotic/fiber interventions and has been responsive to wheat/oat "
            "or fermentable carbohydrate exposure in small feeding studies."
        ),
        "metabolic_function": (
            "Produces acetate and can cross-feed Faecalibacterium prausnitzii; co-culture work showed increased "
            "butyrate production and anti-inflammatory effects."
        ),
        "glucose_evidence": (
            "Direct human glucose-specific evidence for endogenous B. catenulatum is limited; its biological case "
            "is stronger for fiber response and SCFA cross-feeding than for glycemia itself."
        ),
        "caveat": "Do not convert cross-feeding/colitis-model evidence into a direct glucose mechanism.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "moderate_for_diet_function_low_for_glucose",
        "sources": "PMID:35782954; PMCID:PMC7285360; PMCID:PMC5359301",
    },
    "Bifidobacterium_longum": {
        "characterization": "Named, widely studied human gut Bifidobacterium species.",
        "diet_evidence": (
            "Human prebiotic/fiber interventions commonly increase B. longum; it is strongly linked to fermentable carbohydrate use."
        ),
        "metabolic_function": (
            "Saccharolytic fermentation, acetate production and gut ecosystem/barrier effects; mechanisms are strain-dependent."
        ),
        "glucose_evidence": (
            "Prospective human data found lower baseline B. longum in normoglycemic people who later developed T2D and "
            "a negative association with follow-up glucose. A 2026 randomized trial of B. longum BL21 plus metformin "
            "reported a greater HbA1c reduction than placebo."
        ),
        "caveat": "The RCT tests one administered strain, not endogenous species abundance, so it supports plausibility rather than mediation causality.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "moderate_to_strong_plausibility",
        "sources": "PMID:33680988; PMID:41523285; PMID:35782954",
    },
    "Blautia_massiliensis": {
        "characterization": "Named anaerobic human stool isolate in the genus Blautia.",
        "diet_evidence": (
            "Experimental work shows B. massiliensis can metabolize FOS and inulin, supporting a direct fermentable-fiber link."
        ),
        "metabolic_function": (
            "Blautia taxa are SCFA-associated; B. massiliensis has been linked to acetate, but species/strain effects are heterogeneous."
        ),
        "glucose_evidence": (
            "Human metabolic associations are mixed: one recent study found B. massiliensis enriched in metabolic syndrome, "
            "while other work links lower abundance with lower circulating acetate in an inflammatory setting."
        ),
        "caveat": "Do not label B. massiliensis simply 'good' or 'bad'; evidence is context-dependent.",
        "expected_a": 0,
        "expected_b": 0,
        "evidence_strength": "mixed_moderate_for_fiber_low_for_glucose",
        "sources": "PMID:27923606; PMCID:PMC11774028; PMCID:PMC12166849; PMCID:PMC9634083",
    },
    "Clostridium_sp_AF15_49": {
        "characterization": "Named metagenome/culture-linked Clostridium sp. with NCBI taxonomy/genome records (SGB5111 in some MetaPhlAn resources).",
        "diet_evidence": (
            "Direct diet evidence is sparse; it has appeared in intervention/metatranscriptomic studies, but no consistent healthy-diet direction is established."
        ),
        "metabolic_function": "Species-specific metabolic function is insufficiently characterized for a strong mechanism statement.",
        "glucose_evidence": (
            "No convincing direct human glycemic evidence located; one MASLD fiber trial reported context-specific correlations "
            "with TLR2/TLR4 ligands rather than glucose outcomes."
        ),
        "caveat": "Treat as low-evidence exploratory taxon; avoid mechanistic claims.",
        "expected_a": 0,
        "expected_b": 0,
        "evidence_strength": "low",
        "sources": "NCBI Taxonomy:2292997; GenBank:QTXK00000000; PMCID:PMC12721201",
    },
    "Enterocloster_bolteae": {
        "characterization": "Named anaerobic human gut species, formerly Clostridium bolteae.",
        "diet_evidence": (
            "Diet-specific direction is not uniform. E. bolteae can metabolize diverse carbohydrates and can transform "
            "diet-derived urolithin intermediates; this does not make the taxon uniformly beneficial."
        ),
        "metabolic_function": (
            "Known for broad carbohydrate metabolism; reported ethanol production in NASH isolates and specialized urolithin-A pathway chemistry."
        ),
        "glucose_evidence": (
            "Multiple human datasets associate E. bolteae with adverse metabolic phenotypes: enrichment in T2D and negative "
            "correlation with insulin-sensitivity indices in a dietary RCT."
        ),
        "caveat": (
            "This is a key potential 'discordant-but-informative' taxon if the project shows a protective direction. "
            "Association is context-dependent and does not prove pathogenicity."
        ),
        "expected_a": -1,
        "expected_b": 1,
        "evidence_strength": "moderate_adverse_association",
        "sources": "PMID:37298483; PMCID:PMC12056104; PMCID:PMC10687429; PMCID:PMC11760930",
    },
    "GGB3118_SGB4130": {
        "characterization": "Uncharacterized MetaPhlAn species-level genome bin; project taxonomy is the authoritative identity.",
        "diet_evidence": "Very sparse direct literature; appears in population-level microbiome signature databases, without a robust mechanistic diet interpretation.",
        "metabolic_function": "Unknown at species level.",
        "glucose_evidence": "No direct species-level glucose/insulin evidence located.",
        "caveat": "Do not infer function from neighboring taxa or replace the project taxonomy with a newer database label.",
        "expected_a": 0,
        "expected_b": 0,
        "evidence_strength": "very_low",
        "sources": "BugSigDB study PMID-linked record:38968070",
    },
    "GGB9634_SGB15093": {
        "characterization": "Uncharacterized species-level genome bin; higher-level taxonomy can vary across database releases.",
        "diet_evidence": "No strong direct species-level dietary intervention evidence located.",
        "metabolic_function": "Unknown at species level.",
        "glucose_evidence": "No convincing direct species-level glycemic evidence located.",
        "caveat": (
            "Taxonomy is database-version sensitive; retain the exact project taxonomy string and avoid silently reclassifying "
            "Ruminococcaceae/Oscillospiraceae labels from other releases."
        ),
        "expected_a": 0,
        "expected_b": 0,
        "evidence_strength": "very_low",
        "sources": "MetaPhlAn/SGB public profiling resources; no direct mechanistic paper identified",
    },
    "GGB9707_SGB15229": {
        "characterization": "Uncharacterized species-level genome bin; family-level placement is database-version sensitive.",
        "diet_evidence": (
            "Appears among SGBs positively associated with healthy diet/cardiometabolic markers in an updated PREDICT/ZOE-derived "
            "ranking and increased in an avocado dietary intervention supplementary analysis."
        ),
        "metabolic_function": "No validated species-specific mechanism; any family-level SCFA inference would be speculative.",
        "glucose_evidence": (
            "External PREDICT-derived ranking supports a pro-health association, but direct peer-reviewed species-specific glucose mechanism is lacking."
        ),
        "caveat": "The strongest explicit 'pro-health' SGB ranking located is patent-derived; treat it as secondary concordance, not primary biological proof.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "low_to_moderate_association_only",
        "sources": "PMID:33432175 framework; US20260022421A1 Table 1B; Food&Function SI d4fo03806a",
    },
    "GGB9758_SGB15368": {
        "characterization": "Uncharacterized Firmicutes/Clostridia species-level genome bin.",
        "diet_evidence": (
            "Reported to increase after a healthy Australian diet in a recent feeding trial abstract and listed as pro-health "
            "in an updated PREDICT/ZOE-derived SGB ranking."
        ),
        "metabolic_function": "Species-level mechanism remains unknown.",
        "glucose_evidence": (
            "Direct glucose evidence is sparse. It has also been depleted in IBD but appears in colorectal-cancer microbial signatures, "
            "showing that disease associations are context-dependent."
        ),
        "caveat": "Do not call it universally beneficial; external evidence is mixed and partly non-mechanistic.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "low_mixed",
        "sources": "PMCID:PMC12265159; Proceedings of the Nutrition Society 2026 HAD/TAD abstract; US20260022421A1",
    },
    "Lachnospiraceae_unclassified_SGB4882": {
        "characterization": "Uncharacterized Lachnospiraceae SGB; species-level function is unresolved.",
        "diet_evidence": (
            "Listed as a pro-health SGB in an updated PREDICT/ZOE-derived ranking; direct controlled-diet evidence is limited."
        ),
        "metabolic_function": (
            "Lachnospiraceae broadly include carbohydrate fermenters and SCFA producers, but those family-level properties cannot "
            "be assigned to SGB4882 without genomic/functional confirmation."
        ),
        "glucose_evidence": (
            "No direct species-level glucose evidence identified. A recent immune profiling preprint links this SGB/near-equivalent "
            "taxon to multiple immune-cell traits, which is hypothesis-generating only."
        ),
        "caveat": "Family-level 'butyrate producer' claims would be over-interpretation for this uncharacterized bin.",
        "expected_a": 1,
        "expected_b": -1,
        "evidence_strength": "low",
        "sources": "US20260022421A1; medRxiv 2025.12.02.25341510v2",
    },
}


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y", "t"})


def sign_num(x) -> int:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v) or v == 0:
        return 0
    return 1 if v > 0 else -1


def sign_text(x) -> str:
    s = sign_num(x)
    return "+" if s > 0 else "-" if s < 0 else "0/NA"


def get_label(row: pd.Series) -> str:
    if "species_label" in row.index and pd.notna(row["species_label"]):
        return str(row["species_label"]).strip()
    species = str(row["species"])
    if "|s__" in species:
        return species.split("|s__")[-1]
    return species


def load_model(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    for c in KEY:
        d[c] = d[c].astype(str).str.strip()
    d["candidate_mediator_consistent"] = bool_series(d["candidate_mediator_consistent"])
    d["mediation_FDR05_primary"] = bool_series(d["mediation_FDR05_primary"])
    d["species_label_derived"] = d.apply(get_label, axis=1)
    return d


def robust_status(path: Path, core_keys: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not path.is_file():
        out = core_keys[KEY].copy()
        out[f"{prefix}_present"] = False
        out[f"{prefix}_significant"] = np.nan
        out[f"{prefix}_consistent"] = np.nan
        return out
    d = load_model(path)
    x = core_keys[KEY].merge(d, on=KEY, how="left", validate="one_to_one")
    return pd.DataFrame({
        **{c: x[c] for c in KEY},
        f"{prefix}_present": x["indirect_effect"].notna(),
        f"{prefix}_significant": x["mediation_FDR05_primary"],
        f"{prefix}_consistent": x["candidate_mediator_consistent"],
    })


def pathway_phrase(a: float, b: float, score: str) -> str:
    sa, sb = sign_num(a), sign_num(b)
    if sa > 0 and sb < 0:
        return f"higher {score} → higher species abundance → lower glucose_cv"
    if sa < 0 and sb > 0:
        return f"higher {score} → lower species abundance → lower glucose_cv"
    if sa > 0 and sb > 0:
        return f"higher {score} → higher species abundance → higher glucose_cv"
    if sa < 0 and sb < 0:
        return f"higher {score} → lower species abundance → higher glucose_cv"
    return "direction unavailable"


def concordance(a: float, b: float, ea: int, eb: int) -> str:
    if ea == 0 or eb == 0:
        return "not_judged_external_evidence_insufficient_or_mixed"
    sa, sb = sign_num(a), sign_num(b)
    if sa == ea and sb == eb:
        return "concordant_with_external_orientation"
    if sa == -ea and sb == -eb:
        return "opposite_to_external_orientation"
    return "partially_concordant_or_mixed"


def main() -> int:
    if not PRIMARY.is_file():
        raise FileNotFoundError(PRIMARY)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = load_model(PRIMARY)

    cv = p.loc[
        p["cgm_outcome"].eq("glucose_cv")
        & p["diet_score"].isin(["AMED", "hPDI"])
        & p["candidate_mediator_consistent"]
    ].copy()

    amed = set(cv.loc[cv["diet_score"].eq("AMED"), "species"])
    hpdi = set(cv.loc[cv["diet_score"].eq("hPDI"), "species"])
    core = sorted(amed & hpdi)

    if len(core) != 12:
        raise RuntimeError(
            f"Expected 12 recurrent core species, derived {len(core)}. "
            "Check canonical Step10/Step10b outputs before biological annotation."
        )

    long = p.loc[
        p["species"].isin(core)
        & p["diet_score"].isin(["AMED", "hPDI"])
        & p["cgm_outcome"].eq("glucose_cv")
    ].copy()

    # Every core species must contribute exactly two paths.
    counts = long.groupby("species").size()
    if not counts.eq(2).all() or len(long) != 24:
        raise RuntimeError(
            "Expected exactly AMED+hPDI paths for each core species (24 rows total)."
        )

    keep = [
        c for c in [
            "diet_score", "cgm_outcome", "species", "species_label_derived", "N",
            "a_diet_to_microbiome", "b_microbiome_to_cgm", "indirect_effect",
            "indirect_ci_low", "indirect_ci_high", "FDR_BH_within_score_outcome",
            "mediation_FDR05_primary", "candidate_mediator_consistent",
        ] if c in long.columns
    ]
    long = long[keep].copy()
    long["a_sign"] = long["a_diet_to_microbiome"].map(sign_text)
    long["b_sign"] = long["b_microbiome_to_cgm"].map(sign_text)
    long["indirect_sign"] = long["indirect_effect"].map(sign_text)
    long["project_path_interpretation"] = long.apply(
        lambda r: pathway_phrase(
            r["a_diet_to_microbiome"],
            r["b_microbiome_to_cgm"],
            r["diet_score"],
        ),
        axis=1,
    )

    # Attach optional robustness status to the 24 exact core paths.
    for path, prefix in [
        (STRICT, "strict"),
        (BROAD, "broad_abxppi"),
        (PROTOCOL, "protocol_abxppi"),
    ]:
        rs = robust_status(path, long, prefix)
        long = long.merge(rs, on=KEY, how="left", validate="one_to_one")

    long.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")

    # Species-level one-row annotation.
    rows = []
    for species in core:
        g = long.loc[long["species"].eq(species)].sort_values("diet_score")
        label = str(g["species_label_derived"].iloc[0])
        lit = LIT.get(label, {
            "characterization": "No curated species-specific annotation yet.",
            "diet_evidence": "Insufficient.",
            "metabolic_function": "Insufficient.",
            "glucose_evidence": "Insufficient.",
            "caveat": "Manual review required.",
            "expected_a": 0,
            "expected_b": 0,
            "evidence_strength": "very_low",
            "sources": "",
        })

        d = {
            "species": species,
            "species_label": label,
            "literature_evidence_strength": lit["evidence_strength"],
            "characterization": lit["characterization"],
            "diet_evidence": lit["diet_evidence"],
            "metabolic_function_or_plausibility": lit["metabolic_function"],
            "glucose_or_metabolic_evidence": lit["glucose_evidence"],
            "interpretive_caveat": lit["caveat"],
            "literature_sources": lit["sources"],
        }

        for score in ["AMED", "hPDI"]:
            r = g.loc[g["diet_score"].eq(score)].iloc[0]
            d[f"{score}_a"] = r["a_diet_to_microbiome"]
            d[f"{score}_b"] = r["b_microbiome_to_cgm"]
            d[f"{score}_indirect"] = r["indirect_effect"]
            d[f"{score}_a_sign"] = sign_text(r["a_diet_to_microbiome"])
            d[f"{score}_b_sign"] = sign_text(r["b_microbiome_to_cgm"])
            d[f"{score}_indirect_sign"] = sign_text(r["indirect_effect"])
            d[f"{score}_project_pattern"] = r["project_path_interpretation"]
            d[f"{score}_external_concordance"] = concordance(
                r["a_diet_to_microbiome"],
                r["b_microbiome_to_cgm"],
                lit["expected_a"],
                lit["expected_b"],
            )

        # A concise project-oriented interpretation based only on observed signs.
        amed_r = g.loc[g["diet_score"].eq("AMED")].iloc[0]
        hpdi_r = g.loc[g["diet_score"].eq("hPDI")].iloc[0]
        a_same = sign_num(amed_r["a_diet_to_microbiome"]) == sign_num(hpdi_r["a_diet_to_microbiome"])
        b_same = sign_num(amed_r["b_microbiome_to_cgm"]) == sign_num(hpdi_r["b_microbiome_to_cgm"])
        if a_same and b_same:
            d["cross_diet_direction_summary"] = (
                f"AMED and hPDI show the same sign pattern: "
                f"a={sign_text(amed_r['a_diet_to_microbiome'])}, "
                f"b={sign_text(amed_r['b_microbiome_to_cgm'])}, "
                f"indirect={sign_text(amed_r['indirect_effect'])}."
            )
        else:
            d["cross_diet_direction_summary"] = (
                "AMED and hPDI retain direction-consistent negative indirect effects "
                "but differ in at least one component sign; inspect path rows."
            )

        rows.append(d)

    annot = pd.DataFrame(rows)
    annot.to_csv(OUT_ANNOT, index=False, encoding="utf-8-sig")

    # Markdown report.
    md = [
        "# 12-core species biological annotation",
        "",
        "## Scope",
        "",
        "These are the recurrent **AMED/hPDI × glucose_cv** direction-consistent core species. "
        "They are not a core across all CGM outcomes. The mediation analysis is cross-sectional "
        "and is not interpreted as causal.",
        "",
        "## Project direction key",
        "",
        "- `a`: diet score → species CLR-Z",
        "- `b`: species CLR-Z → glucose_cv Z, conditional on diet score and covariates",
        "- `indirect = a × b`",
        "- A negative indirect effect means the species path is aligned with lower glucose variability "
        "for a higher healthy-diet score, but this does not identify a causal mechanism.",
        "",
    ]

    for _, r in annot.iterrows():
        md += [
            f"## {r['species_label']}",
            "",
            f"- **Exact project taxonomy:** `{r['species']}`",
            f"- **Evidence strength:** {r['literature_evidence_strength']}",
            f"- **AMED path:** {r['AMED_project_pattern']} "
            f"(a={r['AMED_a']:.6g}, b={r['AMED_b']:.6g}, indirect={r['AMED_indirect']:.6g})",
            f"- **hPDI path:** {r['hPDI_project_pattern']} "
            f"(a={r['hPDI_a']:.6g}, b={r['hPDI_b']:.6g}, indirect={r['hPDI_indirect']:.6g})",
            f"- **Cross-diet pattern:** {r['cross_diet_direction_summary']}",
            f"- **Diet literature:** {r['diet_evidence']}",
            f"- **Metabolic/function plausibility:** {r['metabolic_function_or_plausibility']}",
            f"- **Glucose/metabolic literature:** {r['glucose_or_metabolic_evidence']}",
            f"- **Caveat:** {r['interpretive_caveat']}",
            f"- **External concordance:** AMED={r['AMED_external_concordance']}; "
            f"hPDI={r['hPDI_external_concordance']}",
            f"- **Sources:** {r['literature_sources']}",
            "",
        ]

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Print compact table for screenshot / chat review.
    display = annot[
        [
            "species_label",
            "literature_evidence_strength",
            "AMED_a_sign",
            "AMED_b_sign",
            "AMED_indirect_sign",
            "hPDI_a_sign",
            "hPDI_b_sign",
            "hPDI_indirect_sign",
            "AMED_external_concordance",
            "hPDI_external_concordance",
        ]
    ].copy()

    lines = [
        "=== STEP 13a 12-CORE BIOLOGICAL ANNOTATION ===",
        f"CORE_N={len(core)}",
        f"PATH_ROWS={len(long)}",
        "",
        display.to_string(index=False),
        "",
        "Interpretation rule: literature annotation is contextual evidence only; "
        "do not promote mediation-style associations to causal mechanisms.",
        "",
        f"LONG_EFFECTS={OUT_LONG}",
        f"ANNOTATION_CSV={OUT_ANNOT}",
        f"ANNOTATION_MD={OUT_MD}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
