"""Explicitly synthetic source tree; never used by the production entry point."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import KEYS, TAGS, DOMAINS, sha256, sample_hash, write_json
from cohort import BASE, covariates
from run_followup import PACKAGE


def fixture(root, full=False):
    root = Path(root)
    dictionary = pd.read_csv(PACKAGE / "config/outcome_dictionary.csv")
    plan = json.loads((PACKAGE / "config/analysis_plan.json").read_text())
    fields = dictionary.field.tolist()
    primary_fields = dictionary[dictionary.family.eq("paper20_primary")].field.tolist()
    domain_map = dictionary.set_index("field").domain.to_dict()
    pid = pd.Series(["SYNTHETIC_{:04d}".format(i) for i in range(160)])
    rng = np.random.RandomState(20260920)
    cgm = pd.DataFrame(dict(participant_id=pid, cohort="10k", research_stage="00_00_visit", array_index="0", connection_id=pid + "_conn"))
    source = cgm.copy()
    for i, f in enumerate(fields):
        values = pd.Series(5 + rng.uniform(0, 60, len(pid)))
        if f == "cgm_gmi":
            values = cgm.cgm_mean_clean * .02 + 3
        values.iloc[10 + i % 4] = np.nan
        source[f] = values
        cgm[f + "_raw"] = values
        cgm[f + "_clean"] = values
        cgm[f + "_z"] = (values - values.mean()) / values.std(ddof=1)

    def csv(frame, rel):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    cgmpath = csv(cgm, "cgm_deal/cgm论文新增17指标/outputs/run_SYNTHETIC/data/07_cgm_paper_extended_phenotypes.csv")
    csv(source, "Transfer/cgm/iglu.csv")
    dg = "diet_microbiome_glucose_analysis/"
    old = pd.DataFrame(dict(participant_id=pid, AHEI_z=rng.normal(size=len(pid)), AMED_z=rng.normal(size=len(pid)), rEDIH_z=rng.normal(size=len(pid))))
    csv(old, dg + "outputs/data/00_current4_diet_cgm_cohort.csv")
    new = pd.DataFrame(dict(participant_id=pid, EAT13_z=rng.normal(size=len(pid)), NOVA4_z=rng.normal(size=len(pid)), Carbohydrate_pct_z=rng.normal(size=len(pid)), mean_daily_energy_kcal=2000.))
    new.loc[5, "mean_daily_energy_kcal"] = np.nan
    csv(new, dg + "outputs/new_diet_extension/data/15d0_new_diet_cgm_cohort.csv")
    hp = pd.DataFrame(dict(participant_id=pid, hpdi_score_energy_adjusted_z=rng.normal(size=len(pid))))
    csv(hp, dg + "outputs/reports/new_diet_extension/hpdi_correction_audit/corrected_scores/hpdi_participant_scores.csv")
    cov = pd.DataFrame({c: np.ones(len(pid)) for c in BASE})
    cov["participant_id"] = pid
    cov["exclude_diabetes_or_a10"] = 0.
    cov.loc[0, "exclude_diabetes_or_a10"] = 1
    cov.loc[1, "exclude_diabetes_or_a10"] = np.nan
    cov.loc[4, "age_years"] = np.nan
    cov["alcohol_intake_g_day"] = 0.
    cov.loc[6, "alcohol_intake_g_day"] = np.nan
    csv(cov, "co-variant/outputs/data/02_covariate_master.csv")
    proto = pd.DataFrame(dict(participant_id=pid, antibiotic_use_protocol_window=0., ppi_use_protocol_window=0.))
    proto.loc[2, "antibiotic_use_protocol_window"] = np.nan
    csv(proto, dg + "outputs/new_diet_extension/mediation/protocol_abxppi/15f2_protocol_abxppi_cohort.csv")
    exposures = ["AHEI", "AMED", "hPDI", "rEDIH", "EAT13", "NOVA4", "Carbohydrate_pct"]
    taxids = ["T023", "T046", "T068", "T096"] + ["TTEST{:03d}".format(i) for i in range(105)]
    taxa = ["k__SYNTHETIC|g__Test|s__Species_" + t for t in taxids]
    rows = []
    if full:
        allowed = {s: DOMAINS if i < 4 else DOMAINS[:2] if i < 21 else [DOMAINS[0]] if i < 39 else [DOMAINS[1]] if i < 107 else [DOMAINS[2]] for i, s in enumerate(taxa)}
        for domain in DOMAINS:
            target = plan["expected"]["domains"][domain]
            count = 0
            for e in exposures:
                for o in [f for f in primary_fields if domain_map[f] == domain]:
                    for s in taxa:
                        if domain in allowed[s] and count < target:
                            rows.append((e, o, s)); count += 1
        seen = set(rows)
        for e in exposures:
            for o in primary_fields:
                for s in taxa:
                    if len(rows) < 1490 and (e, o, s) not in seen:
                        rows.append((e, o, s)); seen.add((e, o, s))
        n = len(rows)
        ix = np.arange(n)
        masks = dict(primary=ix < 1003, strict=ix < 887, protocol=(ix < 871) | ((ix >= 887) & (ix < 956)))
    else:
        for s in taxa[:4]:
            for o in ["cgm_mean", "cgm_lbgi", "cgm_cv", "cgm_cogi"]:
                rows.append(("AMED", o, s))
        for e in ["NOVA4", "hPDI"]:
            for o in ["cgm_cv", "cgm_modd", "cgm_sdw"]:
                rows.append((e, o, taxa[2]))
        n = len(rows)
        masks = {t: np.ones(n, dtype=bool) for t in TAGS}
    micro = pd.DataFrame({s: rng.normal(size=len(pid)) for s in set(r[2] for r in rows)})
    micro["participant_id"] = pid
    micro.loc[3, micro.columns != "participant_id"] = np.nan
    csv(micro, "gut_microbiome_deal/data/08_species_clr_zscore.csv")
    paths = pd.DataFrame(rows, columns=KEYS)
    paths["highest_robustness"] = masks["primary"] & masks["strict"] & masks["protocol"]
    paths["primary_retained_strict"] = masks["primary"] & masks["strict"]
    paths["primary_retained_protocol"] = masks["primary"] & masks["protocol"]
    models, modelpaths = {}, {}
    for tag in TAGS:
        m = paths[KEYS].copy()
        for field in ["robust", "significant", "consistent"]:
            paths[field + "_" + tag] = masks[tag]
        m["a_diet_to_microbiome"] = np.where(np.arange(n) % 2, -.1, .1)
        m["b_microbiome_to_cgm"] = .2
        m["indirect_effect"] = m.a_diet_to_microbiome * .2
        m["direct_effect"] = .02
        m["total_effect"] = m.indirect_effect + .02
        m["indirect_CI95_lower"] = m.indirect_effect - .009
        m["indirect_CI95_upper"] = m.indirect_effect + .009
        m["FDR_BH_within_score_outcome"] = np.where(masks[tag], .01, .5)
        m["FDR_BH_global_all_mediation"] = .04
        m["N"] = 0
        m["sample_sha256"] = ""
        m["covariates"] = ""
        for i, r in m.iterrows():
            excluded = {0, 3, 4, 10 + fields.index(r.outcome_field) % 4}
            # GMI retains mean's missing location too.
            if r.outcome_field == "cgm_gmi":
                excluded.add(10 + fields.index("cgm_mean") % 4)
            if tag == "strict": excluded.add(1)
            if tag == "protocol": excluded.add(2)
            if r.exposure == "hPDI": excluded.add(6)
            if r.exposure in ["EAT13", "NOVA4", "Carbohydrate_pct"]: excluded.add(5)
            keep = pid[~pid.index.isin(excluded)]
            m.loc[i, "N"] = len(keep)
            m.loc[i, "sample_sha256"] = sample_hash(keep)
            m.loc[i, "covariates"] = ";".join(covariates(r.exposure, tag))
        for f in ["N", "indirect_effect"]:
            paths[f + "_" + tag] = m[f]
        path = csv(m, dg + "4New_cgm/outputs/synthetic_" + tag + "/models/model.csv")
        meta = path.parent.parent / "reports/manifest.json"
        meta.parent.mkdir()
        write_json(meta, dict(status="completed", debug_subset=False, final_cgm=str(cgmpath)))
        models[tag], modelpaths[tag] = m, str(path)
    pathsfile = csv(paths, dg + "4New_cgm/outputs/paper20_mediation_threeway_robustness/reports/18f_threeway_path_robustness.csv")
    high = paths[paths.highest_robustness].copy()
    high["phenotype_domain"] = high.outcome_field.map(domain_map)
    breadth = high.groupby("species").phenotype_domain.nunique()
    high["domain_breadth"] = high.species.map(breadth)
    high["taxon_id"] = high.species.map(dict(zip(taxa, taxids)))
    high["display_species"] = high.taxon_id.map(lambda t: "SYNTHETIC_" + t)
    for tag, m in models.items():
        for col in m:
            if col not in KEYS:
                high["model_" + tag + "__" + col] = m.loc[high.index, col]
    folder = root / (dg + "5Step19b_core_taxa_direction/outputs/" + plan["known_step19b_run"])
    highfile = csv(high, str(folder.relative_to(root)) + "/19b_path_direction_details.csv")
    inputs = [dict(path=str(f), label=f.name, sha256=sha256(f)) for f in [pathsfile] + list(map(Path, modelpaths.values()))]
    write_json(folder / "manifest.json", dict(status="PASS", data_kind="SYNTHETIC_TEST_ONLY", all_inputs_unchanged_verified=True,
                                              inputs=inputs, outputs=[dict(file=highfile.name, sha256=sha256(highfile))], selected_models=modelpaths))
    if not full:
        plan["expected"] = dict(candidates=n, retained=len(high), species=4, primary=n, strict_retained=n, protocol_retained=n,
                                domains=high.phenotype_domain.value_counts().to_dict())
    return plan, folder, cgmpath, models
