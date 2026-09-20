"""Descriptive follow-up of frozen results. No model fitting or new FDR."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import KEYS, TAGS, DOMAINS, bools, num, require, need, align


def pairwise(frame, min_n=100):
    """Rank within each pair's complete cases, including ties; never rank then drop."""
    fields = list(frame)
    rho = pd.DataFrame(np.nan, index=fields, columns=fields)
    pearson = rho.copy()
    ns = pd.DataFrame(0, index=fields, columns=fields, dtype=int)
    rows = []
    for i, a in enumerate(fields):
        for j in range(i, len(fields)):
            b = fields[j]
            x, y = num(frame[a]), num(frame[b])
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            ns.loc[a, b] = ns.loc[b, a] = n
            status = "LOW_N" if n < min_n else "CONSTANT" if x[mask].nunique() < 2 or y[mask].nunique() < 2 else "OK"
            r = p = np.nan
            if status == "OK":
                p = float(x[mask].corr(y[mask]))
                r = float(x[mask].rank(method="average").corr(y[mask].rank(method="average")))
                rho.loc[a, b] = rho.loc[b, a] = r
                pearson.loc[a, b] = pearson.loc[b, a] = p
            if i != j:
                rows.append(dict(outcome_1=a, outcome_2=b, pairwise_N=n, spearman=r, pearson=p, status=status))
    return rho, pearson, ns, pd.DataFrame(rows)


def correlations(ctx, cg, masks, dictionary, plan, plot):
    fields = dictionary.field.tolist()
    need(cg, [f + "_z" for f in fields], "all25 standardized CGM fields")
    all_masks = {"all_frozen_CGM": pd.Series(True, index=cg.index)}
    all_masks.update(masks)
    summaries, allpairs, qc = [], [], []
    for name, mask in all_masks.items():
        frame = cg.loc[mask, [f + "_z" for f in fields]].copy()
        frame.columns = fields
        frame = frame.apply(num)
        rho, pearson, ns, pairs = pairwise(frame, plan["min_pair_n"])
        pairs.insert(0, "cohort", name)
        allpairs.append(pairs)
        for label, matrix in [("spearman", rho), ("pearson", pearson), ("pairwise_N", ns)]:
            ctx.csv(matrix.rename_axis("outcome_field").reset_index(), "20_{}_{}.csv".format(name, label))
        for field in fields:
            x = frame[field]
            qc.append(dict(cohort=name, field=field, modeled_column=field + "_z", N_cohort=len(frame),
                           N_valid=int(x.notna().sum()), missing_n=int(x.isna().sum()), unique_values=int(x.nunique()),
                           mean=x.mean(), SD=x.std(ddof=1)))
        summaries.append(dict(cohort=name, N=len(frame), mean_gmi_pair_N=ns.loc["cgm_mean", "cgm_gmi"],
                              mean_gmi_spearman=rho.loc["cgm_mean", "cgm_gmi"],
                              mean_gmi_pearson=pearson.loc["cgm_mean", "cgm_gmi"]))
        if name in ["primary", "strict"] and plot:
            from plots import correlation_plot
            primary = dictionary[dictionary.family.eq("paper20_primary")].field.tolist()
            correlation_plot(ctx, rho.loc[primary, primary], dictionary, name, len(frame))
    combined = pd.concat(allpairs, ignore_index=True)
    ctx.csv(combined, "20_pairwise_correlations_all_cohorts.csv")
    ctx.csv(combined[combined.spearman.abs().ge(plan["high_correlation_threshold"])], "20_high_correlation_pairs.csv")
    ctx.csv(pd.DataFrame(qc), "20_correlation_missingness.csv")
    ctx.csv(pd.DataFrame(summaries), "20_correlation_cohort_summary.csv")
    return dict(cohorts=summaries, value_scale="existing outcome_z; no restandardization; pairwise complete observations", min_pair_N=plan["min_pair_n"])


def safe_ratio(n, d):
    return n / d if d else np.nan


def aggregate_paths(d, metrics):
    candidate, primary = len(d), int(d.robust_primary.sum())
    both = int(d.highest_robustness.sum())
    return dict(registered_metrics=metrics, candidate_paths=candidate, primary_robust=primary,
                retained_strict=int(d.primary_retained_strict.sum()), retained_protocol=int(d.primary_retained_protocol.sum()),
                retained_both=both, both_over_candidates=safe_ratio(both, candidate), both_over_primary=safe_ratio(both, primary),
                unique_species_candidates=d.species.nunique(), unique_species_retained=d.loc[d.highest_robustness, "species"].nunique(),
                unique_diet_species_candidates=d[["exposure", "species"]].drop_duplicates().shape[0],
                unique_diet_species_retained=d.loc[d.highest_robustness, ["exposure", "species"]].drop_duplicates().shape[0])


def domain_analysis(ctx, paths, dictionary, plan, plot):
    primary = dictionary[dictionary.family.eq("paper20_primary")]
    domain_map = dict(zip(primary.field, primary.domain))
    require(paths.outcome_field.isin(domain_map).all(), "Candidate outcome not in frozen paper20 dictionary")
    paths = paths.copy()
    paths["domain"] = paths.outcome_field.map(domain_map)
    panels = {"full20": primary.field.tolist(), "no_GMI19": primary[primary.field.ne("cgm_gmi")].field.tolist(),
              "concept_panel9": plan["representative_fields"]}
    rows, per_outcome, per_exposure = [], [], []
    for panel, fields in panels.items():
        for domain in DOMAINS:
            selected = [f for f in fields if domain_map[f] == domain]
            sub = paths[paths.outcome_field.isin(selected)]
            rows.append(dict(panel=panel, domain=domain, **aggregate_paths(sub, len(selected))))
    for f in primary.field:
        per_outcome.append(dict(outcome_field=f, domain=domain_map[f], **aggregate_paths(paths[paths.outcome_field.eq(f)], 1)))
    for exposure in sorted(paths.exposure.unique()):
        for domain in DOMAINS:
            selected = primary[primary.domain.eq(domain)].field.tolist()
            sub = paths[paths.exposure.eq(exposure) & paths.outcome_field.isin(selected)]
            per_exposure.append(dict(exposure=exposure, domain=domain, **aggregate_paths(sub, len(selected))))
    domains = pd.DataFrame(rows)
    ctx.csv(domains, "20_domain_denominators_and_retention.csv")
    ctx.csv(pd.DataFrame(per_outcome), "20_outcome_denominators_and_retention.csv")
    ctx.csv(pd.DataFrame(per_exposure), "20_diet_domain_denominators_and_retention.csv")
    # Set-overlap quantifies redundancy; correlated paths are not independent replications.
    redundancy = []
    for selection, mask in [("all_candidates", pd.Series(True, index=paths.index)), ("primary_robust", paths.robust_primary), ("retained_both", paths.highest_robustness)]:
        p = paths[mask]
        sets = [set(map(tuple, p.loc[p.outcome_field.eq(f), ["exposure", "species"]].to_numpy())) for f in ["cgm_mean", "cgm_gmi"]]
        a, b = sets
        redundancy.append(dict(stage=selection, mean_paths=len(a), gmi_paths=len(b), intersection=len(a & b), union=len(a | b),
                               Jaccard=safe_ratio(len(a & b), len(a | b)), mean_only=len(a - b), gmi_only=len(b - a)))
    ctx.csv(pd.DataFrame(redundancy), "20_mean_gmi_path_redundancy.csv")
    old_file = ctx.root / "diet_microbiome_glucose_analysis/4New_cgm/outputs/paper20_mediation_phenotype_domains/reports/18g_mean_gmi_redundancy_audit.csv"
    old_status = "NOT_AVAILABLE"
    if old_file.is_file():
        old = ctx.read(old_file, "original_18g_mean_GMI_redundancy")
        require(len(old) == 1, "18g Mean/GMI audit must have one row")
        mapping = {"mean_highest_robust_paths": "mean_paths", "gmi_highest_robust_paths": "gmi_paths",
                   "shared_exposure_species_pairs": "intersection", "union_exposure_species_pairs": "union",
                   "jaccard_mean_vs_gmi": "Jaccard"}
        need(old, mapping, "18g Mean/GMI audit")
        for before, after in mapping.items():
            require(np.isclose(float(old.iloc[0][before]), redundancy[-1][after], equal_nan=True), "18g Mean/GMI differs: " + before)
        ctx.csv(old, "20_original_18g_mean_gmi_audit_verified.csv")
        old_status = "PASS"
    membership = []
    for panel, fields in panels.items():
        selected = paths[paths.highest_robustness & paths.outcome_field.isin(fields)]
        for species in sorted(paths.loc[paths.highest_robustness, "species"].unique()):
            sub = selected[selected.species.eq(species)]
            membership.append(dict(panel=panel, species=species, retained_paths=len(sub),
                                   domain_breadth=sub.domain.nunique(), domains=";".join(d for d in DOMAINS if d in set(sub.domain))))
    ctx.csv(pd.DataFrame(membership), "20_taxon_domain_breadth_redundancy_checks.csv")
    if plot:
        from plots import domain_plot
        domain_plot(ctx, domains)
    return dict(full20=domains[domains.panel.eq("full20")].to_dict("records"), inferential_enrichment_test=False, original_18g_redundancy_comparison=old_status,
                reduced_panels="Post-results descriptive sensitivity; original frozen labels unchanged")


def methods_audit(ctx, paths, models, plan):
    summaries, full, risks, families = [], [], [], []
    for tag in TAGS:
        m = align(paths, models[tag]).copy()
        fields = ["indirect_effect", "direct_effect", "total_effect", "N", "sample_sha256", "covariates",
                  "indirect_CI95_lower", "indirect_CI95_upper", "FDR_BH_within_score_outcome", "FDR_BH_global_all_mediation"]
        need(m, fields, tag + " model")
        for col in ["FDR_BH_within_score_outcome", "FDR_BH_global_all_mediation"]:
            q = num(m[col])
            require(q.dropna().between(0, 1).all(), "Invalid FDR range in " + tag)
        global_q = num(m.FDR_BH_global_all_mediation)
        within = num(m.FDR_BH_within_score_outcome)
        retained = paths.highest_robustness
        summaries.append(dict(model=tag, frozen_candidates=len(m), within_FDR05=int(within.lt(.05).sum()),
                              global_FDR05=int(global_q.lt(.05).sum()), global_FDR_missing=int(global_q.isna().sum()),
                              retained871_global_FDR05=int((retained & global_q.lt(.05)).sum()),
                              N_min=num(m.N).min(), N_median=num(m.N).median(), N_max=num(m.N).max()))
        for (e, o), g in m.groupby(["exposure", "outcome_field"]):
            families.append(dict(model=tag, exposure=e, outcome_field=o, frozen_mediator_family_size=len(g),
                                 FDR_available=int(num(g.FDR_BH_within_score_outcome).notna().sum())))
        total, indirect = num(m.total_effect), num(m.indirect_effect)
        zero = total.abs().le(plan["total_near_zero_threshold"])
        inconsistent = (total * indirect).lt(0)
        for i in np.flatnonzero((zero | inconsistent).to_numpy()):
            risks.append(dict(m.loc[i, KEYS], model=tag, total_effect=total.iloc[i], indirect_effect=indirect.iloc[i],
                              total_near_zero=bool(zero.iloc[i]), opposite_total_indirect=bool(inconsistent.iloc[i]),
                              retained_both=bool(retained.iloc[i])))
        export = m.copy()
        export.insert(0, "model", tag)
        # Model result tables contain aggregate effects, never participant records.
        export["frozen_retained_both"] = retained.to_numpy()
        full.append(export)
    ctx.csv(pd.DataFrame(summaries), "21_existing_global_FDR_sensitivity.csv")
    ctx.csv(pd.DataFrame(families), "21_frozen_mediation_FDR_families.csv")
    ctx.csv(pd.DataFrame(risks, columns=KEYS + ["model", "total_effect", "indirect_effect", "total_near_zero", "opposite_total_indirect", "retained_both"]), "21_mediated_proportion_interpretation_flags.csv")
    ctx.csv(pd.concat(full, ignore_index=True), "21_all1490_three_model_results.csv")
    return dict(original_FDR_only=True, total_near_zero_descriptive_threshold=plan["total_near_zero_threshold"], summaries=summaries)


def mechanism_queue(ctx, high, dictionary):
    seed_file = ctx.package / "config/literature_starting_evidence.csv"
    if seed_file.is_file():
        seed = ctx.read(seed_file, "reviewed_literature_starting_evidence")
        ctx.csv(seed, "22_reviewed_literature_starting_evidence.csv")
    rows = []
    for species, group in high.groupby("species", sort=True):
        r = group.iloc[0]
        for (exposure, domain), g in group.groupby(["exposure", "phenotype_domain"]):
            positive = int(num(g["model_primary__indirect_effect"]).gt(1e-12).sum())
            negative = int(num(g["model_primary__indirect_effect"]).lt(-1e-12).sum())
            rows.append(dict(species=species, taxon_id=r.taxon_id, display_species=r.display_species,
                             domain_breadth=int(r.domain_breadth), exposure=exposure, domain=domain,
                             paths=len(g), indirect_positive=positive, indirect_negative=negative,
                             outcomes=";".join(sorted(g.outcome_field.unique())),
                             literature_review_status="PENDING_PATH_SPECIFIC_REVIEW",
                             species_function_in_HPP="NOT_MEASURED_BY_TAXON_ABUNDANCE",
                             extrapolation_limit="No nearest-species functional assignment" if "SGB" in str(r.display_species) else "Species/strain evidence must match; no causal inference"))
    queue = pd.DataFrame(rows).sort_values(["domain_breadth", "taxon_id", "exposure", "domain"], ascending=[False, True, True, True])
    ctx.csv(queue, "22_mechanism_path_contexts.csv")
    matrix = high[["species", "taxon_id", "display_species", "domain_breadth"]].drop_duplicates().sort_values(["domain_breadth", "taxon_id"], ascending=[False, True])
    for c in ["citation_DOI", "study_population", "study_design", "strain_match", "diet_substrate", "metabolite_pathway", "observed_direction", "concordance_with_HPP", "contradictions", "evidence_level", "reviewer_note"]:
        matrix[c] = ""
    matrix["review_status"] = "NOT_REVIEWED"
    ctx.csv(matrix, "22_species_evidence_matrix_to_complete.csv")
    if seed_file.is_file():
        annotated = matrix[["species", "taxon_id", "display_species", "domain_breadth"]].merge(seed, on="display_species", how="left", validate="many_to_many")
        annotated["HPP_strain_function_match"] = "NOT_ESTABLISHED"
        annotated["link_basis"] = "Display-name literature link only; full HPP taxonomy retained and strain equivalence unverified"
        ctx.csv(annotated, "22_Hpp_taxa_and_external_evidence_links.csv")
    inventory = []
    for top in ctx.root.iterdir():
        if top.is_dir() and any(word in top.name.lower() for word in ["metabol", "microbio", "lipid", "function", "pathway", "humann"]):
            for path in top.rglob("*"):
                if path.is_file() and path.suffix.lower() in [".csv", ".tsv", ".parquet", ".biom"] and any(w in path.name.lower() for w in ["metabol", "pathway", "function", "humann", "gene", "ko_", "lipid"]):
                    inventory.append(dict(path=str(path), bytes=path.stat().st_size, status="FILE_EXISTS_CONTENT_AND_LINKAGE_NOT_AUDITED"))
    ctx.csv(pd.DataFrame(inventory, columns=["path", "bytes", "status"]), "22_function_metabolite_file_inventory.csv")
    return dict(species=len(matrix), path_contexts=len(queue), candidate_function_files=len(inventory),
                evidence_status="Structured worklist ready; species/strain-specific mechanism review remains open")
