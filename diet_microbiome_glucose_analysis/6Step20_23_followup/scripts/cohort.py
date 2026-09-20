"""Reconstruct original complete-case membership; no regressions or imputation."""
import numpy as np
import pandas as pd
from common import FULL_KEY, KEYS, TAGS, need, num, norm_id, unique, require, sample_hash

BASE = ["age_years", "sex", "education_level", "smoking_status", "sleep_duration_hours_day",
        "physical_activity_met_h_week", "vitamin_use", "hormone_use", "bmi", "cgm_device_type"]
CATEGORICAL = {"sex", "education_level", "smoking_status", "cgm_device_type"}
SCORE_CANDIDATES = {
    "AHEI": ["AHEI_z", "ahei_z", "ahei_score_z", "ahei_energy_adjusted_z", "ahei_score_energy_adjusted_z", "mahei7_score_energy_adjusted_z", "mahei7_z"],
    "AMED": ["AMED_z", "amed_z", "amed_score_z", "amed_energy_adjusted_z", "amed_score_energy_adjusted_z"],
    "rEDIH": ["rEDIH_z", "redih_z", "redih_score_z", "redih_energy_adjusted_z", "redih_score_energy_adjusted_z", "redih_energy_adjusted_score_z"],
}


def choose(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError("No supported column among " + str(candidates))


def identified(df, label):
    need(df, ["participant_id"], label)
    df = df.copy()
    df["participant_id"] = norm_id(df["participant_id"])
    unique(df, ["participant_id"], label)
    return df.set_index("participant_id")


def covariates(exposure, tag):
    cols = list(BASE)
    if exposure == "hPDI":
        cols += ["alcohol_intake_g_day"]
    if exposure in ["EAT13", "NOVA4", "Carbohydrate_pct"]:
        cols += ["mean_daily_energy_kcal"]
    if tag == "protocol":
        cols += ["antibiotic_use", "ppi_use"]
    return cols


def valid(s, field):
    if field in CATEGORICAL:
        return s.notna() & s.astype(str).str.strip().ne("") & s.astype(str).str.strip().str.lower().ne("nan")
    return num(s).notna()


def load_population(ctx, cgm, models):
    dg = ctx.root / "diet_microbiome_glucose_analysis"
    old = identified(ctx.read(dg / "outputs/data/00_current4_diet_cgm_cohort.csv", "current4_scores"), "old scores")
    new = identified(ctx.read(dg / "outputs/new_diet_extension/data/15d0_new_diet_cgm_cohort.csv", "new_scores"), "new scores")
    hp = identified(ctx.read(dg / "outputs/reports/new_diet_extension/hpdi_correction_audit/corrected_scores/hpdi_participant_scores.csv", "corrected_hpdi"), "hPDI")
    cov = identified(ctx.read(ctx.root / "co-variant/outputs/data/02_covariate_master.csv", "covariate_master"), "covariates")
    species = sorted(set(models["primary"].species))
    # Only selected species are required for exact path membership. No values exported.
    micro = identified(ctx.read(ctx.root / "gut_microbiome_deal/data/08_species_clr_zscore.csv", "microbiome",
                               usecols=lambda c: c == "participant_id" or c in species), "microbiome")
    proto = identified(ctx.read(dg / "outputs/new_diet_extension/mediation/protocol_abxppi/15f2_protocol_abxppi_cohort.csv", "protocol_medications"), "protocol")
    cg = identified(cgm, "CGM")
    mappings, scores = [], {}
    for exposure, options in SCORE_CANDIDATES.items():
        col = choose(old.columns, options)
        scores[exposure + "_z"] = num(old[col])
        mappings.append(dict(exposure=exposure, source="current4", column=col))
    need(hp, ["hpdi_score_energy_adjusted_z"], "corrected hPDI")
    scores["hPDI_z"] = num(hp.hpdi_score_energy_adjusted_z)
    mappings.append(dict(exposure="hPDI", source="corrected_hpdi", column="hpdi_score_energy_adjusted_z"))
    for e in ["EAT13", "NOVA4", "Carbohydrate_pct"]:
        need(new, [e + "_z"], "new scores")
        scores[e + "_z"] = num(new[e + "_z"])
        mappings.append(dict(exposure=e, source="new_scores", column=e + "_z"))
    score = pd.DataFrame(scores)
    ctx.csv(pd.DataFrame(mappings), "21_score_column_mapping.csv")
    alcohol = choose(cov.columns, ["alcohol_intake_g_day", "alcohol_g_day", "alcohol_intake"])
    need(cov, BASE + ["exclude_diabetes_or_a10", alcohol], "covariates")
    master = score.join(cov[BASE + ["exclude_diabetes_or_a10", alcohol]].rename(columns={alcohol: "alcohol_intake_g_day"}), how="outer")
    need(new, ["mean_daily_energy_kcal"], "energy")
    master = master.join(new[["mean_daily_energy_kcal"]], how="outer").join(cg.filter(regex="^cgm_.*_z$"), how="outer")
    # Eligibility requires an actual covariate row; absent row != unknown exclusion flag.
    flag = num(master.exclude_diabetes_or_a10)
    require(flag.dropna().isin([0, 1]).all(), "Unexpected exclusion flag (must be 0/1/missing)")
    present = master.index.isin(cov.index)
    master["primary_eligible"] = present & ~flag.eq(1)
    master["strict_eligible"] = present & flag.eq(0)
    for field, target in [("antibiotic_use_protocol_window", "antibiotic_use"), ("ppi_use_protocol_window", "ppi_use")]:
        need(proto, [field], "protocol medication")
        x = num(proto[field])
        require(x.dropna().isin([0, 1]).all(), "Nonbinary protocol medication")
        master[target] = x.reindex(master.index)
    flow = []
    masks = {}
    any_score = master[list(scores)].apply(num).notna().any(axis=1)
    common_cov = pd.concat([valid(master[c], c) for c in BASE], axis=1).all(axis=1)
    for tag in ["primary", "strict"]:
        base = master[tag + "_eligible"]
        for name, mask in [("eligible_covariate_record", base), ("plus_CGM_record", base & master.index.isin(cg.index)),
                           ("plus_microbiome_record", base & master.index.isin(cg.index) & master.index.isin(micro.index)),
                           ("plus_any_analyzed_diet", base & master.index.isin(cg.index) & master.index.isin(micro.index) & any_score),
                           ("plus_common_covariates", base & master.index.isin(cg.index) & master.index.isin(micro.index) & any_score & common_cov)]:
            flow.append(dict(cohort=tag, sequential_step=name, N=int(mask.sum())))
        # Descriptive common-base correlation cohort: no all20 completeness or species-specific mask.
        masks[tag] = mask.reindex(cg.index).fillna(False)
    ctx.csv(pd.DataFrame(flow), "21_sequential_common_cohort_flow.csv")
    rows, union_ids = [], {tag: set() for tag in TAGS}
    for tag in TAGS:
        eligible = master["strict_eligible" if tag == "strict" else "primary_eligible"]
        for path_number, (_, r) in enumerate(models[tag].iterrows(), 1):
            if path_number % 250 == 0:
                print("SAMPLE_AUDIT {} {}/{}".format(tag, path_number, len(models[tag])), flush=True)
            e, o, s = r.exposure, r.outcome_field, r.species
            cols = [e + "_z", o + "_z"] + covariates(e, tag)
            need(master, cols, "path inputs")
            need(micro, [s], "species input")
            if "covariates" in r.index:
                recorded = set(str(r["covariates"]).split(";"))
                require(recorded == set(covariates(e, tag)), "Recorded covariate list differs: " + tag + "/" + e)
            before = eligible & master.index.isin(micro.index)
            complete = before.copy()
            for c in cols:
                complete &= valid(master[c], c)
            complete &= num(micro[s]).reindex(master.index).notna()
            ids = pd.Series(master.index[complete], dtype=str)
            n, sh = len(ids), sample_hash(ids)
            same = n == float(r.N) and sh == str(r.sample_sha256)
            union_ids[tag].update(ids)
            rows.append(dict(zip(KEYS, [e, o, s]), model=tag, eligible_microbiome_N=int(before.sum()),
                             complete_case_N=n, excluded_for_missing_N=int(before.sum()) - n,
                             original_N=r.N, sample_sha256=sh, sample_match=bool(same)))
    result = pd.DataFrame(rows)
    ctx.csv(result, "21_path_sample_reconstruction.csv")
    require(result.sample_match.all(), "Path N/sample hash mismatch; see 21_path_sample_reconstruction.csv. No correlation analysis accepted.")
    for tag in ["primary", "strict"]:
        masks[tag + "_path_union"] = pd.Series(cg.index.isin(union_ids[tag]), index=cg.index)
    return cg, masks, dict(paths_checked=len(result), N_hash_matches=int(result.sample_match.sum()),
                           primary_common_base_N=int(masks["primary"].sum()), strict_common_base_N=int(masks["strict"].sum()))


def audit_lineage(ctx, cgm, fields, source_path):
    source = ctx.read(source_path, "official_iglu_export", usecols=lambda c: c in FULL_KEY or c in fields)
    need(cgm, FULL_KEY, "final CGM full connection keys")
    need(source, FULL_KEY, "iglu full connection keys")
    left, right = cgm.copy(), source.copy()
    for c in FULL_KEY:
        for d in [left, right]:
            require(d[c].notna().all(), "Missing connection key")
            d[c] = d[c].astype(str).str.strip()
            if c in ["participant_id", "array_index", "connection_id"]:
                d[c] = d[c].str.replace(r"\.0$", "", regex=True)
    unique(left, FULL_KEY, "CGM keys")
    unique(right, FULL_KEY, "iglu keys")
    joined = left[FULL_KEY].merge(right, on=FULL_KEY, how="left", validate="one_to_one", indicator=True)
    require(joined._merge.eq("both").all(), "Final CGM connection cannot be matched to iglu export")
    rows = []
    for f in fields:
        raw = f + "_raw" if f + "_raw" in cgm else (f if f in cgm else None)
        status, compared, mismatches = "PENDING_COLUMN", 0, None
        if raw and f in joined:
            a, b = num(cgm[raw]), num(joined[f])
            # Illegal source values may correctly have become missing during preprocessing.
            legal = b.ge(0)
            if f in ["cgm_cogi", "cgm_grade_eugly", "cgm_above_140", "cgm_above_180", "cgm_below_70", "cgm_in_range_70_180"]:
                legal &= b.le(100)
            b = b.where(legal)
            eq = np.isclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True)
            mismatches, compared = int((~eq).sum()), int((a.notna() & b.notna()).sum())
            status = "PASS" if mismatches == 0 and compared > 0 else "FAIL"
        clean, z = f + "_clean", f + "_z"
        z_status = "PENDING_COLUMN"
        if clean in cgm and z in cgm:
            c, actual = num(cgm[clean]), num(cgm[z])
            sd = c.std(ddof=1)
            if np.isfinite(sd) and sd > 0:
                expected = (c - c.mean()) / sd
                z_status = "PASS" if np.isclose(expected, actual, rtol=1e-6, atol=1e-7, equal_nan=True).all() else "FAIL"
            else:
                z_status = "UNAVAILABLE_CONSTANT"
        rows.append(dict(outcome_field=f, official_source_column=f, analysis_raw_column=raw,
                         matched_connections=len(joined), numeric_pairs=compared, raw_mismatches=mismatches,
                         raw_lineage_status=status, clean_to_z_status=z_status,
                         historical_iglu_version_parameters="NOT_ESTABLISHED_BY_VALUE_MATCH"))
    report = pd.DataFrame(rows)
    ctx.csv(report, "20_field_lineage_audit.csv")
    return report


def preprocessing_trace(ctx, cgm, cgm_path, fields):
    """Expose cleaning evidence without claiming a historical script was executed."""
    rows = []
    for f in fields:
        raw = num(cgm[f + "_raw"]) if f + "_raw" in cgm else pd.Series(np.nan, index=cgm.index)
        clean = num(cgm[f + "_clean"]) if f + "_clean" in cgm else pd.Series(np.nan, index=cgm.index)
        action = f + "_clean_action"
        rows.append(dict(outcome_field=f, raw_column_present=f + "_raw" in cgm, clean_column_present=f + "_clean" in cgm,
                         raw_valid_N=int(raw.notna().sum()), clean_valid_N=int(clean.notna().sum()),
                         raw_zero_N=int(raw.eq(0).sum()), raw_min=raw.min(), raw_max=raw.max(),
                         finite_raw_to_missing_clean=int((raw.notna() & clean.isna()).sum()),
                         changed_finite_values=int((raw.notna() & clean.notna() & ~np.isclose(raw, clean, equal_nan=True)).sum()),
                         action_column_present=action in cgm,
                         action_counts=str(cgm[action].value_counts(dropna=False).to_dict()) if action in cgm else "UNAVAILABLE"))
    ctx.csv(pd.DataFrame(rows), "21_CGM_raw_clean_audit.csv")
    reports = cgm_path.parent.parent / "reports"
    evidence = []
    for name in ["processing_parameters.csv", "outcomes_used.json", "manifest.json"]:
        file = reports / name
        if file.is_file():
            ctx.record(file, "CGM_preprocessing_record")
            # Only scan record metadata/parameter names. Do not copy participant audit records.
            text = file.read_text(encoding="utf-8")
            evidence.append(dict(path=str(file), contains_sdw_token='"cgm_sdw"' in text or 'cgm_sdw,' in text,
                                 role="candidate_preprocessing_record; execution linkage requires manifest verification"))
    candidates = []
    for file in sorted((ctx.root / "cgm_deal").rglob("*.py")):
        if "outputs" in file.parts or "tests" in file.parts or "__pycache__" in file.parts:
            continue
        content = file.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines(), 1):
            if '"cgm_sdw"' in line or "'cgm_sdw'" in line:
                ctx.record(file, "SDw_source_code_candidate")
                candidates.append(dict(path=str(file), line_number=i, code_excerpt=line.strip()[:300],
                                       interpretation="code reference only; not proof this file generated analyzed values"))
    ctx.csv(pd.DataFrame(evidence, columns=["path", "contains_sdw_token", "role"]), "21_CGM_preprocessing_record_inventory.csv")
    ctx.csv(pd.DataFrame(candidates, columns=["path", "line_number", "code_excerpt", "interpretation"]), "21_SDw_source_code_candidates.csv")
    return dict(preprocessing_records=len(evidence), sdw_code_references=len(candidates),
                historical_iglu_parameters="UNRESOLVED: public defaults are not proof of HPP historical implementation")
