#!/usr/bin/env python3
"""Fixed input contract for the four new CGM outcomes; no outcome recomputation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parents[1]
OUTCOMES = {
    "MAGE": "cgm_mage_z",
    "TAR180": "cgm_above_180_z",
    "TBR70": "cgm_below_70_z",
    "TIR70_180": "cgm_in_range_70_180_z",
}
PROPORTIONS = ["cgm_above_180", "cgm_below_70", "cgm_in_range_70_180"]
EXPOSURES = {
    "AHEI": ("AHEI_z", "original_four"),
    "AMED": ("AMED_z", "original_four"),
    "hPDI": ("hPDI_z", "original_four"),
    "rEDIH": ("rEDIH_z", "original_four"),
    "EAT13": ("EAT13_z", "new_primary"),
    "NOVA4": ("NOVA4_z", "new_primary"),
    "Carbohydrate_pct": ("Carbohydrate_pct_z", "exploratory"),
}
NEW_SOURCE_COLUMNS = {
    "modified_eat_lancet13_z": "EAT13_z",
    "nova4_total_energy_pct_cov80_z": "NOVA4_z",
    "carbohydrate_energy_pct_z": "Carbohydrate_pct_z",
}
CONTINUOUS = ["age_years", "sleep_duration_hours_day", "physical_activity_met_h_week"]
BINARY = ["vitamin_use", "hormone_use"]
CATEGORICAL = ["sex", "education_level", "smoking_status", "cgm_device_type"]
BASE_COVARIATES = CONTINUOUS + BINARY + CATEGORICAL
FLAGS = [prefix + "_cgm_model" + str(model) + "_covariates_complete"
         for prefix in ("amed", "hpdi") for model in (2, 3)]
COV_COLUMNS = BASE_COVARIATES + ["bmi", "alcohol_intake_g_day", "known_diabetes",
                               "a10_medication_use", "exclude_diabetes_or_a10"] + FLAGS
FULL_KEY = ["participant_id", "cohort", "research_stage", "array_index", "connection_id"]
MISSING = {"", "na", "n/a", "nan", "none", "null", "<na>", "#n/a"}


def require_columns(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("%s missing columns: %s" % (label, ", ".join(missing)))


def read_table(path):
    with open(str(path), encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle, strict=True)
        header = next(rows, [])
        if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
            raise ValueError("Empty or duplicate CSV headers: %s" % path)
        for row in rows:
            if row and len(row) != len(header):
                raise ValueError("CSV field count mismatch at line %d: %s" % (rows.line_num, path))
    frame = pd.read_csv(str(path), dtype=str, na_filter=False, encoding="utf-8-sig", low_memory=False)
    require_columns(frame, ["participant_id"], str(path))
    ids = frame["participant_id"].astype(str).str.strip()
    # Same explicit trailing-.0 normalization as original analysis, without
    # numeric CSV inference (which could irreversibly remove leading zeros).
    ids = ids.str.replace(r"\.0$", "", regex=True)
    if ids.str.lower().isin(MISSING).any() or ids.eq("").any():
        raise ValueError("Missing participant_id: %s" % path)
    if ids.duplicated().any():
        raise ValueError("Duplicate participant_id after normalization: %s" % path)
    frame["participant_id"] = ids
    if frame.empty:
        raise ValueError("Empty input table: %s" % path)
    return frame


def numeric(series, name):
    text = series.astype(str).str.strip()
    missing = series.isna() | text.str.lower().isin(MISSING)
    values = pd.to_numeric(text.mask(missing), errors="coerce").astype(float)
    bad = ~missing & (values.isna() | ~np.isfinite(values))
    if bad.any():
        raise ValueError("%s contains %d nonnumeric/nonfinite values" % (name, int(bad.sum())))
    return values


def boolean(series, name):
    text = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    invalid = ~text.isin(set(mapping) | MISSING) & series.notna()
    if invalid.any():
        raise ValueError("%s has invalid boolean values" % name)
    return text.map(mapping).fillna(False).astype(bool)


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def sample_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids.astype(str).tolist()), ensure_ascii=False).encode()).hexdigest()


def load_config(path, data_root=None):
    with open(str(path), encoding="utf-8") as handle:
        config = json.load(handle)
    base = Path(data_root).expanduser().resolve() if data_root else DATA_ROOT
    for key in ("cgm_csv", "old_scores_csv", "new_scores_csv", "microbiome_csv", "covariates_csv"):
        p = Path(config[key]).expanduser()
        config[key] = str(p.resolve() if p.is_absolute() else (base / p).resolve())
    for key in ("expected_species", "minimum_model_n"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("%s must be a positive integer" % key)
    for key in ("run_strict_sensitivity", "run_same_cohort_comparisons"):
        if type(config.get(key)) is not bool:
            raise ValueError("%s must be boolean" % key)
    return config


def validate_cgm(cgm):
    required = FULL_KEY + [m + "_" + v for m in ["cgm_mage"] + PROPORTIONS for v in ("raw", "clean", "z")]
    require_columns(cgm, required, "finalized CGM extension")
    if not (cgm["cohort"].eq("10k") & cgm["research_stage"].eq("00_00_visit")).all():
        raise ValueError("CGM table is not the selected 10k baseline cohort")
    for key in FULL_KEY:
        if cgm[key].astype(str).str.strip().eq("").any():
            raise ValueError("CGM identity is missing: " + key)
    result = cgm.copy()
    audit_rows = []
    for metric in ["cgm_mage"] + PROPORTIONS:
        for version in ("raw", "clean", "z"):
            column = metric + "_" + version
            result[column] = numeric(cgm[column], column)
        clean, z = result[metric + "_clean"], result[metric + "_z"]
        if metric in PROPORTIONS:
            raw = result[metric + "_raw"]
            if (~raw.dropna().between(0, 100)).any():
                raise ValueError(metric + " raw is outside 0–100")
            if not np.allclose(raw, clean, equal_nan=True, rtol=0, atol=0):
                raise ValueError(metric + " clean differs from frozen legality-only raw")
        if not clean.notna().equals(z.notna()):
            raise ValueError(metric + " clean/Z availability differs; use finalized nondegenerate input")
        if z.notna().sum() >= 2:
            if not np.isclose(z.mean(), 0, atol=1e-7) or not np.isclose(z.std(ddof=1), 1, atol=1e-7):
                raise ValueError(metric + " Z is not standardized over the full input cohort")
            expected_z = (clean - clean.mean()) / clean.std(ddof=1)
            if not np.allclose(z, expected_z, equal_nan=True, rtol=1e-7, atol=1e-7):
                raise ValueError(metric + " clean/Z values are inconsistent")
        audit_rows.append({"outcome": metric, "raw_n": int(result[metric + "_raw"].notna().sum()),
                           "clean_n": int(clean.notna().sum()), "z_n": int(z.notna().sum()),
                           "clean_zero_n": int(clean.eq(0).sum()), "z_mean": z.mean(), "z_sample_sd": z.std(ddof=1)})
    raw_props = result[[m + "_raw" for m in PROPORTIONS]]
    complete = raw_props.notna().all(axis=1)
    if ((raw_props.sum(axis=1) - 100).abs().gt(0.1 + 1e-12) & complete).any():
        raise ValueError("CGM complementarity differs from the reviewed input")
    return result, pd.DataFrame(audit_rows)


def prepare_tables(config):
    paths = [config[k] for k in ("cgm_csv", "old_scores_csv", "new_scores_csv", "microbiome_csv", "covariates_csv")]
    fingerprints = [fingerprint(p) for p in paths]
    cgm, old_scores, new_scores, micro, cov = [read_table(p) for p in paths]
    cgm, cgm_summary = validate_cgm(cgm)
    require_columns(old_scores, [EXPOSURES[s][0] for s in list(EXPOSURES)[:4]], "old four scores")
    require_columns(new_scores, list(NEW_SOURCE_COLUMNS) + ["mean_daily_energy_kcal"], "new diet candidate master")
    require_columns(cov, COV_COLUMNS, "covariate master")
    metadata = set(["participant_id", "cohort", "research_stage", "array_index", "shannon_index", "simpson_index"])
    species = [c for c in micro if c not in metadata and ("s__" in c or "|s_" in c)]
    if len(species) != config["expected_species"]:
        species = [c for c in micro if c not in metadata]
    if len(species) != config["expected_species"]:
        raise ValueError("Expected %d species, identified %d; do not silently drop taxa" % (config["expected_species"], len(species)))
    for column in species:
        micro[column] = numeric(micro[column], column)
        if micro[column].isna().any():
            raise ValueError("Processed CLR-Z species matrix contains missing values: " + column)
    old_piece = old_scores[["participant_id"] + [EXPOSURES[s][0] for s in list(EXPOSURES)[:4]]].copy()
    new_piece = new_scores[["participant_id"] + list(NEW_SOURCE_COLUMNS) + ["mean_daily_energy_kcal"]].rename(columns=NEW_SOURCE_COLUMNS)
    scores = old_piece.merge(new_piece, on="participant_id", how="outer", validate="one_to_one")
    for column in [spec[0] for spec in EXPOSURES.values()] + ["mean_daily_energy_kcal"]:
        scores[column] = numeric(scores[column], column)
    cov = cov[["participant_id"] + COV_COLUMNS].copy()
    for column in CONTINUOUS + BINARY + ["bmi", "alcohol_intake_g_day", "known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]:
        cov[column] = numeric(cov[column], column)
    for column in BINARY + ["known_diabetes", "a10_medication_use", "exclude_diabetes_or_a10"]:
        if not cov[column].dropna().isin([0, 1]).all():
            raise ValueError(column + " must be 0/1/missing")
    for column in CATEGORICAL:
        cov[column] = cov[column].mask(cov[column].str.strip().str.lower().isin(MISSING))
    for column in FLAGS:
        cov[column] = boolean(cov[column], column)
    positive = cov["known_diabetes"].eq(1) | cov["a10_medication_use"].eq(1)
    if (positive & ~cov["exclude_diabetes_or_a10"].eq(1)).any():
        raise ValueError("Covariate master exclusion flag contradicts known diabetes/A10 positives")
    # Preserve CGM device precedence used by the original diet cohort. Check
    # the covariate master rather than silently accepting conflicting devices.
    if "cgm_device_type" in cgm.columns:
        devices = cgm[["participant_id", "cgm_device_type"]].merge(cov[["participant_id", "cgm_device_type"]], on="participant_id", suffixes=("_cgm", "_cov"))
        a = devices["cgm_device_type_cgm"].str.strip().str.lower()
        b = devices["cgm_device_type_cov"].str.strip().str.lower()
        if (a.ne(b) & ~a.isin(MISSING) & b.notna()).any():
            raise ValueError("CGM and covariate master device types conflict")
    keep = FULL_KEY + [m + "_" + v for m in ["cgm_mage"] + PROPORTIONS for v in ("raw", "clean", "z")]
    if "cgm_device_type" in cgm:
        keep.append("cgm_device_type")
    base = cgm[keep].merge(cov, on="participant_id", how="left", validate="one_to_one", suffixes=("", "_cov"), indicator="covariate_join")
    base["exclusion_status"] = base["exclude_diabetes_or_a10"]
    base["primary_eligible"] = ~base["exclusion_status"].eq(1)
    base["strict_eligible"] = base["exclusion_status"].eq(0)
    diet = base.merge(scores, on="participant_id", how="inner", validate="one_to_one")
    diet = diet.loc[diet[[v[0] for v in EXPOSURES.values()]].notna().any(axis=1)].copy()
    # Old microbiome analysis obtains the device from the covariate master.
    micro_base = base.drop(columns=["cgm_device_type"]) if "cgm_device_type_cov" in base else base.copy()
    micro_base = micro_base.rename(columns={"cgm_device_type_cov": "cgm_device_type"})
    microbiome = micro_base.merge(micro[["participant_id"] + species], on="participant_id", how="inner", validate="one_to_one")
    if diet.empty or microbiome.empty:
        raise ValueError("Empty CGM/diet or CGM/microbiome overlap; inspect participant IDs and input paths")
    summary = {"cgm_rows": len(cgm), "old_score_rows": len(old_scores), "new_score_rows": len(new_scores),
               "covariate_rows": len(cov), "microbiome_rows": len(micro), "species": len(species)}
    for label, frame in (("diet", diet), ("microbiome", microbiome)):
        summary.update({label + "_overlap_n": len(frame), label + "_primary_n": int(frame["primary_eligible"].sum()),
                        label + "_strict_n": int(frame["strict_eligible"].sum()),
                        label + "_excluded_diabetes_a10_n": int(frame["exclusion_status"].eq(1).sum()),
                        label + "_unknown_exclusion_n": int(frame["exclusion_status"].isna().sum()),
                        label + "_missing_covariate_record_n": int(frame["covariate_join"].eq("left_only").sum())})
    return diet, microbiome, species, cgm_summary, summary, fingerprints
