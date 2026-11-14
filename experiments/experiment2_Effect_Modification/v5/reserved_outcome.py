"""Replay the archived D_Glu outcome and score a frozen prediction pair."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from moderation_models import scores
from person_inputs import digest
from reserved_inputs import HOUSEHOLD, KEY, project_population


W14_MANIFEST = Path("data_qc/w14_eligibility_20261029/final_artifact_manifest.json")
W15_DIRECTORY = Path("data_qc/w15_split_access_20261029")
W15_PREFLIGHT = W15_DIRECTORY / "preflight.json"
W15_MANIFEST = W15_DIRECTORY / "final_artifact_manifest.json"
W15_LOCK = Path("config/w15_split_access_lock.json")
MEMBERSHIP = Path("freeze/module_membership_TRE_ONLY.parquet")
CGM_VISIT = Path("analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet")
ENDPOINT_TABLE = Path("analysis/D_Glu/UPF_bounds/base_table_TRE_ONLY.parquet")

ENDPOINT = "D_Glu"
OUTCOME_ID = "glucose_visit_equal_day_mean"
AGGREGATION = "equal_qualified_day_mean_same_visit_no_food_overlap_requirement"
UNIT = "mg/dL"
DAYS = "n_qualified_CGM_days"
MINIMUM_DAYS = 3
BOOTSTRAP_DRAWS = 1999


def _json_source(path, hashes):
    payload = path.read_bytes()
    hashes[path] = sha256(payload).hexdigest()
    return json.loads(payload)


def _registered_file(path, manifest, hashes):
    records = [record for record in manifest if record["path"] == str(path)]
    if len(records) != 1:
        raise ValueError("RESERVED_OUTCOME_SOURCE_REGISTRY_MISMATCH")
    record = records[0]
    actual = digest(path)
    if path.stat().st_size != record["bytes"] or actual != record["sha256"]:
        raise ValueError("RESERVED_OUTCOME_SOURCE_HASH_MISMATCH")
    if pq.ParquetFile(path, pre_buffer=False).metadata.num_rows != record["rows"]:
        raise ValueError("RESERVED_OUTCOME_SOURCE_ROW_COUNT_MISMATCH")
    hashes[path] = actual


def load_outcome(root, index):
    """Return raw mg/dL in frozen roster order, with NaN for endpoint ineligibility.

    W20's original registration selected the W14 UPF module's raw outcome,
    intersected it with W15 D_Glu membership, and required at least three
    qualified CGM days. Fiber eligibility is determined by the predictor stage.
    """
    key = list(KEY)
    identity = key + [HOUSEHOLD]
    if (index.empty or index[identity].isna().any().any()
            or index.duplicated(key).any() or index.participant_id.duplicated().any()
            or not index.cohort.eq("10k").all() or not index.research_stage.eq("00_00_visit").all()):
        raise ValueError("RESERVED_OUTCOME_INVALID_ROSTER")

    project = Path(root).resolve().parents[2]
    hashes = {}
    lock = _json_source(project / W15_LOCK, hashes)
    preflight = _json_source(project / W15_PREFLIGHT, hashes)
    w14 = _json_source(project / W14_MANIFEST, hashes)
    w15 = _json_source(project / W15_MANIFEST, hashes)
    if (hashes[project / W15_MANIFEST] != lock["production_manifest_sha256"]
            or hashes[project / W14_MANIFEST] != preflight["W14_production_manifest_sha256"]):
        raise ValueError("RESERVED_OUTCOME_MANIFEST_HASH_MISMATCH")
    for relative, manifest in ((MEMBERSHIP, w15), (CGM_VISIT, w14), (ENDPOINT_TABLE, w14)):
        _registered_file(project / relative, manifest, hashes)

    membership = project_population(project / MEMBERSHIP, identity + ["analysis_id"], index,
                                    unique_by=key + ["analysis_id"])
    membership = membership.loc[membership.analysis_id.eq(ENDPOINT), identity]
    members = index[identity].merge(membership, on=key, validate="one_to_one",
                                    suffixes=("", "_w15"))
    if not members[HOUSEHOLD].eq(members[f"{HOUSEHOLD}_w15"]).all():
        raise ValueError("RESERVED_OUTCOME_W15_HOUSEHOLD_MISMATCH")
    result = np.full(len(index), np.nan)
    if members.empty:
        return result, hashes

    qc_columns = [DAYS, "outcome_id", "unit", "aggregation", "phenotype_eligible", "point_qc_verified"]
    cgm = project_population(project / CGM_VISIT, key + qc_columns, members, unique_by=key)
    if (len(cgm) != len(members) or cgm[qc_columns].isna().any().any()
            or not pd.api.types.is_integer_dtype(cgm[DAYS]) or not cgm[DAYS].ge(1).all()
            or not cgm.outcome_id.eq(OUTCOME_ID).all() or not cgm.unit.eq(UNIT).all()
            or not cgm.aggregation.eq(AGGREGATION).all() or not cgm.phenotype_eligible.eq(True).all()
            or not cgm.point_qc_verified.eq(True).all()):
        raise ValueError("RESERVED_OUTCOME_ARCHIVED_CGM_QC_MISMATCH")
    eligible = cgm.loc[cgm[DAYS].ge(MINIMUM_DAYS), key + [DAYS]]
    if eligible.empty:
        return result, hashes

    outcome_columns = ["Y_value", "Y_unit", "outcome_id", "Y_valid_measurements",
                       "phenotype_eligible", "base_eligible", "analysis_id", "exposure_id"]
    outcome = project_population(project / ENDPOINT_TABLE, key + outcome_columns, eligible, unique_by=key)
    outcome = eligible.merge(outcome, on=key, validate="one_to_one")
    canonical = project_population(project / CGM_VISIT, key + ["value"], eligible, unique_by=key)
    outcome = outcome.merge(canonical, on=key, validate="one_to_one")
    if (len(outcome) != len(eligible) or outcome[outcome_columns + ["value"]].isna().any().any()
            or not outcome.Y_unit.eq(UNIT).all() or not outcome.outcome_id.eq(OUTCOME_ID).all()
            or not outcome.Y_valid_measurements.eq(outcome[DAYS]).all()
            or not outcome.phenotype_eligible.eq(True).all() or not outcome.base_eligible.eq(True).all()
            or not outcome.analysis_id.eq(ENDPOINT).all() or not outcome.exposure_id.eq("UPF_bounds").all()
            or not pd.api.types.is_numeric_dtype(outcome.Y_value)
            or not pd.api.types.is_numeric_dtype(outcome.value)
            or not np.isfinite(outcome.Y_value.to_numpy(float)).all()
            or not np.array_equal(outcome.Y_value.to_numpy(float), outcome.value.to_numpy(float))):
        raise ValueError("RESERVED_OUTCOME_ARCHIVED_VALUE_MISMATCH")
    # Align by exact visit key. The caller keeps this array inside the TRE.
    return outcome.set_index(key).Y_value.reindex(pd.MultiIndex.from_frame(index[key])).to_numpy(float), hashes


def summarize(y, baseline, full, groups, contrast):
    """Score fixed predictions using a common denominator and paired households."""
    y, baseline, full, contrast = (np.asarray(a, dtype=float) for a in (y, baseline, full, contrast))
    groups = np.asarray(groups)
    if (y.ndim != 1 or len(y) < 2 or groups.shape != y.shape or pd.isna(groups).any()
            or not all(a.shape == y.shape and np.isfinite(a).all() for a in (y, baseline, full, contrast))):
        raise ValueError("RESERVED_OUTCOME_INVALID_PAIRED_INPUT")
    households = len(np.unique(groups))
    if households < 2:
        raise ValueError("RESERVED_OUTCOME_TOO_FEW_HOUSEHOLDS")
    base_r2, full_r2, delta, interval = scores(y, baseline, full[:, None], groups, draws=BOOTSTRAP_DRAWS)
    contrast_sd = np.std(contrast)
    if not np.isfinite([base_r2, full_r2[0], delta[0], *interval[0], contrast_sd]).all():
        raise ValueError("RESERVED_OUTCOME_NONFINITE_SUMMARY")
    return {
        "people": len(y),
        "households": households,
        "base_r2": float(base_r2),
        "full_r2": float(full_r2[0]),
        "delta_r2": float(delta[0]),
        "delta_r2_ci95": interval[0].tolist(),
        "contrast_sd_10_to_15_mg_dl": float(contrast_sd),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
