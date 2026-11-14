#!/usr/bin/env bash
# Run on the HPP server to summarize frozen predictor availability by role.
(
  set -euo pipefail
  cd -- "${HPP_V5_PACKAGE:-/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1}"
  if [[ -n "${HPP_EXP2_PYTHON:-}" ]]; then
    hpp_python=$HPP_EXP2_PYTHON
  else
    IFS= read -r hpp_python < experiments/diet_effect_modification_20261109_v12/.python_path
  fi
  "$hpp_python" -B - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEY = ["participant_id", "cohort", "research_stage"]
DAY = KEY + ["date"]
HOUSEHOLD = "split_unit_id"
HOST = ["age_calendar_years", "sex_code", "smoking_current_status_label"]
LIFESTYLE = ["activity_walking_10min_days_weekly_numeric",
             "activity_moderate_days_weekly_numeric", "sleep_hours_daily_numeric",
             "alcohol_current_frequency_label"]
NUTRIENTS = ["calories_kcal_complete_sum", "dietary_fiber_g_complete_sum",
             "carbohydrate_g_complete_sum", "lipid_g_complete_sum", "protein_g_complete_sum",
             "sodium_mg_complete_sum"]
LOCK = "locked_W13_independent_full_source_verification_passed"


def emit(tag, **fields):
    print("V5_" + tag, json.dumps(fields, allow_nan=False), flush=True)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def schema(path, expected):
    if not path.is_file():
        return None, expected
    names = pq.ParquetFile(path, pre_buffer=False).schema_arrow.names
    return names, sorted(set(expected) - set(names))


def project(path, columns, roles=None):
    filters = None
    if roles is not None:
        filters = [("cohort", "=", "10k"), ("research_stage", "=", "00_00_visit"),
                   ("participant_id", "in", roles.participant_id.tolist())]
    return pq.read_table(path, columns=columns, filters=filters, pre_buffer=False,
                         use_threads=False).to_pandas(ignore_metadata=True)


def unique(frame, columns, reason):
    require(not frame[columns].isna().any().any() and not frame.duplicated(columns).any(), reason)


def dates(frame, *, drop_missing=False):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date.astype("string").str.slice(0, 10), errors="raise")
    return frame.loc[frame.date.notna()].copy() if drop_missing else frame


def counts(roles, indicator):
    matched = roles.loc[indicator]
    return {"people": len(matched), "households": int(matched[HOUSEHOLD].nunique())}


def read_roles(package):
    path = package / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    pointer = package.parent / "config/w20_simplified_20261101_v2/active_execution.json"
    require(path.is_file() and pointer.is_file(), "FROZEN_ROLE_METADATA_MISSING")
    require(sha256(path.read_bytes()).hexdigest() == read_json(pointer).get("roles_sha256"),
            "FROZEN_ROLE_HASH_MISMATCH")
    columns = KEY + [HOUSEHOLD, "final_role"]
    roles = project(path, columns)
    unique(roles, KEY, "INVALID_FROZEN_ROLE_KEYS")
    require(not roles[columns].isna().any().any() and roles.participant_id.is_unique
            and roles.cohort.eq("10k").all() and roles.research_stage.eq("00_00_visit").all()
            and set(roles.final_role) == {"discovery", "confirmation_reserved"}
            and roles.groupby(HOUSEHOLD).final_role.nunique().max() == 1,
            "INVALID_FROZEN_ROLE_ASSIGNMENT")
    return roles


def audit_food(path, roles):
    flags = ["energy_eligible", "fiber_eligible"]
    food = dates(project(path, DAY + flags, roles))
    unique(food, DAY, "INVALID_FOOD_DAY_KEYS")
    require(all(food[x].dropna().isin([True, False]).all() for x in flags), "INVALID_FOOD_FLAGS")
    energy = food.energy_eligible.eq(True).fillna(False)
    fiber = food.fiber_eligible.eq(True).fillna(False)
    require(not (fiber & ~energy).any(), "FIBER_FLAGS_CONTRADICT_ENERGY_FLAGS")
    days = food.loc[fiber].groupby(KEY, as_index=False, observed=True).size()
    return days.rename(columns={"size": "n_fiber_days"}), food.loc[energy, DAY]


def audit_sites(shared, roles):
    columns = KEY + ["site", "sample_id"]
    registry = project(shared / "sample_registry_TRE_ONLY.parquet", columns + ["sample_lock_status"], roles)
    unique(registry, KEY + ["site"], "INVALID_LOCKED_REGISTRY_KEYS")
    require(registry.site.isin(["oral", "gut"]).all() and registry.sample_id.notna().all()
            and registry.sample_id.is_unique and registry.sample_lock_status.eq(LOCK).all(),
            "INVALID_LOCKED_REGISTRY_STATUS")
    taxon_names = {}
    for site in ("oral", "gut"):
        path = shared / (site + "_selected_abundance_TRE_ONLY.parquet")
        names, missing = schema(path, columns)
        require(names is not None and not missing, "MISSING_SELECTED_MICROBIOME_METADATA")
        taxon_names[site] = [name for name in names if name.startswith(("k__", "s__")) or "|s__" in name]
        matrix = project(path, columns, roles)
        unique(matrix, KEY, "INVALID_SELECTED_MICROBIOME_KEYS")
        require(matrix.sample_id.notna().all() and matrix.sample_id.is_unique and matrix.site.eq(site).all(),
                "INVALID_SELECTED_MICROBIOME_SAMPLE_ID")
        locked = registry.loc[registry.site.eq(site), KEY + ["sample_id"]]
        linked = locked.merge(matrix[KEY + ["sample_id"]], on=KEY, how="outer", validate="one_to_one",
                              suffixes=("_locked", "_matrix"), indicator=True)
        both = linked["_merge"].eq("both")
        if not both.all() or not linked.loc[both, "sample_id_locked"].eq(linked.loc[both, "sample_id_matrix"]).all():
            emit("PREDICTOR_SAMPLE_MISMATCH", site=site,
                 registry_only=int(linked["_merge"].eq("left_only").sum()),
                 matrix_only=int(linked["_merge"].eq("right_only").sum()),
                 differing_sample_ids=int((both & ~linked.sample_id_locked.eq(linked.sample_id_matrix)).sum()))
            raise ValueError("LOCKED_SAMPLE_AND_MATRIX_MISMATCH")
        roles = roles.merge(locked[KEY].assign(**{site: True}), on=KEY, how="left", validate="one_to_one")

    species = {"R_bromii": ("ruminococcus_bromii",),
               "B_adolescentis": ("bifidobacterium_adolescentis",),
               "E_rectale": ("eubacterium_rectale", "agathobacter_rectalis")}
    terminals = [name.rsplit("|", 1)[-1].lower() for name in taxon_names["gut"]]
    matches = {name: sum(t in {"s__" + alias for alias in aliases} for t in terminals)
               for name, aliases in species.items()}
    emit("PREDICTOR_MICROBIOME_SCHEMA", selected_taxon_columns={s: len(t) for s, t in taxon_names.items()},
         fermentation_exact_terminal_match_counts=matches, score_availability_checked=False)
    return roles


def audit_context(shared, host_path, roles, energy_days):
    path = shared / "food_day_qc_TRE_ONLY.parquet"
    _, missing_food = schema(path, DAY + NUTRIENTS)
    _, missing_host = schema(host_path, KEY + LIFESTYLE)
    emit("PREDICTOR_CONTEXT_SCHEMA", food_day_qc_present=path.is_file(),
         missing_food_day_columns=missing_food, missing_lifestyle_columns=missing_host)
    if not path.is_file() or not set(DAY).isdisjoint(missing_food):
        return
    keys = dates(project(path, DAY, roles), drop_missing=True)
    unique(keys, DAY, "INVALID_NUTRIENT_DAY_KEYS")
    linked = energy_days.merge(keys.assign(context_day=True), on=DAY, how="left", validate="one_to_one")
    linked = linked.merge(roles[KEY + ["final_role"]], on=KEY, how="left", validate="many_to_one")
    for role, group in linked.groupby("final_role", observed=True):
        matched = group.context_day.eq(True)
        emit("PREDICTOR_CONTEXT_DAYS", role=role, energy_eligible_days=len(group),
             matching_nutrient_day_keys=int(matched.sum()), anchor_values_checked=False)


def main():
    package = Path.cwd()
    project_root = package.parent
    roles = read_roles(package)
    shared = project_root / "analysis/shared"
    food_path = project_root / "data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet"
    host_path = shared / "covariate_registry_TRE_ONLY.parquet"
    fiber_days, energy_days = audit_food(food_path, roles)
    host = project(host_path, KEY + HOST, roles)
    unique(host, KEY, "INVALID_HOST_KEYS")
    age = pd.to_numeric(host.age_calendar_years, errors="coerce").to_numpy(float, na_value=np.nan)
    sex = pd.to_numeric(host.sex_code, errors="coerce").isin([0, 1])
    smoking = host.smoking_current_status_label.astype("string").fillna("").ne("")
    host = host[KEY].assign(age_sex=np.isfinite(age) & sex, smoking_recorded=smoking)
    roles = roles.merge(fiber_days, on=KEY, how="left", validate="one_to_one")
    roles = roles.merge(host, on=KEY, how="left", validate="one_to_one")
    roles = audit_sites(shared, roles)
    for role, group in roles.groupby("final_role", observed=True):
        fiber = group.n_fiber_days.ge(3).fillna(False)
        age_sex = group.age_sex.eq(True)
        oral, gut = group.oral.eq(True), group.gut.eq(True)
        emit("PREDICTOR_AVAILABILITY", role=role, frozen_role=counts(group, group.final_role.eq(role)),
             fiber_ge3=counts(group, fiber), age_sex=counts(group, age_sex),
             smoking_recorded=counts(group, group.smoking_recorded.eq(True)),
             oral_locked_matrix=counts(group, oral), gut_locked_matrix=counts(group, gut),
             joint_predictor_metadata=counts(group, fiber & age_sex & oral & gut))
    audit_context(shared, host_path, roles, energy_days)
    emit("PREDICTOR_METADATA_DONE", outcome_values_read=False, abundance_values_read=False,
         food_quantities_read=False, row_output=False, files_written=False)


try:
    main()
except (ValueError, OSError, TypeError, KeyError) as exc:
    reason = str(exc)
    if not (reason.isascii() and reason.replace("_", "").isalnum() and reason.isupper()):
        reason = type(exc).__name__
    raise SystemExit("V5_PREDICTOR_METADATA_STOP " + reason) from None
PY
)
