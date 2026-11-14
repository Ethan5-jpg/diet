#!/usr/bin/env bash
# Paste this complete block into the HPP server terminal.
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1 &&
(
  IFS= read -r hpp_python < experiments/diet_effect_modification_20261109_v12/.python_path &&
  "$hpp_python" -B - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.parquet as pq


def emit(tag, **fields):
    print("V5_" + tag, json.dumps(fields, allow_nan=False), flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def project(path, columns):
    return pq.ParquetFile(path, pre_buffer=False).read(
        columns=columns, use_threads=False
    ).to_pandas(ignore_metadata=True)


def checked_roles(package, project_root, key, household):
    pointer = project_root / "config/w20_simplified_20261101_v2/active_execution.json"
    path = package / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    if not path.is_file() or not pointer.is_file():
        raise ValueError("ROLE_METADATA_MISSING")
    if sha256(path.read_bytes()).hexdigest() != read_json(pointer).get("roles_sha256"):
        raise ValueError("ROLE_HASH_MISMATCH")
    cols = key + [household, "final_role"]
    roles = project(path, cols)
    if (roles.empty or roles.duplicated(key).any() or roles[cols].isna().any().any()
            or roles.groupby(household).final_role.nunique().max() != 1):
        raise ValueError("ROLE_METADATA_INVALID")
    discovery = roles.loc[roles.final_role.eq("discovery")]
    reserved = roles.loc[roles.final_role.eq("confirmation_reserved")]
    if len(discovery) + len(reserved) != len(roles) or discovery.empty or reserved.empty:
        raise ValueError("UNEXPECTED_ROLES")
    if set(discovery.participant_id) & set(reserved.participant_id):
        raise ValueError("PARTICIPANT_ROLE_OVERLAP")
    return reserved[key + [household]]


def main():
    package = Path.cwd()
    project_root = package.parent
    key = ["participant_id", "cohort", "research_stage"]
    household = "split_unit_id"
    reserved = checked_roles(package, project_root, key, household)
    source = project_root / "analysis/D_Glu/fiber_density/base_table_TRE_ONLY.parquet"
    food, cgm, host = "n_exposure_days", "n_qualified_CGM_days", "age_sex_available_candidate"
    cols = key + [food, cgm, host]
    if source.is_file() and set(cols) <= set(pq.ParquetFile(source, pre_buffer=False).schema_arrow.names):
        flags = project(source, cols)
        if flags.duplicated(key).any() or flags[key].isna().any().any():
            raise ValueError("D_GLU_FIBER_QC_KEYS_INVALID")
        joined = reserved.merge(flags, on=key, how="inner", validate="one_to_one")
        fiber = joined[food].ge(3).fillna(False)
        glucose = joined[cgm].ge(3).fillna(False)
        age_sex = joined[host].eq(True).fillna(False)

        def counts(frame):
            return {"people": int(frame.participant_id.nunique()),
                    "households": int(frame[household].nunique())}

        emit("RESERVED_QC", scope="W14 technical flags",
             fiber_module=counts(joined), food_ge3=counts(joined.loc[fiber]),
             cgm_ge3=counts(joined.loc[glucose]),
             both_ge3=counts(joined.loc[fiber & glucose]),
             both_ge3_age_sex=counts(joined.loc[fiber & glucose & age_sex]))
    else:
        emit("RESERVED_QC", status="W14_FILE_OR_COLUMNS_UNAVAILABLE")

    path = project_root / "freeze/person_access_status_TRE_ONLY.parquet"
    history_fields = ["historically_unexposed_validation_established",
                      "colleague_history", "other_sessions_or_manual_runs"]
    if path.is_file() and set(key + history_fields) <= set(pq.ParquetFile(path, pre_buffer=False).schema_arrow.names):
        history = project(path, key + history_fields)
        if history.duplicated(key).any() or history[key].isna().any().any():
            raise ValueError("HISTORY_KEYS_INVALID")
        matched = reserved.merge(history, on=key, how="inner", validate="one_to_one")
        emit("HISTORY", scope="archived W15 metadata", reserved_rows=len(reserved),
             rows_matched=len(matched),
             established=int(matched[history_fields[0]].eq(True).sum()),
             colleague_unknown=int(matched[history_fields[1]].eq("unknown_pending_self_confirmation").sum()),
             other_sessions_unverified=int(matched[history_fields[2]].eq("unverified").sum()))
    else:
        emit("HISTORY", status="W15_FILE_OR_COLUMNS_UNAVAILABLE")
    emit("FEASIBILITY_DONE", outcome_values_read=False, exposure_values_read=False)


try:
    main()
except (ValueError, OSError) as exc:
    reason = str(exc)
    if not (reason.isascii() and reason.replace("_", "").isalnum() and reason.isupper()):
        reason = type(exc).__name__
    raise SystemExit("V5_FEASIBILITY_STOP " + reason) from None
PY
)
