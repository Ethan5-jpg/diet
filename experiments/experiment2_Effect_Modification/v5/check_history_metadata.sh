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


def receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_history(package, project, pf):
    identity = ["cohort", "participant_id"]
    flags = ["historically_unexposed_validation_established", "colleague_history",
             "other_sessions_or_manual_runs", "new_W15_association_access"]
    household = "split_unit_id"
    fields = identity + [household, "person_order"] + flags
    missing = sorted(set(fields) - set(pf.schema_arrow.names))
    if missing:
        emit("HISTORY", status="columns_unavailable", missing=missing)
        return False
    path = package / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    pointer = project / "config/w20_simplified_20261101_v2/active_execution.json"
    if not path.is_file() or not pointer.is_file():
        emit("HISTORY", status="role_metadata_unavailable")
        return False
    if sha256(path.read_bytes()).hexdigest() != receipt(pointer).get("roles_sha256"):
        emit("HISTORY", status="role_hash_mismatch")
        return False
    role_columns = identity + [household, "research_stage", "final_role"]
    roles = pq.ParquetFile(path, pre_buffer=False).read(
        columns=role_columns, use_threads=False
    ).to_pandas(ignore_metadata=True)
    history = pf.read(columns=fields, use_threads=False).to_pandas(ignore_metadata=True)
    if (roles.empty or history.empty or roles[role_columns].isna().any().any()
            or history[fields].isna().any().any()
            or roles.duplicated(identity).any() or history.duplicated(identity).any()
            or history.person_order.duplicated().any()
            or not roles.cohort.eq("10k").all() or not roles.research_stage.eq("00_00_visit").all()
            or set(roles.final_role) != {"discovery", "confirmation_reserved"}
            or roles.groupby(household).final_role.nunique().max() != 1):
        raise ValueError("HISTORY_IDENTITY_INVALID")
    joined = roles.merge(history, on=identity, how="outer", validate="one_to_one",
                         suffixes=("", "_w15"), indicator=True)
    if (not joined["_merge"].eq("both").all()
            or not joined[household].eq(joined[f"{household}_w15"]).all()):
        raise ValueError("HISTORY_ROSTER_OR_HOUSEHOLD_MISMATCH")
    reserved = joined.loc[joined.final_role.eq("confirmation_reserved")]
    emit("HISTORY", status="matched", scope="archived W15 only", people=len(reserved),
         households=int(reserved[household].nunique()),
         history_established=int(reserved[flags[0]].eq(True).sum()),
         colleague_unknown=int(reserved[flags[1]].eq("unknown_pending_self_confirmation").sum()),
         other_sessions_unverified=int(reserved[flags[2]].eq("unverified").sum()),
         w15_association_access=int(reserved[flags[3]].eq(True).sum()))
    return True


def main():
    package = Path.cwd()
    project = package.parent
    rows_read = False
    lock_path = project / "config/w15_split_access_lock.json"
    manifest_path = project / "data_qc/w15_split_access_20261029/final_artifact_manifest.json"
    history_path = project / "freeze/person_access_status_TRE_ONLY.parquet"
    checked = (lock_path, manifest_path, history_path)
    if not all(path.is_file() for path in checked):
        emit("HISTORY_SOURCE", lock=checked[0].is_file(), manifest=checked[1].is_file(),
             history=checked[2].is_file(), status="missing_expected_metadata")
    elif sha256(manifest_path.read_bytes()).hexdigest() != receipt(lock_path).get("production_manifest_sha256"):
        emit("HISTORY_SOURCE", status="manifest_hash_mismatch")
    else:
        records = [r for r in receipt(manifest_path)
                   if r.get("path") == str(history_path) or
                   r.get("name") == "person_access_status_TRE_ONLY.parquet"]
        fields = {"participant_id", "cohort", "split_unit_id", "person_order",
                  "historically_unexposed_validation_established", "colleague_history",
                  "other_sessions_or_manual_runs", "new_W15_association_access"}
        pf = pq.ParquetFile(history_path, pre_buffer=False)
        hash_matches = len(records) == 1 and sha256(history_path.read_bytes()).hexdigest() == records[0].get("sha256")
        emit("HISTORY_SOURCE",
             status="present",
             header_rows=pf.metadata.num_rows,
             registered=len(records) == 1,
             history_hash_matches=hash_matches,
             missing_expected_fields=sorted(fields - set(pf.schema_arrow.names)))
        if hash_matches:
            rows_read = summarize_history(package, project, pf)

    status_path = project / "data_qc/w20_simplified_20261101_v2/results/delivery/verified_status.json"
    if status_path.is_file():
        status = receipt(status_path)
        emit("HISTORICAL_DISCOVERY_RECEIPT",
             verified=status.get("status") == "simplified_v2_readback_verified",
             confirmation_y_reads_this_rerun=status.get("confirmation_Y_reads_this_rerun"),
             formal_p_values_computed=status.get("formal_P_values_computed"))
    else:
        emit("HISTORICAL_DISCOVERY_RECEIPT", status="missing")
    for name in ("config", "freeze"):
        path = project / name
        prefixes = ("w22", "w23", "confirmation", "confirm_", "holdout")
        names = sorted(p.name for p in path.iterdir() if p.name.lower().startswith(prefixes)) if path.is_dir() else []
        emit("CONFIRMATION_RECEIPT_NAMES", directory=name, matches=len(names), names=names[:20])
    emit("HISTORY_CHECK_DONE", membership_metadata_join_checked=rows_read,
         individual_rows_exported=False, outcome_values_read=False)


try:
    main()
except (ValueError, OSError, TypeError) as exc:
    detail = str(exc)
    if not (detail.isascii() and detail.replace("_", "").isalnum() and detail.isupper()):
        detail = type(exc).__name__
    raise SystemExit("V5_HISTORY_CHECK_STOP " + detail) from None
PY
)
