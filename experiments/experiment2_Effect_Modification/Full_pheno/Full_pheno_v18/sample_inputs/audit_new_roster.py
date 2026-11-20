"""Count assay and predictor key overlap in candidate evaluation visits."""

from collections import Counter
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(HERE))
from run_main_audit import emit, parse, read_json, require, v1_run, write_json


def identifiers(rows):
    ids = set()
    for row in rows:
        value = row["participant_id"]
        if value is None:
            continue
        identifier = int(value)
        require(value == identifier and abs(identifier) < 2**53, "V18_UNREPRESENTABLE_PARTICIPANT_KEY")
        ids.add(identifier)
    return ids


def main():
    args = parse(__doc__)
    v1_run.init_runtime(1)
    import pyarrow.parquet as pq
    from batch_sources import paths

    out = HERE / "runs/TRE_ONLY" / args.run_id
    audit = read_json(out / "independent_sample_audit.json")
    require(audit["outcome_values_accessed"] is False, "V18_INVALID_SAMPLE_PREFLIGHT")
    handoff = HERE.parents[2]
    project = handoff.parent
    shared = project / "analysis/shared"
    preprocessing = project / "data_qc/preprocessing_20261030_v1"
    assays = paths(handoff)
    registry = handoff / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    pointer = project / "config/w20_simplified_20261101_v2/active_execution.json"
    require(read_json(pointer)["roles_sha256"] == v1_run.file_sha(registry), "V18_ROLE_SOURCE_DIFF")
    role = pq.read_table(registry, columns=["participant_id", "final_role"], use_threads=False).to_pylist()
    registered = {s["participant_id"]: s["final_role"] for s in role}
    require(len(registered) == len(role), "V18_DUPLICATE_ROLE_KEYS")

    def read(path, stage):
        rule = [("cohort", "=", "10k"), ("research_stage", "=", stage)]
        return pq.read_table(path, columns=["participant_id"], filters=rule, use_threads=False).to_pylist()

    result = {}
    for stage in ("00_00_visit", "02_00_visit"):
        daily = preprocessing / "food_qc_v1/food_daily_primary_TRE_ONLY.parquet"
        counts = Counter(r["participant_id"] for r in read(daily, stage))
        three_days = identifiers({"participant_id": i} for i, n in counts.items() if n >= 3)
        groups = ("covariate_registry", "gut_selected_abundance", "oral_selected_abundance")
        people = three_days.intersection(*(identifiers(read(shared / (f + "_TRE_ONLY.parquet"), stage))
                                          for f in groups))
        geometry = preprocessing / "foodid_interval_geometry_v1b/matrix_person_rows_TRE_ONLY.parquet"
        calibrated = identifiers(read(geometry, stage))
        result[stage] = dict(people_with_predictor_keys=len(people), including_foodid_keys=len(people & calibrated),
                             assays={})
        for source in ("nightingale", "serum_lipidomics"):
            available = identifiers(read(assays[source], stage))
            matched = available & people
            result[stage]["assays"][source] = dict(
                people_with_assay_keys=len(available), with_predictor_keys=len(matched),
                with_foodid_keys=len(matched & calibrated), outside_w20_with_predictor_keys=len(matched - registered.keys()),
                outside_w20_with_foodid_keys=len((matched & calibrated) - registered.keys()),
                w20_roles=dict(Counter(registered[i] for i in matched if i in registered)),
            )
    output = dict(commit=args.commit, outcome_values_accessed=False, candidate_visits=result)
    path = out / "candidate_visit_key_overlap.json"
    if path.exists():
        require(read_json(path) == output, "V18_CANDIDATE_VISIT_AUDIT_CHANGED")
    else:
        write_json(path, output)
    emit("V18_CANDIDATE_VISITS", result)


if __name__ == "__main__":
    main()
