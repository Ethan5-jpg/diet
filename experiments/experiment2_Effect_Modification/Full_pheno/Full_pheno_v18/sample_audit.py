"""Freeze candidate contrasts and inspect registered validation-sample metadata."""

from pathlib import Path


CANDIDATES = (
    ("animal_foods", "nightingale:hdl_ce"),
    ("nutrient_density", "nightingale:omega_3_pct"),
    ("fiber_and_sources", "serum_lipidomics:pc_38:2"),
)


def audit_targets(root, study, source_paths, v15, v16, v17):
    import pyarrow.parquet as pq
    from run_v2 import require
    from run_v17 import read_json, v1_run

    before = read_json(v16 / "summary.json")["stable_endpoints"]
    after = read_json(v17 / "summary.json")["stable_endpoints"]
    reads = v15.parent / "reserved-2/score_access.json"
    prior_outcomes = set(read_json(reads)["outcomes"])
    require(all(e in before[g] and e in after[g] and e not in prior_outcomes for g, e in CANDIDATES),
            "V18_CANDIDATE_PREREGISTRATION")
    source_metadata = {}
    for group, e in CANDIDATES:
        platform, label = e.split(":", 1)
        if platform not in source_metadata:
            with pq.ParquetFile(source_paths[platform]) as table:
                metadata = table.read(columns=["cohort", "research_stage"], use_threads=False)
                census = metadata.group_by(["cohort", "research_stage"]).aggregate([("cohort", "count")])
                source_metadata[platform] = census.to_pylist()
        require(label in pq.ParquetFile(source_paths[platform]).schema_arrow.names, "V18_UNKNOWN_CANDIDATE")

    roster_file = root / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    pointer = root.parent / "config/w20_simplified_20261101_v2/active_execution.json"
    require(read_json(pointer)["roles_sha256"] == v1_run.file_sha(roster_file), "V18_ROLE_HASH_CHANGED")
    cols = ["participant_id", "split_unit_id", "final_role"]
    roster = pq.read_table(roster_file, columns=cols, use_threads=False).to_pandas()
    require(roster.participant_id.is_unique and roster.split_unit_id.notna().all()
            and roster.groupby("split_unit_id").final_role.nunique().max() == 1, "V18_ROLE_OVERLAP")
    roles = {k: dict(people=len(f), families=f.split_unit_id.nunique()) for k, f in roster.groupby("final_role")}
    require(roles["discovery"]["people"] == study.n, "V18_ROLE_DISCOVERY_CHANGED")

    legacy_roster = root.parent / "config/w18_preprocessing_20261030_v1/participant_roles_TRE_ONLY.parquet"
    if not legacy_roster.exists():
        legacy_roster = root / "inputs/w18_preprocessing_20261030_v1/participant_roles_TRE_ONLY.parquet"
    stage = dict(present=legacy_roster.is_file())
    if stage["present"]:
        original = pq.read_table(legacy_roster, columns=["participant_id"], use_threads=False).to_pandas()
        stage.update(people=int(original.participant_id.nunique()),
                     additional_people=int((~original.participant_id.isin(roster.participant_id)).sum()),
                     source_sha=v1_run.file_sha(legacy_roster))
    result = dict(
        candidates=[dict(diet_category=g, endpoint=e, available_source=str(source_paths[e.split(":", 1)[0]]))
                    for g, e in CANDIDATES],
        previous_reservation=roles, original_roster_comparison=stage, assay_metadata=source_metadata,
        previous_reserved_predictors=read_json(v15.parent / "reserved-2/FROZEN.json")["households"],
        prior_assays_access_record_sha=v1_run.file_sha(Path(reads)),
        separately_registered_independent_sample=None, outcome_values_accessed=False,
    )
    return result
