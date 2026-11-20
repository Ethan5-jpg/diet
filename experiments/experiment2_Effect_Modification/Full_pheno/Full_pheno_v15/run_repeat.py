"""Freeze the V15 comparison and repeat it on new discovery household partitions."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys


HERE = Path(__file__).resolve().parent
sys.path[1:1] = [str(HERE.parent / name) for name in (
    "Full_pheno_v14", "Full_pheno_v14_class", "Full_pheno_v13"
)]
from run_main import V13, PREVIOUS, REFERENCE, context, emit, parse, register, require, v1_run
from taxonomy import GROUPS, end_group, map_diets, summarize


V14 = HERE.parent / "Full_pheno_v14/runs/TRE_ONLY/main-1"
FRESH = (("fresh-1", 20262107), ("fresh-2", 20262139))
RECIPES = ("reference", "ridge_100")
OLD = "reference:half"
NEW = "ridge_100:half"
ENDPOINTS = (
    "nightingale:m_hdl_pl_pct",
    "nightingale:dha_pct",
    "serum_lipidomics:sm_d18:1/c20:0",
)
MARGIN = .10


def read_json(path):
    return v1_run.read_json(path)


def write_json(path, obj):
    v1_run.atomic_write(path, v1_run.encode(obj))


def discovery_candidates(summary, method):
    return {r["endpoint"] for r in summary["stable"]
            if r["diet"] == method and r["min_gain"] >= MARGIN and r["min_total"] > 0.}


def preflight(root, study, calibration):
    """Check the frozen role roster and discovery FoodID geometry without any outcome values."""
    import numpy as np
    import protocol as p
    import pyarrow.parquet as pq

    roles_path = root / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    pointer = root.parent / "config/w20_simplified_20261101_v2/active_execution.json"
    require(read_json(pointer)["roles_sha256"] == v1_run.file_sha(roles_path), "V15_ROLE_SOURCE_CHANGED")
    fields = [*p.KEY, "split_unit_id", "final_role"]
    roles = pq.ParquetFile(roles_path).read(columns=fields, use_threads=False).to_pandas(ignore_metadata=True)
    require(not roles[fields].isna().any().any() and not roles.duplicated(list(p.KEY)).any()
            and set(roles.final_role) == {"discovery", "confirmation_reserved"}, "V15_ROLE_CATALOG")
    discover = roles.loc[roles.final_role.eq("discovery")]
    reserved = roles.loc[roles.final_role.eq("confirmation_reserved")]
    archived = set(zip(*study.keys, study.family, strict=True))
    present = set(discover[[*p.KEY, "split_unit_id"]].itertuples(index=False, name=None))
    require(archived == present and len(present) == study.n, "V15_DISCOVERY_ROSTER_MISMATCH")
    require(not set(discover.split_unit_id) & set(reserved.split_unit_id), "V15_HOUSEHOLD_OVERLAP")
    require(calibration.weights.shape[0] == study.n
            and np.asanyarray(calibration.qualified).shape == (study.n,), "V15_CALIBRATION_MISMATCH")
    full_day = root.parent / (
        "data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet"
    )
    day = pq.ParquetFile(full_day)
    fiber_fields = {"date", "energy_kcal", "fiber_g", "energy_eligible", "fiber_eligible"}
    require(fiber_fields.issubset(day.schema_arrow.names), "V15_V5_FOOD_SCHEMA_CHANGED")
    return dict(
        discovery_people=len(discover), discovery_households=discover.split_unit_id.nunique(),
        reserved_people=len(reserved), reserved_households=reserved.split_unit_id.nunique(),
        shared_households=0, registered_foodid_rows=calibration.weights.shape[0],
        registered_reserved_foodid_rows=0, legacy_daily_rows=day.metadata.num_rows,
        expanded_predictors_ready=False, independent_scoring_started=False,
        reserved_outcomes_read=False,
    )


def matrix_comparison(parts, domain):
    comparisons = []
    for partition in parts:
        old = {r["endpoint"]: r for r in partition["rows"] if r["diet"] == OLD}
        new = {r["endpoint"]: r for r in partition["rows"] if r["diet"] == NEW}
        require(old.keys() == new.keys(), "V15_DIFFERENT_ENDPOINT_SUPPORT")
        gain = defaultdict(list)
        for e, r in new.items():
            require((old[e]["people"], old[e]["families"]) == (r["people"], r["families"]),
                    "V15_DIFFERENT_FAMILY_SUPPORT")
            gain[domain[e]].append(r["vs_reference_pp"])
        per_domain = {k: median(v) for k, v in sorted(gain.items()) if len(v) >= 5}
        comparisons.append(dict(seed=partition["seed"], scored=len(old), unscored=len(domain)-len(old),
                                domain_median_pp=per_domain, balanced_median_pp=mean(per_domain.values()),
                                negative=[sum(r["diet_total_pp"] < 0. for r in block.values())
                                          for block in (old, new)],
                                severe=[sum(r["diet_total_pp"] <= -MARGIN for r in block.values())
                                        for block in (old, new)]))
    return comparisons


def prespecified(parts):
    from run_expanded_grid import holm

    results = []
    for partition in parts:
        scored = {r["endpoint"]: r for r in partition["rows"] if r["diet"] == NEW}
        rows = [dict(scored[e], scorable=True) if e in scored else
                dict(endpoint=e, p_one_sided=1., scorable=False) for e in ENDPOINTS]
        holm(rows)
        for r in rows:
            r["joint_practical"] = bool(r["scorable"] and r["positive_folds"] >= 3
                                        and r["interaction_pp"] >= MARGIN and r["diet_total_pp"] > 0.)
        results.append(dict(seed=partition["seed"], rows=rows))
    return results


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    import main_fit
    import run_class_controls as class_fit
    from batch_sources import paths, read_diets, read_outcomes
    from interaction_fit import Repartition
    from run_expanded_grid import holm

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, calibration, sites, cal_paths = context()
        references = (REFERENCE, PREVIOUS, V13, V14)
        registrations = [read_json(r / "registration.json") for r in references]
        rounds = [read_json(r / "summary.json") for r in references[1:]]
        v11, _, _, v14 = registrations
        require(all(r["reserved_outcomes_read"] is False for r in rounds)
                and v11["snapshot"] == snapshot.manifest_sha, "V15_ARCHIVE_SCOPE")
        require(not {s for _, s in FRESH} & {s for r in registrations for _, s in r["seeds"]},
                "V15_REPEAT_SEED")
        require(set(ENDPOINTS) <= (discovery_candidates(rounds[-2], "fixed_half")
                                    & discovery_candidates(rounds[-1], NEW)), "V15_CANDIDATE_HISTORY")
        require(v14["code"]["main_fit.py"] == v1_run.file_sha(Path(main_fit.__file__)),
                "V15_BASELINE_IMPLEMENTATION_CHANGED")
        catalog = read_json(REFERENCE / "inventory.json")
        inputs = {Path(f): d for f, d in v11["sources"].items()}
        outcome_paths = paths(root)
        require(set((*outcome_paths.values(), *cal_paths)) <= inputs.keys(), "V15_UNREGISTERED_SOURCE")
        require(all(v1_run.file_sha(f) == d for f, d in inputs.items()), "V15_SOURCE_CHANGED")
        locked = [r / "registration.json" for r in references]
        locked.extend(r / "summary.json" for r in references[1:])
        locked.extend([REFERENCE / "inventory.json", V14 / "archive_replay.json"])
        proof = {str(f): v1_run.file_sha(f) for f in locked}
        require(read_json(V14 / "archive_replay.json")["maximum_score_difference"] < 1e-8,
                "V15_ARCHIVE_REPLAY")
        register(out, dict(commit=args.commit, runtime=runtime, seeds=[list(x) for x in FRESH],
                           scope="discovery re-partition and independent predictor preflight",
                           reference=proof, scope_snapshot=snapshot.manifest_sha,
                           code={f.name: v1_run.file_sha(f) for f in (
                               Path(__file__), HERE / "PROTOCOL.md", Path(main_fit.__file__),
                               Path(class_fit.__file__))},
                           recipes={k: list(main_fit.RECIPES[k]) for k in RECIPES},
                           interactions=main_fit.INTERACTIONS, class_recipes=list(class_fit.RECIPES),
                           class_groups=list(GROUPS), endpoints=list(ENDPOINTS), margin_pp=MARGIN))
        audit = preflight(root, study, calibration)
        write_json(out / "reserved_predictor_preflight.json", audit)
        emit("V15_PREFLIGHT", audit)

        outcome, meta = read_outcomes(outcome_paths, study)
        diets, _, food_paths = read_diets(root, study)
        endpoints = catalog["endpoint_ids"]
        require(list(outcome) == endpoints and [d[0] for d in diets] == catalog["diet_ids"]
                and set(food_paths.values()) <= inputs.keys(), "V15_CATALOG_CHANGED")
        map_diets(catalog["diet_ids"])
        domain = {e: end_group(e, meta["domains"][e]) for e in endpoints}
        require(domain == rounds[0]["endpoint_groups"], "V15_OUTCOME_DOMAINS_CHANGED")
        values = np.column_stack([outcome[e] for e in endpoints])
        parts, classes = [], []
        for name, seed in FRESH:
            whole_path = out / f"{name}.json"
            class_path = out / f"{name}-classes.json"
            if not whole_path.exists() or not class_path.exists():
                work = Repartition(study, sites, calibration, seed, emit)
            if not whole_path.exists():
                emit("V15_WHOLE_START", {"seed": seed})
                moments, _, methods = main_fit.collect(
                    work, diets, values, lambda f, n: emit("V15_WHOLE_FOLD", {"seed": seed, "fold": f, "n": n}),
                    recipes=RECIPES, strata=False)
                rows = main_fit.score_partition(moments, endpoints, methods)
                write_json(whole_path, dict(seed=seed, rows=rows))
            if not class_path.exists():
                emit("V15_CLASSES_START", {"seed": seed})
                moments, methods = class_fit.collect(
                    work, diets, values, lambda f: emit("V15_CLASS_FOLD", {"seed": seed, "fold": f}))
                rows = [dict(r, diet=f"{g}/{r['diet']}")
                        for g, matrix in zip(GROUPS, moments, strict=True)
                        for r in main_fit.score_partition(matrix, endpoints, methods)]
                holm(rows)
                write_json(class_path, dict(seed=seed, rows=rows))
            part, by_class = read_json(whole_path), read_json(class_path)
            require(part["seed"] == by_class["seed"] == seed and len(part["rows"]) >= 1400 * len(RECIPES) * 3
                    and len(by_class["rows"]) >= 1400 * len(GROUPS) * len(class_fit.RECIPES) * 3,
                    "V15_CHECKPOINT_SUPPORT")
            parts.append(part)
            classes.append(by_class)
            emit("V15_PARTITION_COMPLETE", {"seed": seed, "whole": len(part["rows"]), "class": len(by_class["rows"])})

        require(set(snapshot.paths) <= snapshot.verified, "V15_SNAPSHOT_UNVERIFIED")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(f) == d for f, d in inputs.items())
                and all(v1_run.file_sha(Path(f)) == d for f, d in proof.items()), "V15_INPUT_CHANGED")
        result = dict(commit=args.commit, reserved_outcomes_read=False, independent_confirmation=False,
                      predictor_preflight=audit, comparisons=matrix_comparison(parts, domain),
                      prespecified=prespecified(parts),
                      whole=summarize(parts, domain, {r["diet"]: r["diet"] for r in parts[0]["rows"]}),
                      classes=summarize(classes, domain, {r["diet"]: r["diet"] for r in classes[0]["rows"]}))
        write_json(out / "summary.json", result)
    emit("V15_COMPLETE", {"scored": [r["scored"] for r in result["comparisons"]],
                          "prespecified": len(ENDPOINTS), "reserved_outcomes_read": False})


if __name__ == "__main__":
    main()
