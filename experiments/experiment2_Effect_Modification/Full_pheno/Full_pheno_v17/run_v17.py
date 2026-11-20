"""Compare reference-anchored interactions, diary reliability and lipid families."""

from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v16"))
from run_conditional import (PREVIOUS, REFERENCE, V13, V14, V15, context, emit, parse, read_fold, read_json,
                             register, require, stable_sets, v1_run, write_json)
from taxonomy import GROUPS, end_group, map_diets, summarize


V16 = HERE.parent / "Full_pheno_v16/runs/TRE_ONLY/conditional-1"
SEEDS = (20262311, 20262347)


def replay(work, diets, y, masks, endpoints, previous):
    import numpy as np
    import protocol as p
    from adapt_interactions import score_partition
    from reference_updates import REFERENCE_HALF, fold_scores

    scores = [fold_scores(work, f, diets, y, masks)[0] for f in p.FOLDS]
    current = score_partition(np.stack(scores, axis=2), endpoints, (REFERENCE_HALF,))
    old = {r["endpoint"]: r for r in previous["rows"] if r["diet"] == REFERENCE_HALF}
    require(old.keys() == {r["endpoint"] for r in current}
            and all((r["people"], r["families"]) == (old[r["endpoint"]]["people"], old[r["endpoint"]]["families"])
                    for r in current), "V17_REFERENCE_ROSTER_DIFF")
    metrics = ("diet_main_pp", "interaction_pp", "diet_total_pp")
    max_diff = max(abs(r[k] - old[r["endpoint"]][k]) for r in current for k in metrics)
    require(max_diff < 1e-8, "V17_REFERENCE_SCORE_DIFF")
    return dict(seed=previous["seed"], count=len(current), max_diff=max_diff)


def run_partition(out, seed, work, sites, cal, diets, diary, values, masks, endpoints, domains, sources):
    import numpy as np
    import protocol as p
    from adapt_interactions import score_partition
    from reference_updates import METHODS, fold_scores, train_adjustments
    from v3_inputs import InnerContexts

    ctx = InnerContexts(work, sites)
    blocks, audits = [], []
    for fold in p.FOLDS:
        checkpoint = out / f"{seed}-fold-{fold}.npz"
        if not checkpoint.exists():
            weight, selection = train_adjustments(work, ctx, cal, fold, diets, values, masks, domains, sources, seed, emit)
            matrix, reliability = fold_scores(work, fold, diets, values, masks, weight, diary)
            buffer = BytesIO()
            np.savez_compressed(buffer, moments=matrix, metadata=json.dumps(
                dict(seed=seed, fold=fold, selection=dict(anchored=selection, diary=reliability))))
            v1_run.atomic_write(checkpoint, buffer.getvalue())
        matrix, record = read_fold(checkpoint, seed, fold, (len(METHODS), len(endpoints), 10))
        blocks.append(matrix)
        audits.append(record)
        emit("V17_OUTER", dict(seed=seed, fold=fold, active_domains=sum(x["active"] for x in record["anchored"].values()),
                               scorable=int((matrix[0, :, 0] > 0).sum())))
    return dict(seed=seed, rows=score_partition(np.stack(blocks, axis=2), endpoints, METHODS), audits=audits)


def aggregate(parts, domains, v16, values, endpoints):
    from lipid_families import describe
    from reference_updates import ADJUST, DAYS, DIARY, MEAN, METHODS, REFERENCE_HALF

    eligible, stable = stable_sets(parts, METHODS)
    ref = stable[REFERENCE_HALF]
    ref_prior = set(v16["stable_endpoints"][REFERENCE_HALF])
    summary = summarize(parts, domains, {m: m for m in METHODS})
    for method, row in summary["categories"].items():
        require(eligible[method] == eligible[REFERENCE_HALF], "V17_METHOD_SUPPORT_DIFF")
        gains = []
        for part in parts:
            by_domain = defaultdict(list)
            for r in part["rows"]:
                if r["diet"] == method:
                    by_domain[domains[r["endpoint"]]].append(r["vs_fixed_pp"])
            gains.append(mean(median(x) for x in by_domain.values() if len(x) >= 5))
        row.update(retained_reference=len(stable[method] & ref), reference_stable=len(ref),
                   new_stable=len(stable[method] - ref), retained_v16=len(stable[method] & ref_prior),
                   domain_balanced_vs_reference_pp=gains)
    ref_loss = summary["categories"][REFERENCE_HALF]["interaction_loss"]
    for method in (ADJUST, DIARY, DAYS, MEAN):
        row = summary["categories"][method]
        row["meets_predeclared_replacement_gate"] = (row["interaction_loss"] < ref_loss and bool(ref)
                                                     and row["retained_reference"] >= .8 * len(ref)
                                                     and all(x > 0. for x in row["domain_balanced_vs_reference_pp"]))
    return dict(**summary, stable_endpoints={m: sorted(x) for m, x in stable.items()},
                lipid_audit=describe(v16["stable_endpoints"], stable, domains, values, endpoints))


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    from diary_reliability import read_halves
    from interaction_fit import Repartition
    from reference_updates import METHODS, GAIN, LIMIT

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, cal, sites, cal_files = context()
        previous = read_json(V16 / "summary.json")
        require(previous["reserved_outcomes_read"] is False and not (V16 / "RUNNING.lock").exists(), "V17_SCOPE")
        histories = (REFERENCE, PREVIOUS, V13, V14, V15, V16)
        regs = [read_json(h / "registration.json") for h in histories]
        require(snapshot.manifest_sha == regs[0]["snapshot"]
                and not set(SEEDS) & {s for x in regs for _, s in x["seeds"]}, "V17_SEED_OR_SNAPSHOT")
        refs = {str(f): v1_run.file_sha(f) for f in (
            *(h / "registration.json" for h in histories), V16 / "summary.json", V15 / "summary.json",
            REFERENCE / "inventory.json", PREVIOUS / "summary.json", V14 / "archive_replay.json",
        )}
        old_seed = regs[-1]["seeds"][0][1]
        previous_part = V16 / f"{old_seed}.json"
        refs[str(previous_part)] = v1_run.file_sha(previous_part)
        inputs = {Path(k): v for k, v in regs[0]["sources"].items()}
        sources = paths(root)
        require(set((*sources.values(), *cal_files)) <= inputs.keys(), "V17_UNREGISTERED_INPUT")
        require(all(v1_run.file_sha(f) == h for f, h in inputs.items()), "V17_RECORDED_SOURCE_CHANGED")
        register(out, dict(commit=args.commit, runtime=runtime, snapshots=refs, snapshot=snapshot.manifest_sha,
                           seeds=[[f"fresh-{j}", s] for j, s in enumerate(SEEDS, 1)],
                           methods=list(METHODS), max_relative_adjustment=LIMIT, inner_minimum_pp=GAIN,
                           code={f.name: v1_run.file_sha(f) for f in sorted(HERE.iterdir())
                                 if f.is_file() and f.suffix in (".py", ".md")}))
        inv = read_json(REFERENCE / "inventory.json")
        diets, _, diary_sources = read_diets(root, study)
        require([r[0] for r in diets] == inv["diet_ids"] and set(diary_sources.values()) <= inputs.keys(),
                "V17_DIET_CATALOG_DIFF")
        map_diets(inv["diet_ids"])
        diary, counts = read_halves(root, study, set(inv["diet_ids"]), diary_sources["day"])
        emit("V17_DIARY", dict(people_ge3_each=int(((counts[0] >= 3) & (counts[1] >= 3)).sum()),
                              raw_features={k: len(v) for k, v in diary.items()}))
        ends, metadata = read_outcomes(sources, study)
        endpoints = inv["endpoint_ids"]
        require(list(ends) == endpoints, "V17_OUTCOME_CATALOG_DIFF")
        domains = {e: end_group(e, metadata["domains"][e]) for e in endpoints}
        require(domains == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V17_OUTCOME_DOMAINS_DIFF")
        values = np.column_stack([ends[e] for e in endpoints])
        masks = outcome_groups(values)
        source_labels = np.array([metadata["domains"][e] for e in endpoints])
        domain_labels = np.array([domains[e] for e in endpoints])
        replay_path = out / "v16_replay.json"
        if not replay_path.exists():
            work = Repartition(study, sites, cal, old_seed, emit)
            record = replay(work, diets, values, masks, endpoints, read_json(previous_part))
            write_json(replay_path, record)
        record = read_json(replay_path)
        require(record["seed"] == old_seed and record["max_diff"] < 1e-8, "V17_REPLAY_RECORD_INVALID")
        emit("V17_REFERENCE_REPLAY", record)
        parts = []
        for seed in SEEDS:
            saved = out / f"{seed}.json"
            if not saved.exists():
                work = Repartition(study, sites, cal, seed, emit)
                result = run_partition(out, seed, work, sites, cal, diets, diary, values, masks,
                                       endpoints, domain_labels, source_labels)
                write_json(saved, result)
            result = read_json(saved)
            require(result["seed"] == seed and len(result["rows"]) >= 1400 * len(METHODS), "V17_PARTITION_SUPPORT")
            parts.append(result)
            emit("V17_PARTITION", dict(seed=seed, outcomes=len(result["rows"]) // len(METHODS)))
        require(set(snapshot.paths) <= snapshot.verified, "V17_DISCOVERY_VERIFY")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(f) == h for f, h in inputs.items())
                and all(v1_run.file_sha(Path(f)) == h for f, h in refs.items()), "V17_INPUT_CHANGED_DURING_RUN")
        report = aggregate(parts, domains, previous, values, endpoints)
        write_json(out / "summary.json", dict(commit=args.commit, reserved_outcomes_read=False, replay=record, **report))
    emit("V17_COMPLETE", dict(comparisons=report["shared_pairs"], methods=len(METHODS),
                             v16_lipid_candidates=report["lipid_audit"]["v16"]["lipid"],
                             v17_lipid_candidates=report["lipid_audit"]["v17"]["lipid"]))


if __name__ == "__main__":
    main()
