"""Evaluate conditional dietary-category interactions on matched discovery households."""

from collections import Counter
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path[1:1] = [str(HERE.parent / name) for name in (
    "Full_pheno_v15", "Full_pheno_v14", "Full_pheno_v13"
)]
from run_repeat import PREVIOUS, REFERENCE, V13, V14, context, emit, parse, read_json, register, require, v1_run, write_json
from taxonomy import GROUPS, end_group, map_diets, summarize


V15 = HERE.parent / "Full_pheno_v15/runs/TRE_ONLY/repeat-1"
SEEDS = (20262213, 20262247)
MARGIN = .10
INNER_MARGIN = .05
OFF = "interaction_off"
REFERENCE_HALF = "whole_half"
CHOSEN = "domain_selected"
METHODS = (REFERENCE_HALF, *GROUPS, CHOSEN, OFF)


def select_by_domain(fold_gains, valid, domains, sources):
    """Select at most one category per domain using held predictions inside training."""
    import numpy as np

    choices = np.full(len(domains), -1, dtype=int)
    decisions = {}
    for domain in sorted(set(domains)):
        belongs = domains == domain
        supported = belongs & valid
        groups = sorted(set(sources[supported]))
        record = dict(category=OFF, common=int(supported.sum()), sources=len(groups))
        if supported.sum() >= 5:
            by_source = [fold_gains[:, supported & (sources == s), :].mean(axis=1) for s in groups]
            gain = np.mean(by_source, axis=0)
            mean_gain = gain.mean(axis=0)
            positive = (gain > 0.).sum(axis=0)
            eligible = (positive >= 3) & (mean_gain >= INNER_MARGIN)
            if eligible.any():
                choice = int(np.argmax(np.where(eligible, mean_gain, -np.inf)))
                choices[belongs] = choice
                record.update(category=list(GROUPS)[choice], mean_pp=float(mean_gain[choice]),
                              positive_folds=int(positive[choice]))
        decisions[domain] = record
    return choices, decisions


def outer_moments(work, fold, diets, values, groups, choices=None):
    import numpy as np
    from adapt_interactions import matched_predictions
    from batch_fit import fold_moments

    labels = (REFERENCE_HALF,) if choices is None else METHODS
    moments = np.zeros((len(labels), values.shape[1], 10))
    for cols, y, a, b, on, codes, counts in matched_predictions(work, fold, diets, values, groups):
        half = b[:, :, None] + .5 * (on - b[:, :, None])
        if choices is None:
            forecasts = (half[:, :, 0],)
        else:
            valid = choices[cols] >= 0
            chosen = b.copy()
            chosen[:, valid] = half[:, np.flatnonzero(valid), 1 + choices[cols][valid]]
            forecasts = (*np.moveaxis(half, 2, 0), chosen, b)
        for j, c in enumerate(forecasts):
            moments[j, cols] = np.column_stack(fold_moments(y, (a, b, c), codes, counts))
    require(np.isfinite(moments).all() and np.all(moments[:, :, :6] == moments[:1, :, :6]),
            "V16_MAIN_BASELINE_CHANGED")
    if choices is not None:
        require(np.array_equal(moments[-1, :, 6], moments[-1, :, 5]), "V16_OFF_C_EQUALS_B")
    return moments


def train_policy(work, contexts, calibration, outer, diets, values, groups, domains, sources, seed):
    import numpy as np
    from adapt_interactions import DIRECTIONS, gains, inner_profile
    import protocol as p

    require(tuple(DIRECTIONS) == ("whole_diet", *GROUPS), "V16_DIET_DIRECTION_ORDER")
    arms = .5 * np.eye(len(DIRECTIONS))[1:]
    inner_gains, support = [], []
    for inner in p.FOLDS:
        if outer == inner:
            continue
        inner_work = contexts.split(outer, inner)
        inner_work, _ = calibration.inner(inner_work, inner, 100 * seed + 10 * outer + inner)
        bg = inner_work.backgrounds[inner]
        inner_work.backgrounds[inner] = replace(bg, supported=bg.supported & inner_work.active)
        require(np.array_equal(inner_work.active & ~bg.role, work.fold == inner)
                and not np.any(bg.role & (work.fold == outer)), "V16_INNER_OUTER_OVERLAP")
        profile, covered = inner_profile(inner_work, inner, diets, values, groups)
        inner_gains.append(gains(profile, arms))
        support.append(covered)
        emit("V16_INNER", dict(seed=seed, outer=outer, inner=inner, outcomes=int(covered.sum())))
    return select_by_domain(np.stack(inner_gains), np.logical_and.reduce(support), domains, sources)


def read_fold(path, seed, fold, shape):
    import numpy as np

    with np.load(path, allow_pickle=False) as checkpoint:
        data = checkpoint["moments"]
        identity = json.loads(str(checkpoint["metadata"].item()))
    require(identity["seed"] == seed and identity["fold"] == fold and data.shape == shape
            and np.isfinite(data).all() and np.all(data[:, :, :6] == data[:1, :, :6]),
            "V16_FOLD_CHECKPOINT_INVALID")
    return data, identity["selection"]


def fit_partition(out, work, sites, calibration, diets, values, endpoints, domains, sources, seed):
    import numpy as np
    from adapt_interactions import score_partition
    from batch_fit import outcome_groups
    import protocol as p
    from v3_inputs import InnerContexts

    groups = outcome_groups(values)
    contexts = InnerContexts(work, sites)
    moments, selected = [], []
    for fold in p.FOLDS:
        path = out / f"{seed}-fold-{fold}.npz"
        if not path.exists():
            choices, plan = train_policy(work, contexts, calibration, fold, diets, values,
                                         groups, domains, sources, seed)
            values_fold = outer_moments(work, fold, diets, values, groups, choices)
            buffer = BytesIO()
            np.savez_compressed(buffer, moments=values_fold, metadata=json.dumps(
                dict(seed=seed, fold=fold, selection=plan)))
            v1_run.atomic_write(path, buffer.getvalue())
        values_fold, plan = read_fold(path, seed, fold, (len(METHODS), len(endpoints), 10))
        moments.append(values_fold)
        selected.append(plan)
        emit("V16_OUTER", dict(seed=seed, fold=fold, outcomes=int((values_fold[0, :, 0] > 0.).sum())))
    rows = score_partition(np.stack(moments, axis=2), endpoints, METHODS)
    return dict(seed=seed, rows=rows, selection=selected)


def verify_reference(work, diets, values, endpoints, previous):
    import numpy as np
    from adapt_interactions import score_partition
    from batch_fit import outcome_groups
    import protocol as p

    groups = outcome_groups(values)
    by_fold = [outer_moments(work, f, diets, values, groups) for f in p.FOLDS]
    rows = score_partition(np.stack(by_fold, axis=2), endpoints, (REFERENCE_HALF,))
    old = {r["endpoint"]: r for r in previous["rows"] if r["diet"] == "reference:half"}
    require(set(old) == {r["endpoint"] for r in rows}
            and all((r["people"], r["families"]) == (old[r["endpoint"]]["people"], old[r["endpoint"]]["families"])
                    for r in rows), "V16_REFERENCE_SUPPORT_CHANGED")
    metrics = "diet_main_pp", "interaction_pp", "diet_total_pp"
    difference = max(abs(r[k] - old[r["endpoint"]][k]) for r in rows for k in metrics)
    require(difference < 1e-8, "V16_REFERENCE_SCORES_CHANGED")
    return dict(seed=previous["seed"], endpoints=len(rows), max_score_difference=difference)


def stable_sets(parts, methods):
    eligible, positive = {}, {}
    for method in methods:
        rows = [{r["endpoint"]: r for r in p["rows"] if r["diet"] == method} for p in parts]
        eligible[method] = set.intersection(*(set(r) for r in rows))
        positive[method] = {
            e for e in eligible[method]
            if all(x[e]["interaction_pp"] >= MARGIN and x[e]["diet_total_pp"] > 0.
                   and x[e]["positive_folds"] >= 3 for x in rows)
        }
    return eligible, positive


def aggregate(parts, domain, prior):
    eligible, stable = stable_sets(parts, METHODS)
    _, previous = stable_sets(prior, ("reference:half",))
    reference = stable[REFERENCE_HALF]
    identity = {m: m for m in METHODS}
    summary = summarize(parts, domain, identity)
    for method, row in summary["categories"].items():
        require(eligible[method] == eligible[REFERENCE_HALF], "V16_METHOD_SUPPORT_CHANGED")
        members = stable[method]
        union = previous["reference:half"] & eligible[method]
        row.update(retained_current_whole=len(reference & members),
                   new_relative_to_whole=len(members - reference),
                   v15_available=len(union), retained_v15=len(union & members))
        compared = []
        for part in parts:
            by_domain = {}
            for r in part["rows"]:
                if r["diet"] == method:
                    by_domain.setdefault(domain[r["endpoint"]], []).append(r["vs_fixed_pp"])
            compared.append(mean(median(x) for x in by_domain.values() if len(x) >= 5))
        row["balanced_vs_whole_half_pp"] = compared
    selection = {str(p["seed"]): {
        d: dict(Counter(f[d]["category"] for f in p["selection"])) for d in sorted(set(domain.values()))
    } for p in parts}
    return dict(**summary, stable_endpoints={m: sorted(x) for m, x in stable.items()}, selection=selection)


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    import adapt_interactions
    from batch_sources import paths, read_diets, read_outcomes
    import category_fit
    from interaction_fit import Repartition

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, calibration, sites, cal_files = context()
        archives = (REFERENCE, PREVIOUS, V13, V14, V15)
        registered = [read_json(a / "registration.json") for a in archives]
        require(snapshot.manifest_sha == registered[0]["snapshot"]
                and not set(SEEDS) & {s for reg in registered for _, s in reg["seeds"]}, "V16_FOLD_REGISTRATION")
        summary = read_json(V15 / "summary.json")
        require(summary["reserved_outcomes_read"] is False and not (V15 / "RUNNING.lock").exists(),
                "V16_REFERENCE_SCOPE")
        prior_paths = [V15 / f"fresh-{j}.json" for j in (1, 2)]
        prior = [read_json(f) for f in prior_paths]
        inventory = read_json(REFERENCE / "inventory.json")
        sources = {Path(f): h for f, h in registered[0]["sources"].items()}
        outcomes = paths(root)
        require(set((*outcomes.values(), *cal_files)) <= sources.keys(), "V16_UNREGISTERED_INPUT")
        require(all(v1_run.file_sha(f) == h for f, h in sources.items()), "V16_INPUT_CHANGED")
        proof = {str(f): v1_run.file_sha(f) for f in (
            *(a / "registration.json" for a in archives), V15 / "summary.json",
            V14 / "archive_replay.json", REFERENCE / "inventory.json", *prior_paths,
        )}
        code = {f.name: v1_run.file_sha(f) for f in (
            Path(__file__), HERE / "PROTOCOL.md", Path(adapt_interactions.__file__), Path(category_fit.__file__),
        )}
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snapshot.manifest_sha,
                           seeds=[[f"fresh-{j}", seed] for j, seed in enumerate(SEEDS, 1)],
                           methods=list(METHODS), threshold_pp=MARGIN,
                           inner_threshold_pp=INNER_MARGIN, archives=proof, code=code))
        values, meta = read_outcomes(outcomes, study)
        diets, _, food = read_diets(root, study)
        endpoints = inventory["endpoint_ids"]
        require(list(values) == endpoints and [d[0] for d in diets] == inventory["diet_ids"]
                and set(food.values()) <= sources.keys(), "V16_CATALOG_CHANGED")
        map_diets(inventory["diet_ids"])
        domain = {e: end_group(e, meta["domains"][e]) for e in endpoints}
        require(domain == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V16_DOMAIN_CHANGED")
        y = np.column_stack([values[e] for e in endpoints])
        groups = np.array([domain[e] for e in endpoints])
        phenotype_sources = np.array([meta["domains"][e] for e in endpoints])
        replay_path = out / "v15_replay.json"
        if not replay_path.exists():
            work = Repartition(study, sites, calibration, prior[0]["seed"], emit)
            replay = verify_reference(work, diets, y, endpoints, prior[0])
            write_json(replay_path, replay)
        replay = read_json(replay_path)
        require(replay["seed"] == prior[0]["seed"] and replay["max_score_difference"] < 1e-8,
                "V16_INVALID_REPLAY")
        emit("V16_V15_REPLAY", replay)
        fitted = []
        for seed in SEEDS:
            path = out / f"{seed}.json"
            if not path.exists():
                work = Repartition(study, sites, calibration, seed, emit)
                part = fit_partition(out, work, sites, calibration, diets, y, endpoints, groups, phenotype_sources, seed)
                write_json(path, part)
            part = read_json(path)
            require(part["seed"] == seed and len(part["rows"]) >= 1400 * len(METHODS),
                    "V16_PARTITION_SUPPORT")
            fitted.append(part)
            emit("V16_PARTITION", dict(seed=seed, scored=len(part["rows"]) // len(METHODS)))
        require(set(snapshot.paths) <= snapshot.verified, "V16_SNAPSHOT_INCOMPLETE")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(f) == h for f, h in sources.items())
                and all(v1_run.file_sha(Path(f)) == h for f, h in proof.items()), "V16_SOURCE_CHANGED_DURING_RUN")
        output = dict(commit=args.commit, v15_replay=replay, **aggregate(fitted, domain, prior),
                      reserved_outcomes_read=False)
        write_json(out / "summary.json", output)
    emit("V16_COMPLETE", dict(pairs=output["shared_pairs"], methods=len(METHODS), reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
