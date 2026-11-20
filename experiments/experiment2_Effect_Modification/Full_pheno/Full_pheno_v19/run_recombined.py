"""Repeat calibrated-diet and gut-interaction comparisons on matched households."""

from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v18"))
from run_main_audit import PREVIOUS, REFERENCE, V13, V14, V15, V16, V17, context, emit, parse, read_json, register, require, v1_run, write_json
from taxonomy import end_group, map_diets, summarize


V18 = HERE.parent / "Full_pheno_v18/runs/TRE_ONLY/main-audit-1"
SEEDS = (20262509, 20262543)
REFERENCE_HALF = "reference:half"
PRIMARY = "domain_calibrated:half"
MARGIN = .10


def names(mains):
    from main_fit import INTERACTIONS
    return tuple(f"{main}:{arm}" for main in mains for arm in INTERACTIONS)


def verify(replayed, archived, group, target, columns, count_name):
    rows = {r["endpoint"]: r for r in replayed if r["diet"] == group}
    require(rows.keys() == archived.keys() and all(
        (r["people"], r["families"]) == (archived[e]["people"], archived[e][count_name])
        for e, r in rows.items()), "V19_REPLAY_MEMBERSHIP_CHANGED " + target)
    error = max(abs(r[k] - archived[e][old]) for e, r in rows.items() for k, old in columns)
    require(error < 1e-8, "V19_REFERENCE_SCORE_CHANGED " + target)
    return dict(endpoints=len(rows), max_error_pp=error)


def replay(study, sites, cal, diets, values, masks, endpoints, domains, prior17, prior18):
    import numpy as np
    import protocol as p
    from combined_fit import MAINS, collect
    from interaction_fit import Repartition
    from main_fit import score_partition

    work = Repartition(study, sites, cal, prior17["seed"], emit)
    raw = [collect(work, f, diets, values, masks, np.ones(len(endpoints)), mains=("reference",))[0] for f in p.FOLDS]
    rows = score_partition(np.stack(raw, axis=2), endpoints, names(("reference",)))
    previous = {r["endpoint"]: r for r in prior17["rows"] if r["diet"] == "whole_half"}
    columns = tuple((f, f) for f in ("diet_main_pp", "interaction_pp", "diet_total_pp"))
    reference = verify(rows, previous, REFERENCE_HALF, "V17 whole half", columns, "families")

    work = Repartition(study, sites, cal, prior18["seed"], emit)
    raw = []
    for f, policy in zip(p.FOLDS, prior18["amplitudes"], strict=True):
        alpha = np.array([policy[domains[e]]["amplitude"] for e in endpoints])
        raw.append(collect(work, f, diets, values, masks, alpha)[0])
    rows = score_partition(np.stack(raw, axis=2), endpoints, names(MAINS))
    mains = {}
    for arm in MAINS:
        previous = {r["endpoint"]: r for r in prior18["rows"] if r["arm"] == arm}
        mains[arm] = verify(rows, previous, f"{arm}:off", f"V18 {arm}",
                            (("diet_main_pp", "main_pp"),), "households")
    return dict(reference_half=reference, v18_main=mains)


def run_partition(out, seed, work, sites, cal, diets, values, masks, endpoints, domain, source):
    import numpy as np
    import protocol as p
    from combined_fit import MAINS, collect
    from main_attribution import inner_scales
    from main_fit import score_partition
    from v3_inputs import InnerContexts

    methods = names(MAINS)
    context = InnerContexts(work, sites)
    matrices, policies, gaps = [], [], []
    for fold in p.FOLDS:
        path = out / f"{seed}-fold-{fold}.npz"
        if not path.exists():
            alpha, policy = inner_scales(work, context, cal, fold, diets, values, masks, domain, source, seed, emit)
            data, gap = collect(work, fold, diets, values, masks, alpha)
            buffer = BytesIO()
            np.savez_compressed(buffer, moments=data,
                                metadata=json.dumps(dict(seed=seed, fold=fold, selection=policy, update_gap=gap)))
            v1_run.atomic_write(path, buffer.getvalue())
        with np.load(path, allow_pickle=False) as saved:
            data = saved["moments"]
            label = json.loads(str(saved["metadata"].item()))
        require(label["seed"] == seed and label["fold"] == fold and data.shape == (len(methods), len(endpoints), 10)
                and np.isfinite(data).all() and np.all(data[:, :, :5] == data[:1, :, :5]),
                "V19_FOLD_CHECKPOINT_CHANGED")
        policy = label["selection"]
        matrices.append(data)
        policies.append(policy)
        gaps.append(label["update_gap"])
        emit("V19_OUTER", dict(seed=seed, fold=fold, scorable=int((data[0, :, 0] > 0).sum()),
                               adjusted_domains=sum(x["amplitude"] < 1. for x in policy.values())))
    return dict(seed=seed, rows=score_partition(np.stack(matrices, axis=2), endpoints, methods),
                amplitudes=policies, relative_interaction_update_gap=max(gaps))


def aggregate(parts, domains, prior_stable):
    from combined_fit import MAINS
    from run_conditional import stable_sets

    methods = names(MAINS)
    available, stable = stable_sets(parts, methods)
    summary = summarize(parts, domains, {m: m for m in methods})
    reference = stable[REFERENCE_HALF]
    strict = {m: {g: [] for g in ("main", "interaction", "total")} for m in methods}
    primary_by_domain = defaultdict(list)
    split_counts = {}
    for part in parts:
        lookup = {(x["diet"], x["endpoint"]): x for x in part["rows"]}
        split_counts[str(part["seed"])] = len({e for m, e in lookup if m == REFERENCE_HALF})
        for method in methods:
            per_domain = defaultdict(list)
            reference_arm = "reference:" + method.split(":")[1]
            for e in sorted(available[method]):
                here, base = lookup[method, e], lookup[reference_arm, e]
                require((here["people"], here["families"]) == (base["people"], base["families"]),
                        "V19_CROSS_MODEL_SUPPORT_CHANGED")
                effect = [here["diet_main_pp"] - base["diet_main_pp"],
                          here["interaction_pp"] - base["interaction_pp"], here["vs_reference_pp"]]
                per_domain[domains[e]].append(effect)
            pooled = {k: [median(item) for item in zip(*v, strict=True)] for k, v in per_domain.items() if len(v) >= 5}
            for index, key in enumerate(("main", "interaction", "total")):
                strict[method][key].append(mean(val[index] for val in pooled.values()))
            if method == PRIMARY:
                for k, vals in pooled.items():
                    primary_by_domain[k].append(dict(zip(("main", "interaction", "total"), vals, strict=True)))
    for method, z in summary["categories"].items():
        require(available[method] == available[REFERENCE_HALF], "V19_METHOD_CATALOG_MISMATCH")
        here = stable[method]
        groups = [{r["endpoint"]: r for r in part["rows"] if r["diet"] == method} for part in parts]
        z.update(retained_reference=len(here & reference), retained_v17=len(here & prior_stable),
                 new_relative_to_reference=len(here - reference), domain_balanced_delta_pp=strict[method],
                 severe_total_both=sum(all(p[e]["diet_total_pp"] <= -MARGIN for p in groups)
                                       for e in available[method]))
    base = summary["categories"][REFERENCE_HALF]
    target = summary["categories"][PRIMARY]
    checks = dict(
        domain_balanced_total_both_positive=all(x > 0. for x in target["domain_balanced_delta_pp"]["total"]),
        less_severe_total_loss=target["severe_total_both"] < base["severe_total_both"],
        retains_80_percent_of_reference=bool(reference) and target["retained_reference"] >= .8 * len(reference),
    )
    return dict(**summary, primary=PRIMARY, primary_checks=checks, meets_research_gate=all(checks.values()),
                primary_paired_by_domain=dict(primary_by_domain), evaluated_per_split=split_counts,
                max_relative_interaction_update_gap=max(p["relative_interaction_update_gap"] for p in parts),
                stable_endpoints={m: sorted(z) for m, z in stable.items()})


def main():
    args = parse(__doc__)
    threads = v1_run.init_runtime(1)
    import numpy as np
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    from combined_fit import MAINS
    from interaction_fit import Repartition
    import main_attribution
    import main_fit

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, cal, sites, cal_file = context()
        periods = (REFERENCE, PREVIOUS, V13, V14, V15, V16, V17, V18)
        regs = [read_json(t / "registration.json") for t in periods]
        require(snapshot.manifest_sha == regs[0]["snapshot"]
                and not set(SEEDS) & {s for t in regs for _, s in t["seeds"]}, "V19_SEED_OR_SNAPSHOT_CHANGED")
        code = Path(main_attribution.__file__)
        require(v1_run.file_sha(code) == regs[-1]["code"][code.name], "V19_V18_AMPLITUDE_CODE_CHANGED")
        a17, a18 = (p / f"{r['seeds'][0][1]}.json" for p, r in zip((V17, V18), regs[-2:], strict=True))
        u17, u18 = (read_json(p / "summary.json") for p in (V17, V18))
        require(all(x["reserved_outcomes_read"] is False for x in (u17, u18))
                and not any((p / "RUNNING.lock").exists() for p in (V17, V18)), "V19_HISTORICAL_SCOPE_INVALID")
        proof = {str(f): v1_run.file_sha(f) for f in (
            *(p / "registration.json" for p in periods), V17 / "summary.json", V18 / "summary.json",
            REFERENCE / "inventory.json", PREVIOUS / "summary.json", a17, a18,
        )}
        inputs = {Path(p): h for p, h in regs[0]["sources"].items()}
        sources = paths(root)
        require(set((*sources.values(), *cal_file)) <= inputs.keys()
                and all(v1_run.file_sha(f) == h for f, h in inputs.items()), "V19_UNREGISTERED_INPUT")
        register(out, dict(commit=args.commit, runtime=threads, snapshot=snapshot.manifest_sha, archives=proof,
                           seeds=[[f"fresh-{j}", s] for j, s in enumerate(SEEDS, 1)], methods=list(names(MAINS)),
                           code={f.name: v1_run.file_sha(f) for f in (Path(__file__), HERE / "combined_fit.py",
                                 HERE / "PROTOCOL.md", code, Path(main_fit.__file__))}))
        catalog = read_json(REFERENCE / "inventory.json")
        diets, _, food = read_diets(root, study)
        ends, notes = read_outcomes(sources, study)
        endpoints = catalog["endpoint_ids"]
        require(list(ends) == endpoints and [x[0] for x in diets] == catalog["diet_ids"]
                and set(food.values()) <= inputs.keys(), "V19_CATALOG_CHANGED")
        map_diets(catalog["diet_ids"])
        domains = {e: end_group(e, notes["domains"][e]) for e in endpoints}
        require(domains == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V19_DOMAIN_CHANGED")
        values = np.column_stack([ends[e] for e in endpoints])
        masks = outcome_groups(values)
        src, ds = np.array([notes["domains"][e] for e in endpoints]), np.array([domains[e] for e in endpoints])
        replay_path = out / "replay.json"
        if not replay_path.exists():
            record = replay(study, sites, cal, diets, values, masks, endpoints, domains, read_json(a17), read_json(a18))
            write_json(replay_path, record)
        record = read_json(replay_path)
        require(all(y["max_error_pp"] < 1e-8 for y in (record["reference_half"], *record["v18_main"].values())),
                "V19_REPLAY_INVALID")
        emit("V19_REPLAY", record)
        parts = []
        for seed in SEEDS:
            path = out / f"{seed}.json"
            if not path.exists():
                work = Repartition(study, sites, cal, seed, emit)
                part = run_partition(out, seed, work, sites, cal, diets, values, masks, endpoints, ds, src)
                write_json(path, part)
            part = read_json(path)
            require(part["seed"] == seed and len(part["rows"]) >= 1400 * len(names(MAINS)),
                    "V19_PARTITION_INVALID")
            parts.append(part)
            emit("V19_PARTITION", dict(seed=seed, endpoints=len(part["rows"]) // len(names(MAINS))))
        snapshot.verify_manifest()
        require(set(snapshot.paths) <= snapshot.verified and all(v1_run.file_sha(f) == h for f, h in inputs.items())
                and all(v1_run.file_sha(Path(f)) == h for f, h in proof.items()), "V19_INPUT_CHANGED_DURING_RUN")
        summary = aggregate(parts, domains, set(u17["stable_endpoints"]["whole_half"]))
        write_json(out / "summary.json", dict(commit=args.commit, replay=record, reserved_outcomes_read=False, **summary))
    emit("V19_COMPLETE", dict(pairs=summary["shared_pairs"], met_gate=summary["meets_research_gate"],
                             reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
