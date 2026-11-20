"""Apply V19 calibrated diet mains to seven conditional gut interactions."""

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v19"))
from run_recombined import (PREVIOUS, REFERENCE, V13, V14, V15, V16, V17, V18, context,
                            emit, parse, read_json, register, require, v1_run, verify, write_json)
from taxonomy import GROUPS, end_group, map_diets, summarize


V19 = HERE.parent / "Full_pheno_v19/runs/TRE_ONLY/recombine-1"
SEEDS = (20262611, 20262643)
CAL = "domain_calibrated:"


def replay(work, diets, values, masks, endpoints, domain, prior):
    import numpy as np
    import protocol as p
    from category_forecasts import BASELINES, WHOLE, collect, score

    frames = []
    for fold, fitted in zip(p.FOLDS, prior["amplitudes"], strict=True):
        scale = np.array([fitted[domain[e]]["amplitude"] for e in endpoints])
        frames.append(collect(work, fold, diets, values, masks, scale))
    rows = score(frames, endpoints)
    columns = tuple((k, k) for k in ("diet_main_pp", "interaction_pp", "diet_total_pp"))
    return {b: verify(rows, {x["endpoint"]: x for x in prior["rows"] if x["diet"] == b + ":half"},
                      b + ":" + WHOLE, b + " V19 half", columns, "families") for b in BASELINES}


def fit_partition(out, seed, work, sites, cal, diets, y, masks, endpoints, domain, source):
    import numpy as np
    import protocol as p
    from category_forecasts import BASELINES, METHODS, collect, score
    from main_attribution import inner_scales
    from v3_inputs import InnerContexts

    frames = []
    ctx = InnerContexts(work, sites)
    for f in p.FOLDS:
        path = out / f"{seed}-fold-{f}.npz"
        if not path.exists():
            scale, policy = inner_scales(work, ctx, cal, f, diets, y, masks, domain, source, seed, emit)
            data = collect(work, f, diets, y, masks, scale)
            buf = BytesIO()
            np.savez_compressed(buf, moments=data, metadata=json.dumps(dict(seed=seed, fold=f, policy=policy)))
            v1_run.atomic_write(path, buf.getvalue())
        with np.load(path, allow_pickle=False) as saved:
            data = saved["moments"]
            metadata = json.loads(str(saved["metadata"].item()))
        require(metadata["seed"] == seed and metadata["fold"] == f
                and data.shape == (len(BASELINES), len(METHODS), len(endpoints), 12)
                and np.isfinite(data).all() and np.all(data[:, :, :, :5] == data[:1, :1, :, :5])
                and np.all(data[:, :, :, 5] == data[:, :1, :, 5]), "V20_CHECKPOINT_CHANGED")
        frames.append(data)
        emit("V20_OUTER", dict(seed=seed, fold=f, outcomes=int((data[0, 0, :, 0] > 0.).sum()),
                               adjusted_domains=sum(x["amplitude"] < 1. for x in metadata["policy"].values())))
    return dict(seed=seed, rows=score(frames, endpoints))


def alignment_class(rows, field):
    return ("nonpositive_both", "mixed", "positive_both")[sum(x[field] > 0. for x in rows)]


def attribute(indices, common, domains):
    from category_forecasts import METHODS, OFF

    def records(key, endpoint):
        return [p[key, endpoint] for p in indices]

    reasons = defaultdict(Counter)
    interactions = {}
    for m in METHODS:
        by_domain = defaultdict(Counter)
        for e in sorted(common):
            off = records(CAL + OFF, e)
            on = records(CAL + m, e)
            base_negative = sum(x["diet_main_pp"] < 0. for x in off)
            total_negative = sum(x["diet_total_pp"] < 0. for x in on)
            if base_negative == 2 and m == OFF:
                reasons[domains[e]][alignment_class(off, "main_alignment_pp")] += 1
            v = by_domain[domains[e]]
            if total_negative == 2:
                label = ("interaction_added_both", "mixed_base_loss", "already_main_loss_both")[base_negative]
                v[label] += 1
                if base_negative == 0:
                    v["new_loss_alignment_" + alignment_class(on, "interaction_alignment_pp")] += 1
            if base_negative == 2 and total_negative == 0:
                v["recovered_both"] += 1
        total = sum(by_domain.values(), Counter())
        interactions[m] = dict(pooled=dict(total), by_domain={d: dict(v) for d, v in by_domain.items()})
    return dict(main_alignment=dict(sum(reasons.values(), Counter())),
                main_alignment_by_domain={k: dict(v) for k, v in reasons.items()}, conditional=interactions)


def inspect(parts, domains, v16, v17, frozen_targets):
    from category_forecasts import BASELINES, METHODS, OFF, WHOLE
    from run_conditional import stable_sets

    labels = tuple(f"{b}:{m}" for b in BASELINES for m in METHODS)
    eligibility, stability = stable_sets(parts, labels)
    common = eligibility[CAL + OFF]
    require(all(s == common for s in eligibility.values()), "V20_DIFFERENT_ENDPOINT_SUPPORT")
    report = summarize(parts, domains, {m: m for m in labels})
    index = [{(r["diet"], r["endpoint"]): r for r in p["rows"]} for p in parts]
    for label, r in report["categories"].items():
        b, m = label.split(":")
        r["retained_current_reference"] = len(stability[label] & stability[f"reference:{m}"])
        if m in GROUPS:
            repeat = set(v16["stable_endpoints"][m]) & set(v17["stable_endpoints"][m]) & common
            r.update(prior_v16_v17=len(repeat), retained_repeated=len(repeat & stability[label]))
        pooled = []
        for p in index:
            by_domain = defaultdict(list)
            for e in sorted(common):
                by_domain[domains[e]].append(p[label, e]["interaction_pp"])
            pooled.append(mean(median(x) for x in by_domain.values() if len(x) >= 5))
        r["domain_balanced_interaction_pp"] = pooled
        r["large_total_loss_both"] = sum(all(p[label, e]["diet_total_pp"] <= -.10 for p in index) for e in common)
    targets = []
    for m, e in frozen_targets:
        entry = dict(diet=m, endpoint=e, old_stable=all(e in a["stable_endpoints"][m] for a in (v16, v17)),
                     predictions={})
        for b in BASELINES:
            label = f"{b}:{m}"
            mapped = [p[label, e] for p in index if (label, e) in p]
            entry["predictions"][b] = dict(stable=e in stability[label], parts=[
                {key: r[key] for key in ("diet_main_pp", "interaction_pp", "diet_total_pp", "positive_folds")}
                for r in mapped])
        targets.append(entry)
    class_union = {b: sorted(set.union(*(stability[f"{b}:{m}"] for m in GROUPS))) for b in BASELINES}
    class_union_domain = {b: dict(Counter(domains[e] for e in names)) for b, names in class_union.items()}
    attrib = attribute(index, common, domains)
    for m in METHODS:
        counts = attrib["conditional"][m]["pooled"]
        n = sum(counts.get(k, 0) for k in ("interaction_added_both", "mixed_base_loss", "already_main_loss_both"))
        require(n == report["categories"][CAL + m]["total_negative"], "V20_LOSS_ATTRIBUTION_TOTAL_DIFF")
    return dict(**report, endpoints=len(common), stable_endpoints={k: sorted(v) for k, v in stability.items()},
                class_union=class_union, class_union_domain=class_union_domain, targets=targets,
                paired_attribution=attrib, baseline_comparison=CAL + WHOLE)


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    import category_forecasts
    from interaction_fit import Repartition
    import main_attribution
    from sample_audit import CANDIDATES

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, cal, sites, cal_files = context()
        past = (REFERENCE, PREVIOUS, V13, V14, V15, V16, V17, V18, V19)
        registry = [read_json(p / "registration.json") for p in past]
        require(snapshot.manifest_sha == registry[0]["snapshot"]
                and not set(SEEDS) & {s for t in registry for _, s in t["seeds"]}, "V20_SNAPSHOT_OR_SEED_DIFF")
        require(registry[-2]["targets"] == [list(pair) for pair in CANDIDATES], "V20_FROZEN_HYPOTHESES_DIFF")
        code = Path(main_attribution.__file__)
        require(v1_run.file_sha(code) == registry[-1]["code"][code.name], "V20_CALIBRATOR_DIFF")
        history = [read_json(p / "summary.json") for p in (V16, V17, V18, V19)]
        require(all(h["reserved_outcomes_read"] is False for h in history)
                and not (V19 / "RUNNING.lock").exists(), "V20_PREVIOUS_SCOPE")
        prior = V19 / f"{registry[-1]['seeds'][0][1]}.json"
        refs = {str(f): v1_run.file_sha(f) for f in (
            *(p / "registration.json" for p in past),
            *(p / "summary.json" for p in (PREVIOUS, V16, V17, V18, V19)),
            REFERENCE / "inventory.json", prior
        )}
        inputs = {Path(f): h for f, h in registry[0]["sources"].items()}
        source = paths(root)
        require(set((*source.values(), *cal_files)) <= inputs.keys()
                and all(v1_run.file_sha(f) == h for f, h in inputs.items()), "V20_UNREGISTERED_INPUT")
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snapshot.manifest_sha, archives=refs,
                           seeds=[[f"fresh-{j}", s] for j, s in enumerate(SEEDS, 1)],
                           code={f.name: v1_run.file_sha(f) for f in (Path(__file__), HERE / "PROTOCOL.md",
                                 Path(category_forecasts.__file__), code)}))
        cat = read_json(REFERENCE / "inventory.json")
        diets, _, food = read_diets(root, study)
        y_meta, collected = read_outcomes(source, study)
        endpoints = cat["endpoint_ids"]
        require(list(y_meta) == endpoints and [d[0] for d in diets] == cat["diet_ids"]
                and set(food.values()) <= inputs.keys(), "V20_CATALOG_DIFF")
        map_diets(cat["diet_ids"])
        domain = {e: end_group(e, collected["domains"][e]) for e in endpoints}
        require(domain == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V20_DOMAIN_DIFF")
        values = np.column_stack([y_meta[e] for e in endpoints])
        masks = outcome_groups(values)
        groups, platforms = np.array([domain[e] for e in endpoints]), np.array([collected["domains"][e] for e in endpoints])
        check = out / "reference_replay.json"
        if not check.exists():
            previous = read_json(prior)
            work = Repartition(study, sites, cal, previous["seed"], emit)
            write_json(check, replay(work, diets, values, masks, endpoints, domain, previous))
        record = read_json(check)
        require(all(r["max_error_pp"] < 1e-8 for r in record.values()), "V20_REPLAY_FAILED")
        emit("V20_REPLAY", record)
        parts = []
        for seed in SEEDS:
            f = out / f"{seed}.json"
            if not f.exists():
                work = Repartition(study, sites, cal, seed, emit)
                data = fit_partition(out, seed, work, sites, cal, diets, values, masks, endpoints, groups, platforms)
                write_json(f, data)
            data = read_json(f)
            methods = len(category_forecasts.BASELINES) * len(category_forecasts.METHODS)
            require(data["seed"] == seed and len(data["rows"]) >= methods * 1400, "V20_PARTITION_DIFF")
            parts.append(data)
            emit("V20_PARTITION", dict(seed=seed, endpoints=len(data["rows"]) // methods))
        snapshot.verify_manifest()
        require(set(snapshot.paths) <= snapshot.verified
                and all(v1_run.file_sha(f) == h for f, h in inputs.items())
                and all(v1_run.file_sha(Path(f)) == h for f, h in refs.items()), "V20_INPUT_CHANGED")
        result = inspect(parts, domain, history[0], history[1], CANDIDATES)
        write_json(out / "summary.json", dict(commit=args.commit, reserved_outcomes_read=False, replay=record, **result))
    emit("V20_COMPLETE", dict(endpoints=result["endpoints"], reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
