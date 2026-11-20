"""Test V21 source shrinkage with and without a half-strength gut interaction."""

from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v21"))
from run_support_audit import (PREVIOUS, REFERENCE, V13, V14, V15, V16, V17, V18, V19, V20,
                               context, emit, parse, read_json, register, require, v1_run, write_json)
from taxonomy import end_group, map_diets, summarize


V21 = HERE.parent / "Full_pheno_v21/runs/TRE_ONLY/source-refine-1"
BASE = "domain_calibrated"
REFINED = "source_refined"
REFERENCE_HALF = f"{BASE}:half"
TARGET = f"{REFINED}:half"
MODELS = (BASE, REFINED)
TOLERANCE = 1e-8


def stamp(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as item:
        return json.loads(str(item["metadata"].item()))


def replay_v19(study, sites, cal, diets, values, masks, endpoints, domains, archived, prior_fold):
    import numpy as np
    from interaction_fit import Repartition
    from paired_overlay import collect, score
    from run_recombined import verify

    work = Repartition(study, sites, cal, archived["seed"], emit)
    frames = []
    for f, path in enumerate(prior_fold):
        old = stamp(path)
        require(old["seed"] == archived["seed"] and old["fold"] == f, "V22_V19_FOLD_MISMATCH")
        alpha = np.array([old["selection"][domains[e]]["amplitude"] for e in endpoints])
        frames.append(collect(work, f, diets, values, masks, alpha[None,:])[0])
    actual = score(frames, endpoints, (BASE,))
    previous = {r["endpoint"]:r for r in archived["rows"] if r["diet"] == REFERENCE_HALF}
    metrics = tuple((x,x) for x in ("diet_main_pp","interaction_pp","diet_total_pp"))
    return verify(actual, previous, REFERENCE_HALF, "V19 calibrated half", metrics, "families")


def partition(out, seed, work, sites, cal, diets, values, masks, endpoints, domains, sources, archived, prior_fold):
    import numpy as np
    from paired_overlay import ARMS, collect, score
    from run_recombined import verify
    from source_refinement import inner_terms
    from v3_inputs import InnerContexts

    ctx = InnerContexts(work, sites)
    folders, gaps = [], []
    for f, old_file in enumerate(prior_fold):
        path = out / f"{seed}-fold-{f}.npz"
        old = stamp(old_file)
        require(old["seed"] == seed and old["fold"] == f, "V22_V21_FOLD_MISMATCH")
        if not path.exists():
            scales, policy = inner_terms(work, ctx, cal, f, diets, values, masks, domains, sources, seed, emit)
            require(policy == old["policy"], "V22_V21_REFINEMENT_CHANGED")
            matrix, gap = collect(work, f, diets, values, masks, scales)
            buf = BytesIO()
            np.savez_compressed(buf, moments=matrix,
                                metadata=json.dumps(dict(seed=seed,fold=f,policy=policy,update_gap=gap)))
            v1_run.atomic_write(path, buf.getvalue())
        with np.load(path, allow_pickle=False) as z:
            matrix = z["moments"]
            data = json.loads(str(z["metadata"].item()))
        require(data["seed"] == seed and data["fold"] == f and data["policy"] == old["policy"]
                and matrix.shape == (len(MODELS),len(ARMS),len(endpoints),10)
                and np.isfinite(matrix).all()
                and np.all(matrix[:,:,:,:5] == matrix[:1,:1,:,:5])
                and np.all(matrix[:,:,:,5] == matrix[:,:1,:,5]), "V22_CHECKPOINT_CHANGED")
        folders.append(matrix)
        gaps.append(data["update_gap"])
        emit("V22_OUTER", dict(seed=seed,fold=f,scorable=int((matrix[0,0,:,0] > 0.).sum())))
    rows = score(folders, endpoints, MODELS)
    replay = {}
    for m in MODELS:
        force = {r["endpoint"]:r for r in archived["rows"] if r["arm"] == m}
        replay[m] = verify(rows, force, f"{m}:off", f"V21 {m}", (("diet_main_pp","main_pp"),), "households")
    return dict(seed=seed, rows=rows, v21_main_replay=replay, max_scaled_update_difference=max(gaps))


def aggregate(parts, domain, source):
    from paired_overlay import ARMS
    from run_conditional import stable_sets
    from source_refinement import FOCUS

    names = tuple(f"{b}:{c}" for b in MODELS for c in ARMS)
    usable, stable = stable_sets(parts, names)
    common = usable[REFERENCE_HALF]
    require(all(e == common for e in usable.values()), "V22_ENDPOINT_SUPPORT_CHANGED")
    index = [{(r["diet"],r["endpoint"]):r for r in p["rows"]} for p in parts]
    results = summarize(parts, domain, {m:m for m in names})
    for m, v in results["categories"].items():
        v["severe_total_both"] = sum(all(x[m,e]["diet_total_pp"] <= -.10 for x in index) for e in common)
    measures = ("diet_main_pp", "interaction_pp", "diet_total_pp")
    changes = []
    for p in index:
        groups = defaultdict(list)
        for e in sorted(common):
            a,b = p[TARGET,e],p[REFERENCE_HALF,e]
            require((a["people"],a["families"]) == (b["people"],b["families"]), "V22_REFINED_MEMBERSHIP_DIFF")
            diff = [a[m]-b[m] for m in measures]
            require(domain[e] in FOCUS or max(abs(d) for d in diff) < TOLERANCE, "V22_NONFOCUS_PREDICTION_DIFF")
            groups[domain[e]].append(diff)
        domain_metrics = {d:dict(zip(measures, (median(c) for c in zip(*v, strict=True)), strict=True))
                          for d,v in groups.items() if len(v) >= 5}
        change = {m:mean(d[m] for d in domain_metrics.values()) for m in measures}
        changes.append(dict(domain_balanced=change,by_domain=domain_metrics))
    source_audit = {}
    for label in ("sleep_hrv","ecg","retinal_vascular"):
        names_in_source = [e for e in common if source[e] == label]
        if names_in_source:
            source_audit[label] = dict(
                endpoints=len(names_in_source),
                severe=[sum(all(x[m,e]["diet_total_pp"] <= -.10 for x in index)
                            for e in names_in_source) for m in (REFERENCE_HALF,TARGET)],
                paired_median=[median(x[TARGET,e]["diet_total_pp"] - x[REFERENCE_HALF,e]["diet_total_pp"]
                                      for e in names_in_source) for x in index],
            )
    preserved = len(stable[REFERENCE_HALF] & stable[TARGET])
    base = results["categories"][REFERENCE_HALF]
    target = results["categories"][TARGET]
    checks = dict(less_severe_total=target["severe_total_both"] < base["severe_total_both"],
                  retains_80_percent=bool(stable[REFERENCE_HALF]) and preserved >= .8 * len(stable[REFERENCE_HALF]),
                  better_both=all(x["domain_balanced"]["diet_total_pp"] > 0. for x in changes))
    return dict(**results, endpoints=len(common), retained_interaction=preserved, acceptance=checks,
                passes_all=all(checks.values()), paired_pp=changes, sources=source_audit,
                max_scaled_update_difference=max(a["max_scaled_update_difference"] for a in parts),
                stable_endpoints={m:sorted(v) for m,v in stable.items()})


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    import protocol as p
    from batch_fit import outcome_groups
    from batch_sources import paths, read_outcomes, read_diets
    from interaction_fit import Repartition
    import main_attribution
    import paired_overlay
    import source_refinement
    from sample_audit import CANDIDATES

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, cal, sites, cal_files = context()
        history = (REFERENCE, PREVIOUS, V13, V14, V15, V16, V17, V18, V19, V20, V21)
        regs = [read_json(h / "registration.json") for h in history]
        require(snapshot.manifest_sha == regs[0]["snapshot"]
                and regs[-1]["frozen_interaction_targets"] == [list(x) for x in CANDIDATES], "V22_SCOPE_CHANGED")
        for file in (Path(main_attribution.__file__),Path(source_refinement.__file__)):
            require(v1_run.file_sha(file) == regs[-1]["code"][file.name], "V22_V21_CODE_CHANGED")
        require(read_json(V21 / "summary.json")["reserved_outcomes_read"] is False
                and not any((x / "RUNNING.lock").exists() for x in (V19,V21)), "V22_ARCHIVE_INCOMPLETE")
        seed19 = regs[-3]["seeds"][0][1]
        seeds = [s for _,s in regs[-1]["seeds"]]
        parts19 = [V19 / f"{seed19}-fold-{i}.npz" for i in p.FOLDS]
        parts21 = {s:[V21 / f"{s}-fold-{i}.npz" for i in p.FOLDS] for s in seeds}
        refs = {str(f):v1_run.file_sha(f) for f in (
            *(h / "registration.json" for h in history), V19 / "summary.json", V20 / "summary.json",
            V21 / "summary.json", REFERENCE / "inventory.json", PREVIOUS / "summary.json",
            V19 / f"{seed19}.json", *parts19, *(V21 / f"{s}.json" for s in seeds),
            *(f for v in parts21.values() for f in v),
        )}
        inputs = {Path(f):h for f,h in regs[0]["sources"].items()}
        sources = paths(root)
        require(set((*sources.values(),*cal_files)) <= inputs.keys() and
                all(v1_run.file_sha(f) == h for f,h in inputs.items()), "V22_UNREGISTERED_SOURCE")
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snapshot.manifest_sha, archives=refs,
                           seeds=regs[-1]["seeds"], models=list(MODELS),
                           code={f.name:v1_run.file_sha(f) for f in (
                               Path(__file__),HERE / "PROTOCOL.md",Path(paired_overlay.__file__),
                               Path(source_refinement.__file__),Path(main_attribution.__file__))}))
        inventory = read_json(REFERENCE / "inventory.json")
        diets, _, food = read_diets(root, study)
        y, audit = read_outcomes(sources, study)
        endpoints = inventory["endpoint_ids"]
        require(list(y) == endpoints and [x[0] for x in diets] == inventory["diet_ids"]
                and set(food.values()) <= inputs.keys(), "V22_CATALOG_CHANGED")
        map_diets(inventory["diet_ids"])
        raw = audit["domains"]
        domain = {e:end_group(e,raw[e]) for e in endpoints}
        require(domain == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V22_DOMAINS_CHANGED")
        value = np.column_stack([y[e] for e in endpoints])
        masks = outcome_groups(value)
        n, src = np.array([domain[e] for e in endpoints]), np.array([raw[e] for e in endpoints])
        replay = out / "v19_replay.json"
        if not replay.exists():
            record = replay_v19(study, sites, cal, diets, value, masks, endpoints, domain,
                                read_json(V19 / f"{seed19}.json"), parts19)
            write_json(replay, record)
        verified = read_json(replay)
        require(verified["max_error_pp"] < TOLERANCE, "V22_V19_REPLAY_INVALID")
        emit("V22_V19_REPLAY", verified)
        finished = []
        for seed in seeds:
            saved = out / f"{seed}.json"
            if not saved.exists():
                work = Repartition(study, sites, cal, seed, emit)
                result = partition(out, seed, work, sites, cal, diets, value, masks, endpoints,
                                   n, src, read_json(V21 / f"{seed}.json"), parts21[seed])
                write_json(saved, result)
            result = read_json(saved)
            require(result["seed"] == seed and len(result["rows"]) >= 4 * 1400
                    and all(r["max_error_pp"] < TOLERANCE for r in result["v21_main_replay"].values()),
                    "V22_V21_REPLAY_OR_PARTITION_INVALID")
            finished.append(result)
            emit("V22_PARTITION", dict(seed=seed,endpoints=len(result["rows"])//4,
                                      v21_replay=result["v21_main_replay"]))
        snapshot.verify_manifest()
        require(set(snapshot.paths) <= snapshot.verified
                and all(v1_run.file_sha(f) == h for f,h in inputs.items())
                and all(v1_run.file_sha(Path(f)) == h for f,h in refs.items()), "V22_INPUT_CHANGED_DURING_RUN")
        report = aggregate(finished, domain, raw)
        write_json(out / "summary.json", dict(commit=args.commit, v19_replay=verified,
                   v21_main_replay=[h["v21_main_replay"] for h in finished], reserved_outcomes_read=False, **report))
    emit("V22_COMPLETE", dict(endpoints=report["endpoints"],passed=report["passes_all"],reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
