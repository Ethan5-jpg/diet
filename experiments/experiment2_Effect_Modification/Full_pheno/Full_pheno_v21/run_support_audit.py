"""Audit residual diet-main losses and evaluate focused source shrinkage."""

from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v20"))
from run_categories import (PREVIOUS, REFERENCE, V13, V14, V15, V16, V17, V18, V19,
                            context, emit, parse, read_json, register, require, v1_run, write_json)
from taxonomy import end_group, map_diets


V20 = HERE.parent / "Full_pheno_v20/runs/TRE_ONLY/calibrated-categories-1"
SEEDS = (20262707, 20262743)
MARGIN = .10


def replay(work, diets, values, masks, endpoints, domains, archived, checkpoints):
    import numpy as np
    import protocol as p
    from main_attribution import score
    from source_refinement import BASE, collect

    frames = []
    for f, path in zip(p.FOLDS, checkpoints, strict=True):
        with np.load(path, allow_pickle=False) as z:
            stamp = json.loads(str(z["metadata"].item()))
        require(stamp["seed"] == archived["seed"] and stamp["fold"] == f, "V21_BAD_REPLAY_FOLD")
        alpha = np.array([stamp["policy"][domains[e]]["amplitude"] for e in endpoints])
        frames.append(collect(work, f, diets, values, masks, alpha[None, :])[0])
    rows = score(frames, endpoints, (BASE,))
    old = {r["endpoint"]: r for r in archived["rows"] if r["diet"] == "domain_calibrated:interaction_off"}
    require(old.keys() == {r["endpoint"] for r in rows}
            and all((r["people"],r["households"]) == (old[r["endpoint"]]["people"],old[r["endpoint"]]["families"])
                    for r in rows), "V21_REPLAY_ROSTER_DIFF")
    gap = max(abs(r["main_pp"]-old[r["endpoint"]]["diet_main_pp"]) for r in rows)
    require(gap < 1e-8, "V21_REPLAY_SCORE_DIFF")
    return dict(endpoints=len(rows), largest_difference_pp=gap, seed=archived["seed"])


def partition(out, seed, work, sites, cal, diets, values, masks, endpoints, domains, platforms):
    import numpy as np
    import protocol as p
    from main_attribution import score
    from source_refinement import METHODS, collect, inner_terms
    from v3_inputs import InnerContexts

    ctx = InnerContexts(work, sites)
    folds, support, policies = [], [], []
    for fold in p.FOLDS:
        path = out / f"{seed}-fold-{fold}.npz"
        if not path.exists():
            scales, policy = inner_terms(work, ctx, cal, fold, diets, values, masks, domains, platforms, seed, emit)
            m, t = collect(work, fold, diets, values, masks, scales)
            buf = BytesIO()
            np.savez_compressed(buf, main=m, support=t, metadata=json.dumps(dict(seed=seed, fold=fold, policy=policy)))
            v1_run.atomic_write(path, buf.getvalue())
        with np.load(path, allow_pickle=False) as checkpoint:
            m, t = checkpoint["main"], checkpoint["support"]
            trace = json.loads(str(checkpoint["metadata"].item()))
        require(trace["seed"] == seed and trace["fold"] == fold and m.shape == (len(METHODS),len(endpoints),7)
                and t.shape == (len(endpoints),3) and np.isfinite(m).all() and np.isfinite(t).all()
                and np.all(m[:,:,:5] == m[:1,:,:5]), "V21_FOLD_CHECKPOINT_CHANGED")
        folds.append(m)
        support.append(t)
        policies.append(trace["policy"])
        emit("V21_OUTER", dict(seed=seed, fold=fold, supported=int((m[0,:,0] > 0.).sum()),
                               sources_adjusted=sum(k["multiplier"] < 1. for k in trace["policy"].values())))
    kept = np.stack(support).sum(axis=0)
    ix = {e:j for j,e in enumerate(endpoints)}
    rows = score(folds, endpoints, METHODS)
    for r in rows:
        eligible, days, short = kept[ix[r["endpoint"]]]
        people = r["people"]
        require(0 < people <= eligible, "V21_INVALID_AVAILABILITY_DENOMINATOR")
        r.update(availability=people/eligible, mean_diary_days=days/people, short_diary_share=short/people)
    return dict(seed=seed, rows=rows, policies=policies)


def synthesize(parts, domain):
    from source_refinement import BASE, REFINED, FOCUS

    lookups = [{(r["arm"],r["endpoint"]):r for r in p["rows"]} for p in parts]
    keys = [{e for m,e in p if m == BASE} for p in lookups]
    require(all({e for m,e in p if m == REFINED} == a for p,a in zip(lookups, keys, strict=True)),
            "V21_METHOD_SUPPORT_DIFF")
    common = set.intersection(*keys)
    chosen = {}
    stable = {}
    counters = {}
    for arm in (BASE, REFINED):
        pairs = {e: [p[arm,e] for p in lookups] for e in sorted(common)}
        negative = {e for e,z in pairs.items() if all(r["main_pp"] < 0. for r in z)}
        severe = {e for e,z in pairs.items() if all(r["main_pp"] <= -MARGIN for r in z)}
        stable[arm] = {e for e,z in pairs.items() if all(r["main_pp"] >= MARGIN and r["positive_folds"] >= 3 for r in z)}
        chosen[arm] = negative
        counters[arm] = dict(negative=len(negative), severe=len(severe), stable_positive=len(stable[arm]),
                             retain_reference=len(stable[arm] & stable[BASE]),
                             focus={d: dict(negative=sum(domain[e] == d for e in negative),
                                            severe=sum(domain[e] == d for e in severe),
                                            stable_positive=sum(domain[e] == d for e in stable[arm])) for d in FOCUS})
    effects = []
    for p in lookups:
        by_domain = defaultdict(list)
        for e in sorted(common):
            r, b = p[REFINED,e], p[BASE,e]
            require((r["people"],r["households"]) == (b["people"],b["households"])
                    and (domain[e] in FOCUS or r["main_pp"] == b["main_pp"]), "V21_NONTARGETED_CHANGE")
            by_domain[domain[e]].append(r["versus_ref_pp"])
        group_med = {k: median(v) for k,v in by_domain.items() if len(v) >= 5}
        effects.append(dict(domain_balanced=mean(group_med.values()), targets={d: group_med[d] for d in FOCUS}))
    coverage = {}
    for d in FOCUS:
        e0 = {e for e in common if domain[e] == d}
        block = dict(persistent=e0 & chosen[BASE], other=e0 - chosen[BASE])
        cols = ("people", "households", "availability", "mean_diary_days", "short_diary_share")
        coverage[d] = {}
        for label, members in block.items():
            a = dict(endpoints=len(members))
            if len(members) >= 5:
                a.update({k: median(mean(p[BASE,e][k] for p in lookups) for e in members) for k in cols})
            coverage[d][label] = a
    checks = dict(less_severe=counters[REFINED]["severe"] < counters[BASE]["severe"],
                  retains_80_percent=bool(stable[BASE]) and counters[REFINED]["retain_reference"] >= .8 * len(stable[BASE]),
                  better_both=all(x["domain_balanced"] > 0. for x in effects))
    source_activity = defaultdict(list)
    for part in parts:
        for fold in part["policies"]:
            for src, r in fold.items():
                source_activity[src].append(r["multiplier"])
    return dict(endpoints=len(common), metrics=counters, paired_pp=effects, acceptance=checks,
                passes_all=all(checks.values()), coverage=coverage, source_activity=dict(source_activity))


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    import protocol as p
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    from interaction_fit import Repartition
    import main_attribution
    from sample_audit import CANDIDATES
    from source_refinement import FOCUS, METHODS

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, cal, sites, cal_files = context()
        historical = (REFERENCE, PREVIOUS, V13, V14, V15, V16, V17, V18, V19, V20)
        registered = [read_json(t / "registration.json") for t in historical]
        require(snapshot.manifest_sha == registered[0]["snapshot"]
                and not set(SEEDS) & {s for a in registered for _,s in a["seeds"]}, "V21_SNAPSHOT_OR_SEEDS_CHANGED")
        require(registered[-3]["targets"] == [list(x) for x in CANDIDATES], "V21_FROZEN_TARGETS_CHANGED")
        calibrator = Path(main_attribution.__file__)
        require(v1_run.file_sha(calibrator) == registered[-1]["code"][calibrator.name], "V21_PARENT_CALIBRATOR_CHANGED")
        require(read_json(V20 / "summary.json")["reserved_outcomes_read"] is False
                and not (V20 / "RUNNING.lock").exists(), "V21_INVALID_ARCHIVE_SCOPE")
        old_seed = registered[-1]["seeds"][0][1]
        prior = V20 / f"{old_seed}.json"
        checkpoints = [V20 / f"{old_seed}-fold-{f}.npz" for f in p.FOLDS]
        proof = {str(f): v1_run.file_sha(f) for f in (
            *(t / "registration.json" for t in historical), V20 / "summary.json", V18 / "summary.json",
            REFERENCE / "inventory.json", PREVIOUS / "summary.json", prior, *checkpoints)}
        inputs = {Path(f):d for f,d in registered[0]["sources"].items()}
        sources = paths(root)
        require(set((*sources.values(), *cal_files)) <= inputs.keys()
                and all(v1_run.file_sha(f) == h for f,h in inputs.items()), "V21_INPUT_CHANGED")
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snapshot.manifest_sha, archives=proof,
                           seeds=[[f"fresh-{j}",s] for j,s in enumerate(SEEDS,1)], methods=list(METHODS),
                           target_domains=list(FOCUS), frozen_interaction_targets=[list(x) for x in CANDIDATES],
                           code={f.name:v1_run.file_sha(f) for f in (Path(__file__), HERE / "source_refinement.py",
                                 HERE / "PROTOCOL.md", calibrator)}))
        inventory = read_json(REFERENCE / "inventory.json")
        diets, _, food = read_diets(root, study)
        ys, notes = read_outcomes(sources, study)
        endpoints = inventory["endpoint_ids"]
        require(list(ys) == endpoints and [d[0] for d in diets] == inventory["diet_ids"]
                and set(food.values()) <= inputs.keys(), "V21_CATALOG_CHANGED")
        map_diets(inventory["diet_ids"])
        domain = {e:end_group(e,notes["domains"][e]) for e in endpoints}
        require(domain == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V21_DOMAINS_CHANGED")
        values = np.column_stack([ys[e] for e in endpoints])
        masks = outcome_groups(values)
        d, src = np.array([domain[e] for e in endpoints]), np.array([notes["domains"][e] for e in endpoints])
        replay_path = out / "baseline_replay.json"
        if not replay_path.exists():
            work = Repartition(study, sites, cal, old_seed, emit)
            write_json(replay_path, replay(work, diets, values, masks, endpoints, domain, read_json(prior), checkpoints))
        archived = read_json(replay_path)
        require(archived["seed"] == old_seed and archived["largest_difference_pp"] < 1e-8, "V21_REPLAY_INVALID")
        emit("V21_REPLAY", archived)
        parts = []
        for seed in SEEDS:
            f = out / f"{seed}.json"
            if not f.exists():
                work = Repartition(study, sites, cal, seed, emit)
                result = partition(out, seed, work, sites, cal, diets, values, masks, endpoints, d, src)
                write_json(f, result)
            result = read_json(f)
            require(result["seed"] == seed and len(result["rows"]) >= len(METHODS)*1400, "V21_PARTITION_CHANGED")
            parts.append(result)
            emit("V21_PARTITION", dict(seed=seed, outcomes=len(result["rows"])//len(METHODS)))
        snapshot.verify_manifest()
        require(set(snapshot.paths) <= snapshot.verified
                and all(v1_run.file_sha(f) == h for f,h in inputs.items())
                and all(v1_run.file_sha(Path(f)) == h for f,h in proof.items()), "V21_RUNNING_INPUT_CHANGED")
        summary = synthesize(parts, domain)
        write_json(out / "summary.json", dict(commit=args.commit, replay=archived, reserved_outcomes_read=False, **summary))
    emit("V21_COMPLETE", dict(endpoints=summary["endpoints"], passed=summary["passes_all"], reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
