"""Repeat the registered V22 procedure and document the V18 assay probes."""

from io import BytesIO
from pathlib import Path
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v22"))
from run_overlay import (PREVIOUS, REFERENCE, V13, V14, V15, V16, V17, V18, V19, V20, V21, MODELS,
                         TARGET, REFERENCE_HALF, aggregate, context, emit, parse, read_json, register,
                         require, v1_run, write_json)
from taxonomy import end_group, map_diets


V22 = HERE.parent / "Full_pheno_v22/runs/TRE_ONLY/source-overlay-1"
SEEDS = (20262903, 20262927)
SOURCES = ("sleep:sleep_hrv", "cardiovascular:ecg", "cardiovascular:retinal_vascular")


def keep_registered_sources(weights, policy, domains, sources):
    import numpy as np

    group = np.char.add(np.char.add(domains, ":"), sources)
    scope = np.isin(group, SOURCES)
    weights[1,~scope] = weights[0,~scope]
    return {k:v for k,v in policy.items() if k in SOURCES}


def score_probes(folds, candidates):
    import category_forecasts
    from run_expanded_grid import holm

    rows = category_forecasts.score(folds, [e for _,e in candidates])
    selected = []
    by_key = {(r["diet"],r["endpoint"]):r for r in rows}
    for base in category_forecasts.BASELINES:
        shortlist = []
        for g,e in candidates:
            r = by_key.get((f"{base}:{g}",e))
            if r is None:
                shortlist.append(dict(diet=g,endpoint=e,baseline=base,scorable=False,p_one_sided=1.))
            else:
                shortlist.append(dict(r,diet=g,baseline=base,scorable=True))
        holm(shortlist)
        selected.extend(shortlist)
    return selected


def fit_partition(out, seed, work, sites, cal, diets, y, masks, endpoints, domains, sources, candidates):
    import numpy as np
    import protocol as p
    from batch_fit import outcome_groups
    import category_forecasts
    import paired_overlay
    from source_refinement import inner_terms
    from v3_inputs import InnerContexts

    at = {e:i for i,e in enumerate(endpoints)}
    columns = [at[e] for _,e in candidates]
    narrow = y[:,columns]
    small_masks = outcome_groups(narrow)
    ctx = InnerContexts(work, sites)
    cfg, probes, gaps = [], [], []
    for f in p.FOLDS:
        fpath = out / f"{seed}-fold-{f}.npz"
        if not fpath.exists():
            weights, policy = inner_terms(work, ctx, cal, f, diets, y, masks, domains, sources, seed, emit)
            policy = keep_registered_sources(weights, policy, domains, sources)
            require(np.array_equal(weights[0,columns], weights[1,columns]), "V23_LIPID_BASELINE_REFINEMENT_CHANGED")
            data, gap = paired_overlay.collect(work, f, diets, y, masks, weights)
            small = category_forecasts.collect(work, f, diets, narrow, small_masks, weights[1,columns])
            buf = BytesIO()
            np.savez_compressed(buf, moments=data, probes=small,
                                metadata=json.dumps(dict(seed=seed,fold=f,policy=policy,update_gap=gap)))
            v1_run.atomic_write(fpath, buf.getvalue())
        with np.load(fpath, allow_pickle=False) as saved:
            data, small = saved["moments"], saved["probes"]
            m = json.loads(str(saved["metadata"].item()))
        require(m["seed"] == seed and m["fold"] == f
                and data.shape == (len(MODELS),len(paired_overlay.ARMS),len(endpoints),10)
                and small.shape == (len(category_forecasts.BASELINES),len(category_forecasts.METHODS),len(candidates),12)
                and np.isfinite(data).all() and np.isfinite(small).all(), "V23_CHECKPOINT_CHANGED")
        require(np.all(data[:,:,:,:5] == data[:1,:1,:,:5]) and np.all(data[:,:,:,5] == data[:,:1,:,5]),
                "V23_FULL_MATRIX_SUPPORT_CHANGED")
        cfg.append(data)
        probes.append(small)
        gaps.append(m["update_gap"])
        emit("V23_OUTER", dict(seed=seed,fold=f,scorable=int((data[0,0,:,0] > 0.).sum()),
                               activated=sum(t["multiplier"] < 1. for t in m["policy"].values())))
    return dict(seed=seed, rows=paired_overlay.score(cfg,endpoints,MODELS), probes=score_probes(probes,candidates),
                max_scaled_update_difference=max(gaps))


def summarize_probes(parts, candidates):
    lookup = [{(r["baseline"],r["diet"],r["endpoint"]):r for r in x["probes"]} for x in parts]
    rows = []
    for g,e in candidates:
        report = dict(diet=g, endpoint=e, baseline={})
        for b in ("reference","domain_calibrated"):
            variants = [x[b,g,e] for x in lookup]
            stable = all(r["scorable"] and r["interaction_pp"] >= .10 and r["diet_total_pp"] > 0.
                         and r["positive_folds"] >= 3 for r in variants)
            report["baseline"][b] = dict(practical_both=stable, parts=variants)
        rows.append(report)
    return rows


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    import category_forecasts
    from interaction_fit import Repartition
    import main_attribution
    import paired_overlay
    from sample_audit import CANDIDATES
    import source_refinement

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, cal, sites, cal_files = context()
        history = (REFERENCE, PREVIOUS, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22)
        regs = [read_json(h / "registration.json") for h in history]
        require(snapshot.manifest_sha == regs[0]["snapshot"]
                and not set(SEEDS) & {s for x in regs for _,s in x["seeds"]}, "V23_SEEDS_OR_SNAPSHOT_CHANGED")
        prior = read_json(V22 / "summary.json")
        require(prior["passes_all"] and prior["reserved_outcomes_read"] is False
                and not (V22 / "RUNNING.lock").exists() and regs[7]["targets"] == [list(x) for x in CANDIDATES],
                "V23_FROZEN_PREVIOUS_SCOPE")
        methods = [Path(m.__file__) for m in (source_refinement, paired_overlay, main_attribution, category_forecasts)]
        identified = {f.name:v1_run.file_sha(f) for f in methods}
        require(all(regs[-1]["code"][f.name] == identified[f.name] for f in methods[:3])
                and regs[9]["code"][methods[-1].name] == identified[methods[-1].name], "V23_FROZEN_CODE_CHANGED")
        census = V18 / "candidate_visit_key_overlap.json"
        preflight = V18 / "independent_sample_audit.json"
        frozen_protocol = HERE.parent / "Full_pheno_v18/PROTOCOL.md"
        require(v1_run.file_sha(frozen_protocol) == regs[7]["code"][frozen_protocol.name], "V23_V18_PROTOCOL_CHANGED")
        require(read_json(census)["outcome_values_accessed"] is False
                and read_json(preflight)["outcome_values_accessed"] is False, "V23_INDEPENDENCE_PREFLIGHT_SCOPE")
        proofs = {str(f):v1_run.file_sha(f) for f in (
            *(x / "registration.json" for x in history), V22 / "summary.json", PREVIOUS / "summary.json",
            REFERENCE / "inventory.json", census, preflight, frozen_protocol,
        )}
        inputs = {Path(f):v for f,v in regs[0]["sources"].items()}
        sources = paths(root)
        require(set((*sources.values(), *cal_files)) <= inputs.keys()
                and all(v1_run.file_sha(f) == h for f,h in inputs.items()), "V23_UNREGISTERED_INPUT")
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snapshot.manifest_sha,
                           prior_records=proofs, procedure=identified,
                           version22_commit=prior["commit"],
                           frozen_targets=[list(x) for x in CANDIDATES],
                           sources_eligible_for_refinement=list(SOURCES),
                           seeds=[[f"fresh-{j}",s] for j,s in enumerate(SEEDS,1)],
                           local_code={f.name:v1_run.file_sha(f) for f in (HERE / "PROTOCOL.md",Path(__file__))}))
        inv = read_json(REFERENCE / "inventory.json")
        diets, _, food = read_diets(root, study)
        values, meta = read_outcomes(sources, study)
        endpoints = inv["endpoint_ids"]
        require(list(values) == endpoints and [d[0] for d in diets] == inv["diet_ids"]
                and set(food.values()) <= inputs.keys(), "V23_DIET_OR_OUTCOME_CATALOG_CHANGED")
        map_diets(inv["diet_ids"])
        raw = meta["domains"]
        domain = {e:end_group(e,raw[e]) for e in endpoints}
        require(domain == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V23_DOMAINS_CHANGED")
        y = np.column_stack([values[e] for e in endpoints])
        masks = outcome_groups(y)
        labels, platform = np.array([domain[e] for e in endpoints]), np.array([raw[e] for e in endpoints])
        parts = []
        for seed in SEEDS:
            dest = out / f"{seed}.json"
            if not dest.exists():
                work = Repartition(study, sites, cal, seed, emit)
                part = fit_partition(out, seed, work, sites, cal, diets, y, masks, endpoints, labels, platform, CANDIDATES)
                write_json(dest, part)
            part = read_json(dest)
            require(len(part["rows"]) >= 4 * 1400 and part["seed"] == seed
                    and len(part["probes"]) == 2 * len(CANDIDATES), "V23_PARTITION_INCOMPLETE")
            parts.append(part)
            emit("V23_PARTITION", dict(seed=seed,outcomes=len(part["rows"])//4))
        snapshot.verify_manifest()
        require(set(snapshot.paths) <= snapshot.verified
                and all(v1_run.file_sha(f) == h for f,h in inputs.items())
                and all(v1_run.file_sha(Path(f)) == h for f,h in proofs.items()), "V23_INPUT_CHANGED_DURING_RUN")
        summary = aggregate(parts,domain,raw)
        carry = {m: len(set(summary["stable_endpoints"][m]) & set(prior["stable_endpoints"][m]))
                 for m in (REFERENCE_HALF,TARGET)}
        probes = summarize_probes(parts,CANDIDATES)
        write_json(out / "summary.json", dict(commit=args.commit, reserved_outcomes_read=False,
                   independent_scoring_started=False, v22_stable_retention=carry, frozen_probes=probes, **summary))
    emit("V23_COMPLETE", dict(endpoints=summary["endpoints"], passed=summary["passes_all"],
                              registered_targets=len(CANDIDATES),reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
