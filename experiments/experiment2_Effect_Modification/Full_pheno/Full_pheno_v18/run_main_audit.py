"""Diagnose expanded diet-main losses and register new lipid hypotheses."""

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
import json
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v17"))
from run_v17 import PREVIOUS, REFERENCE, V13, V14, V15, V16, context, emit, parse, read_json, register, require, v1_run, write_json
from taxonomy import end_group, map_diets


V17 = HERE.parent / "Full_pheno_v17/runs/TRE_ONLY/compare-1"
SEEDS = (20262409, 20262431)


def replay(work, diets, y, masks, endpoints, prior):
    import protocol as p
    from main_attribution import ARMS, fold_moments, score

    rows = score([fold_moments(work, f, diets, y, masks) for f in p.FOLDS], endpoints, ARMS[:1])
    old = {r["endpoint"]: r for r in prior["rows"] if r["diet"] == "whole_half"}
    require(old.keys() == {r["endpoint"] for r in rows}
            and all((r["people"], r["households"]) == (old[r["endpoint"]]["people"], old[r["endpoint"]]["families"])
                    for r in rows), "V18_REPLAY_MEMBERSHIP_CHANGED")
    diff = max(abs(r["main_pp"] - old[r["endpoint"]]["diet_main_pp"]) for r in rows)
    require(diff < 1e-8, "V18_REFERENCE_MAIN_CHANGED")
    return dict(seed=prior["seed"], count=len(rows), max_difference_pp=diff)


def run_partition(out, seed, work, sites, calibration, diets, y, masks, endpoints, domain, source):
    import numpy as np
    import protocol as p
    from main_attribution import ARMS, fold_moments, inner_scales, score
    from v3_inputs import InnerContexts

    context = InnerContexts(work, sites)
    matrices, policy = [], []
    for fold in p.FOLDS:
        path = out / f"{seed}-fold-{fold}.npz"
        if not path.exists():
            scales, alpha = inner_scales(work, context, calibration, fold, diets, y, masks, domain, source, seed, emit)
            matrix = fold_moments(work, fold, diets, y, masks, scales)
            buf = BytesIO()
            np.savez_compressed(buf, moments=matrix, metadata=json.dumps(dict(seed=seed, fold=fold, scales=alpha)))
            v1_run.atomic_write(path, buf.getvalue())
        with np.load(path, allow_pickle=False) as item:
            desc = json.loads(str(item["metadata"].item()))
            m = item["moments"]
        require(desc["seed"] == seed and desc["fold"] == fold and m.shape == (len(ARMS), len(endpoints), 7)
                and np.isfinite(m).all() and np.all(m[:, :, :5] == m[:1, :, :5]), "V18_CHECKPOINT_MISMATCH")
        matrices.append(m)
        policy.append(desc["scales"])
        emit("V18_OUTER", dict(seed=seed, fold=fold, scored=int((m[0, :, 0] > 0).sum()),
                               calibrated=sum(t["amplitude"] < 1. for t in policy[-1].values())))
    return dict(seed=seed, rows=score(matrices, endpoints, ARMS), amplitudes=policy)


def summarize(parts, domains):
    from main_attribution import ARMS, MARGIN

    ids = []
    for partition in parts:
        index = defaultdict(dict)
        for row in partition["rows"]:
            index[row["arm"]][row["endpoint"]] = row
        require(set(index) == set(ARMS) and all(z.keys() == index["reference"].keys() for z in index.values()),
                "V18_UNPAIRED_SCORE")
        ids.append(index)
    common = set.intersection(*(set(x["reference"]) for x in ids))
    stratified = defaultdict(Counter)
    for e in common:
        rows = [x["reference"][e] for x in ids]
        if all(r["main_pp"] < 0. for r in rows):
            n = sum(r["alignment_pp"] > 0. for r in rows)
            label = ("both_nonpositive", "mixed_alignment", "both_positive_excess_amplitude")[n]
            stratified[domains[e]][label] += 1
    by_method, stable = {}, {}
    for name in ARMS:
        matched = [tuple(a[name][e] for a in ids) for e in sorted(common)]
        stable[name] = {r[0]["endpoint"] for r in matched if all(x["main_pp"] >= MARGIN
                        and x["positive_folds"] >= 3 for x in r)}
        improvement_per_split = []
        for a in ids:
            by_domain = defaultdict(list)
            for e in common:
                by_domain[domains[e]].append(a[name][e]["versus_ref_pp"])
            improvement_per_split.append(mean(median(z) for z in by_domain.values() if len(z) >= 5))
        by_method[name] = dict(
            negative_both=sum(all(x["main_pp"] < 0. for x in r) for r in matched),
            severe_both=sum(all(x["main_pp"] <= -MARGIN for x in r) for r in matched),
            stable_positive=len(stable[name]), balanced_vs_reference_pp=improvement_per_split,
            retain_reference=len(stable[name] & stable["reference"]),
        )
    ref = by_method["reference"]
    for x in by_method.values():
        x["passes_main_research_gate"] = (x["severe_both"] < ref["severe_both"]
                                         and x["retain_reference"] >= .8 * ref["stable_positive"]
                                         and all(d > 0. for d in x["balanced_vs_reference_pp"]))
    all_reasons = Counter()
    for z in stratified.values():
        all_reasons.update(z)
    alpha = defaultdict(list)
    for part in parts:
        for fold in part["amplitudes"]:
            for domain, item in fold.items():
                alpha[domain].append(item["amplitude"])
    return dict(endpoints=len(common), models=by_method, reference_persistent_loss_attribution=dict(all_reasons),
                attribution_by_domain={d: dict(x) for d, x in stratified.items()},
                amplitudes=dict(alpha))


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(1)
    import numpy as np
    from batch_fit import outcome_groups
    from batch_sources import paths, read_diets, read_outcomes
    from interaction_fit import Repartition
    from main_attribution import ARMS
    from sample_audit import CANDIDATES, audit_targets

    out = HERE / "runs/TRE_ONLY" / args.run_id
    with v1_run.run_lock(out):
        root, snapshot, study, calibration, sites, cal_files = context()
        refs = (REFERENCE, PREVIOUS, V13, V14, V15, V16, V17)
        regs = [read_json(h / "registration.json") for h in refs]
        require(snapshot.manifest_sha == regs[0]["snapshot"]
                and not set(SEEDS) & {s for r in regs for _, s in r["seeds"]}, "V18_SEED_OR_REFERENCE_CHANGED")
        require(read_json(V17 / "summary.json")["reserved_outcomes_read"] is False
                and not (V17 / "RUNNING.lock").exists(), "V18_PREVIOUS_SCOPE")
        old_seed = regs[-1]["seeds"][0][1]
        old_part = V17 / f"{old_seed}.json"
        proof = {str(p): v1_run.file_sha(p) for p in (
            *(r / "registration.json" for r in refs), V16 / "summary.json", V17 / "summary.json",
            old_part, PREVIOUS / "summary.json", REFERENCE / "inventory.json"
        )}
        expected = {Path(f): h for f, h in regs[0]["sources"].items()}
        sources = paths(root)
        require(set((*sources.values(), *cal_files)) <= expected.keys()
                and all(v1_run.file_sha(f) == h for f, h in expected.items()), "V18_SOURCE_CHANGED")
        register(out, dict(commit=args.commit, runtime=runtime, snapshots=proof, snapshot=snapshot.manifest_sha,
                           seeds=[[f"fresh-{j}", s] for j, s in enumerate(SEEDS, 1)], models=list(ARMS),
                           targets=[list(x) for x in CANDIDATES],
                           code={p.name: v1_run.file_sha(p) for p in sorted(HERE.iterdir())
                                 if p.is_file() and p.suffix in (".py", ".md")}))
        validation = audit_targets(root, study, sources, V15, V16, V17)
        write_json(out / "independent_sample_audit.json", validation)
        emit("V18_SAMPLE_PREFLIGHT", dict(assays=len(validation["candidates"]),
                                         scope=validation["assay_metadata"],
                                         prior_roles=validation["previous_reservation"],
                                         old_roster=validation["original_roster_comparison"]))
        catalog = read_json(REFERENCE / "inventory.json")
        diets, _, food = read_diets(root, study)
        y, catalog_meta = read_outcomes(sources, study)
        endpoints = catalog["endpoint_ids"]
        require(list(y) == endpoints and [x[0] for x in diets] == catalog["diet_ids"]
                and set(food.values()) <= expected.keys(), "V18_CATALOG_CHANGED")
        map_diets(catalog["diet_ids"])
        domains = {e: end_group(e, catalog_meta["domains"][e]) for e in endpoints}
        require(domains == read_json(PREVIOUS / "summary.json")["endpoint_groups"], "V18_ENDPOINT_DOMAINS_CHANGED")
        values = np.column_stack([y[e] for e in endpoints])
        masks = outcome_groups(values)
        ep_domains = np.array([domains[e] for e in endpoints])
        ep_source = np.array([catalog_meta["domains"][e] for e in endpoints])
        replay_path = out / "reference_replay.json"
        if not replay_path.exists():
            work = Repartition(study, sites, calibration, old_seed, emit)
            write_json(replay_path, replay(work, diets, values, masks, endpoints, read_json(old_part)))
        rep = read_json(replay_path)
        require(rep["seed"] == old_seed and rep["max_difference_pp"] < 1e-8, "V18_REPLAY_INVALID")
        emit("V18_REPLAY", rep)
        parts = []
        for seed in SEEDS:
            path = out / f"{seed}.json"
            if not path.exists():
                work = Repartition(study, sites, calibration, seed, emit)
                result = run_partition(out, seed, work, sites, calibration, diets, values, masks, endpoints,
                                       ep_domains, ep_source)
                write_json(path, result)
            part = read_json(path)
            require(part["seed"] == seed and len(part["rows"]) >= 1400 * len(ARMS), "V18_PARTITION_MISMATCH")
            parts.append(part)
            emit("V18_PARTITION", dict(seed=seed, outcomes=len(part["rows"]) // len(ARMS)))
        snapshot.verify_manifest()
        require(set(snapshot.paths) <= snapshot.verified and all(v1_run.file_sha(p) == h for p, h in expected.items())
                and all(v1_run.file_sha(Path(p)) == h for p, h in proof.items()), "V18_INPUT_CHANGED_DURING_RUN")
        summary = summarize(parts, domains)
        write_json(out / "summary.json", dict(commit=args.commit, replay=rep, reserved_outcomes_read=False,
                                             independent_scoring_started=False, **summary))
    emit("V18_COMPLETE", dict(endpoints=summary["endpoints"], reserved_outcomes_read=False))


if __name__ == "__main__":
    main()
