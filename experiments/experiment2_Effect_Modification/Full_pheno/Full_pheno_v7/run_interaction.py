"""Test fixed and bounded microbiome modifications on the registered cohorts."""

import argparse
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v6_exposure"))
from run_exposure import ANCHOR, archive, emit, v1_run
from run_v2 import directory_id, require


def infer(moments, draws):
    import numpy as np
    from diet_audit import scores
    import protocol as p

    def contrasts(values):
        _, change = scores(values)
        paired = change[..., 1] - change[..., :1, 1]
        return np.concatenate((change, paired[..., None]), axis=-1)

    r2, _ = scores(moments.sum(axis=1))
    points = contrasts(moments.sum(axis=1))
    use = np.any(moments[:, :, 0] > 0, axis=0)
    packed = np.concatenate(moments, axis=1)[use]
    rng = np.random.default_rng(p.BOOTSTRAP_SEED)
    count = len(packed)
    errors = np.empty((draws, *points.shape))
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        weights = rng.multinomial(count, np.full(count, 1./count), size=size)
        trial = (weights @ packed).reshape(size, len(moments), -1)
        errors[start:start+size] = contrasts(trial) - points
    scale = errors.std(axis=0, ddof=1)
    active = scale > 1e-12
    z = np.abs(errors[:, active] / scale[active]).max(axis=1)
    critical = np.quantile(z, .95, method="higher")
    return r2, points, np.where(active, critical*scale, np.abs(errors).max(axis=0))


LABELS = ("diet_main_pp", "interaction_pp", "diet_total_pp", "vs_ref_total_pp", "vs_ref_interaction_pp")


def score(study, fits, all_methods, selected, draws):
    from dataclasses import replace
    import numpy as np
    from diet_audit import household_moments

    _, groups = np.unique(study.family, return_inverse=True)
    indexes = [all_methods.index(x) for x in selected]
    moments = []
    for parts in fits.values():
        cols = [replace(fit, off=fit.off[:, indexes], on=fit.on[:, indexes]) for fit in parts]
        moments.append(household_moments(cols, groups, int(groups.max())+1))
    moments = np.stack(moments)
    r2, points, radius = infer(moments, draws)
    rows = []
    for i, ((exposure, endpoint), parts) in enumerate(fits.items()):
        common = dict(exposure=exposure, endpoint=endpoint, n=int(moments[i, :, 0].sum()),
                      families=int((moments[i, :, 0] > 0).sum()))
        for j, name in enumerate(selected):
            a, b, c = map(float, r2[i, j])
            row = dict(common, method=name, r2_no_diet=a, r2_off=b, r2_on=c)
            for label, value, width in zip(LABELS, points[i, j], radius[i, j], strict=True):
                row[label] = float(value)
                row[label+"_95"] = [float(value-width), float(value+width)]
            k = indexes[j]
            row["positive_folds"] = sum(int(np.sum((f.outcome-f.off[:, k])**2 - (f.outcome-f.on[:, k])**2)>0.)
                                        for f in parts)
            rows.append(row)
    return rows


def coverage_summary(rows, selected):
    summary = []
    for method in selected:
        rest = [r for r in rows if r["method"]==method and (r["exposure"], r["endpoint"]) != ANCHOR]
        gains = [r["interaction_pp"] for r in rest]
        positive, reached, harmed = (sum(x>0. for x in gains), sum(x>=.1 for x in gains), sum(x<=-.1 for x in gains))
        summary.append(dict(method=method, positive=positive, reached=reached, harmed=harmed,
                            passes=positive>=9 and reached>=2 and harmed==0,
                            median_interaction_pp=median(gains),
                            total_negative=sum(r["diet_total_pp"]<0. for r in rest),
                            adjusted_interaction_positive=sum(r["interaction_pp_95"][0]>0. for r in rest),
                            adjusted_interaction_beats_reference=sum(r["vs_ref_interaction_pp_95"][0]>0. for r in rest),
                            adjusted_total_beats_reference=sum(r["vs_ref_total_pp_95"][0]>0. for r in rest)))
    return summary


def ranked(summary):
    return sorted(summary, key=lambda r: (r["passes"], r["harmed"]==0, r["reached"], r["positive"],
                                          r["median_interaction_pp"], -r["total_negative"]), reverse=True)


def report(result):
    for c in ranked(result["coverage"]):
        emit("COVERAGE", c)
    closest = ranked(result["coverage"])[0]["method"]
    for r in result["rows"]:
        if r["method"]==closest and (r["interaction_pp"]>=.1 or (r["exposure"], r["endpoint"])==ANCHOR):
            keys = ("method", "exposure", "endpoint", "n", "interaction_pp", "interaction_pp_95",
                    "diet_total_pp", "vs_ref_total_pp", "vs_ref_total_pp_95", "positive_folds")
            emit("LEADING_PAIR", {k:r[k] for k in keys})
    emit("COMPLETE", dict(rows=len(result["rows"]), seed=result["partition_seed"],
                         frozen_methods=result["frozen_methods"], replay_max=result["replay_max"],
                         reserved_outcomes_read=False))


def verify_reference(previous, rows):
    old = {(r["exposure"], r["endpoint"]):r for r in previous["rows"] if r["workflow"]=="v4_pc5_half"}
    first = [r for r in rows if r["method"]=="pc5_half"]
    errors = []
    for r in first:
        ref = old[r["exposure"], r["endpoint"]]
        require(r["n"]==ref["people"] and r["families"]==ref["households"], "V7_REPLAY_COHORT_MISMATCH")
        errors.extend(abs(r[k]-ref[k]) for k in ("r2_no_diet", "r2_off", "r2_on"))
    require(len(first)==len(old)==18 and max(errors)<=2e-8, "V7_REPLAY_DIFFERENCE")
    return max(errors)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id", type=directory_id, required=True)
    cli.add_argument("--commit")
    cli.add_argument("--partition-seed", type=int, choices=(20261137, 20261183))
    cli.add_argument("--frozen-from", type=directory_id)
    cli.add_argument("--report", action="store_true")
    args = cli.parse_args()
    out = HERE / "runs/TRE_ONLY" / args.run_id
    path = out / "summary.json"
    if args.report:
        report(v1_run.read_json(path))
        return
    require(args.commit and len(args.commit)==40 and all(c in "0123456789abcdef" for c in args.commit), "V7_COMMIT")
    require(bool(args.partition_seed)==bool(args.frozen_from), "V7_SPECIFY_FROZEN_SOURCE_FOR_REPARTITION")
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from interaction_fit import fit_all
    from interaction_rules import METHODS
    from source import Snapshot, Study, required_files
    from upf_refit import load

    root = HERE.parents[2]
    require(root.name==p.HANDOFF, "V7_INVALID_HANDOFF_ROOT")
    snapshot = Snapshot(root, required_files(p.MODES["all"]))
    release, _ = archive.verify_release()
    v5 = HERE.parent / "Full_pheno_v5/runs/TRE_ONLY/discovery-1"
    prior = archive.read_complete(v5, release)
    reg = v1_run.read_json(v5 / "registration.json")
    require(reg["snapshot_sha"]==snapshot.manifest_sha and reg["sources"]==snapshot.report(), "V7_SNAPSHOT_MISMATCH")
    selected, frozen_hash = METHODS, None
    if args.frozen_from:
        source = HERE / "runs/TRE_ONLY" / args.frozen_from / "summary.json"
        frozen = v1_run.read_json(source)
        require(frozen["partition_seed"] is None and frozen["snapshot"]==snapshot.manifest_sha
                and frozen["commit"]==args.commit, "V7_INVALID_FROZEN_SOURCE")
        selected = ("pc5_half", *frozen["frozen_methods"])
        frozen_hash = v1_run.file_sha(source)
    with v1_run.run_lock(out):
        study = Study(snapshot, with_upf=True)
        calibration, paths = load(root, study)
        source_hashes = {str(path):v1_run.file_sha(path) for path in paths}
        dependencies = (HERE.parent / "Full_pheno_v6/diet_audit.py",
                        *(HERE.parent / "Full_pheno_v6_exposure" / x for x in ("exposure_models.py", "upf_refit.py")))
        code = {x.name:v1_run.file_sha(x) for x in (*sorted(HERE.glob("*.py")), *dependencies)}
        identity = dict(commit=args.commit, snapshot=snapshot.manifest_sha, source=source_hashes, code=code,
                        runtime=runtime, methods=list(selected), partition_seed=args.partition_seed, frozen_sha256=frozen_hash)
        meta = out / "registration.json"
        if meta.exists():
            require(v1_run.read_json(meta)==identity, "V7_REGISTRATION_DIFF")
        else:
            v1_run.atomic_write(meta, v1_run.encode(identity))
        if path.exists():
            report(v1_run.read_json(path))
            return
        emit("START", dict(scope="discovery only", run_id=args.run_id, seed=args.partition_seed, methods=len(selected)))
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
        fits, decisions = fit_all(study, outcomes, calibration, args.partition_seed, emit)
        rows = score(study, fits, METHODS, selected, p.BOOTSTRAP_DRAWS)
        coverage = coverage_summary(rows, selected)
        replay_max = verify_reference(prior, rows) if args.partition_seed is None else None
        frozen_methods = [r["method"] for r in ranked(coverage) if r["method"]!="pc5_half"][:2]
        require(set(snapshot.paths)<=snapshot.verified, "V7_UNVERIFIED_INPUTS")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(p))==h for p,h in source_hashes.items()), "V7_UPF_INPUT_CHANGED")
        result = dict(rows=rows, coverage=coverage, upgrades=decisions, frozen_methods=frozen_methods,
                      snapshot=snapshot.manifest_sha, commit=args.commit, partition_seed=args.partition_seed,
                      reserved_outcomes_read=False, replay_max=replay_max, bootstrap_draws=p.BOOTSTRAP_DRAWS)
        v1_run.atomic_write(path, v1_run.encode(result))
    report(result)


if __name__=="__main__":
    main()
