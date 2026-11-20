"""Check small fixed strength changes and repartition frozen recording-precision recipes."""

import argparse
from pathlib import Path

import run_precision as previous
from run_interaction import ANCHOR, coverage_summary, emit, ranked, report, score, v1_run
from run_v2 import directory_id, require


STRENGTHS = (1., 1.15, 1.30, 1.45)
REFERENCE = "pc5_half"


def methods(directions):
    return {REFERENCE:(REFERENCE,1.), **{f"{name}_x{strength:.2f}":(name,strength)
                                        for name in directions for strength in STRENGTHS}}


def rescale(fits, settings):
    import numpy as np
    from diet_audit import Fit

    result = {}
    indexes = [(previous.METHODS.index(name), strength) for name, strength in settings.values()]
    for key, parts in fits.items():
        result[key] = []
        for part in parts:
            off = np.column_stack([part.off[:, j] for j, _ in indexes])
            on = np.column_stack([part.off[:, j] + (1. if key==ANCHOR else strength)*(part.on[:, j]-part.off[:, j])
                                  for j, strength in indexes])
            result[key].append(Fit(part.rows, part.outcome, part.no_diet, off, on))
    return result


def amplitude_ceiling(fits, direction):
    """Use held outcomes only to diagnose whether some uniform scale can cross the point gate."""
    import math
    import numpy as np

    j = previous.METHODS.index(direction)
    points = {0.}
    curves = []
    for key, parts in fits.items():
        if key==ANCHOR:
            continue
        y = np.concatenate([f.outcome for f in parts])
        err = y - np.concatenate([f.off[:, j] for f in parts])
        update = np.concatenate([f.on[:, j]-f.off[:, j] for f in parts])
        sst = float(np.sum((y-y.mean())**2))
        a,b = float(200.*(err@update)/sst), float(100.*(update@update)/sst)
        curves.append((a,b))
        if b==0.:
            continue
        for t in (-.1,0.,.1):
            discriminant = a*a-4.*b*t
            if discriminant>=0.:
                s = math.sqrt(discriminant)
                points.update(x/(2.*b) for x in (a-s,a+s) if x>=0.)
    ends = sorted(points)
    probes = ends + [(x+y)/2 for x,y in zip(ends,ends[1:])] + [ends[-1]+1.]
    feasible = []
    for w in probes:
        gains = [w*(a-b*w) for a,b in curves]
        if sum(g>1e-11 for g in gains)>=9 and all(g>-.1+1e-11 for g in gains):
            feasible.append(sum(g>=.1-1e-11 for g in gains))
    return max(feasible, default=0)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id", type=directory_id, required=True)
    cli.add_argument("--commit")
    cli.add_argument("--partition-seed", type=int, choices=(20261137,20261183))
    cli.add_argument("--frozen-from", type=directory_id)
    cli.add_argument("--report", action="store_true")
    args = cli.parse_args()
    home = previous.HERE / "runs/TRE_ONLY"
    out = home / args.run_id
    aux = out / "summary.json"
    if args.report:
        report(v1_run.read_json(aux))
        return
    require(args.commit and len(args.commit)==40 and all(c in "0123456789abcdef" for c in args.commit), "SCALE_COMMIT")
    require(bool(args.partition_seed)==bool(args.frozen_from), "SCALE_UNFROZEN_REPARTITION")
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from interaction_fit import Repartition
    from source import Snapshot, Study, required_files
    from upf_refit import load
    from v3_inputs import read_sites

    site = previous.HERE.parents[2]
    snapshot = Snapshot(site, required_files(p.MODES["all"]))
    require(site.name==p.HANDOFF, "SCALE_WRONG_HANDOFF")
    source = home / "precision-1/summary.json"
    base = v1_run.read_json(source)
    base_reg = v1_run.read_json(source.with_name("registration.json"))
    require(base_reg["snapshot"]==snapshot.manifest_sha and base["reserved_outcomes_read"] is False,
            "SCALE_BASE_MISMATCH")
    settings = methods(base["frozen_methods"])
    source_sha = v1_run.file_sha(source)
    frozen_sha = None
    if args.frozen_from:
        source = home / args.frozen_from / "summary.json"
        frozen = v1_run.read_json(source)
        require(frozen["commit"]==args.commit and frozen["partition_seed"] is None and frozen["base_sha"]==source_sha,
                "SCALE_FROZEN_MISMATCH")
        frozen_sha = v1_run.file_sha(source)
        chosen = (REFERENCE, *frozen["frozen_methods"])
        settings = {k:settings[k] for k in chosen}
    selected = tuple(settings)
    with v1_run.run_lock(out):
        require(not aux.exists(), "SCALE_COMPLETED")
        study = Study(snapshot, with_upf=True)
        calibration, paths = load(site, study)
        feeds = previous.day_sources(site)
        sources = {str(x):v1_run.file_sha(x) for x in (*paths,*feeds)}
        require(sources==base_reg["source"], "SCALE_PRECISION_INPUTS_CHANGED")
        identity = dict(commit=args.commit, snapshot=snapshot.manifest_sha, runtime=runtime, sources=sources,
                        base_sha=source_sha, frozen_sha=frozen_sha, seed=args.partition_seed, methods=list(selected),
                        code={f.name:v1_run.file_sha(f) for f in (Path(__file__), Path(previous.__file__))})
        reg = out / "registration.json"
        if reg.exists():
            require(v1_run.read_json(reg)==identity, "SCALE_REGISTRATION_MISMATCH")
        else:
            v1_run.atomic_write(reg, v1_run.encode(identity))
        emit("SCALE_START", dict(run_id=args.run_id, seed=args.partition_seed, methods=len(selected)))
        food_days = previous.read_days(feeds,study,calibration)
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
        working = (Repartition(study,read_sites(study),calibration,args.partition_seed,emit)
                   if args.partition_seed is not None else study)
        fits,_ = previous.fit_all(working, outcomes, food_days)
        rows = score(working,rescale(fits,settings),selected,selected,p.BOOTSTRAP_DRAWS)
        coverage = coverage_summary(rows,selected)
        ceiling = {name:amplitude_ceiling(fits,name) for name in base["frozen_methods"]} if args.partition_seed is None else {}
        frozen_methods = [r["method"] for r in ranked(coverage) if r["method"]!=REFERENCE][:2]
        if args.partition_seed is None:
            original = {(r["exposure"],r["endpoint"]):r for r in base["rows"] if r["method"]==REFERENCE}
            err = max(abs(r[k]-original[r["exposure"],r["endpoint"]][k]) for r in rows if r["method"]==REFERENCE
                      for k in ("r2_no_diet","r2_off","r2_on"))
            require(err<=2e-8,"SCALE_REPLAY_MISMATCH")
        else:
            err = None
        require(all(v1_run.file_sha(Path(x))==h for x,h in sources.items()), "SCALE_INPUT_CHANGED")
        require(set(snapshot.paths)<=snapshot.verified, "SCALE_UNVERIFIED_DISCOVERY_INPUTS")
        snapshot.verify_manifest()
        result = dict(commit=args.commit,rows=rows,coverage=coverage,frozen_methods=frozen_methods,base_sha=source_sha,
                      partition_seed=args.partition_seed,replay_max=err,retrospective_max_reached=ceiling,
                      reserved_outcomes_read=False)
        v1_run.atomic_write(aux, v1_run.encode(result))
    emit("SCALE_RETROSPECTIVE_CEILING", ceiling)
    report(result)


if __name__=="__main__":
    main()
