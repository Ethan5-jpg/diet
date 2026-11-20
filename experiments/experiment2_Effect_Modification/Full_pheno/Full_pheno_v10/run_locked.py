"""Vary only the microbiome interaction over the original 18 diet models."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent/"Full_pheno_v9"))
from run_expanded import HISTORY, summarize
from run_interaction import ANCHOR, archive, emit, score, v1_run
from run_v2 import directory_id, require


PARTITIONS = (("archive", None), ("fresh-1", 20261703), ("fresh-2", 20261741))
CANDIDATES = ("pc5_half", "gut_pc2", "gut_pc5", "gut_precision", "oral_curve", "both_sites")
METHODS = ("pc5_half", "v1_fusion", "gut_precision", "inner_pc5", "inner_select", "interaction_off")


def register(out, identity):
    path = out/"registration.json"
    if path.exists():
        require(v1_run.read_json(path) == identity, "V10_REGISTRATION_MISMATCH")
    else:
        v1_run.atomic_write(path, v1_run.encode(identity))


def parse(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-id", type=directory_id, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    require(len(args.commit) == 40 and all(c in "0123456789abcdef" for c in args.commit), "V10_COMMIT")
    return args


def context():
    import protocol as p
    from source import Snapshot, Study, required_files
    from upf_refit import load
    from v3_inputs import read_sites

    root = HERE.parents[2]
    snapshot = Snapshot(root, required_files(p.MODES["all"]))
    release, _ = archive.verify_release()
    discovery = HERE.parent/"Full_pheno_v5/runs/TRE_ONLY/discovery-1"
    archive.read_complete(discovery, release)
    recorded = v1_run.read_json(discovery/"registration.json")
    require(recorded["snapshot_sha"] == snapshot.manifest_sha and recorded["sources"] == snapshot.report(),
            "V10_SOURCE_MISMATCH")
    study = Study(snapshot, with_upf=True)
    calibration, files = load(root, study)
    return root, snapshot, study, calibration, read_sites(study), files


def fit_split(study, exposure, fold, outcomes, days):
    import numpy as np
    import protocol as p
    from breadth import basis, fit_profile
    from diet_audit import Fit
    from expanded_fit import gut_update
    from exposure_models import inputs
    from models import create_design, dose_basis, predict, products
    from run_precision import features
    from v4_models import no_diet_main

    raw, views = basis(study, exposure, fold)
    precise, _ = features(exposure, study.backgrounds[fold], raw[:, 0], days)
    result = {}
    for e, outcome in outcomes.items():
        y, tr, held = inputs(study, exposure, e, outcome, fold)
        require(len(tr) >= 30 and len(held) >= 10 and not np.intersect1d(study.family[tr], study.family[held]).size,
                "V10_LOCKED_SUPPORT")
        rows, n = np.r_[tr, held], len(tr)
        design = create_design(study, exposure, fold, rows, n)
        original = predict(design, y[tr])
        main = design.main
        inv = np.linalg.pinv(main[:n], rcond=p.RANK_TOL)
        residual = y[tr] - main[:n] @ (inv @ y[tr])
        baseline = original[:, 0]
        dose = dose_basis(raw[rows], n)[0]
        plain = no_diet_main(study, rows, n, main, dose)
        nodiet = plain[n:] @ (np.linalg.pinv(plain[:n], rcond=p.RANK_TOL) @ y[tr])
        if (exposure, e) == ANCHOR:
            updates = np.repeat(original[:, 2, None], len(CANDIDATES)+1, axis=1)
        else:
            gut = products(raw[rows, :1], views["gut_pc5"][rows])
            pc5 = fit_profile(gut, main, inv, residual).fit(1.)
            pc2 = fit_profile(gut[:, :2], main, inv, residual).fit(1.)
            reliable = products(precise["person_clipped"][rows, None], views["gut_pc5"][rows])
            oral = products(dose, views["ecology"][rows, :1])
            both = products(raw[rows, :1], views["both_pc5"][rows])
            oral_update = fit_profile(oral, main, inv, residual).fit(.12)
            both_update = fit_profile(both, main, inv, residual).fit(1.)
            updates = baseline[:, None] + np.column_stack((.5*pc5, pc2, pc5, gut_update(reliable, main, residual),
                                                          .65*oral_update, .65*both_update))
            updates = np.column_stack((updates, original[:, 2]))
        result[e] = Fit(held, y[held], nodiet, baseline[:, None], updates)
    return result


def choose(parts, allowed, scales):
    """Choose an interaction only via held-out predictions inside the training households."""
    import numpy as np

    y = np.concatenate([f.outcome for f in parts])
    sst = float(np.sum((y-y.mean())**2))
    if sst <= 0.:
        return 0, 0.
    scores = []
    for j in allowed:
        dot = np.array([np.dot(f.outcome-f.off[:, 0], f.on[:, j]-f.off[:, 0]) for f in parts])
        norm = np.array([np.sum((f.on[:, j]-f.off[:, 0])**2) for f in parts])
        for weight in scales:
            improvement = weight*(2.*dot-weight*norm)
            gain = 100.*improvement.sum()/sst
            if (improvement > 0.).sum() >= 3 and gain >= .015:
                scores.append((gain, -weight, -j))
    if not scores:
        return 0, 0.
    _, w, j = max(scores)
    return -j, -w


def run_partition(study, outcomes, calibration, sites, days, seed):
    import numpy as np
    import protocol as p
    from diet_audit import Fit
    from interaction_fit import Repartition
    from v3_inputs import InnerContexts

    work = study if seed is None else Repartition(study, sites, calibration, seed, emit)
    nested = InnerContexts(work, sites)
    result = {key: [] for key in p.MODES["all"]}
    decisions = []
    receipt = sha256()
    for outer in p.FOLDS:
        for exposure in p.EXPOSURES:
            internal = {e: [] for e in outcomes if (exposure, e) != ANCHOR}
            for inner in p.FOLDS:
                if inner == outer:
                    continue
                ctx = nested.split(outer, inner)
                if exposure == "UPF":
                    ctx, _ = calibration.inner(ctx, inner, 100*(seed or p.BOOTSTRAP_SEED)+10*outer+inner)
                for e, f in fit_split(ctx, exposure, inner, {e: outcomes[e] for e in internal}, days).items():
                    require(np.all(work.fold[f.rows] == inner) and not np.any(work.fold[f.rows] == outer),
                            "V10_INNER_HELD_MEMBERSHIP")
                    internal[e].append(f)
            for e, f in fit_split(work, exposure, outer, outcomes, days).items():
                key = exposure, e
                base = f.off[:, 0]
                if key == ANCHOR:
                    chosen = [f.on[:, -1]]*len(METHODS)
                else:
                    q = choose(internal[e], (0,), (.5, 1.))
                    s = choose(internal[e], range(len(CANDIDATES)), (.35, .65, 1.))
                    j, v = q
                    k, w = s
                    chosen = [f.on[:, 0], f.on[:, -1], f.on[:, 3], base+v*(f.on[:, j]-base),
                              base+w*(f.on[:, k]-base), base]
                    decisions.append(dict(fold=outer, exposure=exposure, endpoint=e,
                                          pc5_weight=v, selected=CANDIDATES[k] if w else None, weight=w))
                off = np.repeat(f.off, len(METHODS), axis=1)
                require(all(np.array_equal(c, base) for c in off.T) and np.all(work.fold[f.rows] == outer),
                        "V10_LOCKED_MAIN_PREDICTION")
                receipt.update(f.rows.astype("<i8").tobytes())
                receipt.update(base.astype("<f8").tobytes())
                result[key].append(Fit(f.rows, f.outcome, f.no_diet, off, np.column_stack(chosen)))
        emit("V10_LOCKED_FOLD", dict(seed=seed, fold=outer))
    return result, decisions, receipt.hexdigest()


def replay(rows, previous):
    reference = {(r["exposure"], r["endpoint"]): r for r in previous["rows"] if r["method"]=="pc5_half"}
    require(len(reference) == 18, "V10_LOCKED_ARCHIVE_PAIRS")
    errors = []
    for r in rows:
        if r["method"] != "pc5_half":
            continue
        old = reference[r["exposure"], r["endpoint"]]
        require((r["n"], r["families"]) == (old["n"], old["families"]), "V10_LOCKED_ARCHIVE_SUPPORT")
        errors.append(max(abs(r[k]-old[k]) for k in ("r2_no_diet", "r2_off", "r2_on")))
    require(max(errors) <= 2e-8, "V10_LOCKED_ARCHIVE_REPLAY")
    return max(errors)


def main():
    args = parse(__doc__)
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from run_precision import day_sources, read_days

    out = HERE/"runs/TRE_ONLY"/args.run_id
    with v1_run.run_lock(out):
        root, snap, study, cal, sites, cal_files = context()
        feeds = day_sources(root)
        sources = {str(f):v1_run.file_sha(f) for f in (*feeds, *cal_files, HISTORY/"archive.json")}
        register(out, dict(commit=args.commit, runtime=runtime, snapshot=snap.manifest_sha,
                           seeds=[list(x) for x in PARTITIONS], methods=list(METHODS), sources=sources,
                           script_sha256=v1_run.file_sha(Path(__file__))))
        days = read_days(feeds, study, cal)
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
        track = []
        for name, seed in PARTITIONS:
            path = out/f"{name}.json"
            if path.exists():
                result = v1_run.read_json(path)
                require(result["seed"] == seed and len(result["rows"]) == 18*len(METHODS),
                        "V10_LOCKED_CHECKPOINT")
            else:
                fits, choices, digest = run_partition(study, outcomes, cal, sites, days, seed)
                rows = score(study, fits, METHODS, METHODS, p.BOOTSTRAP_DRAWS)
                difference = replay(rows, v1_run.read_json(HISTORY/"archive.json")) if seed is None else None
                result = dict(seed=seed, rows=rows, decisions=choices, coverage=summarize(rows, METHODS),
                              fixed_baseline_predictions_sha256=digest, archived_max_r2_error=difference)
                v1_run.atomic_write(path, v1_run.encode(result))
            track.append(dict(name=name, rows=len(result["rows"]), coverage=result["coverage"],
                              archived_max_r2_error=result["archived_max_r2_error"]))
            emit("V10_LOCKED_PARTITION", track[-1])
        require(set(snap.paths) <= snap.verified, "V10_LOCKED_UNVERIFIED_SNAPSHOT")
        snap.verify_manifest()
        require(all(v1_run.file_sha(Path(k)) == v for k,v in sources.items()), "V10_LOCKED_SOURCE_CHANGED")
        summary = dict(partitions=track, commit=args.commit, reserved_outcomes_read=False)
        v1_run.atomic_write(out/"summary.json", v1_run.encode(summary))
    emit("V10_LOCKED_COMPLETE", dict(partitions=len(track), rows=sum(x["rows"] for x in track)))


if __name__ == "__main__":
    main()
