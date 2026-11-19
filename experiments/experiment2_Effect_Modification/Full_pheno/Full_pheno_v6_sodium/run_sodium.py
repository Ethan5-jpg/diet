"""Check additional sodium value on total and visceral fat at two coverage rules."""

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v6_exposure"))
from run_exposure import archive, emit, summarise, v1_run
from run_v2 import directory_id, require


METHODS = ("reference", "sodium_only", "joint_legacy_interaction", "joint_extra_sodium_interaction")
ENDPOINTS = ("D_TotalFat", "D_VAT")
MIN_FRACTIONS = (.8, .5)


def sodium_data(study, root):
    import numpy as np
    import pandas as pd
    from food_features import DAY, KEY, SOURCES, normalise, projected

    paths = tuple((root / SOURCES[name]).resolve(strict=True) for name in ("day", "qc"))
    source_sha = {str(path): v1_run.file_sha(path) for path in paths}
    index = pd.DataFrame(dict(zip(KEY, study.keys, strict=True)))
    index["input_row"] = np.arange(study.n)
    ids = index.participant_id.astype("int64").unique().tolist()
    day = normalise(projected(paths[0], DAY + ["energy_kcal", "energy_eligible"], ids))
    day = day.loc[day.energy_eligible.eq(True) & day.energy_kcal.gt(0)]
    day = day.merge(index, on=KEY, how="inner", validate="many_to_one")
    require(not day.duplicated(DAY).any(), "V6_SODIUM_DUPLICATE_DAY")
    count = np.bincount(day.input_row, minlength=study.n)
    expected = study.numeric["n_energy_days"]
    require(np.array_equal(count[expected >= 3], expected[expected >= 3]), "V6_SODIUM_DAY_COUNTS")
    qc = normalise(projected(paths[1], DAY + ["sodium_mg_complete_sum"], ids))
    day = day.merge(qc, on=DAY, how="inner", validate="one_to_one")
    sodium = day.sodium_mg_complete_sum
    day = day.loc[np.isfinite(sodium) & sodium.ge(0)].copy()
    day["density"] = 1000. * day.sodium_mg_complete_sum / day.energy_kcal
    summary = day.groupby("input_row").density.agg(["size", "mean"]).reindex(range(study.n))
    use = [(summary["size"].ge(3) & summary["size"].ge(expected * f)).to_numpy() for f in MIN_FRACTIONS]
    return {f"sodium_ge{int(f*100)}pct": np.where(ok, summary["mean"], np.nan)[:, None]
            for f, ok in zip(MIN_FRACTIONS, use, strict=True)}, source_sha


def fit_sodium(study, fold, exposure, endpoint, outcome, value, shared):
    import numpy as np
    import protocol as p
    from breadth import fit_profile
    from diet_audit import Fit, ridge_diet
    from exposure_models import inputs, with_feature
    from models import create_design, products, standardize

    y, tr, held = inputs(study, exposure, endpoint, outcome, fold, value)
    n, rows = len(tr), np.r_[tr, held]
    old = with_feature(study, fold, exposure, endpoint, outcome, value, shared)
    raw, views = shared
    pc = views["gut_pc5"][rows]
    legacy = create_design(study, exposure, fold, rows, n).main
    inverse = np.linalg.pinv(legacy[:n], rcond=p.RANK_TOL)
    salt = value[rows]
    joint, adjusted_sodium = ridge_diet(legacy, inverse, salt, y[tr], (.1,))
    doses = np.column_stack((legacy, adjusted_sodium))
    both_inverse = np.linalg.pinv(doses[:n], rcond=p.RANK_TOL)
    legacy_product = products(raw[rows, :1], pc)
    residual = y[tr] - joint[:n]
    original = fit_profile(legacy_product, doses, both_inverse, residual).fit(1.)
    conditional = np.column_stack((doses, legacy_product))
    conditional_inverse = np.linalg.pinv(conditional[:n], rcond=p.RANK_TOL)
    extra = fit_profile(products(standardize(salt, n)[0], pc), conditional, conditional_inverse, residual).fit(1.)
    # The new product is orthogonal to the old interaction on training rows.
    # The comparison keeps the same fitted original interaction in both arms.
    registered_on = joint[n:] + .5*original
    return Fit(held, old.outcome, old.no_diet,
               np.column_stack((old.off[:, 0], old.off[:, 2], joint[n:], registered_on)),
               np.column_stack((old.on[:, 0], old.on[:, 2], registered_on, registered_on + extra)))


def report(result):
    emit("SODIUM_SUPPORT", result["people_with_valid_sodium"])
    for row in result["rows"]:
        if row["method"] == "reference":
            continue
        keys = ("exposure", "endpoint", "feature", "method", "people", "coverage",
                "interaction_pp", "interaction_pp_95", "diet_total_pp", "vs_reference_pp", "vs_reference_pp_95")
        emit("SODIUM_RESULT", {k: row[k] for k in keys})
    emit("COMPLETE", dict(phase="sodium", pairs=len(result["rows"])//len(METHODS), reserved_outcomes_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=directory_id, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    out = HERE / "runs/TRE_ONLY" / args.run_id
    summary_path = out / "summary.json"
    if args.report:
        report(v1_run.read_json(summary_path))
        return
    require(args.commit and len(args.commit) == 40 and all(x in "0123456789abcdef" for x in args.commit),
            "V6_SODIUM_EXACT_COMMIT_REQUIRED")

    runtime = v1_run.init_runtime(2)
    import numpy as np
    import protocol as p
    from breadth import basis
    from exposure_models import inputs
    from source import Snapshot, Study, required_files

    root = HERE.parents[2]
    require(root.name == p.HANDOFF, "V6_SODIUM_HANDOFF_ROOT")
    tasks = tuple((exposure, endpoint) for exposure in p.EXPOSURES for endpoint in ENDPOINTS)
    snapshot = Snapshot(root, required_files(tasks))
    release, _ = archive.verify_release()
    archive.read_complete(HERE.parent / "Full_pheno_v5/runs/TRE_ONLY/discovery-1", release)
    with v1_run.run_lock(out):
        require(not summary_path.exists(), "V6_SODIUM_RUN_ALREADY_COMPLETE")
        study = Study(snapshot, with_upf=True)
        values, source_sha = sodium_data(study, root)
        support = {key: int(np.isfinite(x).sum()) for key, x in values.items()}
        emit("SODIUM_SUPPORT", support)
        helpers = [HERE.parent / "Full_pheno_v6_exposure" / x
                   for x in ("food_features.py", "exposure_models.py", "run_exposure.py")]
        helpers.append(HERE.parent / "Full_pheno_v6/diet_audit.py")
        identity = dict(commit=args.commit, snapshot=snapshot.manifest_sha, source=source_sha, runtime=runtime,
                        code={x.name: v1_run.file_sha(x) for x in (Path(__file__), *helpers)})
        meta = out / "registration.json"
        if meta.exists():
            require(v1_run.read_json(meta) == identity, "V6_SODIUM_REGISTRATION_CHANGED")
        else:
            v1_run.atomic_write(meta, v1_run.encode(identity))

        outcomes = {e: study.outcome(e) for e in ENDPOINTS}
        fits, baseline = {}, {}
        shared = {(x, fold): basis(study, x, fold) for x in p.EXPOSURES for fold in p.FOLDS}
        for exposure, endpoint in tasks:
            target = outcomes[endpoint]
            count = sum(len(inputs(study, exposure, endpoint, target, fold)[2]) for fold in p.FOLDS)
            for name, x in values.items():
                key = exposure, endpoint, name
                fits[key] = [fit_sodium(study, f, exposure, endpoint, target, x, shared[exposure, f]) for f in p.FOLDS]
                baseline[key] = count
        rows = summarise(study, fits, METHODS, baseline, p.BOOTSTRAP_DRAWS)
        for row in rows:
            row["baseline_arm_vs_no_diet_pp"] = row.pop("diet_main_pp")
            row["baseline_arm_vs_no_diet_pp_95"] = row.pop("diet_main_pp_95")
        require(set(snapshot.paths) <= snapshot.verified, "V6_SODIUM_UNVERIFIED_SNAPSHOT")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(path)) == digest for path, digest in source_sha.items()), "V6_SODIUM_SOURCE_CHANGED")
        result = dict(rows=rows, scope="discovery only", people_with_valid_sodium=support, reserved_outcomes_read=False)
        v1_run.atomic_write(summary_path, v1_run.encode(result))
    report(result)


if __name__ == "__main__":
    main()
