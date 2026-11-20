"""Test recording reliability in interaction doses while keeping the prediction arms matched."""

import argparse
from pathlib import Path
import sys
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "Full_pheno_v7"))
from run_interaction import ANCHOR, archive, coverage_summary, emit, ranked, report, score, verify_reference, v1_run
from run_v2 import directory_id, require


class Recipe(NamedTuple):
    name: str
    diet: str
    transform: str
    pc5: float
    pc2: float = 0.


RECIPES = (
    Recipe("pc5_half", "legacy", "raw", .5),
    Recipe("host_pc5_half", "host", "raw", .5),
    Recipe("host_pc5_080", "host", "raw", .8),
    Recipe("legacy_precision_080", "legacy", "count", .8),
    Recipe("host_precision_half", "host", "count", .5),
    Recipe("host_precision_080", "host", "count", .8),
    Recipe("host_dayvariance_080", "host", "person", .8),
    Recipe("host_dayvariance_pc2", "host", "person", .45, .35),
    Recipe("host_clip_080", "host", "clip", .8),
    Recipe("host_clip_dayvariance_080", "host", "person_clipped", .8),
    Recipe("host_clip_dayvariance_pc2", "host", "person_clipped", .45, .35),
)
METHODS = tuple(recipe.name for recipe in RECIPES)


def day_sources(root):
    from food_features import SOURCES
    return [(root / SOURCES[k]).resolve(strict=True) for k in ("day", "nova_day")]


def read_days(paths, study, calibration):
    import numpy as np
    import pandas as pd
    from food_features import DAY, KEY, normalise, projected
    import protocol as p

    index = pd.DataFrame(dict(zip(KEY, study.keys, strict=True)))
    index["row"] = np.arange(study.n)
    people = index.participant_id.astype("int64").unique().tolist()
    day = normalise(projected(paths[0], DAY + ["energy_eligible", "fiber_eligible", "energy_kcal", "fiber_g"], people))
    energy = day.loc[day.energy_eligible.eq(True) & day.energy_kcal.gt(0)].merge(index, on=KEY, validate="many_to_one")
    require(not energy.duplicated(DAY).any(), "PRECISION_DUPLICATE_DAY")
    f = energy.loc[energy.fiber_eligible.eq(True) & energy.fiber_g.ge(0)].copy()
    raw = (1000.*f.fiber_g.to_numpy()/f.energy_kcal.to_numpy()-p.FIBER_DOSE[0])/(p.FIBER_DOSE[1]-p.FIBER_DOSE[0])
    fiber, fc = day_moments(f.row.to_numpy(), raw, study.n)
    expected_f = study.numeric["n_fiber_days"]
    has_fiber = expected_f>=3
    expected_dose = (study.numeric["fiber_density"][has_fiber]-p.FIBER_DOSE[0])/(p.FIBER_DOSE[1]-p.FIBER_DOSE[0])
    require(np.array_equal(fc[has_fiber], expected_f[has_fiber])
            and np.allclose(fiber[has_fiber, 0], expected_dose, atol=1e-7), "PRECISION_FIBER_ARCHIVE_DIFF")

    nova = normalise(projected(paths[1], DAY + ["known_UPF_kcal", "unknown_NOVA_kcal"], people))
    u = energy.merge(nova, on=DAY, validate="one_to_one")
    lower = (u.known_UPF_kcal/u.energy_kcal).to_numpy()
    unknown = (u.unknown_NOVA_kcal/u.energy_kcal).to_numpy()
    owners = u.row.to_numpy()
    lower_summary, uc = day_moments(owners, lower, study.n)
    unknown_summary, _ = day_moments(owners, unknown, study.n)
    mask = calibration.qualified
    require(np.isfinite(lower).all() and np.isfinite(unknown).all() and np.all(lower>=0.) and np.all(unknown>=0.)
            and np.array_equal(uc[mask], study.numeric["n_energy_days"][mask])
            and np.allclose(lower_summary[mask, 0], calibration.lower[mask], atol=1e-7)
            and np.allclose(unknown_summary[mask, 0], (calibration.upper-calibration.lower)[mask], atol=1e-7),
            "PRECISION_UPF_ARCHIVE_DIFF")
    return (fiber[:, 1], fc), (owners, lower, unknown, uc, lower_summary[:, 0], unknown_summary[:, 0])


def day_moments(owners, values, n):
    import numpy as np
    count = np.bincount(owners, minlength=n)
    num = np.bincount(owners, weights=values, minlength=n)
    sq = np.bincount(owners, weights=values**2, minlength=n)
    mean = num/np.maximum(count, 1)
    var = np.maximum((sq-count*mean**2)/np.maximum(count-1, 1), 0.)
    return np.column_stack((mean, var)), count


def features(exposure, bg, raw, days):
    import numpy as np
    if exposure == "fiber_density":
        variance, count = days[0]
    else:
        owners, low, width, count, lower, unknown = days[1]
        expected = raw*bg.upf["scale"] + bg.upf["center"]
        q = np.divide(expected-lower, unknown, out=np.zeros(len(raw)), where=unknown>1e-9)
        eligible = count>=3
        require(np.all((q[eligible]>=-1e-6)&(q[eligible]<=1.+1e-6)), "PRECISION_UPF_BOUNDS")
        q = np.clip(q, 0., 1.)
        daily = (low + width*q[owners])/bg.upf["scale"]
        variance = day_moments(owners, daily, len(raw))[0][:, 1]
    train = bg.role & bg.supported & (count>=3) & np.isfinite(raw) & np.isfinite(variance)
    pooled = float(np.average(variance[train], weights=count[train]-1))
    center = float(raw[train].mean())
    between = max(float(raw[train].var(ddof=1) - np.mean(pooled / count[train])), 1e-8)
    weights = between/(between+pooled/np.maximum(count, 1))
    personal = between/(between+.5*(pooled+variance)/np.maximum(count, 1))
    lo, hi = np.quantile(raw[train], [.05, .95])
    clipped = np.clip(raw, lo, hi)
    vs = {"raw":raw, "count":center+weights*(raw-center), "person":center+personal*(raw-center),
          "clip":clipped, "person_clipped":center+personal*(clipped-center)}
    for key, v in vs.items():
        if key != "raw":
            require(v[train].std()>1e-10, "PRECISION_DEGENERATE_DOSE")
            vs[key] = center + (v-v[train].mean())*(raw[train].std()/v[train].std())
    return vs, dict(median_record_days=float(np.median(count[train])),
                    median_count_weight=float(np.median(weights[train])),
                    median_person_weight=float(np.median(personal[train])))


def fit_all(study, outcomes, days):
    import numpy as np
    import protocol as p
    from breadth import basis, fit_profile
    from diet_audit import Fit, ridge_diet
    from exposure_models import inputs
    from models import create_design, dose_basis, predict, products
    from v4_models import no_diet_main

    results = {task:[] for task in p.MODES["all"]}
    reliability = []
    for fold in p.FOLDS:
        for exposure in p.EXPOSURES:
            raw, views = basis(study, exposure, fold)
            new, diag = features(exposure, study.backgrounds[fold], raw[:, 0], days)
            reliability.append(dict(fold=fold, exposure=exposure, **diag))
            for endpoint, outcome in outcomes.items():
                y, tr, held = inputs(study, exposure, endpoint, outcome, fold)
                n, rows = len(tr), np.r_[tr, held]
                d = create_design(study, exposure, fold, rows, n)
                dose = dose_basis(raw[rows], n)[0]
                base = no_diet_main(study, rows, n, d.main, dose)
                base_inv = np.linalg.pinv(base[:n], rcond=p.RANK_TOL)
                no_diet = base[n:]@(base_inv@y[tr])
                inv = np.linalg.pinv(d.main[:n], rcond=p.RANK_TOL)
                legacy = d.main@(inv@y[tr])
                if (exposure, endpoint)==ANCHOR:
                    off = np.repeat(legacy[n:, None], len(METHODS), axis=1)
                    on = np.repeat(predict(d, y[tr])[:, 2, None], len(METHODS), axis=1)
                else:
                    dietary = np.column_stack((dose, products(dose, d.main[:, 1:3])))
                    host, columns = ridge_diet(base, base_inv, dietary, y[tr], (.1,.5,1.,1.,1.,1.))
                    host_main = np.column_stack((base, columns))
                    setup = {"legacy":(legacy,d.main,inv),
                             "host":(host,host_main,np.linalg.pinv(host_main[:n],rcond=p.RANK_TOL))}
                    cache, off, on = {}, [], []
                    for r in RECIPES:
                        m, nuisance, inverse = setup[r.diet]
                        key = r.diet, r.transform
                        if key not in cache:
                            product = new[r.transform][rows,None]*views["gut_pc5"][rows]
                            resid = y[tr]-m[:n]
                            cache[key] = [fit_profile(product[:, :k], nuisance, inverse, resid).fit(1.) for k in (5,2)]
                        pc5, pc2 = cache[key]
                        off.append(m[n:])
                        on.append(m[n:]+r.pc5*pc5+r.pc2*pc2)
                    off,on = np.column_stack(off),np.column_stack(on)
                results[exposure,endpoint].append(Fit(held,y[held],no_diet,off,on))
        emit("PRECISION_FOLD_DONE", {"fold":fold,"pairs":len(results)})
    return results, reliability


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id", type=directory_id, required=True)
    cli.add_argument("--commit")
    cli.add_argument("--report", action="store_true")
    args = cli.parse_args()
    out = HERE / "runs/TRE_ONLY" / args.run_id
    path = out / "summary.json"
    if args.report:
        report(v1_run.read_json(path))
        return
    require(args.commit and len(args.commit)==40 and all(x in "0123456789abcdef" for x in args.commit), "PRECISION_COMMIT")
    runtime = v1_run.init_runtime(2)
    import protocol as p
    from source import Snapshot, Study, required_files
    from upf_refit import load

    root = HERE.parents[2]
    require(root.name==p.HANDOFF, "PRECISION_HANDOFF")
    snapshot = Snapshot(root, required_files(p.MODES["all"]))
    release, _ = archive.verify_release()
    old = HERE.parent/"Full_pheno_v5/runs/TRE_ONLY/discovery-1"
    previous = archive.read_complete(old, release)
    reg = v1_run.read_json(old/"registration.json")
    require(reg["snapshot_sha"]==snapshot.manifest_sha and reg["sources"]==snapshot.report(), "PRECISION_SNAPSHOT")
    with v1_run.run_lock(out):
        require(not path.exists(), "PRECISION_ALREADY_COMPLETE")
        study = Study(snapshot, with_upf=True)
        cal, cal_paths = load(root, study)
        paths = day_sources(root)
        sources = {str(x):v1_run.file_sha(x) for x in (*paths,*cal_paths)}
        identity = dict(commit=args.commit, snapshot=snapshot.manifest_sha, runtime=runtime, source=sources,
                        code={"run_precision.py":v1_run.file_sha(Path(__file__)),
                              "run_interaction.py":v1_run.file_sha(HERE.parent/"Full_pheno_v7/run_interaction.py")})
        meta = out/"registration.json"
        if meta.exists():
            require(v1_run.read_json(meta)==identity, "PRECISION_REGISTRATION_CHANGED")
        else:
            v1_run.atomic_write(meta, v1_run.encode(identity))
        emit("PRECISION_START", {"run_id":args.run_id,"methods":len(METHODS)})
        days = read_days(paths,study,cal)
        outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
        fits,reliability = fit_all(study,outcomes,days)
        rows = score(study,fits,METHODS,METHODS,p.BOOTSTRAP_DRAWS)
        coverage = coverage_summary(rows,METHODS)
        replay = verify_reference(previous,rows)
        require(set(snapshot.paths)<=snapshot.verified, "PRECISION_UNVERIFIED_INPUTS")
        snapshot.verify_manifest()
        require(all(v1_run.file_sha(Path(x))==h for x,h in sources.items()), "PRECISION_SOURCE_CHANGED")
        result = dict(rows=rows,coverage=coverage,reliability=reliability,replay_max=replay,partition_seed=None,
                      frozen_methods=[r["method"] for r in ranked(coverage) if r["method"]!="pc5_half"][:2],
                      commit=args.commit,reserved_outcomes_read=False)
        v1_run.atomic_write(path,v1_run.encode(result))
    report(result)


if __name__=="__main__":
    main()
