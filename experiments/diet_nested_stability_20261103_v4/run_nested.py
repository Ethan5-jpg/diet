"""Discovery-only nested household CV. No confirmation outcomes or cached CLR fits."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import time
import warnings
import numpy as np
import pandas as pd
import sklearn

SEED = 20261104
SITES = ('host', 'oral', 'oral_gut')
ALGORITHMS = ('hist_boost', 'elastic_net', 'ridge')
PREPROCESS_SHA = 'b8f8e902a0722b5f90672d2cf63811b665074b32649e1b8a87772bb476f1e375'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def dependencies(package, pilot, v3):
    global MicrobiomeCLR, PhenotypeOutliers, host_design, rcs, shared
    shared = module('v3_frozen', v3/'run_screen.py')
    prep = package/'preprocessing/code/preprocessing_v1_20261030.py'
    assert shared.sha(prep) == PREPROCESS_SHA
    pp = module('v4_preprocessing', prep)
    MicrobiomeCLR, PhenotypeOutliers = pp.MicrobiomeCLR, pp.PhenotypeOutliers
    sys.path.insert(0, str(pilot))
    from run_pilot import host_design
    from core import rcs


def folds(groups, k, seed):
    unique = np.array(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(unique)
    lookup = {v: j % k for j, v in enumerate(unique)}
    return np.array([lookup[v] for v in groups])


def separated(groups, train, valid):
    assert train.dtype == valid.dtype == bool
    assert not np.any(train & valid)
    assert not set(groups[train]) & set(groups[valid])


def context(x, raw, taxa, y_raw, training, validation):
    """Fit every learned preprocessing quantity exclusively on training roles."""
    groups = x.split_unit_id.astype(str).to_numpy()
    separated(groups, training, validation)
    assert training.any() and validation.any()
    z, transforms, supported = {}, {}, np.ones(len(x), bool)
    for site in ('oral', 'gut'):
        m = MicrobiomeCLR().fit(raw[site][training], taxa[site],
            x.loc[training, 'participant_id'].tolist(), site=site)
        t = m.transform(raw[site], taxa[site], site=site)
        z[site], transforms[site] = t['values'], m.state
        supported &= t['supported']
    good = training & x.n_fiber_days.ge(3).to_numpy() & np.isfinite(x.fiber_density.to_numpy(float))
    knots_raw = np.quantile(x.loc[good, 'fiber_density'].to_numpy(float), [.1, .5, .9])
    center, scale = float(knots_raw[1]), 5.
    knots = (knots_raw-center)/scale
    u = (x.fiber_density.to_numpy(float)-center)/scale
    h = rcs(u, knots)
    good = training & np.isfinite(y_raw)
    target = PhenotypeOutliers().fit(y_raw[good], np.flatnonzero(good).tolist(), endpoint='D_Glu')
    y = target.transform(y_raw, endpoint='D_Glu')['values']
    eligible = (x.n_fiber_days.ge(3).to_numpy() & supported & np.isfinite(y) &
        np.isfinite(x.age_calendar_years.to_numpy(float)) & x.sex_code.isin([0, 1]).to_numpy() &
        np.isfinite(x.mean_daily_energy_fiber.to_numpy(float)) & np.isfinite(u) & np.isfinite(h))
    tr, va = training & eligible, validation & eligible
    assert tr.sum() > 100 and va.sum() > 30
    host, encoding = host_design(x, tr, u, h)
    designs = dict(host=host, oral=np.column_stack([host, z['oral']]),
        oral_gut=np.column_stack([host, z['oral'], z['gut']]))
    for a in designs.values():
        assert np.isfinite(a[tr | va]).all()
    state = dict(CLR=transforms, diet=dict(center=center, scale=scale, knots=knots),
        target=target.state, host=encoding)
    return designs, y, tr, va, state


def fit_predict(a, y, tr, va, algorithm, setting):
    mean, sd = float(y[tr].mean()), float(y[tr].std())
    assert sd > 0
    est = shared.estimator(algorithm, setting)
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        est.fit(a[tr], (y[tr]-mean)/sd)
        pred = est.predict(a[va])*sd+mean
    assert np.isfinite(pred).all()
    return pred, dict(model=est, y_mean=mean, y_sd=sd), dict(seconds=time.time()-start,
        warnings=[str(w.message) for w in caught], MSE=float(np.mean((y[va]-pred)**2)))


def read_table(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path, pre_buffer=False).read().to_pandas(ignore_metadata=True)


def main(args):
    start = time.time()
    root = Path(__file__).resolve().parent
    package, pilot, v3 = (Path(v).resolve() for v in (args.package, args.pilot_code, args.v3_code))
    assert package.name == 'collaborator_handoff_20261102_v1' and root.parent == package/'experiments'
    dependencies(package, pilot, v3)
    assert json.loads((v3/'results/audit.json').read_text())['status'] == 'PASS'
    data = package/'inputs/discovery_v2'
    input_names = ['discovery_covariates_food_TRE_ONLY.parquet', 'D_Glu_raw_Y_TRE_ONLY.parquet']
    input_names += [s+n for s in ('oral', 'gut') for n in ('_raw_TRE_ONLY.npz', '_taxa.json')]
    paths = [data/n for n in input_names] + [package/'preprocessing/code/preprocessing_v1_20261030.py',
        package/'examples/load_fiber_fold.py', pilot/'run_pilot.py', pilot/'core.py', v3/'run_screen.py']
    paths += [root/n for n in ('run_nested.py', 'verify_nested.py', 'launch.py', 'README.md')]
    hashes = {str(p): shared.sha(p) for p in paths}
    x = read_table(data/input_names[0])
    assert len(x) == 4427 and x.final_role.eq('discovery').all() and x.participant_id.is_unique
    np.testing.assert_array_equal(x.input_row, np.arange(len(x)))
    assert x.split_unit_id.notna().all() and x.groupby('split_unit_id').cv_fold.nunique().max() == 1
    yd = read_table(data/input_names[1])
    rows = yd.input_row.to_numpy(int)
    assert len(set(rows)) == len(rows) and ((rows >= 0) & (rows < len(x))).all()
    for key in ('participant_id', 'cohort', 'research_stage'):
        if key in yd:
            np.testing.assert_array_equal(yd[key].to_numpy(), x.iloc[rows][key].to_numpy())
    assert yd.Y_unit.dropna().unique().tolist() == ['mg/dL']
    y_raw = np.full(len(x), np.nan)
    y_raw[rows] = yd.Y_value.to_numpy(float)
    raw, taxa = {}, {}
    for s in ('oral', 'gut'):
        with np.load(io.BytesIO((data/(s+'_raw_TRE_ONLY.npz')).read_bytes()), allow_pickle=False) as a:
            raw[s] = a['abundance']
        taxa[s] = json.loads((data/(s+'_taxa.json')).read_text())
        assert raw[s].shape == (len(x), len(taxa[s]))
    out = root/'results'
    out.mkdir(exist_ok=False)
    private = out/'TRE_ONLY'
    private.mkdir()
    models = [dict(name=a+'_'+s, algorithm=a, site=s, grid=shared.GRIDS[a]) for s in SITES for a in ALGORITHMS]
    shared.write_json(out/'registration.json', dict(models=models, seed=SEED, repeats=2,
        outer_folds=5, inner_folds=3, expected_inner_fits=1080, expected_outer_fits=90,
        scope='Discovery development stability; prior model-family selection used these participants.',
        source_and_input_sha256=hashes, code_commit=args.code_commit,
        python=sys.version, sklearn=sklearn.__version__, confirmation_outcomes_read=False))
    groups = x.split_unit_id.astype(str).to_numpy()
    outer = np.stack([x.cv_fold.to_numpy(int), folds(groups, 5, SEED)])
    assert not np.array_equal(outer[0], outer[1])
    preds = {m['name']: np.full((2, len(x)), np.nan) for m in models}
    yy, mean_pred = np.full((2, len(x)), np.nan), np.full((2, len(x)), np.nan)
    diagnostics, choices, audits = [], [], []
    # Reproduce the established fold-0 preprocessing before any new fits.
    loader = module('v4_replay_loader', package/'examples/load_fiber_fold.py')
    old = loader.load_fold(0)
    d0, y0, tr0, va0, _ = context(x, raw, taxa, y_raw, outer[0] != 0, outer[0] == 0)
    np.testing.assert_array_equal(tr0, old['train'])
    np.testing.assert_array_equal(va0, old['valid'])
    np.testing.assert_allclose(y0, old['y'], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(d0['oral'][tr0 | va0, -old['oral_z'].shape[1]:], old['oral_z'][tr0 | va0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(d0['oral_gut'][tr0 | va0, -old['gut_z'].shape[1]:], old['gut_z'][tr0 | va0], atol=1e-10, rtol=0)
    print('PREPROCESS_REPLAY_PASS', flush=True)
    for repeat in range(2):
        for fold in range(5):
            tag = f'r{repeat}_f{fold}'
            training, validation = outer[repeat] != fold, outer[repeat] == fold
            separated(groups, training, validation)
            inner = np.full(len(x), -1)
            inner[training] = folds(groups[training], 3, SEED+100*repeat+fold+1)
            ip = {m['name']: np.full((len(x), len(m['grid'])), np.nan) for m in models}
            iy, imask = np.full(len(x), np.nan), np.zeros((3, len(x)), bool)
            losses = {m['name']: np.zeros((3, len(m['grid']))) for m in models}
            for k in range(3):
                it, iv = training & (inner != k), training & (inner == k)
                assert not (it & validation).any() and not (iv & validation).any()
                designs, y, tr, va, state = context(x, raw, taxa, y_raw, it, iv)
                imask[k], iy[va] = tr, y[va]
                with (private/(tag+f'_inner{k}_preprocess.pkl')).open('xb') as f:
                    pickle.dump(state, f, protocol=4)
                for m in models:
                    for j, setting in enumerate(m['grid']):
                        pp, _, diag = fit_predict(designs[m['site']], y, tr, va, m['algorithm'], setting)
                        ip[m['name']][va, j] = pp
                        losses[m['name']][k, j] = diag['MSE']
                        diagnostics.append(dict(stage='inner', repeat=repeat, fold=fold, inner=k,
                            model=m['name'], grid=j, **diag))
                print('INNER_DONE', tag, k, 'fits', len(diagnostics), flush=True)
            shared.save_arrays(private/(tag+'_inner_predictions.npz'), y=iy, inner=inner,
                training_masks=imask, outer_validation=validation, **ip)
            # Read back every inner loss before selecting. No outer outcome is consulted.
            with np.load(io.BytesIO((private/(tag+'_inner_predictions.npz')).read_bytes()), allow_pickle=False) as a:
                for k in range(3):
                    va = (a['inner'] == k) & np.isfinite(a['y'])
                    separated(groups, a['training_masks'][k], va)
                    assert not (va & a['outer_validation']).any()
                    for m in models:
                        actual = np.mean((a['y'][va, None]-a[m['name']][va])**2, axis=0)
                        np.testing.assert_allclose(actual, losses[m['name']][k], atol=1e-10, rtol=0)
            selected = {m['name']: int(losses[m['name']].mean(axis=0).argmin()) for m in models}
            designs, y, tr, va, state = context(x, raw, taxa, y_raw, training, validation)
            yy[repeat, va], mean_pred[repeat, va] = y[va], y[tr].mean()
            with (private/(tag+'_outer_preprocess.pkl')).open('xb') as f:
                pickle.dump(state, f, protocol=4)
            for m in models:
                j = selected[m['name']]
                pp, fitted, diag = fit_predict(designs[m['site']], y, tr, va, m['algorithm'], m['grid'][j])
                preds[m['name']][repeat, va] = pp
                diagnostics.append(dict(stage='outer', repeat=repeat, fold=fold, model=m['name'], grid=j, **diag))
                choices.append(dict(repeat=repeat, fold=fold, model=m['name'], grid=j,
                    inner_mean_MSE=losses[m['name']].mean(axis=0), setting=m['grid'][j]))
                with (private/(tag+'_'+m['name']+'.pkl')).open('xb') as f:
                    pickle.dump(fitted, f, protocol=4)
            audits.append(dict(repeat=repeat, fold=fold, training_n=int(tr.sum()), validation_n=int(va.sum()),
                training_households=len(set(groups[tr])), validation_households=len(set(groups[va])),
                CLR_dimensions={s: len(state['CLR'][s]['output_indices']) for s in ('oral', 'gut')}))
            shared.save_arrays(private/(tag+'_checkpoint.npz'), y=yy, mean=mean_pred, outer=outer, **preds)
            shared.write_json(out/(tag+'_progress.json'), dict(counts=audits[-1], choices=choices[-9:],
                fits_completed=len(diagnostics), elapsed_seconds=time.time()-start))
            print('OUTER_DONE', tag, 'fits', len(diagnostics), 'elapsed_s', round(time.time()-start), flush=True)
    shared.save_arrays(private/'final_predictions.npz', y=yy, mean=mean_pred, outer=outer, **preds)
    summary, comparisons = [], []
    with np.load(io.BytesIO((private/'final_predictions.npz').read_bytes()), allow_pickle=False) as a:
        for repeat in range(2):
            valid = np.isfinite(a['y'][repeat])
            for m in models:
                p = a[m['name']][repeat]
                assert np.array_equal(np.isfinite(p), valid)
                score = shared.metric(a['y'][repeat, valid], p[valid])
                summary.append(dict(repeat=repeat, model=m['name'], n=int(valid.sum()), **score))
                for fold in range(5):
                    va = valid & (a['outer'][repeat] == fold)
                    mse = float(np.mean((a['y'][repeat, va]-p[va])**2))
                    olddiag = [d for d in diagnostics if d['stage']=='outer' and d['repeat']==repeat and d['fold']==fold and d['model']==m['name']]
                    assert len(olddiag) == 1
                    np.testing.assert_allclose(mse, olddiag[0]['MSE'], atol=1e-10, rtol=0)
            pairs = [(alg+'_'+site, 'ridge_'+site) for alg in ('hist_boost','elastic_net') for site in SITES]
            pairs += [(alg+'_oral_gut', alg+'_oral') for alg in ALGORITHMS]
            pairs += [(alg+'_oral', alg+'_host') for alg in ALGORITHMS]
            for model, reference in pairs:
                error = (a['y'][repeat]-a[model][repeat])**2
                base = (a['y'][repeat]-a[reference][repeat])**2
                delta = [float(np.mean(base[valid & (outer[repeat]==f)]-error[valid & (outer[repeat]==f)])) for f in range(5)]
                comparisons.append(dict(repeat=repeat, model=model, reference=reference,
                    MSE_reduction=1-float(error[valid].mean()/base[valid].mean()),
                    delta_R2=float((base[valid].mean()-error[valid].mean())/np.var(a['y'][repeat, valid])),
                    fold_MSE_improvement=delta, improved_folds=sum(v>0 for v in delta)))
    assert len(diagnostics) == 1170 and len(choices) == 90
    assert len(list(private.glob('*_preprocess.pkl'))) == 40
    assert len(list(private.glob('*.pkl'))) == 130
    assert all(shared.sha(p) == value for p, value in hashes.items())
    audit = dict(status='PASS', inner_fits=1080, outer_fits=90, all_1170_losses_recomputed=True,
        preprocessing_fold0_replayed=True, source_and_input_hashes_verified=True,
        household_isolation_verified=True, warning_fits=sum(bool(d['warnings']) for d in diagnostics),
        confirmation_outcomes_read=False, elapsed_seconds=time.time()-start)
    for name, value in [('summary', summary), ('comparisons', comparisons), ('diagnostics', diagnostics),
                        ('choices', choices), ('fold_counts', audits), ('audit', audit)]:
        shared.write_json(out/(name+'.json'), value)
    pd.DataFrame(summary).to_csv(out/'model_comparison.csv', index=False, mode='x')
    lines = ['# V4 nested household stability', '',
        'Discovery development only; model families were chosen using these participants in prior screens.',
        'Two repeats are correlated, not independent replications. Do not infer significance from fold counts.',
        'Targets and eligibility are refitted per training split; use paired within-repeat comparisons.', '',
        '| Repeat | Model | n | R2 | RMSE |', '| --- | --- | ---: | ---: | ---: |']
    lines += [f"| {r['repeat']} | {r['model']} | {r['n']} | {r['R2']:.6f} | {r['RMSE']:.6f} |" for r in summary]
    lines += ['', '| Repeat | Model | Reference | Delta R2 | MSE reduction | Improved folds |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    lines += [f"| {r['repeat']} | {r['model']} | {r['reference']} | {r['delta_R2']:.6f} | {r['MSE_reduction']:.3%} | {r['improved_folds']}/5 |" for r in comparisons]
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print('NESTED_AUDIT_PASS', json.dumps(audit), flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('package', 'pilot-code', 'v3-code', 'code-commit'):
        parser.add_argument('--'+key, required=True)
    main(parser.parse_args())
