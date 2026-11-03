"""Matched-input discovery screen. Individual data and fitted objects stay in TRE."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
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
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261103
GRIDS = {
    'hist_boost': [dict(leaves=7, l2=10., iterations=160, rate=.04),
                   dict(leaves=15, l2=30., iterations=160, rate=.04),
                   dict(leaves=31, l2=30., iterations=240, rate=.03)],
    'random_forest': [dict(leaf=10, features=.5), dict(leaf=20, features=1.)],
    'ridge': [dict(alpha=a) for a in (1., 100., 10000.)],
    'elastic_net': [dict(alpha=a, l1_ratio=r) for a in (.001, .01, .1) for r in (.2, .8)],
}
REPRESENTATIONS = ('PCA10', 'PCA30', 'PCA50', 'CLR')
SITES = ('oral', 'gut', 'oral_gut')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False, default=convert)
        f.write('\n')


def save_arrays(path, **arrays):
    # Mounted TRE storage requires sequential writes; ZIP may seek internally.
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with Path(path).open('xb') as stream:
        stream.write(buffer.getvalue())


def specs():
    inputs = [('host', 'host')] + [(s, r) for s in SITES for r in REPRESENTATIONS]
    return [dict(name=f'{a}_{r}_{s}', algorithm=a, site=s, representation=r, grid=g)
            for s, r in inputs for a, g in GRIDS.items()]


def estimator(algorithm, setting):
    if algorithm == 'hist_boost':
        return HistGradientBoostingRegressor(max_iter=setting['iterations'],
            learning_rate=setting['rate'], max_leaf_nodes=setting['leaves'],
            min_samples_leaf=30, l2_regularization=setting['l2'],
            early_stopping=False, random_state=SEED)
    if algorithm == 'random_forest':
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=setting['leaf'],
            max_features=setting['features'], n_jobs=2, random_state=SEED)
    if algorithm == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=setting['alpha'], solver='svd'))
    if algorithm == 'elastic_net':
        return make_pipeline(StandardScaler(), ElasticNet(**setting, max_iter=20000,
            tol=1e-5, selection='cyclic', random_state=SEED))
    raise ValueError(algorithm)


def pca_map(z, training):
    """Fit once per site and fold; all smaller PCA designs are exact prefixes."""
    good = np.isfinite(z).all(1)
    assert good[training].all()
    model = PCA(n_components=50, svd_solver='full').fit(z[training])
    transformed = np.full((len(z), 50), np.nan)
    transformed[good] = model.transform(z[good])
    return transformed, model


def design(host, clr, pcs, site, representation):
    if site == 'host':
        return host
    sites = ('oral', 'gut') if site == 'oral_gut' else (site,)
    parts = [clr[s] if representation == 'CLR' else pcs[s][:, :int(representation[3:])]
             for s in sites]
    return np.column_stack([host] + parts)


def metric(y, prediction):
    mse = float(np.mean((y-prediction)**2))
    return dict(R2=1-mse/float(np.var(y)), MSE=mse, RMSE=float(np.sqrt(mse)))


def report(summary, audit):
    rows = sorted(summary['models'], key=lambda r: -r['R2'])
    lines = ['# V3 matched-input prediction screen', '',
        'Exploratory development results; selection and reporting use the same household folds.',
        'This screen compares predictive models, not causal dietary effects or microbial interactions.', '',
        f"Audit: {audit['status']}. Families: {len(rows)}. Candidate fits: {audit['candidate_fits']}.",
        f"Baseline R2: {summary['baseline']['R2']:.6f}.", '',
        '| Model | R2 | Delta vs baseline | Delta vs same-input Ridge | Improved folds |',
        '| --- | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['model']} | {r['R2']:.6f} | {r['delta_R2_vs_v1_null']:.6f} | "
              f"{r['delta_R2_vs_same_input_ridge']:.6f} | {r['folds_improved_vs_v1_null']} |" for r in rows]
    lines += ['', '## Interpretation rules', '',
        '- PCA10/PCA30/PCA50 use prefixes of one train-only PCA50 map per site, on the existing fold-specific CLR design. They are not assumed identical to v2 archived PC10.',
        '- Compare site inputs within the same algorithm and representation; compare algorithms within the same input design.',
        '- PCA joint-site designs concatenate k oral and k gut scores, so total microbial dimensions are 2k.',
        '- Ridge/Elastic Net standardize all design columns on training rows. Their penalties differ from the v1 partially penalized background model.',
        '- Do not interpret ranking differences as statistical significance. Repeat-seed, nested validation or untouched confirmation is still needed.',
        '- Numerical fitting warnings and every candidate score are retained.',
        '- Patient-level predictions, labels, household identifiers and fitted models are excluded from public output.', '']
    return '\n'.join(lines)


def audit_and_summarize(output, specifications, identity, elapsed, source_hashes, anchors):
    # Independent readback from completed candidate arrays, not in-memory score rows.
    with np.load(io.BytesIO((output/'candidate_predictions_TRE_ONLY.npz').read_bytes()), allow_pickle=False) as a:
        y, baseline, fold = a['y'], a['baseline'], a['fold']
        arrays = {m['name']: a[m['name']] for m in specifications}
    common = np.isfinite(y)
    assert common.sum() == 3343
    assert identity.loc[common, 'split_unit_id'].nunique() == 3059
    assert identity.loc[common].groupby('split_unit_id').cv_fold.nunique().max() == 1
    base = metric(y[common], baseline[common])
    diagnostics = json.loads((output/'diagnostics.json').read_text())
    observed = {(d['model'], d['grid'], d['fold']): d for d in diagnostics}
    expected = 5*sum(len(m['grid']) for m in specifications)
    assert len(diagnostics) == len(observed) == expected == 910
    assert len(list((output/'fits_TRE_ONLY').glob('*.pkl'))) == 915
    selected, grid = [], []
    held = identity.loc[common].copy()
    held['Y_processed'], held['baseline_prediction'] = y[common], baseline[common]
    for m in specifications:
        pred = arrays[m['name']]
        assert pred.shape == (4427, len(m['grid'])) and np.isfinite(pred[common]).all()
        losses = np.array([[np.mean((y[fold==f]-pred[fold==f, j])**2) for f in range(5)]
                           for j in range(len(m['grid']))])
        for j, setting in enumerate(m['grid']):
            for f in range(5):
                np.testing.assert_allclose(losses[j, f], observed[m['name'], j, f]['MSE'], atol=1e-10, rtol=0)
            grid.append(dict(model=m['name'], setting=setting, grid=j,
                             mean_fold_MSE=float(losses[j].mean()), fold_MSE=losses[j].tolist()))
        chosen = int(losses.mean(1).argmin())
        score = metric(y[common], pred[common, chosen])
        selected.append(dict(model=m['name'], algorithm=m['algorithm'], site=m['site'],
            representation=m['representation'], selected_grid=chosen,
            selected_setting=m['grid'][chosen], **score, delta_R2_vs_v1_null=score['R2']-base['R2'],
            residual_MSE_reduction_vs_v1_null=1-score['MSE']/base['MSE'],
            fold_MSE=losses[chosen].tolist(), folds_improved_vs_v1_null=int(sum(
                losses[chosen, f] < np.mean((y[fold==f]-baseline[fold==f])**2) for f in range(5)))))
        held[m['name']] = pred[common, chosen]
    ridge = {(r['site'], r['representation']): r['R2'] for r in selected if r['algorithm']=='ridge'}
    for row in selected:
        row['delta_R2_vs_same_input_ridge'] = row['R2']-ridge[row['site'], row['representation']]
    for path, digest in source_hashes.items():
        assert sha(path) == digest, 'Source changed during run: '+path
    assert anchors == 60
    summary = dict(status='complete', baseline=base, models=selected, elapsed_seconds=elapsed,
        families=len(specifications), candidate_fits=expected, confirmation_outcomes_read=False,
        interpretation='Development screen; same folds used for tuning and reporting.')
    audit = dict(status='PASS', candidate_fits=expected, families=52, n_people=3343, n_households=3059,
        every_candidate_MSE_recomputed=True, source_hashes_verified=True,
        v2_anchor_predictions_verified=anchors, warning_fits=sum(bool(d['warnings']) for d in diagnostics))
    write_json(output/'summary.json', summary)
    write_json(output/'grid_scores.json', grid)
    write_json(output/'audit.json', audit)
    pd.DataFrame(selected).sort_values('R2', ascending=False).to_csv(output/'model_comparison.csv', index=False, mode='x')
    buffer = io.BytesIO()
    held.to_parquet(buffer, index=False)
    (output/'selected_predictions_TRE_ONLY.parquet').write_bytes(buffer.getvalue())
    (output/'ANALYSIS.md').write_text(report(summary, audit))
    print('SCREEN_AUDIT_PASS', json.dumps(audit), flush=True)
    print(pd.DataFrame(selected).sort_values('R2', ascending=False)[['model','R2','delta_R2_vs_same_input_ridge']].head(12).to_string(index=False), flush=True)


def main(args):
    start = time.time()
    source = Path(__file__).resolve().parent
    package, pilot, breadth = map(lambda s: Path(s).resolve(), (args.package, args.pilot_code, args.breadth_code))
    assert package.name == 'collaborator_handoff_20261102_v1'
    assert source.parent == package/'experiments'
    output = source/'results'
    sys.path.insert(0, str(pilot))
    from run_pilot import host_design
    spec = importlib.util.spec_from_file_location('existing_loader', package/'examples/load_fiber_fold.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    assert Path(loader.ROOT).resolve() == package
    # The previous breadth screen remains read-only and supplies exact replay anchors.
    assert json.loads((breadth/'postrun_audit.json').read_text())['status']=='PASS'
    with np.load(io.BytesIO((breadth/'results/all_candidate_predictions_TRE_ONLY.npz').read_bytes()), allow_pickle=False) as a:
        previous = {k: a[k] for k in a.files}
    output.mkdir(exist_ok=False)
    (output/'fits_TRE_ONLY').mkdir()
    files = [source/n for n in ('run_screen.py','verify_screen.py','launch.py','README.md')]
    files += [pilot/'run_pilot.py', pilot/'core.py', package/'examples/load_fiber_fold.py', breadth/'run_breadth.py']
    hashes = {str(p): sha(p) for p in files}
    models = specs()
    write_json(output/'registration.json', dict(models=models, source_sha256=hashes,
        code_commit=args.code_commit, python=sys.version, sklearn=sklearn.__version__, seed=SEED,
        started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        scope='v2-informed discovery follow-up; same-fold hyperparameter selection',
        confirmation_outcomes_read=False, PCA='unwhitened full SVD on training CLR; site-wise shared prefix maps'))
    pred = {m['name']: np.full((4427,len(m['grid'])), np.nan) for m in models}
    y_oof, base_oof, folds = np.full(4427,np.nan), np.full(4427,np.nan), np.full(4427,-1)
    identity, diagnostics, anchors = None, [], 0
    for f in range(5):
        d = loader.load_fold(f)
        tr, va, idx, y = d['train'], d['valid'], d['index'], d['y']
        assert len(idx)==4427 and idx.final_role.eq('discovery').all()
        assert int(va.sum())==[661,677,668,665,672][f]
        assert not set(idx.loc[tr,'split_unit_id']) & set(idx.loc[va,'split_unit_id'])
        current = idx[['input_row','split_unit_id','cv_fold']].copy()
        if identity is None:
            identity = current
        else:
            pd.testing.assert_frame_equal(identity,current)
        np.testing.assert_array_equal(previous['folds']==f, va)
        np.testing.assert_allclose(previous['y'][va],y[va],atol=0,rtol=0)
        y_oof[va], base_oof[va], folds[va] = y[va], previous['baseline'][va], f
        host, encoding = host_design(idx,tr,d['u'],d['h'])
        clr, pcs, maps, pca_info = {}, {}, {}, {}
        for site in ('oral','gut'):
            clr[site] = d[site+'_z']
            pcs[site], maps[site] = pca_map(clr[site],tr)
            pca_info[site] = dict(CLR_dimensions=clr[site].shape[1],
                explained_variance_by_k={str(k):float(maps[site].explained_variance_ratio_[:k].sum()) for k in (10,30,50)})
        ym, ys = float(y[tr].mean()), float(y[tr].std())
        write_json(output/f'fold{f}_encoding.json',dict(host=encoding,pca=pca_info,
            training_n=int(tr.sum()),validation_n=int(va.sum()),y_mean=ym,y_sd=ys))
        with (output/'fits_TRE_ONLY'/f'fold{f}_PCA.pkl').open('xb') as stream:
            pickle.dump(maps,stream,protocol=4)
        for m in models:
            xx = design(host,clr,pcs,m['site'],m['representation'])
            for j, setting in enumerate(m['grid']):
                tick = time.time()
                est = estimator(m['algorithm'],setting)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    est.fit(xx[tr],(y[tr]-ym)/ys)
                    pp = np.asarray(est.predict(xx[va])).ravel()*ys+ym
                assert np.isfinite(pp).all()
                pred[m['name']][va,j] = pp
                if m['site']=='host' or m['representation']=='CLR':
                    count = {'hist_boost':2,'random_forest':1}.get(m['algorithm'],0)
                    if j<count:
                        rep = 'PC10' if m['site']=='host' else 'CLR'
                        key = f"{m['algorithm']}_{rep}_{m['site']}"
                        np.testing.assert_allclose(pp,previous[key][va,j],atol=2e-8,rtol=0)
                        anchors += 1
                diagnostics.append(dict(fold=f,model=m['name'],grid=j,seconds=time.time()-tick,
                    warnings=[str(w.message) for w in caught],MSE=float(np.mean((y[va]-pp)**2))))
                with (output/'fits_TRE_ONLY'/f"fold{f}_{m['name']}_{j}.pkl").open('xb') as stream:
                    pickle.dump(dict(model=est,y_mean=ym,y_sd=ys,specification=m,grid=j),stream,protocol=4)
            print('MODEL_DONE',f,m['name'],flush=True)
        save_arrays(output/f'checkpoint_fold{f}_TRE_ONLY.npz',y=y_oof,baseline=base_oof,fold=folds,**pred)
        print('FOLD_DONE',f,'elapsed_s',round(time.time()-start),'anchors',anchors,flush=True)
    save_arrays(output/'candidate_predictions_TRE_ONLY.npz',y=y_oof,baseline=base_oof,fold=folds,**pred)
    write_json(output/'diagnostics.json',diagnostics)
    audit_and_summarize(output,models,identity,time.time()-start,hashes,anchors)
    print('SCREEN_COMPLETE',round(time.time()-start),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('package','pilot-code','breadth-code','code-commit'):
        parser.add_argument('--'+key,required=True)
    main(parser.parse_args())
