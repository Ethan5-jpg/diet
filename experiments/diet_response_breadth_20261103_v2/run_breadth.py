"""Broad discovery screen: prediction algorithms and explicit varying coefficients.

All participant-level material remains in TRE. Requires the existing v1 pilot
source and the collaborator fold interface. No confirmation outcomes are read.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.svm import SVR

SEED = 20261103
SITES = ('host', 'oral', 'gut', 'oral_gut')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=convert)
        stream.write('\n')


def specs():
    """Each family gets a small initial grid, rather than exhaustive tuning."""
    grids = {
        'random_forest': [dict(leaf=10), dict(leaf=40)],
        'extra_trees': [dict(leaf=10), dict(leaf=40)],
        'hist_boost': [dict(leaves=7, l2=10.), dict(leaves=15, l2=30.)],
        'gradient_boost': [dict(depth=2), dict(depth=3)],
        'rbf_svr': [dict(C=1.), dict(C=10.)],
        'rbf_features': [dict(alpha=1.), dict(alpha=30.)],
        'pls': [dict(k=3), dict(k=8)],
        'mlp': [dict(alpha=1.), dict(alpha=20.)],
        'additive_spline': [dict(alpha=10.), dict(alpha=100.)],
    }
    out = []
    for algorithm, grid in grids.items():
        # Both tree and smooth families start on the same PC10 interface.
        for site in SITES:
            out.append(dict(name=f'{algorithm}_PC10_{site}', track='prediction',
                            algorithm=algorithm, site=site, representation='PC10', grid=grid))
    # Tree splits can also find signals discarded by unsupervised PCA.
    for algorithm in ('random_forest', 'extra_trees', 'hist_boost'):
        for site in SITES[1:]:
            out.append(dict(name=f'{algorithm}_CLR_{site}', track='prediction',
                            algorithm=algorithm, site=site, representation='CLR', grid=grids[algorithm]))
    for representation in ('spline_PC10', 'rbf_PC10', 'rbf_CLR'):
        for site in SITES[1:]:
            out.append(dict(name=f'varying_{representation}_{site}', track='varying',
                            representation=representation, site=site,
                            grid=[dict(penalty=x) for x in (.01, .1, 1., 10.)]))
    return out


def estimator(algorithm, setting, width):
    if algorithm in ('random_forest', 'extra_trees'):
        cls = RandomForestRegressor if algorithm == 'random_forest' else ExtraTreesRegressor
        return cls(n_estimators=200, min_samples_leaf=setting['leaf'],
                   max_features=.5, n_jobs=2, random_state=SEED)
    if algorithm == 'hist_boost':
        return HistGradientBoostingRegressor(max_iter=160, learning_rate=.04,
            max_leaf_nodes=setting['leaves'], min_samples_leaf=30,
            l2_regularization=setting['l2'], early_stopping=False, random_state=SEED)
    if algorithm == 'gradient_boost':
        return GradientBoostingRegressor(n_estimators=160, learning_rate=.03,
            max_depth=setting['depth'], min_samples_leaf=30, random_state=SEED)
    if algorithm == 'rbf_svr':
        return make_pipeline(StandardScaler(), SVR(C=setting['C'], epsilon=.1, gamma='scale'))
    if algorithm == 'rbf_features':
        return make_pipeline(StandardScaler(), RBFSampler(gamma=1/max(width, 1),
            n_components=256, random_state=SEED), Ridge(alpha=setting['alpha']))
    if algorithm == 'pls':
        return PLSRegression(n_components=min(setting['k'], width), max_iter=1000)
    if algorithm == 'mlp':
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,),
            activation='tanh', solver='lbfgs', alpha=setting['alpha'], max_iter=600,
            max_fun=12000, tol=1e-4, early_stopping=False, random_state=SEED))
    if algorithm == 'additive_spline':
        return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3,
            knots='quantile', include_bias=False, extrapolation='linear'),
            Ridge(alpha=setting['alpha']))
    raise ValueError(algorithm)


def microbiome(data, site, representation):
    sites = ('oral', 'gut') if site == 'oral_gut' else (site,)
    if site == 'host':
        return np.empty((len(data['y']), 0))
    key = '_z' if representation.endswith('CLR') else '_PC'
    parts = [data[s+key] if key == '_z' else data[s+key][:, :10] for s in sites]
    return np.column_stack(parts)


def feature_map(z, training, representation):
    scaler = StandardScaler().fit(z[training])
    good = np.isfinite(z).all(1)
    transformed = scaler.transform(z[good])
    train_scaled = scaler.transform(z[training])
    if representation.startswith('spline'):
        mapping = SplineTransformer(n_knots=4, degree=3, knots='quantile',
            include_bias=False, extrapolation='linear').fit(train_scaled)
    else:
        mapping = RBFSampler(gamma=1/z.shape[1], n_components=128,
            random_state=SEED).fit(train_scaled)
    values = mapping.transform(transformed)
    raw = np.full((len(z), values.shape[1]), np.nan)
    raw[good] = values
    center = raw[training].mean(0)
    scale = raw[training].std(0)
    keep = scale > 1e-8
    phi = (raw[:, keep]-center[keep])/scale[keep]
    return phi, dict(scaler=scaler, mapping=mapping, center=center, scale=scale, keep=keep)


def shifted_host(host, encoding, index, u, h):
    result = host.copy()
    numeric = index[['age_calendar_years', 'sex_code', 'mean_daily_energy_fiber']].to_numpy(float)
    numeric = (numeric-encoding['numeric_means'])/encoding['numeric_scales']
    replacements = dict(diet_linear=np.full(len(host), u), diet_RCS=np.full(len(host), h),
                        diet_age=u*numeric[:, 0], diet_sex=u*numeric[:, 1])
    for column, label in enumerate(encoding['labels']):
        if label in replacements:
            result[:, column] = replacements[label]
    return result


def main(args):
    started = time.time()
    package, pilot = Path(args.package).resolve(), Path(args.pilot_code).resolve()
    source, output = Path(__file__).resolve().parent, Path(args.output).resolve()
    assert package.name == 'collaborator_handoff_20261102_v1'
    assert package/'experiments' in output.parents
    sys.path.insert(0, str(pilot))
    from core import Background, metrics, rcs, save_npz
    from run_pilot import host_design
    spec = importlib.util.spec_from_file_location('existing_loader', package/'examples/load_fiber_fold.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    assert Path(loader.ROOT).resolve() == package
    models = specs()
    output.mkdir(parents=True, exist_ok=False)
    fits = output/'fits_TRE_ONLY'
    fits.mkdir()
    hashes = {str(p): digest(p) for p in [source/'run_breadth.py', source/'verify_breadth.py',
        pilot/'core.py', pilot/'run_pilot.py', package/'examples/load_fiber_fold.py']}
    write_json(output/'registration.json', dict(status='registered_before_fits', models=models,
        source_sha256=hashes, code_commit=args.code_commit, python=sys.version,
        sklearn=sklearn.__version__, seed=SEED, scope='development_same_fold_selection',
        confirmation_outcomes_read=False, started_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    prior = package/'experiments/diet_response_heterogeneity_20261103_v1_attempt02/results'
    n = 4427
    prediction = {m['name']: np.full((n, len(m['grid'])), np.nan) for m in models}
    contrast = {m['name']: np.full((n, len(m['grid'])), np.nan) for m in models}
    matched = {m['name']: np.full(n, np.nan) for m in models if m['track'] == 'varying'}
    y_oof, baseline = np.full(n, np.nan), np.full(n, np.nan)
    all_folds = np.full(n, -1)
    diagnostics, supports = [], []
    identity = None
    for fold in range(5):
        d = loader.load_fold(fold)
        tr, va, index, y = d['train'], d['valid'], d['index'], d['y']
        assert len(index) == n and index.final_role.eq('discovery').all()
        assert int(va.sum()) == [661, 677, 668, 665, 672][fold]
        assert not set(index.loc[tr, 'split_unit_id']) & set(index.loc[va, 'split_unit_id'])
        this_identity = index[['input_row', 'split_unit_id', 'cv_fold']].copy()
        if identity is None:
            identity = this_identity
        else:
            pd.testing.assert_frame_equal(identity, this_identity)
        y_oof[va], all_folds[va] = y[va], fold
        host, encoding = host_design(index, tr, d['u'], d['h'])
        base = np.column_stack([host, d['oral_PC'][:, :5], d['gut_PC'][:, :5]])
        zfull = np.column_stack([d['oral_z'], d['gut_z']])
        background = Background(base, zfull, y, tr, 1.)
        baseline[va] = background.null()['prediction'][va]
        # Check the numerical baseline against the actually deployed prior fits.
        with np.load(io.BytesIO((prior/'fits_TRE_ONLY'/f'fold{fold}_ridge_PC10_gut_grid1.npz').read_bytes()), allow_pickle=False) as old:
            np.testing.assert_array_equal(old['validation'], va)
            previous = background.interaction(d['u'][:, None]*d['gut_PC'][:, :10]).ridge(.1)
            np.testing.assert_allclose(previous['prediction'][va], old['prediction'][va], rtol=0, atol=2e-8)
        diet = json.loads((package/f'preprocessing/folds/fold{fold}/diet_basis.json').read_text())['fiber_density']
        fiber = d['u']*diet['scale']+diet['center']
        support = np.quantile(fiber[tr], [.05, .95])
        supports.append(dict(fold=fold, q05=float(support[0]), q95=float(support[1]),
                             contrast_supported=bool(support[0] <= 10 < 15 <= support[1])))
        lo, hi = (np.array([10., 15.])-diet['center'])/diet['scale']
        hlo, hhi = rcs(np.array([lo, hi]), diet['knots'])
        host_lo = shifted_host(host, encoding, index, lo, hlo)
        host_hi = shifted_host(host, encoding, index, hi, hhi)
        # Fold-local y scaling also makes SVR/MLP regularization comparable.
        ym, ys = float(y[tr].mean()), float(y[tr].std())
        write_json(output/f'fold{fold}_encoding.json', dict(host=encoding, diet=diet,
            train_n=int(tr.sum()), valid_n=int(va.sum()), y_mean=ym, y_sd=ys))
        for m in models:
            name = m['name']
            z = microbiome(d, m['site'], m['representation'])
            if m['track'] == 'prediction':
                xx = np.column_stack([host, z])
                xlo, xhi = np.column_stack([host_lo, z]), np.column_stack([host_hi, z])
                for j, setting in enumerate(m['grid']):
                    begin = time.time()
                    model = estimator(m['algorithm'], setting, xx.shape[1])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(xx[tr], (y[tr]-ym)/ys)
                        pred = np.asarray(model.predict(xx[va])).ravel()*ys+ym
                        diff = (np.asarray(model.predict(xhi[va])).ravel()-np.asarray(model.predict(xlo[va])).ravel())*ys
                    assert np.isfinite(pred).all() and np.isfinite(diff).all()
                    prediction[name][va, j], contrast[name][va, j] = pred, diff
                    # Persist fitted objects only in TRE, using sequential writes.
                    import pickle
                    with (fits/f'fold{fold}_{name}_{j}.pkl').open('xb') as stream:
                        stream.write(pickle.dumps(dict(model=model, y_mean=ym, y_sd=ys,
                            host_encoding=encoding, specification=m, setting=setting), protocol=4))
                    diagnostics.append(dict(fold=fold, model=name, grid=j, seconds=time.time()-begin,
                        warnings=[str(w.message) for w in caught], MSE=float(np.mean((y[va]-pred)**2))))
            else:
                phi, mapping = feature_map(z, tr, m['representation'])
                # Include the SAME nonlinear microbial main effects in both models.
                bg = Background(base, np.column_stack([zfull, phi]), y, tr, 1.)
                matched[name][va] = bg.null()['prediction'][va]
                block = bg.interaction(d['u'][:, None]*phi)
                for j, setting in enumerate(m['grid']):
                    fit = block.ridge(setting['penalty'])
                    pred = fit['prediction'][va]
                    diff = (hi-lo)*(phi@fit['interaction_coef'])
                    diff -= diff[tr].mean()
                    assert np.isfinite(pred).all() and np.isfinite(diff[va]).all()
                    prediction[name][va, j], contrast[name][va, j] = pred, diff[va]
                    save_npz(fits/f'fold{fold}_{name}_{j}.npz', base_coef=fit['base_coef'],
                        main_coef=fit['main_coef'], interaction_coef=fit['interaction_coef'])
                    diagnostics.append(dict(fold=fold, model=name, grid=j, warnings=[],
                                            MSE=float(np.mean((y[va]-pred)**2))))
                import pickle
                with (fits/f'fold{fold}_{name}_map.pkl').open('xb') as stream:
                    stream.write(pickle.dumps(mapping, protocol=4))
            print('MODEL_DONE', fold, name, flush=True)
        # Persist checkpoint arrays after every complete fold; never append in TRE.
        save_npz(output/f'checkpoint_fold{fold}_TRE_ONLY.npz', y=y_oof,
                 baseline=baseline, folds=all_folds, **prediction)
        print('FOLD_DONE', fold, 'elapsed_s', round(time.time()-started), flush=True)
    mask = np.isfinite(y_oof)
    assert mask.sum() == 3343 and identity.loc[mask, 'split_unit_id'].nunique() == 3059
    base_score = metrics(y_oof[mask], baseline[mask])
    summary, per_setting = [], []
    held = identity.loc[mask].copy()
    held['Y_processed'], held['baseline_prediction'] = y_oof[mask], baseline[mask]
    for m in models:
        name = m['name']
        assert np.isfinite(prediction[name][mask]).all()
        fold_losses = np.array([[np.mean((y_oof[all_folds == f]-prediction[name][all_folds == f, j])**2)
                                for f in range(5)] for j in range(len(m['grid']))])
        chosen = int(fold_losses.mean(1).argmin())
        pred, diff = prediction[name][mask, chosen], contrast[name][mask, chosen]
        score = metrics(y_oof[mask], pred)
        row = dict(model=name, track=m['track'], site=m['site'], representation=m['representation'],
            selected_setting=m['grid'][chosen], **score, delta_R2_vs_v1_null=score['R2']-base_score['R2'],
            contrast_SD=float(diff.std()), contrast_q10=float(np.quantile(diff, .1)),
            contrast_q90=float(np.quantile(diff, .9)),
            contrast_kind='total_conditional_prediction' if m['track']=='prediction' else 'centered_microbiome_modifier',
            folds_improved_vs_v1_null=int(sum(np.mean((y_oof[all_folds==f]-prediction[name][all_folds==f, chosen])**2)
                < np.mean((y_oof[all_folds==f]-baseline[all_folds==f])**2) for f in range(5))))
        if m['track'] == 'varying':
            match_score = metrics(y_oof[mask], matched[name][mask])
            row['matched_no_interaction_R2'] = match_score['R2']
            row['interaction_delta_R2'] = score['R2']-match_score['R2']
            held[name+'_matched_null'] = matched[name][mask]
        summary.append(row)
        held[name+'_prediction'], held[name+'_contrast'] = pred, diff
        per_setting.extend(dict(model=name, grid=j, setting=m['grid'][j],
            mean_fold_MSE=float(v.mean()), fold_MSE=v.tolist()) for j, v in enumerate(fold_losses))
    write_json(output/'summary.json', dict(status='complete', baseline=base_score,
        model_families=len(models), candidate_fits=len(diagnostics), models=summary,
        elapsed_seconds=time.time()-started, contrast_support=supports,
        interpretation='Exploratory same-fold tuning. No formal selection-adjusted intervals.'))
    write_json(output/'diagnostics.json', diagnostics)
    write_json(output/'grid_scores.json', per_setting)
    table = pd.DataFrame(summary).sort_values('R2', ascending=False)
    with (output/'model_comparison.csv').open('x') as stream:
        table.to_csv(stream, index=False)
    buffer = io.BytesIO()
    held.to_parquet(buffer, index=False)
    with (output/'predictions_TRE_ONLY.parquet').open('xb') as stream:
        stream.write(buffer.getvalue())
    save_npz(output/'all_candidate_predictions_TRE_ONLY.npz', y=y_oof,
             baseline=baseline, folds=all_folds, **prediction)
    print('BREADTH_COMPLETE', 'families', len(models), 'fits', len(diagnostics), flush=True)
    print(table[['model', 'R2', 'delta_R2_vs_v1_null']].head(15).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--pilot-code', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--code-commit', required=True)
    main(parser.parse_args())
