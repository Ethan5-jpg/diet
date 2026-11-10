"""Bounded nested-household functional pilot. Native data never leave TRE."""
import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from functional import Design
from native import sha

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    os.replace(str(temporary), str(path))


def save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.stem+'.tmp-'+uuid.uuid4().hex+'.npz')
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def design(base_engine, data, train, valid, config):
    data.dataset.api.separated(data.dataset.index.split_unit_id.astype(str).to_numpy(), train, valid)
    base = base_engine.design(data.dataset, train, valid)
    s = base.role.split
    if not np.array_equal(s.train_rows, np.flatnonzero(train)) or not np.array_equal(s.rows, np.flatnonzero(valid)):
        raise ValueError('Predefined functional members changed during frozen preprocessing')
    return base


def wrapper(base, data, route, config):
    role = 'coverage' if route == 'coverage_confidence' else 'abundance'
    return Design(base, route, data.signals[role], data.technical, config, data.frozen.features)


def outer(data, config, repeat, fold, log=print):
    ds, frozen = data.dataset, data.frozen
    groups = ds.index.split_unit_id.astype(str).to_numpy()
    assignment = ds.reference['outer'][repeat]
    train = data.available & (assignment != fold)
    valid = data.available & (assignment == fold)
    ds.api.separated(groups, train, valid)
    inner = np.full(len(groups), -1, int)
    inner[train] = ds.api.folds(groups[train], config['inner_folds'], config['outer_seed']+100*repeat+fold+1)
    engine = frozen.engine.Engine(frozen.config, None)
    log('START r%d_f%d inner training roles' % (repeat, fold))
    bases = [design(engine, data, train & (inner != k), train & (inner == k), config) for k in range(config['inner_folds'])]
    target = np.concatenate([b.role.split.yv for b in bases])
    selections = {}
    for route in config['routes']:
        views = [wrapper(base, data, route, config) for base in bases]
        predictions = [np.column_stack([frozen.models.fit_main(v, p).predict() for p in config['main_penalties']]) for v in views]
        loss = np.mean((target[:,None]-np.concatenate(predictions))**2, axis=0)
        chosen = int(loss.argmin())
        main_penalty = config['main_penalties'][chosen]
        spec = dict(name=route)
        candidates = [None]+config['interaction_penalties']
        augmented = [np.column_stack([predictions[k][:,chosen]] + [frozen.models.fit_augmented(v, spec, main_penalty, p).predict() for p in config['interaction_penalties']]) for k,v in enumerate(views)]
        interaction_loss = np.mean((target[:,None]-np.concatenate(augmented))**2, axis=0)
        selections[route] = dict(main_penalty=main_penalty, interaction_penalty=candidates[int(interaction_loss.argmin())],
                                 inner_main_MSE=loss.tolist(), inner_interaction_MSE=interaction_loss.tolist())
    del bases, views, predictions, augmented, target
    log('FIT r%d_f%d outer models; selection fixed' % (repeat, fold))
    base = design(engine, data, train, valid, config)
    s = base.role.split
    arrays = dict(rows=s.rows, y=s.yv, raw_y=ds.y[s.rows])
    main, full, total, delta, component, supported = [], [], [], [], [], []
    audits = {}
    for route in config['routes']:
        view = wrapper(base, data, route, config)
        setting = selections[route]
        baseline = frozen.models.fit_main(view, setting['main_penalty'])
        model = frozen.models.fit_augmented(view, dict(name=route), setting['main_penalty'], setting['interaction_penalty'], baseline)
        main.append(baseline.predict())
        full.append(model.predict())
        baseline_contrast, _ = baseline.contrasts()
        contrast, raw_component = model.contrasts()
        total.append(contrast)
        delta.append(contrast-baseline_contrast)
        component.append(raw_component)
        supported.append(view.support())
        audits[route] = view.function.audit(data.labels)
    arrays.update(main=np.asarray(main), interaction=np.asarray(full), total_contrast=np.asarray(total),
                  delta_contrast=np.asarray(delta), raw_component=np.asarray(component), support=np.asarray(supported, bool))
    if not all(np.isfinite(arrays[k]).all() for k in ('y','raw_y','main','interaction','total_contrast','delta_contrast','raw_component')):
        raise ValueError('Nonfinite outer values')
    metadata = dict(repeat=repeat, fold=fold, people=len(s.rows), train_people=len(s.train_rows), selections=selections, function_audits=audits,
                    endpoint_preprocessing=s.preprocessing.get('target',{}) if hasattr(s,'preprocessing') else dict(synthetic_only=True))
    log('OUTER_COMPLETE r%d_f%d people=%d routes=%d' % (repeat, fold, len(s.rows), len(config['routes'])))
    return arrays, metadata


def checkpoint(private, tag, identity_hash, compute):
    complete = private/(tag+'_COMPLETE.json')
    array_path, meta_path = private/(tag+'.npz'), private/(tag+'.json')
    if complete.exists():
        proof = json.loads(complete.read_text())
        if proof['identity_hash'] != identity_hash or proof['arrays_sha256'] != sha(array_path) or proof['metadata_sha256'] != sha(meta_path):
            raise ValueError('Checkpoint identity or checksum mismatch')
        with np.load(str(array_path), allow_pickle=False) as archive:
            return {name:archive[name] for name in archive.files}, json.loads(meta_path.read_text())
    start = time.monotonic()
    arrays, metadata = compute()
    metadata['seconds'] = time.monotonic()-start
    save_arrays(array_path, arrays)
    write_json(meta_path, metadata)
    write_json(complete, dict(identity_hash=identity_hash, arrays_sha256=sha(array_path), metadata_sha256=sha(meta_path)))
    return arrays, metadata


def metric(y, prediction):
    sst = float(np.sum((y-y.mean())**2))
    if not np.isfinite(sst) or sst <= 0:
        raise ValueError('Outcome variance unavailable')
    mse = float(np.mean((y-prediction)**2))
    return dict(R2=1-len(y)*mse/sst, MSE=mse)


def paired_intervals(y, main, full, groups, draws, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 30:
        raise ValueError('Too few households for conditional bootstrap')
    delta = (y[None,:]-main)**2-(y[None,:]-full)**2
    table = np.column_stack([np.bincount(inverse), np.bincount(inverse,weights=y), np.bincount(inverse,weights=y*y)] + [np.bincount(inverse,weights=v) for v in delta])
    rng = np.random.default_rng(seed)
    result = []
    for begin in range(0,draws,100):
        weights = rng.multinomial(len(unique), np.full(len(unique),1./len(unique)), size=min(100,draws-begin))
        totals = weights @ table
        denominator = totals[:,2]-totals[:,1]**2/totals[:,0]
        good = denominator > 0
        result.append(totals[good,3:]/denominator[good,None])
    samples = np.concatenate(result)
    if len(samples) < .95*draws:
        raise ValueError('Insufficient nondegenerate household draws')
    return np.quantile(samples,[.025,.975],axis=0).T


def summarize(data, config, chunks):
    n, routes = len(data.available), config['routes']
    names = ('y','raw_y','main','interaction','total_contrast','delta_contrast','raw_component','support')
    all_arrays = {}
    for name in names:
        shape = (config['outer_repeats'],n) if name in ('y','raw_y') else (config['outer_repeats'],len(routes),n)
        all_arrays[name] = np.full(shape,np.nan)
    seen = np.zeros((config['outer_repeats'],n),int)
    selected = {name:[0]*config['outer_repeats'] for name in routes}
    for arrays, meta in chunks:
        repeat, rows = meta['repeat'], arrays['rows']
        seen[repeat,rows] += 1
        for name in names:
            if name in ('y','raw_y'):
                all_arrays[name][repeat,rows] = arrays[name]
            else:
                all_arrays[name][repeat][:,rows] = arrays[name]
        for route in routes:
            selected[route][repeat] += meta['selections'][route]['interaction_penalty'] is not None
    if not np.all(seen[:,data.available] == 1) or np.any(seen[:,~data.available]):
        raise ValueError('Outer evaluation membership missing or duplicated')
    groups = data.dataset.index.loc[data.available,'split_unit_id'].astype(str).to_numpy()
    scores = []
    for repeat in range(config['outer_repeats']):
        y = all_arrays['y'][repeat,data.available]
        main = all_arrays['main'][repeat][:,data.available]
        full = all_arrays['interaction'][repeat][:,data.available]
        intervals = paired_intervals(y,main,full,groups,config['prediction_bootstraps'],config['seed']+repeat)
        for j,route in enumerate(routes):
            baseline, augmented = metric(y,main[j]), metric(y,full[j])
            raw_y = all_arrays['raw_y'][repeat,data.available]
            raw_delta = metric(raw_y,full[j])['R2']-metric(raw_y,main[j])['R2']
            scores.append(dict(route=route,repeat=repeat,N=int(data.available.sum()),baseline=baseline,interaction=augmented,
                delta_R2=augmented['R2']-baseline['R2'],MSE_reduction=baseline['MSE']-augmented['MSE'],
                conditional_household_delta_R2_CI95=intervals[j].tolist(),raw_y_same_prediction_delta_R2=raw_delta,
                interaction_selected_folds=selected[route][repeat],
                total_model_contrast_SD=float(all_arrays['total_contrast'][repeat,j,data.available].std()),
                added_model_contrast_SD=float(all_arrays['delta_contrast'][repeat,j,data.available].std()),
                local_contrast_support_fraction=float(all_arrays['support'][repeat,j,data.available].mean())))
    stability = {}
    for j,route in enumerate(routes):
        a,b = [all_arrays['delta_contrast'][r,j,data.available] for r in (0,1)]
        stability[route] = float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None
    return dict(status='COMPLETE',synthetic_only=data.synthetic_only,claim_type=config['claim_type'],
                scope='Same discovery cohort; fixed predictions conditional intervals; no formal interaction P value or causal response.',
                population=data.audit,primary_route=config['primary_route'],primary_repeat=0,scores=scores,
                same_people_added_contrast_split_correlation=stability,
                historical_discovery_search=True,independent_confirmation=False,confirmation_outcomes_read=False), all_arrays


def report(summary):
    print(('SYNTHETIC ONLY' if summary['synthetic_only'] else 'DISCOVERY FUNCTION PILOT')+' | fiber_density -> D_Glu (mg/dL)')
    print('Population:',json.dumps(summary['population'],sort_keys=True))
    print('PRIMARY abundance_relative, repeat 0; other encodings and repeat 1 are sensitivity only.')
    print('route repeat base_R2 full_R2 delta_R2 conditional_HH_CI95 selected_folds rawY_delta support added_contrast_SD')
    for row in summary['scores']:
        lo,hi = row['conditional_household_delta_R2_CI95']
        print('%s %d %.6f %.6f %+.6f [%+.6f,%+.6f] %d %+.6f %.3f %.3f' % (
            row['route'],row['repeat'],row['baseline']['R2'],row['interaction']['R2'],row['delta_R2'],lo,hi,
            row['interaction_selected_folds'],row['raw_y_same_prediction_delta_R2'],row['local_contrast_support_fraction'],row['added_model_contrast_SD']))
    print('Same-person split contrast correlation:',json.dumps(summary['same_people_added_contrast_split_correlation'],sort_keys=True))
    print('Matched baselines are refit within each route on the same functional subset; do not compare with old N=3343 R2.')
    print('CIs resample households with OOF predictions fixed; no learning/selection uncertainty or formal interaction P value.')
    print('Null encoding is an assumption; contrasts are model-generated associations, not measured individual diet responses.')
    print('Eligibility stays fixed; outcome clipping limits use training roles only, with no new validation-outcome exclusions.')


def execute(data, config, run, threads=2, resume=False, log=print):
    run = Path(run)
    run.mkdir(parents=True,exist_ok=True)
    private = run/'TRE_ONLY'
    private.mkdir(exist_ok=True)
    code = {str(p):sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.sh') and p.name not in ('terminal_install_and_run.sh','verification.json')}
    identity = dict(config=config,code_hashes=code,source_hashes=data.source_hashes,versions=data.versions,
                    python=sys.version,threads=threads,population=data.audit,synthetic_only=data.synthetic_only)
    identity_hash = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    registration, lock = run/'registration.json', run/'RUNNING.lock'
    if registration.exists():
        if json.loads(registration.read_text()) != identity:
            raise ValueError('Resume source/config/environment identity mismatch')
        if (run/'summary.json').exists():
            summary = json.loads((run/'summary.json').read_text())
            proof = json.loads((run/'COMPLETE.json').read_text())
            if proof['identity_hash'] != identity_hash or proof['summary_sha256'] != sha(run/'summary.json') or proof['arrays_sha256'] != sha(private/'final_predictions.npz') or proof['cohort_flags_sha256'] != sha(private/'cohort_flags.npz'):
                raise ValueError('Completed summary checksum mismatch')
            return summary
        if not resume:
            raise ValueError('Existing incomplete run; use identical sources with --resume')
    else:
        if resume:
            raise ValueError('No registered run to resume')
        write_json(registration,identity)
        (run/'source_audit.txt').write_text(data.audit_text)
    if lock.exists():
        previous = json.loads(lock.read_text())
        if not resume or previous['host'] != socket.gethostname():
            raise RuntimeError('Run lock exists; inspect its owner')
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError('Run lock owner still exists')
    with lock.open('x') as stream:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname()),stream)
    chunks=[]
    try:
        flag_path=private/'cohort_flags.npz'
        expected_flags=dict(original_eligible=np.isfinite(data.dataset.reference['y'][0]),functional_available=data.available,
                            original_eligible_without_function=np.isfinite(data.dataset.reference['y'][0]) & ~data.available)
        if flag_path.exists():
            with np.load(str(flag_path),allow_pickle=False) as previous_flags:
                if any(not np.array_equal(previous_flags[n],v) for n,v in expected_flags.items()):
                    raise ValueError('Stored cohort flags differ from registered population')
        else:
            save_arrays(flag_path,expected_flags)
        with threadpool_limits(limits=threads):
            for repeat in range(config['outer_repeats']):
                for fold in range(config['outer_folds']):
                    tag='r%d_f%d' % (repeat,fold)
                    chunks.append(checkpoint(private,tag,identity_hash,lambda repeat=repeat,fold=fold:outer(data,config,repeat,fold,log)))
            for path,digest in data.source_hashes.items():
                if sha(path) != digest:
                    raise ValueError('Source changed during run; summary not published')
            summary, arrays = summarize(data,config,chunks)
            save_arrays(private/'final_predictions.npz',arrays)
            write_json(run/'summary.json',summary)
            write_json(run/'COMPLETE.json',dict(identity_hash=identity_hash,summary_sha256=sha(run/'summary.json'),arrays_sha256=sha(private/'final_predictions.npz'),cohort_flags_sha256=sha(flag_path)))
    except BaseException:
        raise
    else:
        lock.unlink()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
    parser.add_argument('--raw-root',default='/home/ec2-user/studies/hpp_datasets')
    parser.add_argument('--run-id',default='function_20261110_v1')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.run_id) or not 1 <= args.threads <= 8:
        raise ValueError('Invalid run ID or thread count')
    run=ROOT/'runs'/args.run_id
    if args.report:
        proof=json.loads((run/'COMPLETE.json').read_text())
        if sha(run/'summary.json') != proof['summary_sha256']:
            raise ValueError('Summary checksum mismatch')
        report(json.loads((run/'summary.json').read_text()))
        return
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).name != name or sha(ROOT/name) != digest:
            raise ValueError('Functional source manifest mismatch')
    config=json.loads((ROOT/'config.json').read_text())
    from native import load
    print('START native functional pilot; source audit first; no confirmation outcomes.',flush=True)
    data=load(args.project,args.raw_root,config)
    print('SOURCE_READY functional_people=%d original_people=%d' % (data.audit['functional_people'],data.audit['original_people']),flush=True)
    summary=execute(data,config,run,args.threads,args.resume,lambda message:print(message,flush=True))
    report(summary)


if __name__ == '__main__':
    main()
