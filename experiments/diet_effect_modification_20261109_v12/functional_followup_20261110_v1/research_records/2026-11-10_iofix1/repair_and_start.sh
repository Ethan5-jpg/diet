#!/usr/bin/env bash
bash <<'HPP_FUNCTION_IO_REPAIR1'
set -euo pipefail
hpp_io_root="${HPP_FUNCTION_IO_ROOT:-/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/functional_followup_20261110_v1}"
cd "$hpp_io_root"
hpp_io_python=python
if [ -s ../.python_path ]; then
    IFS= read -r hpp_io_python < ../.python_path
fi
"$hpp_io_python" -B - <<'HPP_IO_PREFLIGHT'
import hashlib,json,os,socket
from pathlib import Path
root=Path.cwd()
OLD=json.loads(r'''{"files": {"IMPLEMENTATION_PLAN.md": "ba8fb78c33145fe08eac0ce7895519b59a6463e6cd87c64a6ccbb24bd7ca58fb", "PROTOCOL.md": "bb845b41d395cf6748a401e7077b1c86f52432e2f16c4bfc815326c1091ab123", "README.md": "3be7666891a8a3e2a5bf1f716838f5ac05934a8de0eb460dd0db38557b91ed03", "audit_driver.py": "0c2186ecfc4b42456437f5145847addf807d4cff9979088e30f57bcccfbcf92c", "check_synthetic.py": "10e90a306d1c64fedb5a577193adfc33aa6391a549fe9e040a97d1d9b4b6655d", "config.json": "5f4fe8a03443b1623a9c2a980e3e8aadf3188148ab65290982ddf3cd898e3102", "demo_data.py": "4a25c6af3289ed250f42307ffa1ccebbe9ddc72abee81f293b0ebedf94a2efab", "demo_files.py": "ec6f1e5a4f8a972b1473c425827e7a4424a15e4cfaa5d4f11cfa4d9a9c57b965", "frozen.py": "783382e5bae6ddd26d97226569867be69ec7e7cacfd2ed29cbc32fe812c52eea", "functional.py": "fcefca3fa10d03ac95abe9b286c573dfbcbd280371ed8d3fc5e7dc7c280d3d3a", "native.py": "761492bdf30fb074d18d48a159a93d239089fe6ecd79e31927beab00dc271884", "pilot.py": "46417d533e03873e58214fcc2029e8ab2d8016b7d5310792c26b71361557a471", "server_run.sh": "8e7d81c72f09a45d9dc5a55e0772410a007dad2bf48e19d6018c5ab9461cc1fd"}, "native_HPP_inputs_embedded": false, "original_V12_files_modified": false, "version": "functional_followup_20261110_v1"}''')
NEW=json.loads(r'''{"analysis_configuration_unchanged": true, "files": {"IMPLEMENTATION_PLAN.md": "ba8fb78c33145fe08eac0ce7895519b59a6463e6cd87c64a6ccbb24bd7ca58fb", "PROTOCOL.md": "bb845b41d395cf6748a401e7077b1c86f52432e2f16c4bfc815326c1091ab123", "README.md": "e9b1879e6af5c3999c33e4090b3b44c93b506f1b78ce5aaa0496ef34c48e3158", "audit_driver.py": "0c2186ecfc4b42456437f5145847addf807d4cff9979088e30f57bcccfbcf92c", "check_synthetic.py": "a41f7f96ab80dc55204ed14511a456e1725d14b65cca65dad70804d258f31f8b", "config.json": "5f4fe8a03443b1623a9c2a980e3e8aadf3188148ab65290982ddf3cd898e3102", "demo_data.py": "4a25c6af3289ed250f42307ffa1ccebbe9ddc72abee81f293b0ebedf94a2efab", "demo_files.py": "ec6f1e5a4f8a972b1473c425827e7a4424a15e4cfaa5d4f11cfa4d9a9c57b965", "frozen.py": "783382e5bae6ddd26d97226569867be69ec7e7cacfd2ed29cbc32fe812c52eea", "functional.py": "fcefca3fa10d03ac95abe9b286c573dfbcbd280371ed8d3fc5e7dc7c280d3d3a", "native.py": "761492bdf30fb074d18d48a159a93d239089fe6ecd79e31927beab00dc271884", "pilot.py": "5740847289b97775e53ea90e10bd68b19aa75a197ee770360e5d287bc38c4ffd", "server_run.sh": "8e7d81c72f09a45d9dc5a55e0772410a007dad2bf48e19d6018c5ab9461cc1fd"}, "native_HPP_inputs_embedded": false, "original_V12_files_modified": false, "storage_repair": "In-memory ZIP assembly followed by sequential hash-checked writes; fresh IO-fix run ID preserves failed registration.", "version": "functional_followup_20261110_v1_iofix1"}''')
OLD_MANIFEST_SHA='5e5ac520f43b6bc68fd9448f85b460ce3a483b01c3dc17a623ce59f6924c2280'
NEW_MANIFEST_SHA='3a99e664252c1ecd82c48d3c0cabb2f7098f03bd8724c18f84a684eb2d7820bd'
CHANGED=['pilot.py', 'check_synthetic.py', 'README.md']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def stop(message):
    raise SystemExit('IO_REPAIR_STOP: '+message)
if root.name!='functional_followup_20261110_v1':
    stop('unexpected module directory')
for name in OLD['files']:
    path=root/name
    if path.is_symlink() or not path.is_file() or digest(path) not in (OLD['files'][name],NEW['files'][name]):
        stop('unknown source bytes: '+name)
manifest=root/'manifest.json'
if manifest.is_symlink() or digest(manifest) not in (OLD_MANIFEST_SHA,NEW_MANIFEST_SHA):
    stop('unknown source manifest')
oldrun=root/'runs/function_20261110_v1'
newrun=root/'runs/function_20261110_v1_iofix1'
for run in (oldrun,newrun):
    lock=run/'RUNNING.lock'
    if lock.exists():
        value=json.loads(lock.read_text())
        if value.get('host')!=socket.gethostname():
            stop('cannot verify a lock from another host')
        pid=value.get('pid')
        if not isinstance(pid,int) or pid<=0:
            stop('invalid lock owner')
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            pass
        except PermissionError:
            stop('cannot verify lock owner')
        else:
            stop('a run process is still active; no sources changed')
if (oldrun/'COMPLETE.json').exists() or (oldrun/'summary.json').exists() or list((oldrun/'TRE_ONLY').glob('r*_f*_COMPLETE.json')):
    stop('old run already has completed scientific work; do not replace its source identity')
registration=oldrun/'registration.json'
if registration.exists():
    value=json.loads(registration.read_text())
    if value.get('config')!=json.loads((root/'config.json').read_text()) or value.get('synthetic_only') is not False:
        stop('old run configuration/scope differs')
archive=root/'research_records/2026-11-10_iofix1/server_previous_sources'
if archive.is_symlink():
    stop('source backup directory is a symlink')
archive.mkdir(parents=True,exist_ok=True)
for name in CHANGED+['manifest.json']:
    path=root/name
    expected=OLD_MANIFEST_SHA if name=='manifest.json' else OLD['files'][name]
    backup=archive/name
    if backup.is_symlink() or backup.exists() and digest(backup)!=expected:
        stop('unexpected previous-source backup: '+name)
    if digest(path)==expected and not backup.exists():
        with backup.open('xb') as stream:
            stream.write(path.read_bytes())
        if digest(backup)!=expected:
            stop('previous-source backup verification failed')
state=dict(old_run_id='function_20261110_v1',new_run_id='function_20261110_v1_iofix1',
    old_registration_sha256=digest(registration) if registration.exists() else None,
    old_lock_sha256=digest(oldrun/'RUNNING.lock') if (oldrun/'RUNNING.lock').exists() else None,
    statistical_config_sha256=digest(root/'config.json'),old_completed_fold_count=0,old_summary_present=False)
state_path=archive.parent/'original_failed_run_state.json'
if state_path.exists():
    if json.loads(state_path.read_text())!=state:
        stop('failed-run state changed since initial repair guard')
else:
    with state_path.open('x') as stream:
        stream.write(json.dumps(state,sort_keys=True,indent=2)+'\n')
print('IO_REPAIR_PREFLIGHT_OK: known sources, no completed old folds, no active owner; failed run retained.')
HPP_IO_PREFLIGHT
hpp_io_write() {
    local hpp_name="$1"
    local hpp_old="$2"
    local hpp_new="$3"
    local hpp_temporary
    hpp_temporary="$(mktemp "$PWD/.iofix-source.XXXXXXXX")"
    cat > "$hpp_temporary"
    "$hpp_io_python" -B - "$hpp_name" "$hpp_old" "$hpp_new" "$hpp_temporary" <<'HPP_IO_SOURCE'
import hashlib,os,sys
from pathlib import Path
name,old,new,incoming=sys.argv[1],sys.argv[2],sys.argv[3],Path(sys.argv[4])
try:
    target=Path.cwd()/name
    if Path(name).name!=name or target.is_symlink():
        raise SystemExit('IO_REPAIR_STOP: invalid source target')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(incoming)!=new:
        raise SystemExit('IO_REPAIR_STOP: incoming checksum mismatch')
    current=digest(target)
    if current==new:
        pass
    elif current==old:
        backup=Path.cwd()/'research_records/2026-11-10_iofix1/server_previous_sources'/name
        if not backup.is_file() or digest(backup)!=old:
            raise SystemExit('IO_REPAIR_STOP: old source backup missing')
        os.replace(str(incoming),str(target))
        if digest(target)!=new:
            raise SystemExit('IO_REPAIR_STOP: source readback mismatch')
    else:
        raise SystemExit('IO_REPAIR_STOP: source changed after preflight')
finally:
    if incoming.exists():
        incoming.unlink()
HPP_IO_SOURCE
}
hpp_io_write 'pilot.py' '46417d533e03873e58214fcc2029e8ab2d8016b7d5310792c26b71361557a471' '5740847289b97775e53ea90e10bd68b19aa75a197ee770360e5d287bc38c4ffd' <<'HPP_IO_SOURCE_00_END'
"""Bounded nested-household functional pilot. Native data never leave TRE."""
import argparse
import hashlib
import io
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
    # ZIP needs backwards seeks while it is assembled; the mounted destination
    # may only support sequential writes. Finish the archive before opening it.
    with io.BytesIO() as memory:
        np.savez_compressed(memory, **arrays)
        payload = memory.getvalue()
    try:
        with temporary.open('xb') as stream:
            written = stream.write(payload)
        if written != len(payload) or sha(temporary) != hashlib.sha256(payload).hexdigest():
            raise OSError('Sequential checkpoint readback mismatch')
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


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
    parser.add_argument('--run-id',default='function_20261110_v1_iofix1')
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
HPP_IO_SOURCE_00_END
hpp_io_write 'check_synthetic.py' '10e90a306d1c64fedb5a577193adfc33aa6391a549fe9e040a97d1d9b4b6655d' 'a41f7f96ab80dc55204ed14511a456e1725d14b65cca65dad70804d258f31f8b' <<'HPP_IO_SOURCE_01_END'
"""Deterministic checks with generated inputs only; no native HPP reader calls."""
import copy
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import native
from demo_data import make
from frozen import modules
from functional import Design, FixedMemberTarget, FunctionState, TechnicalState
from pilot import checkpoint, execute, outer, paired_intervals, report, save_arrays, sha, summarize

ROOT=Path(__file__).resolve().parent


def config():
    c=json.loads((ROOT/'config.json').read_text())
    c.update(outer_folds=3,inner_folds=2,prediction_bootstraps=99,functional_main_components=5,
             main_penalties=[.01,.1],interaction_penalties=[.03,.3],support_neighbors=40)
    return c


def expect_error(action, fragment):
    try:
        action()
    except (ValueError,RuntimeError) as error:
        assert fragment in str(error),(fragment,str(error))
    else:
        raise AssertionError('Expected rejection: '+fragment)


def check_sequential_archive(passed):
    original_open=io.open
    writers=[]
    class ForwardOnly:
        def __init__(self,stream,fail_write=False):
            self.stream,self.fail_write=stream,fail_write
            self.seeks=[]
            self.backward_rejected=False
        def __getattr__(self,name):
            return getattr(self.stream,name)
        def __enter__(self):
            return self
        def __exit__(self,*args):
            self.stream.close()
        def seek(self,offset,whence=0):
            self.seeks.append((offset,whence))
            target=offset if whence==0 else self.stream.tell()+offset if whence==1 else os.fstat(self.stream.fileno()).st_size+offset
            if target < self.stream.tell() and not self.backward_rejected:
                # Permit later destructor cleanup after reproducing the first
                # unsupported backwards seek, avoiding ignored-finalizer noise.
                self.backward_rejected=True
                raise OSError(95,'Simulated unsupported backwards write seek')
            return self.stream.seek(offset,whence)
        def write(self,payload):
            if self.fail_write:
                self.stream.write(payload[:5])
                raise OSError(5,'Simulated interrupted sequential write')
            return self.stream.write(payload)
    def hook(fail_write=False):
        def opened(path,mode='r',*args,**kwargs):
            result=original_open(path,mode,*args,**kwargs)
            if isinstance(path,(str,os.PathLike)) and str(path).endswith('.npz') and any(c in mode for c in 'wxa'):
                result=ForwardOnly(result,fail_write)
                writers.append(result)
            return result
        return opened
    with tempfile.TemporaryDirectory(prefix='hpp_npz_forward_only_synthetic_') as tmp:
        folder=Path(tmp)
        arrays=dict(flags=np.array([True,False,True]),values=np.arange(18,dtype=float).reshape(6,3))
        reproduced=False
        with patch('io.open',hook()):
            try:
                np.savez_compressed(str(folder/'old_direct.npz'),**arrays)
            except OSError as error:
                reproduced=error.errno==95
        passed('original_path_zip_writer_reproduces_errno95',reproduced and any(w.backward_rejected for w in writers))
        writers.clear()
        target=folder/'new_sequential.npz'
        with patch('io.open',hook()):
            save_arrays(target,arrays)
        passed('new_archive_writer_never_seeks_target_file',bool(writers) and all(not w.seeks for w in writers))
        with np.load(str(target),allow_pickle=False) as loaded:
            passed('sequential_npz_roundtrip_preserves_flags_values_and_shapes',all(np.array_equal(loaded[name],value) and loaded[name].dtype==value.dtype for name,value in arrays.items()))
        before=target.read_bytes()
        failed=False
        with patch('io.open',hook(True)):
            try:
                save_arrays(target,dict(flags=np.array([False])))
            except OSError as error:
                failed=error.errno==5
        passed('interrupted_write_preserves_completed_file_and_cleans_temporary',failed and target.read_bytes()==before and not list(folder.glob('new_sequential.tmp-*')))


def native_adapter_fixture():
    """Run the copied source audit on generated files, mock only the old model reader."""
    import demo_files as fixture
    with tempfile.TemporaryDirectory(prefix='hpp_function_adapter_synthetic_') as tmp:
        state=fixture.fixture(Path(tmp))
        for which,path in enumerate(state['paths']):
            with pa.memory_map(str(path),'r') as stream:
                table=ipc.open_file(stream).read_all()
            rows=table.to_pylist()
            for row in rows:
                if row['participant_id']==101:
                    for name in ('PWY-1234:demo','GLYCOLYSIS-PWY:demo','VALID-OTHER-FEATURE'):
                        row[name]=None
                if row['participant_id']==102:
                    row['GLYCOLYSIS-PWY:demo']=None
            result=pa.Table.from_pylist(rows,schema=table.schema)
            with pa.OSFile(str(path),'wb') as out:
                with ipc.new_file(out,result.schema) as writer:
                    writer.write_table(result)
        index=pq.ParquetFile(state['cov']).read().to_pandas(ignore_metadata=True)
        with np.load(str(state['reference']),allow_pickle=False) as archive:
            reference=dict(y=archive['y'])
        dataset=SimpleNamespace(index=index,reference=reference,paths=[state['cov']],y=np.arange(5,dtype=float),api=SimpleNamespace(PhenotypeOutliers=lambda:None))
        frozen=modules()
        gate=Path(tmp)/'existing_summary.json'
        gate.write_text('{"synthetic_only":true}')
        fake=SimpleNamespace(root=frozen.root,config=frozen.config,upload_check=frozen.upload_check,
            runtime=SimpleNamespace(environment=lambda c:dict(synthetic_only=True),readiness=lambda *a,**k:dict(v11_summary=str(gate))),
            bridge=SimpleNamespace(load=lambda *a:dataset,adapter_sources=lambda:[]))
        before=native.modules
        old_handler=signal.getsignal(signal.SIGALRM)
        native.modules=lambda:fake
        try:
            cfg=config()
            cfg.update(original_people=3,functional_people=2,candidate_fields=3,source_input_rows=5,original_households=2)
            data=native.load(state['project'],state['raw'],cfg)
        finally:
            native.modules=before
        assert signal.getsignal(signal.SIGALRM)==old_handler
        assert np.flatnonzero(data.available).tolist()==[1,2]
        assert np.isnan(data.signals['abundance'][0]).all()
        assert data.signals['abundance'][1,1] == 2.  # sorted public labels: glycolysis, PWY, other
        assert data.audit['excluded_whole_candidate_null']==1
        assert 'PATHWAY_NULL_CONTRACT_DONE' in data.audit_text
        assert not data.synthetic_only  # native adapter path was exercised, all files were generated


def main():
    checks=[]
    def passed(name, condition=True):
        assert condition,name
        checks.append(name)
    cfg=config()
    check_sequential_archive(passed)
    class OriginalTarget:
        def fit(self,y,participant_ids,*,endpoint,role='training'):
            self.state=dict(kind='synthetic_original',clip_low=-5.,clip_high=5.)
            return self
        def transform(self,y,*,endpoint):
            a=np.asarray(y,float)
            return dict(values=np.where(abs(a)>8,np.nan,np.clip(a,-5,5)),excluded_8sd=abs(a)>8)
    target=FixedMemberTarget(OriginalTarget()).fit(np.array([-1.,0.,1.]),[0,1,2],endpoint='D_Glu')
    transformed=target.transform(np.array([100.,np.nan,-100.,1.]),endpoint='D_Glu')
    passed('fixed_eligible_extreme_outcomes_clipped_not_rescreened',np.array_equal(transformed['values'][[0,2,3]],np.array([5.,-5.,1.])) and not transformed['excluded_8sd'].any())
    passed('nonfinite_outcomes_not_manufactured',np.isnan(transformed['values'][1]) and transformed['original_rule_excluded_8sd'][0] and transformed['winsorized_high'][0] and transformed['winsorized_low'][2])
    data=make(cfg)
    frozen=modules()
    a=data.signals['abundance'][:300]
    state=FunctionState('abundance_relative',cfg).fit(a)
    multiplier=np.linspace(.01,100,len(a))[:,None]
    passed('relative_signal_invariant_to_positive_sample_scale',np.allclose(state.transform(a),state.transform(a*multiplier),atol=1e-10))
    ranks=FunctionState('abundance_unknown_null',cfg).fit(a)
    b=a.copy()
    b[:,0]=np.nan
    passed('unknown_null_encoding_retains_reporting_information',not np.allclose(ranks.raw(a),ranks.raw(b)))
    rank_scaled=FunctionState('abundance_unknown_null',cfg).fit(a*np.arange(1,a.shape[1]+1))
    passed('observed_rank_representation_invariant_to_feature_units',np.allclose(ranks.raw(a),rank_scaled.raw(a*np.arange(1,a.shape[1]+1))))
    empty=np.ones((100,5))
    constant=FunctionState('abundance_relative',cfg).fit(empty)
    passed('constant_relative_spectrum_has_zero_components',constant.transform(empty).shape==(100,0))
    never=np.full((100,4),np.nan)
    never[:,0]=1.
    rare=FunctionState('coverage_confidence',cfg).fit(never)
    passed('training_missing_fields_not_selected',rare.keep.tolist()==[0])
    expect_error(lambda:state.transform(np.full((2,a.shape[1]),np.nan)),'flagged')
    passed('entire_null_profile_not_made_biological_zero')
    expect_error(lambda:FunctionState('coverage_confidence',cfg).fit(np.full((50,4),2.)),'range')
    passed('coverage_range_checked')
    tech=TechnicalState().fit(data.technical[:300])
    old_median=tech.median.copy()
    transformed=tech.transform(np.full((5,3),np.nan))
    passed('technical_missing_values_train_fitted_and_finite',np.array_equal(tech.median,old_median) and np.isfinite(transformed).all())
    train=data.available & (data.dataset.reference['outer'][0] != 0)
    valid=data.available & ~train
    base=data.frozen.engine.Engine(None,None).design(data.dataset,train,valid)
    view=Design(base,'abundance_relative',data.signals['abundance'],data.technical,cfg,frozen.features)
    spec=dict(name='abundance_relative')
    zt,zv,imap=view.interactions(spec)
    passed('functional_interaction_residualized_against_matched_main',np.max(abs(view.xt.T@zt))<1e-6)
    baseline=frozen.models.fit_main(view,.1)
    no_interaction=frozen.models.fit_augmented(view,spec,.1,None,baseline)
    passed('no_interaction_is_exact_same_fitted_baseline',no_interaction is baseline and np.array_equal(no_interaction.predict(),baseline.predict()))
    fit=frozen.models.fit_augmented(view,spec,.1,.03)
    total,component=fit.contrasts()
    passed('counterfactual_model_contrast_matches_direct_prediction_difference',np.allclose(total,fit.predict(diet=15)-fit.predict(diet=10)))
    passed('main_effect_contains_all_interaction_pc_coordinates',view.pcs[0].shape[1]>=zt.shape[1] and np.array_equal(view.xt[:,0],np.ones(len(view.xt))))
    passed('functional_nonlinear_main_terms_in_both_models',view.function_curves is not None and len(view.function_curves.keep)>0 and baseline.design is fit.design)
    unchanged=state.pca.components_.copy()
    state.transform(np.where(np.isnan(a[:5]),np.nan,a[:5]*100000))
    passed('heldout_function_values_do_not_refit_pca',np.array_equal(state.pca.components_,unchanged))

    first,meta=outer(data,cfg,0,0,lambda *a:None)
    altered=make(cfg)
    outer_valid=altered.available & (altered.dataset.reference['outer'][0]==0)
    altered.dataset.y[outer_valid] += 1000000.
    second,meta2=outer(altered,cfg,0,0,lambda *a:None)
    passed('outer_validation_outcome_does_not_change_selection',meta['selections']==meta2['selections'])
    passed('outer_validation_outcome_does_not_change_predictions',np.array_equal(first['main'],second['main']) and np.array_equal(first['interaction'],second['interaction']))
    passed('outer_outcome_mutation_reaches_only_evaluation_target',not np.array_equal(first['y'],second['y']))
    no_variation=make(cfg)
    for values in no_variation.signals.values():
        values[no_variation.available]=1.
    same_prediction,constant_meta=outer(no_variation,cfg,0,0,lambda *a:None)
    passed('zero_dimensional_functions_exactly_revert_to_main_in_nested_selection',np.array_equal(same_prediction['main'],same_prediction['interaction']) and all(v['interaction_penalty'] is None for v in constant_meta['selections'].values()))
    native_adapter_fixture()
    passed('native_adapter_full_generated_file_lineage_and_row_mapping')
    with tempfile.TemporaryDirectory(prefix='hpp_function_pilot_synthetic_') as tmp:
        folder=Path(tmp)
        failed_run=folder/'failed_before_models'
        failure_logs=[]
        startup_failure=False
        with patch('pilot.save_arrays',side_effect=OSError(95,'Simulated old startup failure')):
            try:
                execute(data,cfg,failed_run,threads=1,log=failure_logs.append)
            except OSError as error:
                startup_failure=error.errno==95
        preserved={name:(failed_run/name).read_bytes() for name in ('registration.json','RUNNING.lock','source_audit.txt')}
        passed('cohort_write_failure_occurs_before_any_model_or_complete_fold',startup_failure and not failure_logs and not list((failed_run/'TRE_ONLY').glob('*_COMPLETE.json')) and not (failed_run/'summary.json').exists())
        summary=execute(data,cfg,folder/'run',threads=1,log=lambda *a:None)
        passed('separate_repaired_run_preserves_failed_registration_and_lock',all((failed_run/name).read_bytes()==value for name,value in preserved.items()) and summary['status']=='COMPLETE')
        primary=[r for r in summary['scores'] if r['route']==cfg['primary_route']]
        passed('known_synthetic_interaction_gains_on_both_household_splits',all(r['delta_R2']>0 for r in primary))
        passed('all_routes_and_splits_reported',len(summary['scores'])==6 and all(r['N']==420 for r in summary['scores']))
        passed('synthetic_results_explicitly_labeled',summary['synthetic_only'] is True)
        passed('six_fold_atomic_completion_receipts',len(list((folder/'run/TRE_ONLY').glob('*_COMPLETE.json')))==6)
        again=execute(data,cfg,folder/'run',threads=1,log=lambda *a:None)
        passed('completed_run_replayed_without_refit',again==summary)
        tampered=folder/'run/TRE_ONLY/final_predictions.npz'
        with tampered.open('ab') as stream:
            stream.write(b'changed')
        expect_error(lambda:execute(data,cfg,folder/'run',threads=1,log=lambda *a:None),'checksum')
        passed('tampered_completed_arrays_rejected')
        ck=folder/'chunks'
        ck.mkdir()
        arrays=dict(rows=np.array([1]),y=np.array([2.]))
        checkpoint(ck,'one','same',lambda:(arrays,dict(synthetic_only=True)))
        expect_error(lambda:checkpoint(ck,'one','changed',lambda:None),'identity')
        passed('changed_checkpoint_identity_rejected')
        (ck/'one.json').write_text('{}')
        expect_error(lambda:checkpoint(ck,'one','same',lambda:None),'checksum')
        passed('tampered_fold_metadata_rejected')
        preview=[]
        from contextlib import redirect_stdout
        import io
        output=io.StringIO()
        with redirect_stdout(output):
            report(summary)
        preview=output.getvalue()
        passed('report_contains_scope_and_no_individual_identifiers','SYNTHETIC ONLY' in preview and 'SYNTHETIC_HH_' not in preview and 'formal interaction P value' in preview)
    # Explicitly adverse predictions: every planned negative result must survive aggregation.
    chunks=[]
    y=data.dataset.y
    for repeat in range(2):
        for fold in range(cfg['outer_folds']):
            rows=np.flatnonzero(data.available & (data.dataset.reference['outer'][repeat]==fold))
            main=np.tile(y[rows],(3,1))
            arrays=dict(rows=rows,y=y[rows],raw_y=y[rows],main=main,interaction=main+1.,total_contrast=np.zeros_like(main),delta_contrast=np.zeros_like(main),raw_component=np.zeros_like(main),support=np.ones_like(main,dtype=bool))
            chunks.append((arrays,dict(repeat=repeat,fold=fold,selections={route:dict(interaction_penalty=.03) for route in cfg['routes']})))
    negative,_=summarize(data,cfg,chunks)
    passed('negative_incremental_R2_preserved_for_every_route',all(row['delta_R2']<0 for row in negative['scores']))
    passed('constant_split_contrast_correlation_is_missing_not_zero',all(v is None for v in negative['same_people_added_contrast_split_correlation'].values()))
    error_chunks=chunks+[chunks[0]]
    expect_error(lambda:summarize(data,cfg,error_chunks),'duplicated')
    passed('duplicate_outer_members_rejected')
    groups=data.dataset.index.loc[data.available,'split_unit_id'].astype(str).to_numpy()
    noise=np.tile(y[data.available],(3,1))
    interval=paired_intervals(y[data.available],noise,noise+1.,groups,99,7)
    passed('paired_household_interval_has_correct_adverse_direction',np.all(interval<0))
    for p in ROOT.glob('*.py'):
        compile(p.read_text(),str(p),'exec')
    passed('all_python_sources_compile')
    destination=ROOT/'research_records/local_validation'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'synthetic_preview.txt').write_text('GENERATED DATA ONLY; NOT HPP FINDINGS\n\n'+preview)
    result=dict(status='passed',synthetic_only=True,check_count=len(checks),checks=checks,native_HPP_inputs_read=False,
        native_environment_execution_verified=False,scope='Generated matrices/files; frozen source imports; native reader exercised on generated fixtures only.')
    (destination/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FUNCTIONAL_SYNTHETIC_CHECKS_OK checks=%d; no native HPP data read.' % len(checks))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
HPP_IO_SOURCE_01_END
hpp_io_write 'README.md' '3be7666891a8a3e2a5bf1f716838f5ac05934a8de0eb460dd0db38557b91ed03' 'e9b1879e6af5c3999c33e4090b3b44c93b506f1b78ce5aaa0496ef34c48e3158' <<'HPP_IO_SOURCE_02_END'
# 口腔功能探索 v1

此模块是V12完成后的独立子目录，不修改父目录51份冻结交付文件或旧结果。方案见[PROTOCOL.md](PROTOCOL.md)。主问题仍为纤维密度与个人平均D_Glu的关联异质性；只做固定范围的探索性折外预测比较，不生成逐通路显著性排行榜。

原3343人来源已匹配；六个候选功能项全空者单独留标记，功能可用3337人为本次评价人群。各路线均重新拟合相同人群/折的主效应参照，不直接与旧3343人的R²比较。主要路线为相对丰度信号，覆盖度和未知空值编码仅作敏感性；二轮均是同人家庭分折，不能称独立重复。

部分空格按未报告0处理是有依据的工作假设，不是对HPP聚合格式的认证。相对信号表示不声称已知RPK/CPM，覆盖度不称反应完成率。六人不造全零功能谱；观察值内常量列不全局删除。训练内筛选/PCA和选型均不看外层验证Y。固定人群内只按训练稳健阈值截尾结局，不按新阈值再次剔除成员；原8SD规则的标记保留作审计。技术读段主效应及其饮食项进入两个比较模型。

安装指令会只创建本子目录，已有相同文件跳过，不同内容拒绝覆盖。默认运行命令为：

```bash
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/functional_followup_20261110_v1
bash server_run.sh --threads 2
```

启动依次核对新模块文件、生成数据检查、原51份源码、既有环境与来源资格，再进入两轮五折的家庭隔离探索。原生数据从不发送到本地；最终数组和资格标记保存在`runs/function_20261110_v1_iofix1/TRE_ONLY/`，只展示不超过三屏的聚合报告。耗时取决于TRE计算资源，不预先承诺完成时间。

2026-11-10存储修复：首次原生运行在模型拟合前保存NPZ时报seek不支持。修复版在内存中组装ZIP，随后顺序写入并核对SHA；统计方案、配置、特征变换、人口、分折和惩罚网格保持原样。默认新运行标识为`function_20261110_v1_iofix1`，旧失败目录`runs/function_20261110_v1/`及其锁/部分文件保留，不把不同代码的旧注册强行当作同源恢复。修复指令对已知旧源作备份，拒绝未知改动或活动进程。

十个外折有原子检查点。失败后保留锁和已完成折；仅当同一主机旧PID不存在、代码/配置/输入/环境完全相同时可恢复：

```bash
bash server_run.sh --threads 2 --resume
```

查看完成汇总不重新读取参与者数据：

```bash
bash server_run.sh --threads 2 --report
```

报告保留所有路线的两轮负值、无交互选择、匹配主效应R²、增量及条件家庭自助区间。区间不包含重训/选型不确定性，也不是正式交互P值。10→15对比为模型生成的关联差，不是实际饮食干预获益。主要路线缺乏两轮一致增益时按预定停止规则报告，不再改网格寻找好结果。

`check_synthetic.py`和`demo_*.py`只生成测试数据，不触碰原生HPP输入；验证输出存本模块`research_records/local_validation/`并醒目标为生成数据。真实服务器运行目前尚未执行，任何本地模拟分数都不是HPP结果。
HPP_IO_SOURCE_02_END
hpp_io_write 'manifest.json' '5e5ac520f43b6bc68fd9448f85b460ce3a483b01c3dc17a623ce59f6924c2280' '3a99e664252c1ecd82c48d3c0cabb2f7098f03bd8724c18f84a684eb2d7820bd' <<'HPP_IO_SOURCE_03_END'
{
  "analysis_configuration_unchanged": true,
  "files": {
    "IMPLEMENTATION_PLAN.md": "ba8fb78c33145fe08eac0ce7895519b59a6463e6cd87c64a6ccbb24bd7ca58fb",
    "PROTOCOL.md": "bb845b41d395cf6748a401e7077b1c86f52432e2f16c4bfc815326c1091ab123",
    "README.md": "e9b1879e6af5c3999c33e4090b3b44c93b506f1b78ce5aaa0496ef34c48e3158",
    "audit_driver.py": "0c2186ecfc4b42456437f5145847addf807d4cff9979088e30f57bcccfbcf92c",
    "check_synthetic.py": "a41f7f96ab80dc55204ed14511a456e1725d14b65cca65dad70804d258f31f8b",
    "config.json": "5f4fe8a03443b1623a9c2a980e3e8aadf3188148ab65290982ddf3cd898e3102",
    "demo_data.py": "4a25c6af3289ed250f42307ffa1ccebbe9ddc72abee81f293b0ebedf94a2efab",
    "demo_files.py": "ec6f1e5a4f8a972b1473c425827e7a4424a15e4cfaa5d4f11cfa4d9a9c57b965",
    "frozen.py": "783382e5bae6ddd26d97226569867be69ec7e7cacfd2ed29cbc32fe812c52eea",
    "functional.py": "fcefca3fa10d03ac95abe9b286c573dfbcbd280371ed8d3fc5e7dc7c280d3d3a",
    "native.py": "761492bdf30fb074d18d48a159a93d239089fe6ecd79e31927beab00dc271884",
    "pilot.py": "5740847289b97775e53ea90e10bd68b19aa75a197ee770360e5d287bc38c4ffd",
    "server_run.sh": "8e7d81c72f09a45d9dc5a55e0772410a007dad2bf48e19d6018c5ab9461cc1fd"
  },
  "native_HPP_inputs_embedded": false,
  "original_V12_files_modified": false,
  "storage_repair": "In-memory ZIP assembly followed by sequential hash-checked writes; fresh IO-fix run ID preserves failed registration.",
  "version": "functional_followup_20261110_v1_iofix1"
}
HPP_IO_SOURCE_03_END
"$hpp_io_python" -B - <<'HPP_IO_FINAL'
import hashlib,json
from pathlib import Path
root=Path.cwd()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files'].items():
    if Path(name).name!=name or digest(root/name)!=expected:
        raise SystemExit('IO_REPAIR_STOP: repaired source manifest mismatch')
archive=root/'research_records/2026-11-10_iofix1'
state=json.loads((archive/'original_failed_run_state.json').read_text())
oldrun=root/'runs/function_20261110_v1'
for name,key in [('registration.json','old_registration_sha256'),('RUNNING.lock','old_lock_sha256')]:
    actual=digest(oldrun/name) if (oldrun/name).exists() else None
    if actual!=state[key]:
        raise SystemExit('IO_REPAIR_STOP: old failed run changed')
if digest(root/'config.json')!=state['statistical_config_sha256']:
    raise SystemExit('IO_REPAIR_STOP: statistical configuration changed')
receipt=dict(status='STORAGE_PATCH_INSTALLED',analysis_configuration_unchanged=True,old_failed_run_preserved=True,
    new_run_id='function_20261110_v1_iofix1',manifest_sha256=digest(root/'manifest.json'),native_storage_success_not_yet_verified=True)
with (archive/'repair_installed.json').open('w') as stream:
    stream.write(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print('IO_REPAIR_INSTALLED: memory ZIP plus sequential writes; statistical plan unchanged; old run preserved.')
HPP_IO_FINAL
if [ "${HPP_FUNCTION_IO_REPAIR_ONLY:-0}" = 1 ]; then
    echo 'IO_REPAIR_ONLY_DONE; no models started.'
    exit 0
fi
if [ -f runs/function_20261110_v1_iofix1/COMPLETE.json ]; then
    bash server_run.sh --threads 2 --run-id function_20261110_v1_iofix1 --report
elif [ -f runs/function_20261110_v1_iofix1/registration.json ]; then
    bash server_run.sh --threads 2 --run-id function_20261110_v1_iofix1 --resume
else
    bash server_run.sh --threads 2 --run-id function_20261110_v1_iofix1
fi
HPP_FUNCTION_IO_REPAIR1
