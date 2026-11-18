#!/usr/bin/env python3
"""V7 server entry point. All checkpoints and person-level arrays stay in the TRE."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
V6 = ROOT.parent / 'v6'
VERSION = 'effect_modification_v7_ratio_pc10_20261118_expanded'
RUN = ROOT / 'runs' / 'TRE_ONLY' / VERSION
MANIFEST = ROOT / 'upload_manifest.json'
HIST_DIR = ROOT.parent.parent / 'experiments/diet_response_heterogeneity_20261103_v1_attempt02/results'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_release():
    manifest = json.loads(MANIFEST.read_text())
    if manifest['version'] != VERSION:
        raise ValueError('Unexpected V7 upload version')
    for key, folder in [('v7', ROOT), ('v6', V6)]:
        for name, expected in manifest[key].items():
            path = folder / name
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f'Upload mismatch: {key}/{name}')
    return manifest


def emit(tag, value):
    print('EXP2_V7_' + tag, json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def artifacts():
    files = ['registration.json', 'summary.json', 'all_routes.csv']
    return files + [f'r{r}_f{f}/{filename}' for r in range(2) for f in range(5)
                    for filename in ('arrays.npz', 'metadata.json', 'receipt.json')]


def read_complete(run, release):
    receipt = json.loads((run / 'COMPLETE.json').read_text())
    if receipt['version'] != VERSION or set(receipt['artifacts']) != set(artifacts()):
        raise ValueError('Incomplete V7 receipt')
    for name, expected in receipt['artifacts'].items():
        if digest(run / name) != expected:
            raise ValueError('Completed artifact changed: ' + name)
    registration = json.loads((run / 'registration.json').read_text())
    if registration['release'] != release or registration['version'] != VERSION:
        raise ValueError('Completed run belongs to a different release')
    return json.loads((run / 'summary.json').read_text())


def write_aggregate_csv(path, summary):
    fields = ['repeat','route','baseline_r2','full_r2','delta_r2','delta_r2_ci95',
              'delta_r2_percentage_points','ratio_over_full_pct','ratio_over_full_ci95_pct',
              'ratio_over_full_status','beats_historical_point','beats_historical_ci',
              'ratio_minus_reference_pp','ratio_minus_reference_ci95_pp','active_folds',
              'threshold_excess_pp','threshold_excess_simul_lower95_pp','exceeds_benchmark_simul_lower']
    with tempfile.NamedTemporaryFile('w', newline='', dir=path.parent, delete=False,
                                     encoding='utf-8') as file:
        temp = Path(file.name)
        try:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            for rep in summary['repeats']:
                for name, row in rep['routes'].items():
                    values = {'repeat':rep['repeat'], 'route':name}
                    for key in fields[2:]:
                        val = row.get(key) if key.startswith(('threshold_', 'exceeds_benchmark_simul')) else row[key]
                        values[key] = json.dumps(val) if isinstance(val, (list, dict)) else val
                    writer.writerow(values)
            file.flush()
            file.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def execute(source, run, release, threads, *, draws=1999, historical_dir=None):
    import run_v6 as store
    import v7_eval as ev
    import v7_models as models
    import v7_reference as ref
    registration = dict(version=VERSION, release=release, draws=draws,
                        families=models.families(), routes=list(ev.route_names()),
                        exposure='fiber_density', outcome='D_Glu', reserved_outcomes_read=False,
                        runtime=store.runtime_identity(threads), cohort=store.cohort_identity(source),
                        source_hashes={str(k):v for k,v in source.source_hashes.items()})
    with store.lock(run):
        reg = run / 'registration.json'
        if reg.exists():
            if store.read_json(reg) != registration:
                raise ValueError('V7 run registration differs. Preserve existing results.')
        else:
            if any(run.glob('r*_f*')) or (run/'COMPLETE.json').exists():
                raise ValueError('V7 artifacts without registration')
            store.write_json(reg, registration)
        if (run / 'COMPLETE.json').exists():
            return read_complete(run, release)
        folds = {}
        for r in range(2):
            for f in range(5):
                if (run / f'r{r}_f{f}').exists():
                    folds[r, f] = store.read_fold(run, r, f)
                    emit('REUSED', {'repeat':r,'fold':f})
                    continue
                start = time.monotonic()
                emit('START', {'repeat':r,'fold':f})
                arrays, meta = ev.run_fold(source, r, f)
                source.verify_sources()
                store.save_fold(run, r, f, arrays, meta)
                folds[r, f] = arrays, meta
                emit('OUTER_COMPLETE', {'repeat':r,'fold':f,'people':len(arrays['y']),
                                        'seconds':round(time.monotonic()-start, 1)})
        summary = ev.summarize(source, folds, draws=draws)
        summary['reference_concordance_r0'] = (ref.historical_check(historical_dir, folds, digest)
            if historical_dir is not None else {'status':'generated_fixture'})
        source.verify_sources()
        store.write_json(run / 'summary.json', summary)
        write_aggregate_csv(run / 'all_routes.csv', summary)
        hashes = {name:digest(run / name) for name in artifacts()}
        store.write_json(run / 'COMPLETE.json', {'version':VERSION,'artifacts':hashes})
        return read_complete(run, release)


def compact(summary):
    def fmt(x):
        return 'NA' if x is None else f'{x:+.3f}'
    def ci(x):
        return 'NA' if x is None else '[' + ','.join(map(fmt,x)) + ']'
    emit('REFERENCE', summary['reference_concordance_r0'])
    print('EXP2_V7_SCREEN_BEGIN | discovery | delta in R2 percentage points | ratio = 100 * delta/full R2')
    for rep in summary['repeats']:
        print(f"r{rep['repeat']} | N={rep['people']} | households={rep['households']}")
        routes = rep['routes']
        names = ['reference_pc10_p01', 'ratio_selector', 'ratio_selector_original35', 'ratio_selector_guarded']
        for kind in ('exploratory_observed_max', 'exploratory_stable_max'):
            best = rep[kind]
            print(f"  {kind}: {best['route'] if best else 'none'}")
            if best and best['route'] not in names:
                names.append(best['route'])
        for name in names:
            d = routes[name]
            delta_ci = d['delta_r2_ci95']
            delta_ci_pp = None if delta_ci is None else [100*x for x in delta_ci]
            print(f"  {name}: off/on {d['baseline_r2']:.5f}/{d['full_r2']:.5f}  delta {fmt(d['delta_r2_percentage_points'])} pp CI {ci(delta_ci_pp)}")
            print(f"    ratio {fmt(d['ratio_over_full_pct'])}% CI {ci(d['ratio_over_full_ci95_pct'])}  "
                  f"vs fixed PC10 {fmt(d['ratio_minus_reference_pp'])} pp CI {ci(d['ratio_minus_reference_ci95_pp'])}")
            print(f"    exceeds 6.45689% (point/CI lower): {d['beats_historical_point']}/{d['beats_historical_ci']}")
        paired = rep['expanded_vs_original35']
        print(f"  EXPANDED_MINUS_ORIGINAL35 | ratio {fmt(paired['ratio_difference_pp'])} pp "
              f"CI {ci(paired['ratio_difference_ci95_pp'])} | delta R2 {fmt(paired['delta_r2_difference_pp'])} pp "
              f"CI {ci(paired['delta_r2_difference_ci95_pp'])}")
        threshold = rep['threshold_screen']
        print(f"  SIMULTANEOUS_THRESHOLD | families={threshold['family_count']} "
              f"supported={','.join(threshold['supported_families']) or 'none'} "
              f"critical_margin={fmt(threshold['critical_margin_pp'])} R2 pp")
    emit('POSITIVE_BOTH_TOP3', summary['positive_both_repeats'][:3])
    print('EXP2_V7_SCREEN_END | repeats use the same people; maxima are exploratory', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--report', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    release = verify_release()
    emit('RELEASE_OK', VERSION)
    if args.report:
        compact(read_complete(RUN, release))
        return
    sys.path.insert(1, str(V6))
    if sys.version_info[:2] != (3,11):
        raise RuntimeError('Use the V12 Python 3.11 environment')
    for pkg, version in release['pins'].items():
        if importlib.metadata.version(pkg) != version:
            raise RuntimeError(f'Use the pinned V12 environment: expected {pkg}=={version}')
    # Import numerical libraries before limiting their native BLAS and OpenMP pools.
    import v7_models as models
    import v7_eval
    import scipy.linalg
    import sklearn.decomposition
    from adapters.person_inputs import load_discovery
    from v7_source import wrap_source
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        families = models.families()
        emit('DESIGN', dict(families=len(families), settings=sum(map(len,families.values())),
                            routes=len(v7_eval.route_names()), threads=args.threads))
        if args.preflight:
            tests = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
            with open(os.devnull, 'w') as muted, contextlib.redirect_stdout(muted):
                outcome = unittest.TextTestRunner(verbosity=1).run(tests)
            if not outcome.wasSuccessful() or not outcome.testsRun:
                raise RuntimeError('V7 generated preflight failed')
            emit('PREFLIGHT_DONE', dict(tests=outcome.testsRun, skipped=len(outcome.skipped), scope='generated only'))
        else:
            source = wrap_source(load_discovery(ROOT))
            compact(execute(source, RUN, release, args.threads, historical_dir=HIST_DIR))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit('STOP', {'error':type(exc).__name__,'detail':str(exc)})
        raise
