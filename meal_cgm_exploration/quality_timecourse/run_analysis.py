#!/usr/bin/env python3
"""Read-only quality sensitivity and early/late meal response extension."""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meal-run', help='Original completed meal_... output')
    p.add_argument('--seven-run', help='Completed seven_diets_... output')
    p.add_argument('--data-root', help='Data directory; default original root')
    p.add_argument('--diary-file', help='Optional explicit daily diary table path')
    p.add_argument('--output-root', default=str(PACKAGE / 'outputs'))
    p.add_argument('--stage', choices=['check', 'all'], default='all')
    p.add_argument('--show', help='Print an existing quality_timecourse_... SUMMARY.md')
    args = p.parse_args(argv)
    if args.show:
        try:
            print((Path(args.show).expanduser() / 'SUMMARY.md').read_text(encoding='utf-8'))
            return 0
        except OSError as e:
            print('Cannot read summary: ' + str(e), file=sys.stderr)
            return 2
    if not args.meal_run or not args.seven_run:
        p.error('--meal-run and --seven-run are required')
    if sys.version_info < (3, 8):
        print('Python >=3.8 required', file=sys.stderr)
        return 2
    run, out, result = None, None, None
    state = dict(status='RUNNING', stage=args.stage)
    try:
        import pandas as pd
        from hppmeal.common import Run, require, write_json, sha256
        from seven_diet_meals.inputs import record_expected, unchanged, load
        from quality_timecourse import VERSION
        from quality_timecourse.inputs import bundle, inspect, load_diary
        from quality_timecourse.coverage import build
        from quality_timecourse.analysis import run_models
        from quality_timecourse.reporting import summary, write_results, figures
        print('CHECK frozen packages and source provenance; hashing original cache', flush=True)
        expected = bundle()
        out_root = Path(args.output_root).expanduser().resolve()
        for source in [Path(args.meal_run).expanduser().resolve(), Path(args.seven_run).expanduser().resolve()]:
            require(out_root != source and source not in out_root.parents, 'Output must be outside previous run directories')
        out = out_root / datetime.now(timezone.utc).strftime('quality_timecourse_%Y%m%dT%H%M%S_%fZ')
        out.mkdir(parents=True, exist_ok=False)
        print('RUN_DIR=' + str(out), flush=True)
        root, c, paths, checked, audit = inspect(args.meal_run, args.seven_run, args.data_root, out, args.diary_file)
        expected.update(checked)
        run = Run(root, out, c)
        state['version'] = VERSION
        run.status['quality_timecourse'] = state
        run.save()
        summary(run, state)
        record_expected(run, expected)
        run.table(audit, '00_preflight.csv')
        write_json(out / 'analysis_spec.json', dict(version=VERSION, data_kind=c['data_kind'], meal_run=str(Path(args.meal_run).resolve()), seven_run=str(Path(args.seven_run).resolve()),
            planned_models=294, quality_settings=['source_primary', 'aligned', 'cgm90', 'log80', 'joint', 'joint_people_all_meals'],
            cgm_day_coverage=.9, minimum_reference_days=7, diary_reference_coverage=.8, cutoff_minutes=60,
            thresholds='fixed operational sensitivity definitions; not proof of complete logging',
            fdr='per quality setting three families of14; per timecourse setting one family of21; retain unestimated rows',
            interpolation='original within-window gap rules; split piecewise-linear area at60; no imputation of missing windows or meals'))
        meals, people = load(run, paths)
        diary = load_diary(run, paths['diary'])
        previous = pd.read_csv(paths['previous_results'])
        if args.stage == 'check':
            unchanged(run)
            state.update(status='CHECK_PASS', reason='Inputs and diary rows checked; full curve/model reproduction requires --stage all')
        else:
            augmented, person_info = build(run, meals, paths, diary, lambda s: print(s, flush=True))
            result = run_models(augmented, people, previous, lambda s: print(s, flush=True))
            write_results(run, result, previous)
            figures(run, result)
            print('VERIFY all source files unchanged', flush=True)
            unchanged(run)
            estimated = int(result['models'].status.eq('ESTIMATED').sum() + result['timecourse'].status.eq('ESTIMATED').sum())
            state.update(status='COMPLETE' if estimated == 294 else 'COMPLETE_WITH_LIMITATIONS', estimated_models=estimated, planned_models=294,
                         reason='' if estimated == 294 else 'Some fixed specifications not estimable; reasons retained in model tables')
            state['result_sha256'] = {str(f.relative_to(out)): sha256(f) for folder in ['tables', 'private', 'figures'] for f in sorted((out / folder).glob('*')) if f.is_file()}
        run.save()
        summary(run, state, result)
        print('QUALITY_TIMECOURSE={}\nSUMMARY={}'.format(state['status'], out / 'SUMMARY.md'), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        reason = 'Interrupted; start a new run' if isinstance(e, KeyboardInterrupt) else str(e)
        state.update(status='INTERRUPTED' if isinstance(e, KeyboardInterrupt) else 'FAILED', reason=reason)
        if out is not None:
            (out / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
            if run is not None:
                run.save()
                summary(run, state)
            else:
                (out / 'SUMMARY.md').write_text('# QUALITY_TIMECOURSE FAILED\n\n' + reason + '\n', encoding='utf-8')
        print('QUALITY_TIMECOURSE ERROR: ' + reason, file=sys.stderr, flush=True)
        return 130 if isinstance(e, KeyboardInterrupt) else 2


if __name__ == '__main__':
    sys.exit(main())
