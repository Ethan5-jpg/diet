#!/usr/bin/env python3
"""Nutrition adjustment, positive area and original20 CGM bridge."""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-run', help='Completed quality_timecourse output directory')
    parser.add_argument('--data-root', help='Data directory; default previous run root')
    parser.add_argument('--cgm-file', help='Explicit original finalized CGM20 table, if relocated')
    parser.add_argument('--output-root', default=str(PACKAGE / 'outputs'))
    parser.add_argument('--stage', choices=['check', 'all'], default='all')
    parser.add_argument('--show', help='Existing output path or latest')
    parser.add_argument('--page', type=int, default=1, help='1..54 short pages; 0 full summary')
    args = parser.parse_args(argv)
    if sys.version_info < (3, 8):
        print('Python >=3.8 required', file=sys.stderr)
        return 2
    if args.show:
        try:
            from nutrition_bridge.reporting import show
            show(args.show, args.output_root, args.page)
            return 0
        except Exception as e:
            print('SHOW ERROR: ' + str(e), file=sys.stderr)
            return 2
    if not args.quality_run:
        parser.error('--quality-run is required')
    run, out = None, None
    state = dict(status='RUNNING', stage=args.stage)
    try:
        from hppmeal.common import Run, require, write_json, sha256
        from seven_diet_meals.inputs import record_expected, unchanged
        from nutrition_bridge import VERSION
        from nutrition_bridge.inputs import bundle, inspect, load
        from nutrition_bridge.analysis import run_models
        from nutrition_bridge.reporting import summary, save_tables, pages, figures
        print('CHECK packages, previous run, original CGM and source hashes', flush=True)
        expected = bundle()
        source = Path(args.quality_run).expanduser().resolve()
        root_out = Path(args.output_root).expanduser().resolve()
        prior_spec = json.loads((source / 'analysis_spec.json').read_text(encoding='utf-8'))
        for p in [source, Path(prior_spec['meal_run']).resolve(), Path(prior_spec['seven_run']).resolve(), Path(__file__).resolve().parent]:
            require(root_out != p and p not in root_out.parents, 'Output root must be outside source runs and package')
        out = root_out / datetime.now(timezone.utc).strftime('nutrition_bridge_%Y%m%dT%H%M%S_%fZ')
        out.mkdir(parents=True, exist_ok=False)
        print('RUN_DIR=' + str(out), flush=True)
        root, c, paths, checked, audit = inspect(source, args.data_root, out, args.cgm_file)
        expected.update(checked)
        run = Run(root, out, c)
        state['version'] = VERSION
        run.status['nutrition_bridge'] = state
        run.save()
        summary(run, state)
        record_expected(run, expected)
        run.table(audit, '00_preflight.csv')
        write_json(out / 'analysis_spec.json', dict(version=VERSION, data_kind=c['data_kind'], quality_run=str(source),
            whole_cgm=str(paths['whole_cgm']), sources={k: str(v) for k, v in paths.items()},
            stage=args.stage, settings=['source_primary', 'joint'], reproduced_models=84,
            planned_models=dict(adjustment=140, positive=42, bridge=120),
            fdr='per setting: adjustment70, positive21, bridge60; unestimated retained in denominator',
            positive_missing='no meal deletion: specification NOT_ESTIMATED if any required meal missing',
            bridge='same visit and connection; no dietary score selection; BASE adjusted; paired-sample meal SD; original whole CGM Z',
            historical_cgm='exact previously archived path; current hash recorded, not proof of historical numerical identity',
            inference='exploratory; no coefficient change test, no mediation fraction, no independent replication'))
        meals, people, cgm, previous, members = load(run, paths)
        print('INPUTS ready meals={}, CGM people={}'.format(len(meals), len(cgm)), flush=True)
        if args.stage == 'check':
            unchanged(run)
            state.update(status='CHECK_PASS', reason='Inputs checked; model reproduction requires full run')
        else:
            result = run_models(meals, people, cgm, previous, members, lambda s: print(s, flush=True))
            save_tables(run, result)
            figures(run, result)
            print('VERIFY all original inputs unchanged', flush=True)
            unchanged(run)
            estimated = sum(int(result[k].status.eq('ESTIMATED').sum()) for k in ['adjustment', 'positive', 'bridge'])
            state.update(status='COMPLETE' if estimated == 302 else 'COMPLETE_WITH_LIMITATIONS',
                reproduced=len(result['reproduction']), estimated_models=estimated, planned_models=302,
                reason='' if estimated == 302 else 'Some fixed specifications unavailable; see page30 and full tables')
            pages(run, result, state)
            state['result_sha256'] = {str(f.relative_to(out)): sha256(f) for folder in ['tables', 'private', 'figures', 'pages']
                for f in sorted((out / folder).glob('*')) if f.is_file()}
        run.save()
        summary(run, state)
        print('NUTRITION_BRIDGE={}\nSUMMARY={}'.format(state['status'], out / 'SUMMARY.md'), flush=True)
        print('VIEW: python3 nutrition_bridge/run_analysis.py --show "{}" --page 1'.format(out), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        state.update(status='INTERRUPTED' if isinstance(e, KeyboardInterrupt) else 'FAILED', reason=str(e) or 'Interrupted')
        if out is not None:
            (out / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
            if run is not None:
                run.save()
                summary(run, state)
            else:
                (out / 'SUMMARY.md').write_text('# NUTRITION_BRIDGE FAILED\n\n' + state['reason'] + '\n', encoding='utf-8')
        print('NUTRITION_BRIDGE ERROR: ' + state['reason'], file=sys.stderr, flush=True)
        return 130 if isinstance(e, KeyboardInterrupt) else 2


if __name__ == '__main__':
    sys.exit(main())
