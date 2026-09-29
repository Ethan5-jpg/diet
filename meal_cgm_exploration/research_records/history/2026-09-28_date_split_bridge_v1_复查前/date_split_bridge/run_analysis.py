#!/usr/bin/env python3
"""Date-separated window metrics and bounded, prespecified associations."""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--followup-run',default=str(PACKAGE/'outputs/response_followup_20260927T155611_920880Z'))
    p.add_argument('--data-root');p.add_argument('--output-root',default=str(PACKAGE/'outputs'))
    p.add_argument('--stage',choices=['check','all'],default='all')
    p.add_argument('--show');p.add_argument('--brief',action='store_true');p.add_argument('--page',type=int,choices=[1,2,3])
    a=p.parse_args(argv)
    if sys.version_info<(3,8):print('Python>=3.8 required',file=sys.stderr);return 2
    if a.show:
        try:
            from date_split_bridge.reporting import show
            show(a.show,a.output_root,a.page,a.brief);return 0
        except Exception as e:print('SHOW ERROR: '+str(e),file=sys.stderr);return 2
    if a.brief or a.page is not None:p.error('--brief/--page require --show')
    run=None;out=None;state=dict(status='RUNNING',stage=a.stage)
    try:
        from hppmeal.common import Run,require,write_json,sha256
        from seven_diet_meals.inputs import record_expected,unchanged
        from response_followup.inputs import relocate
        from date_split_bridge import VERSION
        from date_split_bridge.inputs import bundle,inspect,load
        from date_split_bridge.build import build,descriptive
        from date_split_bridge.models import analyze
        from date_split_bridge.reporting import save,pages,summary
        source=Path(a.followup_run).expanduser().resolve();output=Path(a.output_root).expanduser().resolve()
        require(source.is_dir(),'Follow-up run missing: '+str(source))
        prior=json.loads((source/'manifest.json').read_text(encoding='utf-8'));spec=json.loads((source/'analysis_spec.json').read_text(encoding='utf-8'))
        oldroot=Path(prior['root']);root=Path(a.data_root or oldroot).expanduser().resolve()
        nutrition=relocate(spec['nutrition_run'],oldroot,root)
        ns=json.loads((nutrition/'analysis_spec.json').read_text(encoding='utf-8'))
        quality=relocate(ns['quality_run'],oldroot,root)
        qs=json.loads((quality/'analysis_spec.json').read_text(encoding='utf-8'))
        protected=[source,nutrition,quality,relocate(qs['meal_run'],oldroot,root),relocate(qs['seven_run'],oldroot,root)]
        protected += [PACKAGE/x for x in ['date_split_bridge','response_followup','nutrition_bridge','quality_timecourse','seven_diet_meals','hppmeal','scripts','research_records']]
        for f in protected:
            f=f.resolve();require(output!=f and f not in output.parents,'Output root must be outside frozen run/code/archive directories')
        out=output/datetime.now(timezone.utc).strftime('date_split_bridge_%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
        print('RUN_DIR='+str(out),flush=True);print('CHECK upload, frozen lineage and source hashes',flush=True)
        expected=bundle();root,c,paths,checked,audit=inspect(source,a.data_root,out);expected.update(checked)
        run=Run(root,out,c);state.update(version=VERSION);run.status['date_split_bridge']=state;run.save()
        record_expected(run,expected);run.table(audit,'00_preflight.csv')
        contract=json.loads((PACKAGE/'date_split_bridge/CONTRACT.json').read_text(encoding='utf-8'))
        write_json(out/'analysis_spec.json',dict(version=VERSION,data_kind=c['data_kind'],stage=a.stage,
            followup_run=str(source),nutrition_run=str(nutrition),contract=contract,contract_sha256=sha256(PACKAGE/'date_split_bridge/CONTRACT.json')))
        meals,people,cgm,tables=load(run,paths)
        if a.stage=='check':
            unchanged(run);state.update(status='CHECK_PASS',reason='Inputs verified/loaded; no windows or models run')
        else:
            print('REBUILD frozen cuts/memberships and compute new grid metrics',flush=True)
            result=build(run,paths,meals,people,cgm,tables,lambda s:print(s,flush=True))
            result.update(descriptive(result['person_features']))
            result.update(analyze(result['person_features'],lambda s:print(s,flush=True)))
            save(run,result)
            models=result['models'];n=int(models.loc[models.target.eq('late_primary'),'status'].eq('ESTIMATED').sum())
            state.update(status='COMPLETE_WITH_LIMITATIONS',primary_estimated=n,primary_planned=24,
                secondary_planned=48,prior_count_checks=14,metric_contract='NEW_GRID_RECIPE_V1',
                reason='New fixed window definitions; legacy iglu exact equivalence not asserted. Internal associations, not external validation. All planned rows retained.')
            pages(run,result,state)
            print('VERIFY all frozen inputs unchanged',flush=True);unchanged(run)
            state['result_sha256']={str(f.relative_to(out)):sha256(f) for folder in ['tables','private','pages'] for f in sorted((out/folder).glob('*')) if f.is_file()}
        run.save();summary(run,state)
        print('DATE_SPLIT_BRIDGE={}\nDATA={}'.format(state['status'],c['data_kind']),flush=True)
        if a.stage=='all':print('VIEW: python3 date_split_bridge/run_analysis.py --show "{}" --brief'.format(out),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as e:
        state.update(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED',reason=str(e) or 'Interrupted')
        if out is not None:
            (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            if run is not None:run.save();summary(run,state)
        print('DATE_SPLIT_BRIDGE ERROR: '+state['reason'],file=sys.stderr,flush=True)
        return 130 if isinstance(e,KeyboardInterrupt) else 2


if __name__=='__main__':sys.exit(main())
