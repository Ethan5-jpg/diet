#!/usr/bin/env python3
"""Frozen B interpretation and disjoint-date feasibility; no new association fits."""
import argparse
import json
import sys
import traceback
from datetime import datetime,timezone
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nutrition-run',default=str(PACKAGE/'outputs/nutrition_bridge_20260926T071736_080048Z'))
    p.add_argument('--data-root')
    p.add_argument('--output-root',default=str(PACKAGE/'outputs'))
    p.add_argument('--stage',choices=['check','all'],default='all')
    p.add_argument('--show');p.add_argument('--page',type=int,default=1)
    a=p.parse_args(argv)
    if sys.version_info<(3,8):print('Python >=3.8 required',file=sys.stderr);return 2
    if a.show:
        try:
            from response_followup.reporting import show
            show(a.show,a.output_root,a.page);return 0
        except Exception as e:print('SHOW ERROR: '+str(e),file=sys.stderr);return 2
    out=None;run=None;state=dict(status='RUNNING',stage=a.stage)
    try:
        from hppmeal.common import Run,require,write_json,sha256
        from seven_diet_meals.inputs import record_expected,unchanged
        from response_followup import VERSION
        from response_followup.inputs import bundle,inspect,load,relocate
        from response_followup.support import analyze as interaction
        from response_followup.feasibility import analyze as feasibility
        from response_followup.reporting import save,pages,figures,summary
        source=Path(a.nutrition_run).expanduser().resolve();output=Path(a.output_root).expanduser().resolve()
        require(source.is_dir(),'Frozen nutrition run missing: '+str(source))
        prior=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        spec=json.loads((source/'analysis_spec.json').read_text(encoding='utf-8'))
        root=Path(a.data_root or prior['root']).expanduser().resolve();oldroot=Path(prior['root'])
        quality=relocate(spec['quality_run'],oldroot,root)
        qspec=json.loads((quality/'analysis_spec.json').read_text(encoding='utf-8'))
        protected=[source,quality,relocate(qspec['meal_run'],oldroot,root),relocate(qspec['seven_run'],oldroot,root)]
        protected += [PACKAGE/x for x in ['response_followup','nutrition_bridge','quality_timecourse','seven_diet_meals','hppmeal','research_records','scripts']]
        for f in protected:
            f=f.resolve();require(output!=f and f not in output.parents,'Output root must be outside frozen runs and code/archive folders')
        print('CHECK frozen packages, source/result hashes and data contracts',flush=True)
        expected=bundle()
        out=output/datetime.now(timezone.utc).strftime('response_followup_%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
        print('RUN_DIR='+str(out),flush=True)
        root,c,paths,checked,audit=inspect(source,a.data_root,out);expected.update(checked)
        run=Run(root,out,c);state['version']=VERSION;run.status['response_followup']=state;run.save();summary(run,state)
        record_expected(run,expected);run.table(audit,'00_preflight.csv')
        write_json(out/'analysis_spec.json',dict(version=VERSION,data_kind=c['data_kind'],nutrition_run=str(source),
            stage=a.stage,planned_interaction_transformations=42,new_association_models=0,
            support='Outcome-blind score Q25/Q75; overlap of group meal-energy IQRs then carbohydrate P10/P90; pooled conditional carb Q25/Q75; nonoverlapping +/-5g bands; >=50 per endpoint cell and >=30 within-person crossers/group',
            uncertainty='Pointwise CI of original single interaction scaled, conditional on observed contrast anchors; no new p/q',
            disjoint='Provisional first-half local-calendar dates for meals, later half for CGM; same connection; actual endpoint interpolation brackets must precede cut',
            feasibility_thresholds=dict(x_meals=5,x_dates=3,y_full_days=3,y_daily_coverage=.9,MODD_proxy_adjacent_pairs=2,MODD_proxy_common_clock_bins=48),
            original_CGM_definition_status='Not confirmed for window recomputation; no new phenotype or inference calculated',
            sources={k:str(v) for k,v in paths.items()}))
        meals,people,cgm,tables=load(run,paths)
        if a.stage=='check':
            unchanged(run);state.update(status='CHECK_PASS',reason='Sources loaded; membership/feasibility requires all stage')
        else:
            result=interaction(meals,people,tables,lambda x:print(x,flush=True))
            result.update(feasibility(run,paths,meals,people,cgm,tables,lambda x:print(x,flush=True)))
            save(run,result);figures(run,result)
            count=int(result['contrasts'].status.eq('SUPPORTED_CONTRAST').sum())
            state.update(status='COMPLETE_WITH_LIMITATIONS',supported_contrasts=count,planned_contrasts=42,
                membership_checks=14,planned_new_models=0,reason='Follow-up completed; date-split counts remain provisional and original CGM window definitions need review')
            pages(run,result,state)
            print('VERIFY frozen inputs unchanged',flush=True);unchanged(run)
            state['result_sha256']={str(f.relative_to(out)):sha256(f) for folder in ['tables','private','figures','pages'] for f in sorted((out/folder).glob('*')) if f.is_file()}
        run.save();summary(run,state)
        print('RESPONSE_FOLLOWUP={}\nDATA={}\nSUMMARY={}'.format(state['status'],c['data_kind'],out/'SUMMARY.md'),flush=True)
        print('VIEW: python3 response_followup/run_analysis.py --show "{}" --page 1'.format(out),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as e:
        state.update(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED',reason=str(e) or 'Interrupted')
        if out is not None:
            (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
            if run is not None:run.save();summary(run,state)
        print('RESPONSE_FOLLOWUP ERROR: '+state['reason'],file=sys.stderr,flush=True)
        return 130 if isinstance(e,KeyboardInterrupt) else 2

if __name__=='__main__':sys.exit(main())
