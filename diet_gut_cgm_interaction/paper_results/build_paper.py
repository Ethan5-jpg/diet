#!/usr/bin/env python3
"""Export paper figures/tables from verified aggregate results; never refit models."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

PACKAGE=Path(__file__).resolve().parent
DEFAULT_OLD='run_20261010T085601_096554Z_4d43e0'
DEFAULT_NEW='extension_20261010T105930_496220Z_ff8085'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path)
    p.add_argument('--extension-run',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--preview',action='store_true',help='Use packaged screenshot values, clearly labelled as preview')
    p.add_argument('--allow-synthetic',action='store_true',help='Testing only: requires SYNTHETIC input runs and labels every artifact')
    args=p.parse_args(argv)
    if args.preview and (args.reference_run or args.extension_run or args.allow_synthetic):p.error('--preview cannot be combined with server/test inputs')
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
    try:
        from paperlib import VERSION
        from paperlib.data import load_preview,load_server,recheck_sources,sha,true,KEY,BETA,LOW,HIGH
        from paperlib.figures import make_figures
        from paperlib.report import write_report,num
    except ImportError as error:
        print('Dependency missing: {}\nInstall with: {} -m pip install -r {}'.format(error,sys.executable,PACKAGE/'requirements.txt'),file=sys.stderr)
        return 2
    old=(args.reference_run or PACKAGE.parent/'outputs'/DEFAULT_OLD).expanduser().resolve()
    new=(args.extension_run or PACKAGE.parent/'extension_sensitivity/outputs_extension_sensitivity'/DEFAULT_NEW).expanduser().resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    default=PACKAGE/'preview_from_screenshots' if args.preview else PACKAGE/('outputs_test' if args.allow_synthetic else 'outputs_server')/('paper_'+stamp)
    out=(args.output_dir or default).expanduser().resolve()
    if out.exists():p.error('Output directory already exists; choose a new directory: '+str(out))
    if not args.preview and any(out==r or r in out.parents or out in r.parents for r in [old,new]):p.error('Output must be separate from source runs')
    out.mkdir(parents=True)
    for folder in ['figures','tables','source_data']:(out/folder).mkdir()
    code={str(p.relative_to(PACKAGE)):sha(p) for pattern in ['*.py','*.sh','paperlib/*.py','requirements.txt','screenshot_sources/*.json'] for p in PACKAGE.glob(pattern)}
    manifest=dict(version=VERSION,status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat())
    def save(): (out/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        bundle=load_preview(PACKAGE) if args.preview else load_server(old,new,args.allow_synthetic)
        print('PAPER INPUT_CHECK | '+bundle['meta']['mode'],flush=True)
        figures=make_figures(bundle,out)
        tables=write_report(bundle,out,figures)
        recheck_sources(bundle)
        if not all(sha(PACKAGE/name)==digest for name,digest in code.items()):raise ValueError('Export code changed during execution')
        meta=bundle['meta'];primary=bundle['primary'];st=bundle['stability'];focus=primary[primary.family.eq('focus')]
        selected=tables['Table_4_targets_retained_all_six']
        one=['PAPER 1/2 | '+meta['mode']+' | EXPORT_COMPLETE',
             'Source runs: '+meta['original_run']+' / '+meta['followup_run'],
             'Primary rows available={}/84; focus={}/24; original extension targets={}'.format(len(primary),len(focus),len(st)),
             'Focus qFocus24<0.05={}; primary qAll84<0.05={}; all-six retained={}'.format(int(focus.q_focus24.lt(.05).sum()),int(primary.q_all84.lt(.05).sum()),len(selected)),
             'Figures={} (PDF/SVG/PNG); tables={}; each figure has its own Source Data CSV'.format(len(figures),len(tables)),
             'Source hashes checked; source runs unchanged; no model fits or participant-data export.']
        if not args.preview:
            one.append('Saved reference reproduction={} / 204; full output table coverage verified.'.format(meta['reference_reproduced']))
        else:one.append('PREVIEW ONLY: 41 unseen primary rows and sensitivity coefficients not reconstructed.')
        two=['PAPER 2/2 | TARGETS RETAINED IN ALL SIX | '+meta['mode'],
             'Diet Taxon Y N Primary_beta [pointwise 95% CI] qAll84 maxSensitivityQ']
        for _,r in selected.iterrows():
            two.append('{} {} {} {} {} [{}, {}] {} {}'.format(r.exposure,r.taxon,r.outcome,int(r.n),num(r[BETA]),num(r[LOW]),num(r[HIGH]),num(r.q_all84),num(r.max_sensitivity_q_all84)))
        check=primary[(primary.exposure=='rEDIH')&(primary.taxon=='T096')&(primary.outcome=='CV')]
        if len(check):two.append('Saved primary q: rEDIH/T096/CV qAll84={:.10g}'.format(check.q_all84.iloc[0]))
        extras=tables.get('Table_S7_new_signals_outside_original')
        if extras is not None:
            for _,r in extras.head(8).iterrows():two.append('New outside original: {} {} {} {} q={}'.format(r.model,r.exposure,r.taxon,r.outcome,num(r.q_all84)))
        two.append('rEDIH provenance/timing and same-cohort selection limits remain; see Results_and_methods.md.')
        for i,lines in enumerate([one,two],1):(out/('SCREEN_'+str(i)+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (out/'SCREENSHOT_SUMMARY.txt').write_text('\n\n'.join('\n'.join(x) for x in [one,two])+'\n',encoding='utf-8')
        # Public provenance contains aggregate file identities, never input paths or participant hashes.
        source_hashes={}
        for index,(path,digest) in enumerate(meta['source_files'].items(),1):
            part=Path(path); source_hashes[str(index)+'_'+part.parent.name+'_'+part.name]=digest
        manifest.update(status='EXPORT_COMPLETE',data_mode=meta['mode'],source_run_ids=[meta['original_run'],meta['followup_run']],
            source_aggregate_sha256=source_hashes,code_sha256=code,figures=figures,
            table_rows={n:len(frame) for n,frame in tables.items()},primary_rows_available=len(primary),
            focus_significant=int(focus.q_focus24.lt(.05).sum()),extension_targets=len(st),retained_all_six=len(selected),
            limitations=meta.get('limitations',[]),source_records=meta.get('source_records',{}),
            reference_reproduced=meta['reference_reproduced'])
        manifest['output_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='export_manifest.json'}
        save()
        pointer=PACKAGE/('LATEST_PREVIEW.txt' if args.preview else 'LATEST_TEST.txt' if args.allow_synthetic else 'LATEST_SERVER.txt')
        pointer.write_text(str(out)+'\n',encoding='utf-8')
        print('\n'+(out/'SCREENSHOT_SUMMARY.txt').read_text())
        print('PAPER_RESULT_DIR='+str(out))
        return 0
    except Exception as error:
        manifest.update(status='FAILED',error=str(error));save()
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('FAILED: '+str(error)+'\nDetails: '+str(out/'error_traceback.txt'),file=sys.stderr)
        return 2


if __name__=='__main__':sys.exit(main())
