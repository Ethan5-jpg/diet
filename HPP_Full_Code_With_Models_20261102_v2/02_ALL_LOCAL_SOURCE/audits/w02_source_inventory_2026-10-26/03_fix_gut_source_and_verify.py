"""Resolve the observed gut filename; preserve initial evidence and verify W02.

All remote manifests/dictionaries remain inside TRE. No participant rows read.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path

LOG = Path('/tmp/hpp_w02_b1da099d.log')
EXPECTED_INITIAL = 'b1da099df73e101f36410987cb0bc7d210f912303bf0de919876c20a3c26e5f0'
OLD_REL = 'gut_microbiome/abundance/metaphlan4/aggregated/abundance_species.parquet'
NEW_REL = 'gut_microbiome/abundance/metaphlan4/aggregated/metaphlan_abundance_species.parquet'

def main():
    initial = json.loads([x[len('W02_SUMMARY '):] for x in LOG.read_text().splitlines() if x.startswith('W02_SUMMARY ')][-1])
    manifest = Path(initial['source_manifest'])
    bundle = json.loads(manifest.read_text())
    root = Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
    run = initial['run_id']
    previous_code = Path(initial['code_path'])
    code = previous_code.read_bytes()
    assert hashlib.sha256(code).hexdigest() == EXPECTED_INITIAL
    assert code.count(OLD_REL.encode()) == 1
    corrected = previous_code.with_name('02_source_inventory_corrected.py')
    corrected.write_bytes(code.replace(OLD_REL.encode(), NEW_REL.encode()))
    expected_corrected = hashlib.sha256(code.replace(OLD_REL.encode(), NEW_REL.encode())).hexdigest()
    assert hashlib.sha256(corrected.read_bytes()).hexdigest() == expected_corrected
    namespace = runpy.run_path(str(corrected))
    work = namespace['L'] / run / 'gut_source_correction'
    work.mkdir(parents=True, exist_ok=False)
    new = namespace['guarded_meta'](namespace['R']/NEW_REL, work, 0)
    if new['status'] != 'metadata_read':
        namespace['dump'](root/'data_qc'/run/'gut_source_correction_failed.json', new)
        print('W02_GUT_CORRECTION_FAILED', new['status'], flush=True)
        return 3
    new.update(role='gut_species', required=True, historical_comparison={},
               delta_status='planned_filename_corrected_and_metadata_verified',
               previous_planned_path=str(namespace['R']/OLD_REL),
               quality_state='source_schema_available_semantics_and_participant_QC_pending')
    for artifact in (manifest, Path(initial['source_delta']), Path(initial['read_plan']),
                     Path(initial['report']), Path(initial['memo']), manifest.parent/'summary.json', manifest.parent/'artifact_manifest.json'):
        backup = artifact.with_name(artifact.stem + '.initial_partial' + artifact.suffix)
        assert not backup.exists(), 'Correction already attempted; inspect existing evidence'
        backup.write_bytes(artifact.read_bytes())
        assert backup.read_bytes() == artifact.read_bytes()
    bundle['sources'] = [new if x['role']=='gut_species' else x for x in bundle['sources']]
    bundle['revision'] = 2
    bundle['correction'] = {'reason': 'old planned filename did not exist; exact observed filename carries metaphlan_ prefix',
                            'initial_exit_code': 2, 'participant_rows_read': 0, 'new_source_only_rechecked': 'gut_species',
                            'corrected_collector': str(corrected), 'corrected_collector_sha256': expected_corrected}
    namespace['dump'](manifest, bundle)
    import csv
    delta_path = Path(initial['source_delta'])
    with delta_path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row['role']=='gut_species':
            row.update(path=new['path'],status=new['status'],delta_status=new['delta_status'],
                       inheritance='old planned path corrected; current metadata verified, abundance QC still pending')
    namespace['csv_save'](delta_path, rows, fields)
    plan_path = Path(initial['read_plan'])
    plan = json.loads(plan_path.read_text())
    for row in plan['future_data_reads']:
        if row['role']=='gut_species':
            row.update(path=new['path'],candidate_columns=new['candidate_selected_columns'],
                       projected_uncompressed_bytes=new['selected_uncompressed_bytes'])
    plan['revision'] = 2
    namespace['dump'](plan_path, plan)
    summary = dict(initial)
    summary.update(metadata_read=sum(x['status']=='metadata_read' for x in bundle['sources']),
                   required_errors=[x['role'] for x in bundle['sources'] if x['required'] and x['status']!='metadata_read'],
                   status='W02_SOURCE_INVENTORY_COMPLETE', revision=2,
                   corrected_gut_path=new['path'], corrected_collector=str(corrected),
                   corrected_collector_sha256=expected_corrected, initial_inventory_exit_code=2)
    assert summary['metadata_read']==summary['source_files'] and not summary['required_errors']
    repair_path = previous_code.with_name('03_fix_gut_source_and_verify.py')
    repair_path.write_bytes(Path(__file__).read_bytes())
    summary['verification_code_path'] = str(repair_path)
    report = Path(initial['report'])
    report.write_text('# W02 source inventory: completed after targeted path correction\n\n'+json.dumps(summary,indent=2)+
                      '\n\nInitial 25/26 metadata reads and exit 2 preserved. Only gut species was reread after correcting its observed filename. Final 26/26 metadata sources; participant rows read 0.\n'
                      'Matching historical metadata is not proof of unchanged data-page values. Clinical semantics, cohort N, CGM point quality, FoodID/NOVA coverage and microbiome technical QC remain pending.\n')
    memo = Path(initial['memo'])
    memo.write_text(memo.read_text()+
        '\n\n## Targeted correction (revision 2)\n\nCode: '+str(repair_path)+
        '\nCorrected collector: '+str(corrected)+'\nCorrected collector SHA256: '+expected_corrected+
        '\nObserved gut source: '+new['path']+
        '\nInitial collector exit 2 remains historical; targeted correction succeeded. No other source was rescanned; 0 participant rows read.\n'
        'Final results: '+str(manifest)+', '+str(delta_path)+', '+str(plan_path)+
        '\n\nOriginal code / initial_partial files preserved. Final confirmation or scientific model has not been executed.\n')
    namespace['dump'](manifest.parent/'summary.json', summary)
    logs = root/'logs'/run
    logs.mkdir(exist_ok=True)
    (logs/'02_inventory_console.log').write_bytes(LOG.read_bytes())
    (logs/'02_inventory_exit_code.txt').write_text('2\n')
    artifacts = [manifest,delta_path,plan_path,report,memo,previous_code,corrected,repair_path,
                 manifest.parent/'summary.json',logs/'02_inventory_console.log',logs/'02_inventory_exit_code.txt']
    receipts = [{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts]
    namespace['dump'](manifest.parent/'artifact_manifest.json', receipts)
    verified = all(Path(x['path']).stat().st_size==x['bytes'] and
                   hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
    assert verified
    verification = {'run_id':run,'revision':2,'status':'W02_SOURCE_INVENTORY_COMPLETE',
                    'initial_exit_code':2,'targeted_correction_exit_code':0,
                    'artifact_hashes_verified':verified,'artifacts_verified':len(receipts),
                    'metadata_sources_read':summary['metadata_read'],'source_files':summary['source_files'],
                    'historical_compared':summary['historical_compared'],
                    'historical_metadata_matches':summary['historical_metadata_matches'],
                    'dictionary_files':summary['dictionary_files'],'dictionary_candidates':summary['dictionary_candidates'],
                    'cgm_fragment_files':summary['cgm_fragment_files'],
                    'participant_rows_read':0,'gut_source':new['path'],
                    'corrected_collector_sha256':expected_corrected,'code_directory':str(previous_code.parent),
                    'manifest_directory':str(manifest.parent),'results_directory':str(report.parent),
                    'read_plan':str(plan_path),'memo':str(memo),
                    'scientific_QC_or_models_completed':False}
    namespace['dump'](report.parent/'verification.json', verification)
    print('\n'*35+'W02 FINAL VERIFICATION',flush=True)
    for key in ('run_id','status','metadata_sources_read','source_files','historical_metadata_matches',
                'dictionary_files','dictionary_candidates','cgm_fragment_files','participant_rows_read',
                'artifact_hashes_verified','gut_source','code_directory','manifest_directory','results_directory','memo'):
        print(key, verification[key],flush=True)
    print('W02_TARGETED_CORRECTION_EXIT_CODE_0',flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
