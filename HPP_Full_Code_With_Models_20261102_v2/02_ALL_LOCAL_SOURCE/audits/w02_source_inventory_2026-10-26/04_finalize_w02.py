"""Finalize W02 log snapshots/readback and project handoff, entirely inside TRE."""
import hashlib
import json
from pathlib import Path

P = Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
RUN = 'w02_sources_20261026T130908Z'
M = P/'manifests'/RUN
Q = P/'data_qc'/RUN

def write_json(path, obj):
    b = (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    path.write_bytes(b)
    assert path.read_bytes()==b

def main():
    summary = json.loads((M/'summary.json').read_text())
    assert summary['status']=='W02_SOURCE_INVENTORY_COMPLETE'
    assert summary['metadata_read']==summary['source_files']==26
    assert summary['participant_rows_read']==0
    exit_path = Path('/tmp/hpp_w02_fix_368e34b9.exit')
    assert exit_path.read_text().strip()=='0', 'Targeted correction has not exited successfully'
    receipts = json.loads((M/'artifact_manifest.json').read_text())
    assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
    code = P/'scripts'/RUN/'04_finalize_w02.py'
    code.write_bytes(Path(__file__).read_bytes())
    assert code.read_bytes()==Path(__file__).read_bytes()
    logs = P/'logs'/RUN
    (logs/'03_correction_console.log').write_bytes(Path('/tmp/hpp_w02_fix_368e34b9.log').read_bytes())
    (logs/'03_correction_exit_code.txt').write_bytes(exit_path.read_bytes())
    memo = Path(summary['memo'])
    memo.write_text(memo.read_text()+'\n\n## Finalization\nCode: '+str(code)+
                    '\nInitial collector exit 2 preserved; targeted single-source correction actual exit 0 independently read back.\n'
                    'All source metadata reads: 26/26. Historical structural comparisons: 8/8 matching. Participant rows read: 0.\n'
                    'Both console logs and actual exit records saved under '+str(logs)+
                    '\nFinalization checks files and traceability only; clinical units, FoodID mapping, actual cohort N, raw CGM QC, models and split remain pending.\n')
    report = Q/'REPORT.md'
    report.write_text(report.read_text()+'\n\nFinal readback: correction exit 0 verified, original exit 2 retained, logs snapshotted to V2. W01 full rerun waived by user; current host/paths verified.\n')
    v = json.loads((Q/'verification.json').read_text())
    v.update(finalizer_code=str(code), actual_targeted_correction_exit_code=0,
             initial_collector_exit_code=2, w01_full_rerun='waived_by_user_reused_existing_environment',
             all_participant_data_remained_in_TRE=True)
    files = {x['path'] for x in receipts}
    files.update(str(p) for p in (code, logs/'03_correction_console.log',logs/'03_correction_exit_code.txt'))
    updated = [{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in sorted(files)]
    write_json(M/'artifact_manifest.json', updated)
    assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in updated)
    v['artifact_hashes_verified']=True
    v['artifacts_verified']=len(updated)
    write_json(Q/'verification.json',v)
    status_path = P/'PROJECT_EXECUTION_STATE.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status['W01_execution_note']='User requested reuse of the same server, no full rerun; minimal current host/path verification and W02 readback completed.'
    status['W02']={'status':'source_inventory_complete','run_id':RUN,
                  'manifest':str(M/'source_manifest.json'),'read_plan':summary['read_plan'],
                  'result_directory':str(Q),'code_directory':str(code.parent),'verification':str(Q/'verification.json'),
                  'scope':'metadata and dictionary inventory only; 0 participant rows; no scientific QC or models',
                  'next':'W03 identity/join contract; W04 feature spec and W05 nutrient semantics; do not use old gut filename'}
    write_json(status_path,status)
    handoff = P/'W02_HANDOFF.md'
    handoff.write_text('# W02 completed: source inventory\n\nRun '+RUN+'\n\nCode: '+str(code.parent)+
                       '\n\nManifest: '+str(M/'source_manifest.json')+'\nRead plan: '+summary['read_plan']+
                       '\nResult: '+str(Q)+'\nMemo: '+str(memo)+
                       '\n\n26/26 source metadata reads; 8/8 historical metadata matches. 29 dictionary files inventoried; 991 term-match candidates, not 991 validated variables.\n'
                       'Gut source filename was corrected to metaphlan_abundance_species.parquet; preserve old failure evidence.\n'
                       'W01 full rerun waived by user. No participant rows read, actual cohorts/split/QC/model remain pending.\n')
    print('\n'*35+'W02 ACCEPTED / SOURCE METADATA ONLY',flush=True)
    for k in ('run_id','metadata_sources_read','source_files','historical_metadata_matches','dictionary_files',
              'dictionary_candidates','participant_rows_read','artifact_hashes_verified','artifacts_verified',
              'code_directory','manifest_directory','results_directory'):
        print(k, v[k],flush=True)
    print('CODE_SHA256',hashlib.sha256(code.read_bytes()).hexdigest(),flush=True)
    print('ACTUAL_CORRECTION_EXIT_CODE',v['actual_targeted_correction_exit_code'],flush=True)
    print('W02_FINALIZATION_EXIT_CODE_0',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
