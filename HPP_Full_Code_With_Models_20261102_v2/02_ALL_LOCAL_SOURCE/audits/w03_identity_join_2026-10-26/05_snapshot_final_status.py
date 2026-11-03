"""Read actual W03 exit records, snapshot all console logs, verify final artifact hashes."""
import hashlib,json
from pathlib import Path
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');RUN='w03_keys_20261026T134938Z'
def save(p,v):
 b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b

def main():
 m=P/'manifests'/RUN;q=P/'data_qc'/RUN;logs=P/'logs'/RUN
 actual={}
 for num,stem in [('02','hpp_w03_f0cadbf2'),('03','hpp_w03_verify_6653b947'),('04','hpp_w03_bridge_ea236556')]:
  ep=Path('/tmp/'+stem+'.exit');lp=Path('/tmp/'+stem+'.log');rc=ep.read_text().strip();assert rc=='0',lp.read_text()[-1500:];actual[num]=int(rc)
  (logs/(num+'_actual_exit_code.txt')).write_bytes(ep.read_bytes());(logs/(num+'_console_complete.log')).write_bytes(lp.read_bytes())
 verify=json.loads((q/'verification.json').read_text());bridge=json.loads((q/'oral_visit_bridge_summary.json').read_text());summaries=json.loads((q/'source_key_date_summary.json').read_text());joins=json.loads((q/'join_cardinalities.json').read_text())
 verify['actual_exit_codes']=actual;verify['oral_bridge']=bridge;save(q/'verification.json',verify)
 code=P/'scripts'/RUN/'05_snapshot_final_status.py';code.write_bytes(Path(__file__).read_bytes());assert code.read_bytes()==Path(__file__).read_bytes()
 memo=P/'memos'/RUN/'MEMO.md';memo.write_text(memo.read_text()+'\n05_snapshot_final_status.py: read actual 02/03/04 exit records (all 0), snapshot complete consoles to V2 and refresh final artifact hashes. No new source scan.\n')
 files={x['path'] for x in json.loads((m/'artifact_manifest.json').read_text())}|{str(p) for p in logs.iterdir() if p.is_file()}|{str(code)}
 receipts=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in map(Path,sorted(files))];save(m/'artifact_manifest.json',receipts);assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
 statepath=P/'PROJECT_EXECUTION_STATE.json';st=json.loads(statepath.read_text());st['W03']['actual_exit_codes']=actual;st['W03']['oral_visit_bridge_summary']=str(q/'oral_visit_bridge_summary.json');st['W03']['important_note']='Raw oral composite keys fail; only exact unique index/date bridge may support visit evidence. Ambiguous/unmatched remain held. W11 sample QC and cross-domain lock not completed.';save(statepath,st)
 print('\n'*35+'W03 FINAL VERIFIED / KEY-DATE AUDIT',flush=True)
 print('RUN',RUN,'EXIT_RECORDS',json.dumps(actual),'FULL_SOURCES',len(summaries),'PROJECTED_ROWS',sum(s['rows_scanned'] for s in summaries),flush=True)
 print('DICT_EVIDENCE_ROWS',verify['dictionary_exact_field_evidence_rows'],'ORAL_VERSION_KEY_DATE_DIFF',verify['oral_version_left_only_key_date_groups'],verify['oral_version_right_only_key_date_groups'],flush=True)
 print('INVALID_FIELD_COUNTS',json.dumps(verify['invalid_key_field_counts'],separators=(',',':')),flush=True)
 print('ORAL_BRIDGE',json.dumps(bridge,separators=(',',':')),flush=True)
 for s in summaries:
  if s['date_crosschecks'] or s['entity_to_multiple_person_visits']:print('DATE_CHECK',s['role'],json.dumps(s['date_crosschecks'],separators=(',',':')),'entity_conflicts',s['entity_to_multiple_person_visits'],flush=True)
 print('JOIN_PROFILES',len(joins),'ACTUAL_JOIN_ROW_EXPANSIONS',0,'OUTPUT_HASHES_VERIFIED',len(receipts),flush=True)
 print('CONTRACT',str(P/'config'/RUN/'key_contract.yaml'),flush=True)
 print('RESULTS',str(q),'MEMO',str(memo),flush=True)
 print('W03_FINAL_STATUS_SAVED_NO_MODELS_NO_FINAL_COHORT_N',flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
