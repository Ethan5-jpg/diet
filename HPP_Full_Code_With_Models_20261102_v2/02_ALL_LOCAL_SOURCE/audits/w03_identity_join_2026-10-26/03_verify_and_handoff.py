"""Verify W03 readback/conservation; dictionary provenance; snapshot logs and handoff."""
import csv,hashlib,json,re,sqlite3,sys
from collections import Counter
from pathlib import Path
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L=Path('/home/ec2-user/multisystem_runtime_20261021')

def save(p,v):
 b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b

def main():
 exit_path=Path('/tmp/hpp_w03_f0cadbf2.exit');assert exit_path.read_text().strip()=='0'
 log=Path('/tmp/hpp_w03_f0cadbf2.log');run=re.search(r'^W03_RUN (\S+)',log.read_text(),re.M).group(1)
 m=P/'manifests'/run;q=P/'data_qc'/run;cfg=P/'config'/run;work=L/run
 summary=json.loads((m/'summary.json').read_text());sources=json.loads((q/'source_key_date_summary.json').read_text())
 assert summary['sources_full_scanned']==summary['sources_planned']==14 and not summary['failures']
 receipts=json.loads((m/'artifact_manifest.json').read_text());assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
 conn=sqlite3.connect(work/'identity.sqlite')
 conservation=[]
 for s in sources:
  good=conn.execute('SELECT COALESCE(SUM(n),0) FROM records WHERE role=?',(s['role'],)).fetchone()[0]
  assert good+s['invalid_identity_rows']==s['rows_scanned']
  conservation.append({'role':s['role'],'raw_projected_rows':s['rows_scanned'],'valid_rows':good,'invalid_rows':s['invalid_identity_rows'],'passed':True})
 invalid_fields=Counter()
 for role,detail in conn.execute("SELECT role,detail FROM exceptions WHERE reason='invalid_identity_or_visit'"):
  values=json.loads(detail)
  for field,value in values.items():
   if value is None or str(value).strip().lower() in {'','undefined','none','null','nan','na','n/a','unknown','<na>'} or str(value)!=str(value).strip():invalid_fields[(role,field)]+=1
 invalid_breakdown=[{'role':r,'field':f,'rows':n} for (r,f),n in sorted(invalid_fields.items())]
 # Version-key/date comparison does not imply equality of nonprojected feature values.
 x=conn.execute("SELECT COUNT(*) FROM (SELECT pv,entity,day,n FROM records WHERE role='oral_index' EXCEPT SELECT pv,entity,day,n FROM records WHERE role='oral_index_alternate')").fetchone()[0]
 y=conn.execute("SELECT COUNT(*) FROM (SELECT pv,entity,day,n FROM records WHERE role='oral_index_alternate' EXCEPT SELECT pv,entity,day,n FROM records WHERE role='oral_index')").fetchone()[0]
 manifest=json.loads((P/'manifests/w02_sources_20261026T130908Z/source_manifest.json').read_text())
 definitions=[]; wanted={'participant_id','cohort','research_stage','connection_id','collection_date','collection_timestamp','local_timestamp','timezone','sample_id','uuid','visit_id'}
 for item in manifest['dictionary_inventory']:
  p=Path(item['path'])
  if item.get('status')!='csv_dictionary_read' or p.stat().st_size>4*1024**2:continue
  with p.open(encoding='utf-8-sig',newline='') as f:
   for row_no,row in enumerate(csv.DictReader(f),2):
    if row_no>10001:break
    if any(str(value) in wanted for value in row.values()):definitions.append({'path':str(p),'module':item['module'],'line':row_no,'row':row,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 save(q/'key_date_dictionary_evidence_TRE_ONLY.json',definitions)
 contract=json.loads((cfg/'key_contract.yaml').read_text())
 contract.update(dictionary_evidence=str(q/'key_date_dictionary_evidence_TRE_ONLY.json'),
  oral_version_key_date_comparison={'left_only_groups':x,'right_only_groups':y,'scope':'projected keys/date/multiplicity only; do not merge versions or infer all feature values equal'},
  join_exception_disposition={'invalid_person_or_visit':'quarantine source row; never impute undefined','multivalued_person_visit':'known grain, aggregate/select in dedicated downstream task; never join both many sides','entity_to_multiple_person_visits':'stop affected connection/sample mapping until explicit source evidence resolves; never transfer identities','multiple_dates':'retain; not exclusion on its own','unmatched_abundance_and_index':'hold unresolved source records for W11; no nearest-person/date identity invention'})
 save(cfg/'key_contract.yaml',contract)
 verify={'run_id':run,'actual_audit_exit':0,'full_sources':14,'rows_projected_scanned':summary['rows_projected_scanned'],'source_row_conservation':conservation,'invalid_key_field_counts':invalid_breakdown,'dictionary_exact_field_evidence_rows':len(definitions),'oral_version_left_only_key_date_groups':x,'oral_version_right_only_key_date_groups':y,'no_outcome_values_read':True,'no_raw_many_to_many_join_materialized':True}
 save(q/'verification.json',verify)
 code=P/'scripts'/run/'03_verify_and_handoff.py';code.write_bytes(Path(__file__).read_bytes());assert code.read_bytes()==Path(__file__).read_bytes()
 logs=P/'logs'/run
 (logs/'02_audit_console.log').write_bytes(log.read_bytes());(logs/'02_audit_exit_code.txt').write_bytes(exit_path.read_bytes())
 for f in work.glob('*.log'):(logs/f.name).write_bytes(f.read_bytes())
 memo=P/'memos'/run/'MEMO.md';memo.write_text(memo.read_text()+'\nVerifier: '+str(code)+'\nActual audit exit 0 read back. All 14 source row conservations verified; dictionary evidence and projected oral version differences recorded.\nConsole and per-source logs snapshotted under '+str(logs)+'\nNo final sample choice or discovery/confirmation roles frozen.\n')
 report=q/'REPORT.md';report.write_text(report.read_text()+'\n\nVerification:\n'+json.dumps(verify,indent=2))
 files={x['path'] for x in receipts}|{str(q/'verification.json'),str(q/'key_date_dictionary_evidence_TRE_ONLY.json'),str(code)}|{str(p) for p in logs.iterdir() if p.is_file()}
 receipts=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in map(Path,sorted(files))]
 save(m/'artifact_manifest.json',receipts);assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
 statepath=P/'PROJECT_EXECUTION_STATE.json';state=json.loads(statepath.read_text());state['W03']={'status':'key_contract_complete_with_downstream_gates','run_id':run,'config':str(cfg/'key_contract.yaml'),'result_directory':str(q),'memo':str(memo),'actual_audit_exit':0,'not_final_eligible_cohorts':True};save(statepath,state)
 (P/'W03_HANDOFF.md').write_text('# W03 complete with downstream gates\n\n'+json.dumps(state['W03'],indent=2)+'\n\nKeep individual exceptions and SQLite in TRE. Local aggregate notes may be read without reconnecting. W06 ingestion/date semantics, W09 raw points, W10 phenotype choice and W11 biological sample/version/QC remain gated.\n')
 print('\n'*20+'W03_VERIFIED',run,'AUDIT_ACTUAL_EXIT',exit_path.read_text().strip(),flush=True)
 for s in sources:
  print('SOURCE',s['role'],s['rows_scanned'],s['baseline_10k_person_visits'],s['invalid_identity_rows'],s['person_visits_multiple_rows'],s['person_visits_multiple_dates'],flush=True)
 print('TOTAL_ROWS',summary['rows_projected_scanned'],'IDENTITY_INVALID',summary['invalid_identity_rows'],'CONSERVATION_PASS',len(conservation),'DICT_ROWS',len(definitions),'HASHES_PASS',len(receipts),flush=True)
 print('ORAL_VERSION_DIFF_GROUPS',x,y,flush=True)
 print('INVALID_KEY_FIELDS',json.dumps(invalid_breakdown,separators=(',',':')),flush=True)
 for s in sources:
  if s['date_crosschecks'] or s['entity_to_multiple_person_visits']:print('DATE_OR_ENTITY',s['role'],json.dumps(s['date_crosschecks']),s['entity_to_multiple_person_visits'],flush=True)
 print('W03_VERIFIER_FINISHED',str(q),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
