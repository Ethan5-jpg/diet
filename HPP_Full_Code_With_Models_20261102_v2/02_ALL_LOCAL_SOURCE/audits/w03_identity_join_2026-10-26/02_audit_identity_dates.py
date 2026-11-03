"""W03 TRE-only bounded key/date scan and guarded join contract (no outcomes).

Full projection scan for 14 sources; raw food events reserved for W06 editing QC.
Individual keys/rows remain in TRE. Does not select microbiome samples or train.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import csv, hashlib, json, resource, sqlite3, subprocess, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone, date
from pathlib import Path
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L=Path('/home/ec2-user/multisystem_runtime_20261021')
W02='w02_sources_20261026T130908Z'
KEY=['participant_id','cohort','research_stage']
DATE=['collection_date','collection_timestamp','local_timestamp']
EXTRA=['timezone','connection_id','sample_id','uuid','visit_id']
ROLES=['food_index','food_events','cgm_connections','cgm_window_legacy','cgm_daily_legacy','anthropometrics','body_composition','liver_detail','liver_aggregated','oral_index','oral_index_alternate','gut_index','oral_species','gut_species']
BAD={'','undefined','null','none','nan','na','n/a','unknown','<na>'}

def write_json(p,v):
    b=(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n').encode(); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b); assert p.read_bytes()==b

def safe_date(x):
    if isinstance(x,datetime): return x.date().isoformat()
    if isinstance(x,date): return x.isoformat()
    return ''

def valid_key(vals):
    if any(x is None or str(x).strip().lower() in BAD or str(x)!=str(x).strip() for x in vals): return False
    try: return int(vals[0])>0
    except (ValueError,TypeError): return False

def ident(vals): return json.dumps([str(x) for x in vals],separators=(',',':'))

def worker(manifest,role,dbpath,output):
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    import pyarrow as pa
    import pyarrow.parquet as pq
    e=next(x for x in json.loads(Path(manifest).read_text())['sources'] if x['role']==role)
    path=Path(e['path']); st=path.stat()
    assert st.st_size==e['bytes'] and st.st_mtime_ns==e['mtime_ns'],'Source changed since W02'
    pf=pq.ParquetFile(path,pre_buffer=False)
    assert pf.metadata.num_rows==e['rows_metadata']
    cols=[c for c in KEY+DATE+EXTRA if c in pf.schema_arrow.names]
    assert all(c in cols for c in KEY),'Missing explicit person/cohort/stage; never infer'
    projection_bytes=sum(pf.metadata.row_group(g).column(c).total_uncompressed_size for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.row_group(g).num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in cols)
    assert projection_bytes<=1024**3,'Projection exceeds 1GiB streaming budget'
    conn=sqlite3.connect(dbpath); conn.execute('PRAGMA cache_size=-12000')
    conn.execute('CREATE TABLE IF NOT EXISTS records(role TEXT,pv TEXT,entity TEXT,day TEXT,n INTEGER,PRIMARY KEY(role,pv,entity,day)) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS exceptions(role TEXT,ordinal INTEGER,reason TEXT,detail TEXT)')
    n=invalid=0; flags=Counter(); stages=Counter(); zones=Counter(); date_stats={c:Counter() for c in DATE if c in cols}; cross=Counter()
    numeric_dates=[c for c in DATE if c in cols and not (pa.types.is_timestamp(pf.schema_arrow.field(c).type) or pa.types.is_date(pf.schema_arrow.field(c).type))]
    date_types={c:str(pf.schema_arrow.field(c).type) for c in DATE if c in cols}
    primary='collection_date' if 'collection_date' in cols else 'collection_timestamp' if 'collection_timestamp' in cols and 'collection_timestamp' not in numeric_dates else None
    start=time.monotonic()
    for b in pf.iter_batches(batch_size=2048,columns=cols,use_threads=False):
        values={c:b.column(b.schema.get_field_index(c)).to_pylist() for c in cols}; groups=Counter(); errors=[]
        for i in range(b.num_rows):
            n+=1; kv=[values[c][i] for c in KEY]
            if not valid_key(kv):
                invalid+=1; errors.append((role,n,'invalid_identity_or_visit',json.dumps(dict(zip(KEY,kv)),default=str))); continue
            pv=ident(kv); stages[(str(kv[1]),str(kv[2]))]+=1
            if 'timezone' in cols and values['timezone'][i] is not None: zones[str(values['timezone'][i])]+=1
            dates={}
            for c,counts in date_stats.items():
                x=values[c][i]
                if x is None: counts['missing']+=1; dates[c]=''; continue
                if c in numeric_dates: counts['numeric_or_unparsed_not_interpreted']+=1; dates[c]=''; continue
                d=safe_date(x); dates[c]=d
                if not d: counts['unsupported_value']+=1
                else:
                    counts['valid']+=1
                    if isinstance(x,datetime):
                        counts['aware' if x.tzinfo is not None else 'naive']+=1
                        if x.hour==x.minute==x.second==x.microsecond==0: counts['midnight']+=1
            if dates.get('collection_date') and dates.get('collection_timestamp'):
                cross['date_vs_aware_timestamp_compared']+=1
                if dates['collection_date']!=dates['collection_timestamp']: cross['date_vs_aware_timestamp_conflicts']+=1
            if dates.get('collection_date') and dates.get('local_timestamp'):
                cross['date_vs_local_timestamp_compared']+=1
                if dates['collection_date']!=dates['local_timestamp']: cross['date_vs_local_timestamp_conflicts']+=1
            entity=''
            for c in ['connection_id','sample_id','uuid','visit_id']:
                if c in cols and values[c][i] is not None and str(values[c][i]).strip().lower() not in BAD:
                    entity=c+':'+str(values[c][i]); break
            if 'connection_id' in cols and not entity:
                flags['missing_connection_id']+=1; errors.append((role,n,'missing_connection_id',pv))
            groups[(role,pv,entity,dates.get(primary,''))]+=1
        conn.executemany('INSERT INTO records VALUES(?,?,?,?,?) ON CONFLICT(role,pv,entity,day) DO UPDATE SET n=n+excluded.n',[(r,p,e,d,k) for (r,p,e,d),k in groups.items()])
        conn.executemany('INSERT INTO exceptions VALUES(?,?,?,?)',errors); conn.commit()
        if time.monotonic()-start>1200: raise RuntimeError('Full projection scan exceeded 1200s; incomplete not accepted')
    assert n==pf.metadata.num_rows,'Incomplete scan'
    assert path.stat().st_size==st.st_size and path.stat().st_mtime_ns==st.st_mtime_ns,'Source changed during scan'
    scalar=lambda sql,args=():conn.execute(sql,args).fetchone()[0]
    pv_count=scalar('SELECT COUNT(*) FROM (SELECT pv FROM records WHERE role=? GROUP BY pv)',(role,))
    multi=scalar('SELECT COUNT(*) FROM (SELECT pv FROM records WHERE role=? GROUP BY pv HAVING SUM(n)>1)',(role,))
    multiday=scalar("SELECT COUNT(*) FROM (SELECT pv FROM records WHERE role=? AND day<>'' GROUP BY pv HAVING COUNT(DISTINCT day)>1)",(role,))
    entity_conflicts=scalar("SELECT COUNT(*) FROM (SELECT entity FROM records WHERE role=? AND entity<>'' GROUP BY entity HAVING COUNT(DISTINCT pv)>1)",(role,))
    base=0
    for (pv,) in conn.execute('SELECT DISTINCT pv FROM records WHERE role=?',(role,)):
        k=json.loads(pv)
        if k[1:] == ['10k','00_00_visit']: base+=1
    out={'role':role,'path':str(path),'rows_scanned':n,'columns':cols,'projection_uncompressed_bytes':projection_bytes,'invalid_identity_rows':invalid,
         'person_visits':pv_count,'baseline_10k_person_visits':base,'person_visits_multiple_rows':multi,'person_visits_multiple_dates':multiday,
         'entity_to_multiple_person_visits':entity_conflicts,'flags':dict(flags),'date_types':date_types,'date_counts':{k:dict(v) for k,v in date_stats.items()},
         'date_crosschecks':dict(cross),'timezone_values':dict(zones),'cohort_stage_counts':[{'cohort':k[0],'research_stage':k[1],'rows':v} for k,v in stages.items()],
         'numeric_date_columns_not_interpreted':numeric_dates,'primary_calendar_date':primary,'explicit_sample_or_connection_fields':[c for c in EXTRA[1:] if c in cols],
         'elapsed_s':round(time.monotonic()-start,2),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'full_scan':True}
    write_json(Path(output),out); conn.close(); return 0

def join_profile(conn,left,right,grain):
    # Aggregate both sides BEFORE matching to audit potential expansion without materializing it.
    fields='pv,day' if grain=='person_visit_day' else 'pv,entity' if grain=='connection' else 'pv'
    a={tuple(row[:-1]):row[-1] for row in conn.execute('SELECT '+fields+',SUM(n) FROM records WHERE role=? GROUP BY '+fields,(left,))}
    b={tuple(row[:-1]):row[-1] for row in conn.execute('SELECT '+fields+',SUM(n) FROM records WHERE role=? GROUP BY '+fields,(right,))}
    common=a.keys()&b.keys()
    many_many=sum(a[k]>1 and b[k]>1 for k in common)
    return {'left':left,'right':right,'grain':grain,'left_keys':len(a),'right_keys':len(b),'matched_keys':len(common),'left_unmatched_keys':len(a.keys()-b.keys()),'right_unmatched_keys':len(b.keys()-a.keys()),
            'raw_matched_row_pairs':sum(a[k]*b[k] for k in common),'many_to_many_keys_if_raw_join':many_many,
            'guard':'Never materialize raw many-to-many. Reduce/select to declared grain, require unique right keys and assert post-join row conservation.'}

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker': return worker(*sys.argv[2:])
    assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
    run='w03_keys_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); work=L/run; work.mkdir(parents=True,exist_ok=False)
    for name in ['scripts','manifests','config','data_qc','memos','logs']: (P/name/run).mkdir(parents=True,exist_ok=True)
    code=P/'scripts'/run/'02_audit_identity_dates.py'; code.write_bytes(Path(__file__).read_bytes()); assert code.read_bytes()==Path(__file__).read_bytes()
    manifest=P/'manifests'/W02/'source_manifest.json'; db=work/'identity.sqlite'; summaries=[]; failures=[]
    print('W03_RUN',run,flush=True)
    for role in ROLES:
        out=work/(role+'.json'); err=work/(role+'.log')
        with err.open('wb') as f:
            proc=subprocess.Popen([sys.executable,str(code),'--worker',str(manifest),role,str(db),str(out)],stdout=f,stderr=subprocess.STDOUT)
            t=time.monotonic(); stop=None
            while proc.poll() is None:
                try:
                    rss=int(next(x.split()[1] for x in Path('/proc/'+str(proc.pid)+'/status').read_text().splitlines() if x.startswith('VmRSS:')))
                    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
                    if rss>768*1024 or available<1536*1024: stop='memory_guard'
                except (FileNotFoundError,StopIteration): pass
                if time.monotonic()-t>1260: stop='timeout'
                if stop:
                    proc.terminate(); proc.wait(); break
                time.sleep(.3)
        rc=proc.returncode
        if rc or not out.exists():
            failures.append({'role':role,'exit':rc,'guard':stop,'log':str(err)}); print('SOURCE_FAILED',role,rc,flush=True)
        else:
            s=json.loads(out.read_text()); summaries.append(s); print('SOURCE_PASS',role,s['rows_scanned'],s['baseline_10k_person_visits'],'invalid',s['invalid_identity_rows'],flush=True)
    q=P/'data_qc'/run; cfg=P/'config'/run; m=P/'manifests'/run
    conn=sqlite3.connect(db); joins=[]
    present={s['role'] for s in summaries}
    specs=[('food_events','food_index','person_visit_day'),('cgm_daily_legacy','cgm_connections','connection'),('cgm_window_legacy','cgm_connections','connection'),('oral_species','oral_index','person_visit'),('oral_index','oral_index_alternate','person_visit'),('gut_species','gut_index','person_visit'),('liver_aggregated','liver_detail','person_visit'),('food_events','cgm_daily_legacy','person_visit_day'),('body_composition','food_events','person_visit'),('liver_aggregated','food_events','person_visit')]
    for a,b,g in specs:
        if a in present and b in present: joins.append(join_profile(conn,a,b,g))
    write_json(q/'source_key_date_summary.json',summaries); write_json(q/'join_cardinalities.json',joins)
    import pyarrow as pa
    import pyarrow.parquet as pq
    exception_file=q/'identity_exceptions_TRE_ONLY.parquet'
    sch=pa.schema([('role',pa.string()),('source_row_ordinal',pa.int64()),('reason',pa.string()),('detail_TRE_ONLY',pa.string())])
    writer=pq.ParquetWriter(exception_file,sch); cur=conn.execute('SELECT role,ordinal,reason,detail FROM exceptions'); exception_count=0
    while True:
        rows=cur.fetchmany(2048)
        if not rows: break
        exception_count+=len(rows); writer.write_table(pa.Table.from_pylist([dict(zip(sch.names,row)) for row in rows],schema=sch))
    # Add duplicate/multi-date groups as explicitly distinct issues, not identity failures.
    for s in summaries:
        for reason,condition in [('multiple_rows_requires_grain_resolution','SUM(n)>1'),('multiple_dates_not_identity_failure',"COUNT(DISTINCT NULLIF(day,''))>1")]:
            cursor=conn.execute('SELECT pv,SUM(n) FROM records WHERE role=? GROUP BY pv HAVING '+condition,(s['role'],))
            while True:
                rows=cursor.fetchmany(2048)
                if not rows:break
                exception_count+=len(rows)
                writer.write_table(pa.Table.from_pylist([{'role':s['role'],'source_row_ordinal':None,'reason':reason,'detail_TRE_ONLY':pv+';count='+str(n)} for pv,n in rows],schema=sch))
    writer.close()
    assert pq.ParquetFile(exception_file).metadata.num_rows==exception_count
    date_path=q/'date_provenance.csv'
    with date_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['role','field','arrow_type','primary_calendar_field','valid','missing','naive','aware','midnight','numeric_uninterpreted','timezone_source','handling']); w.writeheader()
        for s in summaries:
            for field,ty in s['date_types'].items():
                c=s['date_counts'][field]; numeric=field in s['numeric_date_columns_not_interpreted']
                w.writerow({'role':s['role'],'field':field,'arrow_type':ty,'primary_calendar_field':s['primary_calendar_date'],'valid':c.get('valid',0),'missing':c.get('missing',0),'naive':c.get('naive',0),'aware':c.get('aware',0),'midnight':c.get('midnight',0),'numeric_uninterpreted':c.get('numeric_or_unparsed_not_interpreted',0),'timezone_source':'Arrow timezone or explicit timezone column; no silent localization','handling':'do not guess epoch/unit' if numeric else 'preserve provided date; timezone-aware uses its represented local date; naive stays naive'})
    contract={'run_id':run,'format':'JSON valid YAML 1.2','status':'frozen_key_rules_with_explicit_downstream_gates' if not failures else 'incomplete_sources_do_not_use',
       'scope':'W03 key/date contract; not W11 final sample registry, W06 food cleaning or W09 point QC',
       'person_visit_key':KEY,'target_scope':{'cohort':'10k','research_stage':'00_00_visit'},
       'key_validation':'Positive participant_id; explicit nonblank cohort/stage, reject placeholder/whitespace keys. No fallback from participant alone and no conversion of undefined to baseline.',
       'visit_evidence':'Observed explicit research_stage and cohort; no independent clinical visit_id implied. Multiple dates/lag do not invalidate same-stage membership; ambiguous keys are quarantined.',
       'source_manifest':str(manifest),'source_fingerprints':'Before and after scan require W02 bytes/mtime and rows_metadata; data-page cryptographic equality not established.',
       'date_policy':{'collection_date':'retain as source calendar date, not ingestion time','collection_timestamp':'use only when Arrow is typed timestamp; keep timezone; no naive/aware merge','local_timestamp':'naive source time; do not localize without definition; disagreements flagged for W06','numeric_timestamp':'never assume seconds/ms/epoch; body uses explicit collection_date for day descriptors','DST':'Arrow Asia/Jerusalem aware semantics retained; no fixed UTC offset; point-day DST coverage and ingestion-time validation remain W09/W06, not passed by W03'},
       'microbiome':{'admission':'same person/cohort/explicit stage and later QC; no fixed date lag or unique-date requirement','sample_mapping':'If no explicit sample ID, identity is participant/visit plus source-version row locator; not a validated biological sample ID. Duplicate abundance rows are held for W11; never arbitrarily pick first or merge versions.','cross_module_lock':'W11 publishes one source-version and selected row/sample map reused across all outcome modules'},
       'join_rules':joins,'execution_rules':['Check declared grain before join; right key uniqueness required for m:1.','Both many sides must be aggregated/selected separately; reject unexpected row expansion.','Match losses recorded by source; no global four-outcome intersection.','Liver aggregate without dates cannot be assigned first detail date; preserve one-to-many dates until W10 resolves.','CGM connection_id links to explicit same person/cohort/stage; overlapping/multiple connections require W09 rule, not arbitrary dedup.'],
       'downstream_gates':['W06 resolve food date discrepancy and actual ingestion semantics','W09 scan raw CGM points and actual local-day coverage/DST','W10 phenotype-specific date/measurement and validity selection','W11 sample version/QC and duplicate resolution','family clustering ID provenance remains separate; family_history is not family ID'],
       'participant_discovery_confirmation_roles_frozen':False,'outcome_values_read':False,'scientific_model_trained':False}
    write_json(cfg/'key_contract.yaml',contract)
    total=sum(s['rows_scanned'] for s in summaries)
    summary={'run_id':run,'status':'W03_CONTRACT_COMPLETE_WITH_DOWNSTREAM_GATES' if not failures else 'W03_PARTIAL',
       'sources_planned':len(ROLES),'sources_full_scanned':len(summaries),'rows_projected_scanned':total,'invalid_identity_rows':sum(s['invalid_identity_rows'] for s in summaries),
       'identity_exception_records':exception_count,'join_profiles':len(joins),'raw_many_to_many_join_profiles':sum(j['many_to_many_keys_if_raw_join']>0 for j in joins),
       'failures':failures,'participant_data_remains_TRE':True,'outcome_values_read':False,'actual_eligible_N_recomputed':False,
       'key_contract':str(cfg/'key_contract.yaml'),'date_provenance':str(date_path),'result_directory':str(q),'code_directory':str(code.parent),'memo':str(P/'memos'/run/'MEMO.md')}
    write_json(m/'summary.json',summary)
    report='# W03 identity/date/join audit\n\n'+json.dumps(summary,indent=2)+'\n\nFull projected keys/dates scan, no outcome values. Counts are source availability, not eligible analysis N. Raw food version editing reserved for W06; raw CGM point/day coverage reserved for W09.\n\n'+json.dumps(joins,indent=2)
    (q/'REPORT.md').write_text(report)
    (P/'memos'/run/'MEMO.md').write_text('# W03 code memo\n\nCode: '+str(code)+'\nFunction: Full bounded key/date scans, source grain and join assertions, quarantine invalid and multivalued groups, freeze noncausal key policy with explicit unresolved downstream gates.\nInput: W02 current manifest and 14 source projections. Raw abundance feature values and all outcomes excluded.\nCommand: '+sys.executable+' -u '+str(code)+'\nResults: '+str(q)+'\nContract: '+str(cfg/'key_contract.yaml')+'\nRuntime individual-key SQLite stays TRE: '+str(db)+'\nCode SHA256: '+hashlib.sha256(code.read_bytes()).hexdigest()+'\nNo model/roles/sample-QC selection. Full worker logs retained under '+str(work)+'\nDo not merge abundance versions, guess numeric timestamps, infer undefined visit or silently deduplicate.\n')
    files=[code,cfg/'key_contract.yaml',q/'source_key_date_summary.json',q/'join_cardinalities.json',exception_file,date_path,q/'REPORT.md',m/'summary.json',P/'memos'/run/'MEMO.md']
    receipts=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    write_json(m/'artifact_manifest.json',receipts); assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in receipts)
    print('\n'*12+'W03_FINAL_SUMMARY',json.dumps(summary,separators=(',',':')),flush=True)
    for s in summaries: print('KEY_RESULT',s['role'],'rows',s['rows_scanned'],'base_pv',s['baseline_10k_person_visits'],'invalid',s['invalid_identity_rows'],'multirow_pv',s['person_visits_multiple_rows'],'multidate_pv',s['person_visits_multiple_dates'],flush=True)
    for j in joins: print('JOIN_RESULT',j['left'],j['right'],j['grain'],'matched',j['matched_keys'],'left_lost',j['left_unmatched_keys'],'right_lost',j['right_unmatched_keys'],'raw_many_many',j['many_to_many_keys_if_raw_join'],flush=True)
    print('W03_ARTIFACT_HASHES_VERIFIED',len(receipts),flush=True)
    return 0 if not failures else 2

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        traceback.print_exc(); raise SystemExit(3)
