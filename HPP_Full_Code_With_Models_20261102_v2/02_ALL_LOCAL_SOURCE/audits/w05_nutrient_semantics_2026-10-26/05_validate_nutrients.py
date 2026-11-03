"""W05: TRE-only full projected nutrient audit; source values never reweighted.
No outcome reads, no participant exports, no raw/processed pooling.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import csv,hashlib,json,resource,subprocess,sys,time,traceback,socket
from pathlib import Path
from datetime import datetime,timezone
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L=Path('/home/ec2-user/multisystem_runtime_20261021')
R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')
FIELDS=['weight_g','calories_kcal','carbohydrate_g','lipid_g','protein_g','alcohol_g','dietary_fiber_g']
URL='https://knowledgebase.pheno.ai/datasets/005-diet_logging.html'

def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n').encode();p.write_bytes(b);assert p.read_bytes()==b

def fingerprint(p):
    s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(source,out):
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=R/(source+'.parquet');before=fingerprint(path);pf=pq.ParquetFile(path,pre_buffer=False)
    cols=FIELDS+['food_id','cohort','research_stage'];assert set(cols)<=set(pf.schema_arrow.names)
    budget=sum(pf.metadata.row_group(g).column(c).total_uncompressed_size for g in range(pf.metadata.num_row_groups) for c in range(pf.metadata.num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in cols)
    assert budget<=1536*1024**2,'Projected uncompressed budget exceeded'
    stats={s:{f:dict(rows=0,missing=0,nonfinite_nonmissing=0,negative=0,zero=0,positive=0,sum_finite=0.,minimum=None,maximum=None) for f in FIELDS} for s in ['all_rows','baseline_10k_rows']}
    quality={s:dict(rows=0,positive_energy_missing_or_nonpositive_weight=0,fiber_exceeds_positive_weight=0,weight_above_dict_max_2000g=0,energy_above_dict_max_5000kcal=0) for s in stats}
    energy={s:{name:dict(n=0,sum_source_kcal=0.,sum_reconstructed_kcal=0.,sum_absolute_difference=0.,absolute_difference_gt_1kcal=0,absolute_difference_gt_10kcal=0) for name in ['4C_9L_4P','4C_9L_4P_7A','4C_9L_4P_7A_minus_2F','4C_9L_4P_7A_plus_2F']} for s in stats}
    extremes={};examples=[];example_ids=set();n=0;t0=time.monotonic()
    for b in pf.iter_batches(batch_size=8192,columns=cols,use_threads=False):
        d=b.to_pandas(ignore_metadata=True);a={f:d[f].to_numpy(dtype=float,na_value=np.nan) for f in FIELDS}
        masks={'all_rows':np.ones(len(d),dtype=bool),'baseline_10k_rows':((d.cohort=='10k')&(d.research_stage=='00_00_visit')).to_numpy()}
        E=a['calories_kcal'];W=a['weight_g'];F=a['dietary_fiber_g'];A=a['alcohol_g']
        macro=4*a['carbohydrate_g']+9*a['lipid_g']+4*a['protein_g']
        recon={'4C_9L_4P':macro,'4C_9L_4P_7A':macro+7*A,'4C_9L_4P_7A_minus_2F':macro+7*A-2*F,'4C_9L_4P_7A_plus_2F':macro+7*A+2*F}
        for scope,m in masks.items():
            q=quality[scope];q['rows']+=int(m.sum())
            q['positive_energy_missing_or_nonpositive_weight']+=int((m&np.isfinite(E)&(E>0)&(~np.isfinite(W)|(W<=0))).sum())
            q['fiber_exceeds_positive_weight']+=int((m&np.isfinite(F)&np.isfinite(W)&(W>0)&(F>W)).sum())
            q['weight_above_dict_max_2000g']+=int((m&np.isfinite(W)&(W>2000)).sum())
            q['energy_above_dict_max_5000kcal']+=int((m&np.isfinite(E)&(E>5000)).sum())
            for f in FIELDS:
                v=a[f][m];z=stats[scope][f];finite=v[np.isfinite(v)]
                z['rows']+=len(v);z['missing']+=int(np.isnan(v).sum());z['nonfinite_nonmissing']+=int(np.isinf(v).sum())
                z['negative']+=int((finite<0).sum());z['zero']+=int((finite==0).sum());z['positive']+=int((finite>0).sum());z['sum_finite']+=float(finite.sum())
                if len(finite):
                    lo=float(finite.min());hi=float(finite.max());z['minimum']=lo if z['minimum'] is None else min(lo,z['minimum']);z['maximum']=hi if z['maximum'] is None else max(hi,z['maximum'])
            for name,v in recon.items():
                ok=m&np.isfinite(E)&np.isfinite(v);z=energy[scope][name];diff=np.abs(E[ok]-v[ok]);z['n']+=int(ok.sum());z['sum_source_kcal']+=float(E[ok].sum());z['sum_reconstructed_kcal']+=float(v[ok].sum());z['sum_absolute_difference']+=float(diff.sum());z['absolute_difference_gt_1kcal']+=int((diff>1).sum());z['absolute_difference_gt_10kcal']+=int((diff>10).sum())
        # Portion consistency is corroboration, never the source of unit semantics.
        ok=np.isfinite(W)&(W>0)&np.isfinite(E)&(E>0)&d.food_id.notna().to_numpy()
        v=d.loc[ok,['food_id']+FIELDS].copy();v['source_row_ordinal_0based']=np.flatnonzero(ok)+n
        for which in ['idxmin','idxmax']:
            ix=getattr(v.groupby('food_id',sort=False).weight_g,which)()
            for row in v.loc[ix].to_dict('records'):
                fid=int(row['food_id']);pair=extremes.setdefault(fid,[None,None]);j=0 if which=='idxmin' else 1
                if pair[j] is None or (row['weight_g']<pair[j]['weight_g'] if j==0 else row['weight_g']>pair[j]['weight_g']):pair[j]=row
        assert len(extremes)<=50000,'Too many food IDs for bounded portion audit'
        if len(examples)<6:
            for row in v.to_dict('records'):
                fid=int(row['food_id'])
                if fid in example_ids or abs(row['weight_g']-100)<1e-8:continue
                if not all(pd.notna(row[f]) and np.isfinite(row[f]) and row[f]>=0 for f in FIELDS):continue
                example_ids.add(fid);examples.append(row)
                if len(examples)==6:break
        n+=len(d)
        if time.monotonic()-t0>900:raise RuntimeError('Full scan timeout')
    assert n==pf.metadata.num_rows and fingerprint(path)==before
    portion={f:dict(varying_weight_food_ids=0,per_g_close_1pct=0,absolute_value_close_1pct=0) for f in ['calories_kcal','dietary_fiber_g']};pairs=[]
    for fid,(lo,hi) in extremes.items():
        if lo is None or hi is None or hi['weight_g']<2*lo['weight_g']:continue
        for f,z in portion.items():
            x=lo[f];y=hi[f]
            if not (pd.notna(x) and pd.notna(y) and np.isfinite(x) and np.isfinite(y) and x>0 and y>0):continue
            z['varying_weight_food_ids']+=1
            z['per_g_close_1pct']+=int(abs(x/lo['weight_g']-y/hi['weight_g'])<=.01*max(x/lo['weight_g'],y/hi['weight_g']))
            z['absolute_value_close_1pct']+=int(abs(x-y)<=.01*max(x,y))
        if len(pairs)<10:pairs.extend([dict(pair_id=len(pairs)//2,side='low',**lo),dict(pair_id=len(pairs)//2,side='high',**hi)])
    # Examples stay in TRE. Source passthrough versus the erroneous second scaling.
    for row in examples:
        for f in FIELDS[1:]:
            row[f+'_intake']=row[f];row[f+'_WRONG_second_weight_scaling']=row[f]*row['weight_g']/100
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('event_examples_TRE_ONLY',examples),('portion_pairs_TRE_ONLY',pairs)]:pq.write_table(pa.Table.from_pylist(rows),out/(source+'_'+name+'.parquet'))
    result=dict(source=source,fingerprint=before,schema_sha256=hashlib.sha256(str(pf.schema_arrow).encode()).hexdigest(),rows_metadata=pf.metadata.num_rows,rows_scanned=n,projection_uncompressed_bytes=budget,full_scan=True,fields=stats,quality_flags=quality,energy_diagnostics=energy,portion_consistency=portion,examples=len(examples),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_s=round(time.monotonic()-t0,2))
    put(out/(source+'_summary.json'),result);print('SOURCE_DONE',source,n,'RSS_KiB',result['peak_rss_kib'],flush=True)

def main():
    assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
    import pyarrow as pa
    import pyarrow.parquet as pq
    run='w05_nutrients_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');work=L/run;work.mkdir(parents=True,exist_ok=False)
    for name in ['scripts','config','data_qc','memos','logs','manifests']:(P/name/run).mkdir(parents=True,exist_ok=False)
    q=P/'data_qc'/run;code=P/'scripts'/run/'05_validate_nutrients.py';code.write_bytes(Path(__file__).read_bytes());assert sha(code)==sha(__file__)
    put(P/'W05_LATEST_RUN.json',dict(run=run,work=str(work),status='running'))
    dicts={};evidence=[];sources=[]
    for source in ['diet_logging_events','raw_diet_logging_events']:
        dp=R/'metadata'/('diet_logging__'+source+'_bulk_dictionary.csv')
        with dp.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        fields={r['tabular_field_name']:r for r in rows};assert all(f in fields for f in FIELDS)
        for f in FIELDS:
            r=fields[f];assert r['units']==('kcal' if f=='calories_kcal' else 'g')
            text=(r.get('field_string','')+' '+r.get('description_string','')).lower()
            assert ('food item logged' in text if f in ['weight_g','calories_kcal'] else 'intake per food logged' in text),(source,f,text)
            evidence.append(dict(source=source,field=f,units=r['units'],description=r['description_string'],dictionary=str(dp),dictionary_sha256=sha(dp)))
        dicts[source]=dict(path=str(dp),sha256=sha(dp))
        pf=pq.ParquetFile(R/(source+'.parquet'),pre_buffer=False)
        sources.append(dict(source=source,**fingerprint(R/(source+'.parquet')),rows=pf.metadata.num_rows,columns=pf.schema_arrow.names))
    index=R/'diet_logging.parquet';pf=pq.ParquetFile(index,pre_buffer=False)
    assert all(f not in pf.schema_arrow.names for f in FIELDS),'Daily index unexpectedly contains nutrient inputs'
    put(q/'dictionary_evidence_TRE_ONLY.json',evidence)
    put(P/'manifests'/run/'inputs.json',dict(sources=sources,index=dict(**fingerprint(index),rows=pf.metadata.num_rows,columns=pf.schema_arrow.names),dictionaries=dicts,host=socket.gethostname(),python=sys.executable,code_sha256=sha(code),pyarrow=pa.__version__))
    summaries=[];exitcodes={}
    for source in dicts:
        import psutil
        assert psutil.virtual_memory().available>=2500*1024**2,'Insufficient available memory'
        log=work/(source+'.log')
        with log.open('wb') as f:
            proc=subprocess.Popen([sys.executable,str(code),'--worker',source,str(q)],stdout=f,stderr=subprocess.STDOUT)
            started=time.monotonic()
            while proc.poll() is None:
                try:
                    if psutil.Process(proc.pid).memory_info().rss>768*1024**2 or psutil.virtual_memory().available<2*1024**3 or time.monotonic()-started>1000:
                        proc.terminate();proc.wait(timeout=15);raise RuntimeError('Worker resource/time guard')
                except psutil.NoSuchProcess:pass
                time.sleep(.25)
        exitcodes[source]=proc.returncode;put(work/(source+'_exit.json'),dict(exit_code=proc.returncode))
        assert proc.returncode==0,log.read_text()[-2500:]
        summaries.append(json.loads((q/(source+'_summary.json')).read_text()))
    rows=[]
    for s in summaries:
        for scope,fs in s['fields'].items():
            for field,v in fs.items():
                assert v['rows']==v['missing']+v['nonfinite_nonmissing']+v['negative']+v['zero']+v['positive']
                rows.append(dict(source=s['source'],scope=scope,field=field,**v))
    audit=q/'nutrient_conversion_audit.parquet';pq.write_table(pa.Table.from_pylist(rows),audit);assert pq.read_metadata(audit).num_rows==28
    # Independently re-read original source row groups for all stored examples.
    verified=0
    for s in summaries:
        source=s['source'];pf=pq.ParquetFile(R/(source+'.parquet'),pre_buffer=False)
        examples=pq.read_table(q/(source+'_event_examples_TRE_ONLY.parquet')).to_pylist()
        offsets=[];total=0
        for g in range(pf.metadata.num_row_groups):offsets.append(total);total+=pf.metadata.row_group(g).num_rows
        for row in examples:
            ordinal=row['source_row_ordinal_0based'];g=max(i for i,o in enumerate(offsets) if o<=ordinal)
            # Guard full row-group read; selected seven numeric columns only.
            size=sum(pf.metadata.row_group(g).column(c).total_uncompressed_size for c in range(pf.metadata.num_columns) if pf.metadata.row_group(g).column(c).path_in_schema in FIELDS)
            assert size<64*1024**2,'Example readback row group too large'
            original=pf.read_row_group(g,columns=FIELDS,use_threads=False).slice(ordinal-offsets[g],1).to_pylist()[0]
            for f in FIELDS:assert original[f]==row[f]
            for f in FIELDS[1:]:assert row[f+'_intake']==original[f]
            assert row['calories_kcal_WRONG_second_weight_scaling']!=row['calories_kcal_intake']
            verified+=1
    assert verified==12
    rules={f:dict(unit='kcal' if f=='calories_kcal' else 'g',basis='logged_food_event_intake' if f!='weight_g' else 'logged_food_event_weight',multiplier=1.0,weight_rescaling=False,missing='preserve_NA',negative_or_nonfinite='flag_and_hold_for_W06_no_silent_zero',evidence='current TRE dictionary plus public official field definition') for f in FIELDS}
    semantics=dict(version=1,status='event_unit_semantics_verified_quality_cleaning_pending',source_url=URL,primary_table='diet_logging_events.parquet',raw_table='diagnostic_only_do_not_pool_with_processed',fields=rules,source_energy='preserve calories_kcal; reconstructed energy is diagnostic, never a replacement',alcohol='Only observed alcohol_g; no energy residual, assumed ABV, g-to-mL or missing-to-zero recovery',fiber_energy='Unknown carbohydrate fiber convention; 4/9/4/7 and +/-2F diagnostics do not identify the true convention',recipes_cooking='Use event amounts as supplied; no generic raw/cooked or recipe mass conversion; underlying recipe composition unverified',daily_index='Index only, no nutrient fields',remaining_gates=['W06 event edit/duplicate/missing/extreme QC','W07 NOVA food-ID/version mapping','No equal-energy carbohydrate-fat closure until source convention or justified separate specification'],run=run)
    put(P/'config'/run/'nutrient_semantics.yaml',semantics)
    put(P/'config'/'nutrient_semantics.yaml',semantics)
    manifest=[]
    for p in sorted(q.iterdir()):manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    summary=dict(run=run,status='passed_unit_semantics_not_full_diet_QC',exit_codes=exitcodes,full_sources=2,total_event_rows_scanned=sum(s['rows_scanned'] for s in summaries),independent_example_readbacks=verified,source_summaries=summaries,artifact_count=len(manifest),semantics=semantics)
    put(q/'verification.json',dict(passed=True,independent_example_readbacks=verified,parquet_audit_rows=28,actual_worker_exit_codes=exitcodes,artifacts=manifest))
    put(q/'summary.json',summary)
    memo=f'''# W05 nutrient semantics — {run}\n\nCode: {code}\nFunction: verify current local dictionaries, fully scan selected numeric columns in both event layers separately, audit missing/negative/range flags and diagnostic macro-energy formulas, and independently re-read 12 source examples. No outcomes or microbiome values read.\n\nRows scanned: {summary['total_event_rows_scanned']}; workers actual exit: {exitcodes}; 12 independent example readbacks passed.\n\nResult: {q}\nConfig: {P/'config'/run/'nutrient_semantics.yaml'}\nSource nutrients already represent event intake; multiplier 1, never multiply by weight/100 again. Keep missing unknown. Preserve source energy. Raw and processed never pooled.\n\nNot completed: W06 recording/edit/duplicate/missing/extreme cleaning, W07 food mapping, recipe/cooking composition, precise fiber/polyol energy convention, protocol freeze or individual diet features. Quality flags are not automatic deletions. 4/9/4/7 closure is a diagnostic, not unit proof. Alcohol is not reconstructed from residuals or assumed ABV. Examples and dictionary details stay in TRE.\n\nRe-run in pheno: {sys.executable} {code}; creates a new run and retains prior evidence. Runtime logs: {work}.\n'''
    (P/'memos'/run/'MEMO.md').write_text(memo);(q/'MEMO.md').write_text(memo);(P/'W05_HANDOFF.md').write_text(memo)
    put(P/'W05_LATEST_RUN.json',dict(run=run,work=str(work),result=str(q),status='completed'))
    state_path=P/'PROJECT_EXECUTION_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    state['W05']=dict(status=summary['status'],run=run,result=str(q),config=str(P/'config'/run/'nutrient_semantics.yaml'),memo=str(P/'memos'/run/'MEMO.md'));put(state_path,state)
    put(work/'result_pointer.json',dict(run=run,result=str(q)))
    print('\n'*10+'W05_AUDIT_PASS',run,'ROWS',summary['total_event_rows_scanned'],'EXAMPLES',verified,flush=True)
    for s in summaries:
        print('SOURCE',s['source'],'rows',s['rows_scanned'],'base',s['quality_flags']['baseline_10k_rows']['rows'],'RSS_KiB',s['peak_rss_kib'],flush=True)
        for f in ['weight_g','calories_kcal','dietary_fiber_g','alcohol_g']:
            z=s['fields']['all_rows'][f];print('FIELD',f,'missing',z['missing'],'negative',z['negative'],'nonfinite',z['nonfinite_nonmissing'],'zero',z['zero'],'min',z['minimum'],'max',z['maximum'],flush=True)
        print('FLAGS',json.dumps(s['quality_flags']['all_rows']),flush=True)
        print('PORTION',json.dumps(s['portion_consistency']),flush=True)
    print('RESULT',q,'CONFIG',P/'config'/run/'nutrient_semantics.yaml',flush=True)
    return 0

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(*sys.argv[2:]);sys.exit(0)
    sys.exit(main())
