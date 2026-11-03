# -*- coding: utf-8 -*-
"""Recover null short names only through observed unambiguous FoodID; audit, no overwrite."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import ast,json,hashlib,resource,time
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w04_mapping_review_20261027'
L=Path('/home/ec2-user/multisystem_runtime_20261021/w04_mapping_review_20261027')
B=L/'06_artifacts';B.mkdir(exist_ok=True)
keys=['participant_id','cohort','research_stage','date']
tree=ast.parse((P/'scripts/w04_mapping_review_20261027/03_audit_event_mapping_days.py').read_text())
catalog=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CATALOG' for t in n.targets))
norm=lambda x:str(x).strip().casefold() if pd.notna(x) else ''
fidnorm=lambda x:str(int(x)) if pd.notna(x) else ''
mp={norm(n):v for n,v in catalog}
food={}
for x in json.loads((Q/'food_id_name_category_candidates_TRE_ONLY.json').read_text()):
 k=str(int(Decimal(x['food_id'])));z=food.setdefault(k,{'names':set(),'categories':set()});z['names'].update(x['short_names']);z['categories'].update(x['categories'])
lookup={}
for k,z in food.items():
 v={mp.get(norm(n)) for n in z['names']}
 if len(v)==1 and None not in v:lookup[k]=next(iter(v))
conflicts=sum(len({mp.get(norm(n)) for n in z['names']} - {None})>1 for z in food.values())
assert conflicts==0
print('CANONICAL_FOOD_IDS',len(food),'UNAMBIGUOUS_NOVA_IDS',len(lookup),'CONFLICTS',conflicts,flush=True)
# Recovering the previous independent computations from their saved outputs.
D=pq.ParquetFile(Q/'day_mapping_diagnostics_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True)
D['date']=D.date.astype(str)
b=D[(D.cohort=='10k')&(D.research_stage=='00_00_visit')&D.index_matched].copy()
validE=(b.energy_bad_rows==0)&(b.energy_kcal>0)
aggregation={}
for label,ok,num,scale in [('UPF',validE&(b.NOVA_unknown_rows==0),'UPF_kcal_known',100),('fiber',validE&(b.fiber_bad_rows==0),'fiber_g_known',1000)]:
 v=b.loc[ok].copy();v['daily']=scale*v[num]/v.energy_kcal
 g=v.groupby(['participant_id','cohort','research_stage']).agg(n=('daily','size'),equal_day=('daily','mean'),numerator=(num,'sum'),E=('energy_kcal','sum'))
 g['cumulative']=scale*g.numerator/g.E;g['difference']=g.equal_day-g.cumulative;q=g[g.n>=7]
 aggregation[label]={'n_ge7':len(q),'median_absolute_difference':float(q.difference.abs().median()),'p95_absolute_difference':float(q.difference.abs().quantile(.95)),'spearman':float(q.equal_day.rank().corr(q.cumulative.rank()))}
 out=B/(label+'_aggregation_comparison_TRE_ONLY.parquet')
 pq.write_table(pa.Table.from_pandas(g.reset_index(),preserve_index=False),out)
 assert pq.ParquetFile(out).metadata.num_rows==len(g)
print('AGGREGATION_SAME_DAYS',json.dumps(aggregation),flush=True)
# Use exact paths returned by directory inventory, never retype a discovered filename.
search=json.loads((Q/'search_mapping_candidates_TRE_ONLY.json').read_text());old=[]
for x in search['found']:
 p=Path(x['path'])
 if x['directory'] or not x['path'].startswith('/home/ec2-user/studies/diet_gut_cgm/') or p.suffix!='.py':continue
 t=p.read_text();assert 0<len(t)<200000
 hits=[{'line':i,'text':line[:260]} for i,line in enumerate(t.splitlines(),1) if any(s in line.lower() for s in ['read_csv','read_excel','nova','mapping_file','mapping_path','short_food_name','food_category','xlsx'])]
 old.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'hits':hits})
 print('OLD_MAPPING_SOURCE',str(p),flush=True)
 for h in hits[:10]:print('OLD_MAPPING_HIT',h['line'],h['text'],flush=True)
(Q/'old_mapping_script_evidence_TRE_ONLY.json').write_text(json.dumps(old,indent=2))
idx=set(tuple(x) for x in D.loc[D.index_matched,keys].itertuples(index=False,name=None))
new={};un=defaultdict(lambda:[0,0.]);n=0;start=time.monotonic();recovered_events=0;recovered_energy=0.
R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging/diet_logging_events.parquet')
cols=['participant_id','cohort','research_stage','collection_date','food_id','short_food_name','product_name','food_category','calories_kcal']
for rb in pq.ParquetFile(R,pre_buffer=False).iter_batches(batch_size=8192,columns=cols,use_threads=False):
 d=rb.to_pandas(ignore_metadata=True);n+=len(d)
 d['date']=pd.to_datetime(d.collection_date,errors='coerce').dt.strftime('%Y-%m-%d')
 keep=[k in idx for k in d[['participant_id','cohort','research_stage','date']].itertuples(index=False,name=None)]
 d=d.loc[keep].copy()
 if d.empty:continue
 names=d.short_food_name.map(norm);nova=names.map(mp)
 candidate=d.food_id.map(fidnorm).map(lookup)
 recover=(names=='')&nova.isna()&candidate.notna()
 after=nova.where(~recover,candidate)
 e=pd.to_numeric(d.calories_kcal,errors='coerce');d['_E']=e.where(np.isfinite(e)&(e>=0),0)
 d['_unknownE']=d._E.where(~after.isin([1,2,3,4]),0)
 d['_unknownN']=(~after.isin([1,2,3,4])).astype(int)
 d['_K']=d._E.where(after==4,0)
 d['_recoveredN']=recover.astype(int);d['_recoveredE']=d._E.where(recover,0)
 for k,v in zip((g:=d.groupby(keys,sort=False)[['_unknownE','_unknownN','_K','_recoveredN','_recoveredE']].sum()).index,g.to_numpy()):
  if k in new:new[k]+=v
  else:new[k]=v
 for (fid,name,product,cat),g in d.loc[~after.isin([1,2,3,4])].groupby(['food_id','short_food_name','product_name','food_category'],dropna=False,sort=False):
  k=(fidnorm(fid),str(name),str(product),str(cat));z=un[k];z[0]+=len(g);z[1]+=float(g._E.sum())
 assert len(un)<100000
 if time.monotonic()-start>600:raise RuntimeError('Recovery scan timeout')
assert n==2640959
Z=D.loc[D.index_matched].copy()
values=np.array([new[tuple(k)] for k in Z[keys].itertuples(index=False,name=None)])
assert len(values)==len(Z)
for j,c in enumerate(['unknownE_after','unknownN_after','UPF_E_after','recoveredN','recoveredE']):Z[c]=values[:,j]
assert (Z.unknownE_after<=Z.NOVA_unknown_kcal+1e-7).all()
summary={}
for scope,m in [('all_source',np.ones(len(Z),dtype=bool)),('baseline_10k',(Z.cohort=='10k')&(Z.research_stage=='00_00_visit'))]:
 z=Z.loc[m];ok=(z.energy_bad_rows==0)&(z.energy_kcal>0)
 modes={'legacy_all_rows_classified_before':ok&(z.NOVA_unknown_rows==0),'all_rows_classified_after_FoodID':ok&(z.unknownN_after==0),'zero_unknown_energy_after_FoodID':ok&(z.unknownE_after==0)}
 for pct in [.90,.95,.99]:modes['coverage_ge_'+str(int(pct*100))+'_bounds_only']=ok&((1-z.unknownE_after/z.energy_kcal)>=pct)
 counts={}
 for label,mask in modes.items():
  a=z.loc[mask].groupby(['participant_id','cohort','research_stage']).size()
  counts[label]={'days':int(mask.sum()),'people_ge1':len(a),'people_ge7':int((a>=7).sum())}
 summary[scope]={'recovered_events':int(z.recoveredN.sum()),'recovered_known_kcal':float(z.recoveredE.sum()),'known_energy_coverage_after':float(1-z.unknownE_after.sum()/z.energy_kcal.sum()),'counts':counts}
 print('RECOVERY',scope,json.dumps(summary[scope]),flush=True)
out=B/'day_mapping_after_foodid_diagnostic_TRE_ONLY.parquet'
pq.write_table(pa.Table.from_pandas(Z,preserve_index=False),out)
assert pq.ParquetFile(out).metadata.num_rows==len(Z)
u=[dict(food_id=k[0],short_name=k[1],product_name=k[2],category=k[3],rows=v[0],known_energy_kcal=v[1]) for k,v in un.items()]
u.sort(key=lambda x:x['known_energy_kcal'],reverse=True)
(Q/'remaining_unclassified_foodID_products_TRE_ONLY.json').write_text(json.dumps(u,ensure_ascii=False,indent=2))
result={'canonical_food_IDs':len(food),'unambiguous_NOVA_IDs':len(lookup),'NOVA_ID_conflicts':conflicts,'IDs_multiple_names':sum(len(z['names'])>1 for z in food.values()),'IDs_multiple_categories':sum(len(z['categories'])>1 for z in food.values()),'aggregation':aggregation,'recovery':summary,'remaining_product_groups':len(u),'previous_04_actual_exit':(L/'04_actual_exit_code').read_text(),'previous_05_actual_exit':(L/'05_actual_exit_code').read_text(),'binary_results_runtime':str(B),'no_formal_protocol_change':True,'not_final_analysis_N':True,'no_outcomes_read':True}
(Q/'recovery_and_aggregation_summary.json').write_text(json.dumps(result,indent=2))
print('REMAINING_UNCLASSIFIED_PRODUCT_GROUPS',len(u),flush=True)
for x in u[:5]:print('UNKNOWN_PRODUCT_TOP',json.dumps(x,ensure_ascii=False),flush=True)
# Mirror through ordinary byte writes, then independently verify; do not assume FUSE Parquet writing works.
mirrors=[]
for p in B.glob('*.parquet'):
 target=Q/p.name
 with p.open('rb') as src,target.open('wb') as dst:
  while True:
   chunk=src.read(1024*1024)
   if not chunk:break
   dst.write(chunk)
 time.sleep(1)
 h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 good=target.stat().st_size==p.stat().st_size and h(target)==h(p)
 mirrors.append({'runtime':str(p),'cloud':str(target),'bytes':p.stat().st_size,'sha256':h(p),'cloud_readback_verified':good})
 if good:assert pq.ParquetFile(target).metadata.num_rows==pq.ParquetFile(p).metadata.num_rows
(Q/'binary_mirror_verification.json').write_text(json.dumps(mirrors,indent=2))
print('BINARY_MIRRORS',json.dumps([{'file':Path(x['cloud']).name,'bytes':x['bytes'],'verified':x['cloud_readback_verified']} for x in mirrors]),flush=True)
print('RECOVERY_AUDIT_DONE',flush=True)
