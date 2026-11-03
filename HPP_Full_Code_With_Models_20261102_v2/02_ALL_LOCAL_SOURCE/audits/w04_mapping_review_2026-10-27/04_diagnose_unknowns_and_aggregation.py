# -*- coding: utf-8 -*-
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import ast,json,hashlib,resource,re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/w04_mapping_review_20261027'
source=P/'scripts/w04_mapping_review_20261027/03_audit_event_mapping_days.py'
tree=ast.parse(source.read_text())
cat=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CATALOG' for t in n.targets))
norm=lambda x:str(x).strip().casefold() if pd.notna(x) else ''
mp={norm(k):v for k,v in cat}
foods=json.loads((Q/'food_id_name_category_candidates_TRE_ONLY.json').read_text())
food_summary={'IDs_total':len(foods),'IDs_all_names_valid_NOVA':sum(x['all_names_have_valid_NOVA'] for x in foods),'unique_short_names':len({n for x in foods for n in x['short_names']}),'unique_food_categories':len({n for x in foods for n in x['categories']})}
print('FOOD_CATALOG',json.dumps(food_summary),flush=True)
D=pq.read_table(Q/'day_mapping_diagnostics_TRE_ONLY.parquet').to_pandas(ignore_metadata=True)
b=D[(D.cohort=='10k')&(D.research_stage=='00_00_visit')&D.index_matched].copy()
assert len(b)==116906
ix=set(zip(b.participant_id,b.cohort,b.research_stage,b.date))
validE=(b.energy_bad_rows==0)&(b.energy_kcal>0)
comparison={}
for label,ok,num,scale in [('UPF',validE&(b.NOVA_unknown_rows==0),'UPF_kcal_known',100),('fiber',validE&(b.fiber_bad_rows==0),'fiber_g_known',1000)]:
 v=b.loc[ok].copy();v['daily']=scale*v[num]/v.energy_kcal
 g=v.groupby(['participant_id','cohort','research_stage']).agg(n=('daily','size'),equal_day=('daily','mean'),numerator=(num,'sum'),E=('energy_kcal','sum'))
 g['cumulative']=scale*g.numerator/g.E
 g['difference']=g.equal_day-g.cumulative
 q=g[g.n>=7]
 comparison[label]={'diagnostic_person_visits_ge7':len(q),'equal_mean':float(q.equal_day.mean()),'cumulative_mean':float(q.cumulative.mean()),'median_signed_difference':float(q.difference.median()),'absolute_difference_quantiles':dict(zip(['p50','p90','p95','max'],map(float,q.difference.abs().quantile([.5,.9,.95,1])))),'spearman':float(q.equal_day.rank().corr(q.cumulative.rank()))}
 g.reset_index().to_parquet(Q/(label+'_aggregation_comparison_TRE_ONLY.parquet'),index=False)
 print('AGGREGATION',label,json.dumps(comparison[label]),flush=True)
unknown=defaultdict(lambda:[0,0.,set()]);missing_class=defaultdict(lambda:[0,0.]);categories=defaultdict(lambda:[0,0.]);allnames=set();matchedold=set();n=0
R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging/diet_logging_events.parquet')
cols=['participant_id','cohort','research_stage','collection_date','food_id','short_food_name','food_category','calories_kcal']
for rb in pq.ParquetFile(R,pre_buffer=False).iter_batches(batch_size=8192,columns=cols,use_threads=False):
 d=rb.to_pandas(ignore_metadata=True);n+=len(d)
 d=d[(d.cohort=='10k')&(d.research_stage=='00_00_visit')].copy()
 d['date']=pd.to_datetime(d.collection_date,errors='coerce').dt.strftime('%Y-%m-%d')
 keep=[k in ix for k in zip(d.participant_id,d.cohort,d.research_stage,d.date)]
 d=d.loc[keep].copy();e=pd.to_numeric(d.calories_kcal,errors='coerce');d['E']=e.where(np.isfinite(e)&(e>=0),0)
 for name,c,g in ((str(name),c,g) for (name,c),g in d.groupby(['short_food_name','food_category'],dropna=False,sort=False)):
  allnames.add(name);k=norm(name);val=mp.get(k);z=categories[str(c)];z[0]+=len(g);z[1]+=float(g.E.sum())
  if k in mp:matchedold.add(k)
  if val not in [1,2,3,4]:
   z=unknown[(name,str(c))];z[0]+=len(g);z[1]+=float(g.E.sum());z[2].update(str(x) for x in g.food_id.dropna().unique())
assert n==2640959
un=[{'short_food_name':k[0],'food_category':k[1],'rows':v[0],'known_energy_kcal':v[1],'food_ID_count':len(v[2]),'in_old_catalog':norm(k[0]) in mp} for k,v in unknown.items()]
un.sort(key=lambda x:x['known_energy_kcal'],reverse=True)
(Q/'unclassified_foods_baseline_indexed_TRE_ONLY.json').write_text(json.dumps(un,ensure_ascii=False,indent=2))
print('UNKNOWN_NAME_GROUPS',len(un),'BASELINE_SHORT_NAMES',len(allnames),'OLD_CATALOG_NAMES_CONSUMED',len(matchedold),flush=True)
for x in un[:12]:print('UNKNOWN_TOP',json.dumps(x,ensure_ascii=False),flush=True)
print('CATEGORIES_BASELINE',json.dumps(sorted(categories)),flush=True)
# Read only located project source, not run it. Save full bounded source inside TRE.
old=[]
for s in ['/home/ec2-user/studies/diet_gut_cgm/script/amd/03_prepare_amd_mapping.py','/home/ec2-user/studies/diet_gut_cgm/new_diet_indicators/03_prepare_diet_indicator_mapping.py']:
 p=Path(s);t=p.read_text();assert len(t)<200000
 hits=[{'line':i,'text':line[:260]} for i,line in enumerate(t.splitlines(),1) if any(x in line.lower() for x in ['read_csv','read_excel','nova','mapping_file','mapping_path','short_food_name','food_category','xlsx'])]
 old.append({'path':s,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'line_count':len(t.splitlines()),'hits':hits})
 print('OLD_SCRIPT',s,'LINES',len(t.splitlines()),flush=True)
 for h in hits[:18]:print('OLD_CODE_HIT',h['line'],h['text'],flush=True)
(Q/'old_mapping_script_evidence_TRE_ONLY.json').write_text(json.dumps(old,indent=2))
summary={'catalog':food_summary,'aggregation':comparison,'unknown_name_groups_baseline_indexed':len(un),'baseline_short_names':len(allnames),'old_catalog_names_consumed':len(matchedold),'top_unknown_foods':un[:12],'not_final_QC_or_admission':True}
(Q/'unknowns_and_aggregation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('DIAGNOSE_UNKNOWNS_DONE',flush=True)
