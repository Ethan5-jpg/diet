# -*- coding: utf-8 -*-
"""Independent point-accounting, source hashes, unit, DST, daily-recalculation checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,time,resource
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w09_cgm_point_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028');K=['participant_id','cohort','research_stage'];KC=K+['connection_id'];KD=K+['date'];TZ='Asia/Jerusalem'
assert (L/'09_actual_exit_code').read_text().strip()=='0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return pq.ParquetFile(p).read(use_threads=False).to_pandas(ignore_metadata=True)
s=json.loads((Q/'summary_TRE_ONLY.json').read_text());m=json.loads((Q/'fragment_processing_manifest_TRE_ONLY.json').read_text());assert len(m)==s['fragments']==9497
counts=Counter()
for x in m:
 p=Path(x['source_path']);assert p.stat().st_size==x['source_bytes'] and p.stat().st_mtime_ns==x['source_mtime_ns'] and sha(p)==x['source_sha256'];assert x['status']=='processed' and pq.ParquetFile(p).metadata.num_rows==x['rows'];counts.update(x['counts'])
 assert x['counts']['input_points']==sum(v for k,v in x['counts'].items() if k!='input_points')
assert dict(counts)==s['point_accounting']
for x in s['artifacts']:
 assert sha(x['cloud'])==sha(x['runtime'])==x['sha256'] and pq.ParquetFile(x['cloud']).metadata.num_rows==x['rows']
C=load(P/'data_qc/cgm_point_connection_QC.parquet');D=load(Q/'cgm_daily_all_connections_TRE_ONLY.parquet');S=load(P/'analysis/shared/cgm_daily_TRE_ONLY.parquet');assert len(C)==9497 and not D.duplicated(KC+['date']).any() and not S.duplicated(KD).any()
assert S.point_qc_verified.all() and set(S.unit)=={'mg/dL'} and set(S.timezone)=={TZ};assert D.coverage_fraction.between(0,1+1e-12).all() and D.day_length_seconds.isin([82800,86400,90000]).all()
assert (D.n_points.sum()==counts['valid_unique_points'])
expect=(D.n_points>0)&D.coverage_fraction.ge(.8)&D.max_uncovered_gap_seconds.le(7200)&~D.has_timestamp_conflict&D.cadence_verified
bad=set(C.loc[C.missing_timestamp_points>0,'connection_id']);expect &= ~D.connection_id.isin(bad);assert expect.equals(D.cgm_day_eligible)
assert C.loc[C.selected_connection].groupby(K).size().eq(1).all();sort=C.sort_values(K+['qualified_food_common_days','first_observed','connection_id'],ascending=[True]*3+[False,True,True]);assert set(sort.drop_duplicates(K).connection_id)==set(C.loc[C.selected_connection,'connection_id'])
# Independent interval-union implementation on 3 entire real fragments, every calendar day.
checks=0;curves=[]
for ordinal in [0,len(m)//2,len(m)-1]:
 raw=load(m[ordinal]['source_path']);t=pd.to_datetime(raw.collection_timestamp,utc=True).dt.tz_convert(TZ);raw=raw.assign(t=t,g=pd.to_numeric(raw.glucose,errors='coerce')).dropna(subset=['t']);raw=raw.drop_duplicates(['t','g']);bad=raw.groupby('t',dropna=False).g.nunique(dropna=False);bad=set(bad[bad>1].index);good=raw.loc[~raw.t.isin(bad)&raw.g.between(40,500)&np.isfinite(raw.g)].sort_values('t');c=str(raw.connection_id.iloc[0]);r=D.loc[D.connection_id==c]
 for z in r.itertuples():
  vals=good.loc[good.t.dt.strftime('%Y-%m-%d')==str(z.date),'g'];assert len(vals)==z.n_points
  if len(vals):assert np.isclose(float(vals.mean()),z.glucose_mean_mg_dL,rtol=0,atol=1e-10)
  a=pd.Timestamp(z.date,tz=TZ).timestamp();b=(pd.Timestamp(z.date,tz=TZ)+pd.DateOffset(days=1)).timestamp();events=[]
  for x in good.t:
   left=max(a,x.timestamp()-450);right=min(b,x.timestamp()+450)
   if left<right:events.extend([(left,1),(right,-1)])
  events.sort();level=0;prev=a;covered=0.;maxgap=0.;gapstart=a
  for x,v in events:
   if level>0:covered+=x-prev
   else:maxgap=max(maxgap,x-gapstart)
   level+=v
   if level==0:gapstart=x
   prev=x
  maxgap=max(maxgap,b-gapstart) if level==0 else maxgap
  assert abs(covered-z.covered_seconds)<1e-7 and abs(maxgap-z.max_uncovered_gap_seconds)<1e-7
  assert z.day_length_seconds==b-a;checks+=1
 curves.append((ordinal,good))
# Fixed-source curves stay in TRE, no individual export. Inspectable static HTML.
html=['<!doctype html><meta charset="utf-8"><title>W09 fixed curves QC — TRE ONLY</title><h1>W09 fixed 3 source-ordinal curves</h1><p>Chosen before glucose values: first / middle / last source ordinal. Individual dates/IDs stay in TRE.</p>']
for ordinal,d in curves:
 g=d.g.to_numpy();t=np.array([x.timestamp() for x in d.t]);x=(t-t.min())/max(1.,t.max()-t.min())*1000+40;y=230-(g-40)/460*210;poly=' '.join(f'{a:.2f},{b:.2f}' for a,b in zip(x,y));html.append(f'<h2>Source ordinal {ordinal}, {len(d)} valid points</h2><svg viewBox="0 0 1080 260"><rect x="40" y="20" width="1000" height="210" fill="white" stroke="black"/><polyline fill="none" stroke="#127783" stroke-width="1" points="{poly}"/></svg>')
html.append('<p>Automated full-fragment unit, interval, conflict and range tests accompany these curves. Visual inspection is not asserted by script.</p>');(Q/'fixed_curve_QC_TRE_ONLY.html').write_text('\n'.join(html))
v={'verified':True,'build_actual_exit':0,'production_calendar_repair_09_exit':0,'date_key_representation':'ISO YYYY-MM-DD string','source_fragments_hash_verified':len(m),'all_point_count_conservation_verified':True,'mirrors_verified':len(s['artifacts']),'independent_source_fragments':3,'independent_source_days':checks,'selected_unique_day_keys':True,'candidate_rule_80pct_2h_verified':True,'full_protocol_freeze':False,'curve_html_created':True,'curve_visual_review_completed':False,'codes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}};(Q/'verification.json').write_text(json.dumps(v,indent=2));print('W09_INDEPENDENT_PASS',json.dumps(v),flush=True)
