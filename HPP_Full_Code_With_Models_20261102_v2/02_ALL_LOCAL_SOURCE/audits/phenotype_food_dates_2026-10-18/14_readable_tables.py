"""English labels for remote viewer without CJK fonts; no analysis changes."""
from pathlib import Path
import json, hashlib, sys
import pandas as pd
OUT=Path('/home/ec2-user/studies/diet_gut_cgm_v2/phenotype_food_dates_20261018')
assert '/envs/pheno/' in sys.executable
assert (OUT/'13_exit_code').read_text().strip()=='0'
s=json.loads((OUT/'summary.json').read_text())
labels={'anthropometrics':'Body: weight / BMI / waist','body_composition':'DXA body composition','liver_ultrasound':'Liver ultrasound','oral':'Oral microbiome','gut':'Gut microbiome'}
rows=[]
for r,label in labels.items():
    a=s['timing'][r]['diet'];c=s['timing'][r]['cgm']
    rows.append({'Dataset':label,'With dates':s['people'][r],'Paired with diet':a['unique_point_date'],'Median gap to diet (days)':a['nearest_observed_day_abs']['median'],'Before / during / after diet span':f"{a['before_span']} / {a['inside_span']} / {a['after_span']}",'Within 30 days of diet':a['within_days']['30'],'Median gap to CGM (days)':c['nearest_observed_day_abs']['median']})
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>HPP actual dates and daily food tables</title><style>body{font:15px Arial,sans-serif;max-width:1300px;margin:25px auto;padding:16px;color:#182638}table{border-collapse:collapse;font-size:13px;width:100%;margin:18px 0}th,td{border:1px solid #cdd7e2;padding:8px;text-align:left}th{background:#edf3f8}p{line-height:1.6}.note{background:#fff2d7;padding:12px}</style><h1>HPP: actual dates and daily food tables</h1><p class="note">TRE ONLY: real individual dates below. Scope: 10k baseline visit. Date availability does not establish complete diet logging, meal timing, CGM coverage, or causal ordering. English labels are used because this remote viewer lacks Chinese fonts.</p>',pd.DataFrame(rows).to_html(index=False,border=0),'<p>Each point-measurement row uses its own matched sample. CGM and diet columns can have different denominators. Gaps use nearest actual observed days. "During" uses the first-to-last diet-date span.</p><h2>Real-date examples</h2><p>First five sorted person/visit keys with at least seven common diet-CGM dates and one unique date for every point-measurement source. Examples are not representative cases. Participant IDs are not displayed.</p>']
p=pd.read_csv(OUT/'person_visit_all_dates_TRE_ONLY.csv');q=p.loc[p.common_days.ge(7)]
for r in labels:q=q.loc[q[r+'_distinct_dates'].eq(1)]
q=q.sort_values(['participant_id','cohort','research_stage']).head(5)
d=pd.read_csv(OUT/'daily_food_CGM_TRE_ONLY.csv',usecols=['participant_id','cohort','research_stage','date','food_event_rows','cgm_observed'])
for i,(_,r) in enumerate(q.iterrows(),1):
    a=[{'Dataset':'Food records','Actual date / span':r.diet_first+' to '+r.diet_last,'Observed days':int(r.diet_days)},{'Dataset':'CGM daily records','Actual date / span':r.cgm_first+' to '+r.cgm_last,'Observed days':int(r.cgm_days)},{'Dataset':'Actual common dates','Actual date / span':'Intersection of observed date sets; not inferred from spans','Observed days':int(r.common_days)}]
    a.extend({'Dataset':label,'Actual date / span':r[role+'_dates'],'Observed days':1} for role,label in labels.items())
    parts+=['<h3>Example '+str(i)+'</h3>',pd.DataFrame(a).to_html(index=False,border=0)]
    if i==1:
        k=['participant_id','cohort','research_stage'];x=d.loc[(d[k]==r[k]).all(axis=1),['date','food_event_rows','cgm_observed']].copy()
        x['cgm_observed']=x.cgm_observed.map({1:'Yes',0:'No'})
        parts+=['<h3>Example 1: each actual food-record date</h3><p>One row per observed food date. Food-event rows are not meal counts. Days without food records are not included.</p>',x.rename(columns={'date':'Actual food-record date','food_event_rows':'Food-event rows','cgm_observed':'CGM observed on this date?'}).to_html(index=False,border=0)]
parts+=['<h2>Full daily file</h2><p>daily_food_CGM_TRE_ONLY.csv: 125,374 person-days from 9,737 people. Median 14 food-record days per person; median 16 food-event rows per person-day. Event source includes food names, food IDs, weights, timestamps and nutrient fields. Nutrient sums require field-definition and portion-conversion review before scientific use.</p></html>']
b=''.join(parts).encode();target=OUT/'TABLES_TRE_ONLY.html';backup=OUT/'TABLES_ZH_TRE_ONLY.html'
assert not backup.exists();backup.write_bytes(target.read_bytes());target.write_bytes(b);assert target.read_bytes()==b
source=Path(__file__).read_bytes();(OUT/'14_readable_tables.py').write_bytes(source)
note='\n## 14_readable_tables.py\n\nRemote Firefox lacks CJK glyphs. Kept the original TABLES_ZH_TRE_ONLY.html and wrote English labels to TABLES_TRE_ONLY.html, preserving actual dates and selection rules. No statistics changed; readback verified. 14_exit_code=0.\n'
memo=OUT/'MEMO.md';b=(memo.read_text()+note).encode();memo.write_bytes(b);assert memo.read_bytes()==b
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')};(OUT/'code_sha256.json').write_text(json.dumps(hashes,indent=2))
(OUT/'14_exit_code').write_text('0\n')
print('READABLE_TABLES_SAVED_AND_READBACK_VERIFIED')
print('REAL_DATE_EXAMPLES',len(q),'SOURCE_SHA256',hashlib.sha256(source).hexdigest())
print('ANALYSIS_UNCHANGED; ORIGINAL_CHINESE_HTML_PRESERVED; INDIVIDUALS_TRE_ONLY')
print('READABLE_TABLES_EXIT_CODE_0')
