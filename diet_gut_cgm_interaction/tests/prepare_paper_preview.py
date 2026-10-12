"""Normalize archived screenshot evidence; does not fabricate unseen table rows."""
from pathlib import Path
import hashlib
import json
import copy

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT/'research_records'
SOURCES = {
 'focus':RECORDS/'2026-10-10_服务器结果/截图显示数值与证据边界.json',
 'follow':RECORDS/'2026-10-10_服务器结果/后续核对截图显示数值.json',
 'check':RECORDS/'2026-10-10_服务器结果/17.40.57_补充核对数值.json',
 'extension':RECORDS/'2026-10-10_扩展敏感性/19.01_服务器结果/截图显示数值与核对.json',
 'qfix':RECORDS/'2026-10-10_扩展敏感性/19.10_q值核对/核对数值与更正.json'}
d={k:json.loads(p.read_text()) for k,p in SOURCES.items()}
models=['strict','protocol_same_sample','protocol_adjusted','quadratic','covariate_interactions','common_all7_sample']
keys=['exposure','taxon','outcome']
def clean(row):
 row=copy.deepcopy(row)
 for src,dst in [('ci_low_native','interaction_ci_low_native'),('ci_high_native','interaction_ci_high_native'),('q_focus','q_focus24'),('q_all','q_all84')]:
  if src in row:row[dst]=row.pop(src)
 for name in ['display','transcription_status','supersedes','correction_reason','status']:row.pop(name,None)
 row.update(status='OK',model=row.get('model','primary'),family='focus' if row['exposure'] in ['hPDI','NOVA4'] else 'extension')
 row['unit']='percentage points' if row['outcome']=='CV' else 'mg/dL'
 return row
primary=[clean(r) for r in d['focus']['focus24_primary_displayed']+d['follow']['extension_primary_q_all_below_005_displayed']]
lookup={tuple(r[k] for k in keys):r for r in primary}
for r in d['follow']['nova4_t068_interactions_displayed']:
 if r['model']=='primary':
  target=lookup[tuple(r[k] for k in keys)]; target.update(clean(r))
# Apply the independently checked 19:10 row, preserving displayed numeric precision.
q=d['qfix']['old_primary_displayed']; row=lookup[('rEDIH','T096','CV')]
for src,dst in [('Beta','interaction_beta_native'),('L95','interaction_ci_low_native'),('U95','interaction_ci_high_native'),('p','interaction_p'),('qAll84','q_all84')]:row[dst]=float(q[src])
sens=[clean(r) for r in d['follow']['nova4_t068_interactions_displayed'] if r['model']!='primary']
for i,r in enumerate(sens):
 if (r['model'],r['outcome'])==('protocol_adjusted','CV'): sens[i]=clean(d['check']['corrected_protocol_cv_row'])
for diag in d['check']['primary_diagnostics']:
 lookup[tuple(diag[k] for k in keys)].update({k:v for k,v in diag.items() if k not in ['display','model']})
for effect in d['check']['primary_double_iqr_contrasts']:
 row=lookup[tuple(effect[k] for k in keys)]
 row.update(double_iqr_beta_native=effect['double_iqr_beta_native'],double_iqr_ci_low_native=effect['ci_low_native'],double_iqr_ci_high_native=effect['ci_high_native'],double_iqr_status=effect['double_iqr_status'])
slopes=[]
for r in d['follow']['primary_conditional_diet_slopes_displayed']:
 row={k:v for k,v in r.items() if k not in ['display','transcription_status']}
 row['n']=lookup[tuple(row[k] for k in keys)]['n'];slopes.append(row)
for item in d['check']['extension_conditional_slopes']:
 for point in item['slopes']:
  row={k:item[k] for k in keys+['n','model','unit']}
  row.update({k:v for k,v in point.items() if k!='display'});slopes.append(row)
stability=[]
for item in d['extension']['original_targets_displayed_strings']:
 key=('Carbohydrate_pct' if item['Diet']=='Carb_pct' else item['Diet'],item['Taxon'],item['Y'])
 pr=lookup[key]
 row=dict(zip(keys,key),primary_n=pr['n'],reference_beta_native=pr['interaction_beta_native'],reference_q_all84=pr['q_all84'],min_sensitivity_n=int(item['Nmin']),retained_sensitivity_count=int(item['Keep']),retained_all_six=item['Keep']=='6',same_direction_all_six=item['Same6']=='yes',all_six_estimable=True)
 for code,model in zip('SPAQCU',models):row[model+'__q_all84']=float(item[code])
 row['max_sensitivity_q_all84']=max(row[m+'__q_all84'] for m in models)
 stability.append(row)
summary=[]
for item in d['extension']['summary_displayed_strings']:
 lo,hi=item['Nrange'].split('..')
 summary.append(dict(model=models[list('SPAQCU').index(item['Spec'])],extension_planned=60,extension_fitted=60,extension_q_all84_below_alpha=int(item['qAll84_below_alpha']),reference_hits=19,reference_hits_retained=int(item['OldRetained']),new_significant_outside_reference=int(item['NewOutsideOld']),n_min=int(lo),n_max=int(hi),limited_support_tests=int(item['SupportFlags'])))
meta=dict(mode='SCREENSHOT_PREVIEW',original_run='run_20261010T085601_096554Z_4d43e0',followup_run='extension_20261010T105930_496220Z_ff8085',reference_reproduced=204,actual_fits=588,planned_fits=588,counts=d['focus']['summary'],corrections=['17:40 protocol CV interval corrected','19:10 rEDIH/T096/CV q corrected to 0.04872153914'],source_records={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()},limitations=['Screenshot-rounded aggregate values; unseen 41 primary extension rows, individual sensitivity coefficients and missing p values are not reconstructed.','41 unseen primary extension results require server export; no placeholder zero rows.'])
tables=dict(primary=primary,original=primary+sens,extension=[],all_followup=[],slopes=slopes,stability=stability,summary=summary,fdr=[],reproduction=[])
assert len(primary)==43 and len(slopes)==44 and len(stability)==19
assert sum(r['retained_all_six'] for r in stability)==7
path=ROOT/'paper_results/screenshot_sources/snapshot.json'
path.write_text(json.dumps(dict(meta=meta,tables=tables),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(path)
