"""Reconcile unfiltered raw medication rows against the explicit valid-visit key contract."""
import runpy
from pathlib import Path
B=runpy.run_path(str(Path(__file__).with_name('03_source_QC.py')),run_name='auditlib');globals().update({k:v for k,v in B.items() if not k.startswith('__')})
import re
raw=readpath('/home/ec2-user/studies/hpp_datasets/medications/medications.parquet',K+['atc5'],purpose='X05 reconcile raw invalid-key records before independent class join');t=read('medication_timing',K+['atc5']);W=read('W15_order',K);bad={'','undefined','none','null','nan','na','n/a','unknown','<na>'};masks={}
for c in K:
 s=raw[c].astype('string');masks[c]=(s.notna()&~s.str.lower().isin(bad)&s.eq(s.str.strip())).fillna(False)
pid=pd.to_numeric(raw.participant_id,errors='coerce');masks['participant_id'] &= np.isfinite(pid)&pid.gt(0)&pid.eq(np.floor(pid));good=np.logical_and.reduce([v.to_numpy(dtype=bool) for v in masks.values()]);ad=raw.loc[good].copy();ad['participant_id']=pd.to_numeric(ad.participant_id).astype('int64');ad.cohort=ad.cohort.astype(str);ad.research_stage=ad.research_stage.astype(str);r=ad.groupby(K).size().rename('raw').to_frame().join(t.groupby(K).size().rename('timing'),how='outer').fillna(0);rowdiff=int(r.raw.ne(r.timing).sum());rules=json.loads((P/'config/medication_rules_W14_20261029.json').read_text())['class_patterns']
def atcs(x):
 if isinstance(x,(np.ndarray,list,tuple,set)):return set().union(*(atcs(v) for v in x)) if len(x) else set()
 return set(re.findall(r'(?<![A-Z0-9])[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}(?![A-Z0-9])',str(x)))
def counts(d):
 out=Counter()
 for v in d.itertuples(index=False):
  cs=atcs(v.atc5);k=tuple(getattr(v,c) for c in K)
  for cl,ps in rules.items():
   if any(c==p if cl=='metformin' else c.startswith(p) for c in cs for p in ps):out[k+(cl,)]+=1
 return out
ac=counts(ad);tc=counts(t);diff=sum(ac[k]!=tc[k] for k in set(ac)|set(tc));invalid=raw.loc[~good];knownpk=keys(W,['cohort','participant_id']);sourceknown=int(sum((r.cohort,r.participant_id) in knownpk for r in invalid.itertuples()));out={'raw_rows':len(raw),'raw_valid_explicit_visit_key_rows':len(ad),'timing_rows':len(t),'raw_invalid_visit_key_rows':len(invalid),'invalid_by_key_field':{c:int((~v).sum()) for c,v in masks.items()},'valid_key_row_count_mismatches':rowdiff,'valid_key_ATC_class_count_mismatches':diff,'invalid_rows_matching_a_W15_cohort_participant_without_valid_visit':sourceknown,'invalid_keys_repaired_or_imputed':False,'interpretation':'unfiltered raw and timing table have intentionally different source-key eligibility; unknown-visit records cannot be attached to a baseline visit from participant alone','published_window_state_reconstruction_mismatch_total':sum(json.loads((Q/'X05_medication_independent_merge.json').read_text())['mismatches'].values())};assert len(raw)-len(ad)==799 and len(ad)==len(t)==42618 and rowdiff==diff==0;save('X05_raw_key_eligibility_reconciliation.json',out);print('X05_RAW_KEY_RECONCILIATION',json.dumps(out),flush=True)
