# -*- coding: utf-8 -*-
"""W10 source-defined per-outcome eligibility and registry. No association/model."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,shutil,csv,resource,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets');Q=P/'data_qc/w10_outcome_qc_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w10_outcome_qc_20261028');K=['participant_id','cohort','research_stage'];BAD={'','undefined','null','none','nan','na','n/a','unknown','<na>'}
assert sys.executable=='/home/ec2-user/miniconda/envs/pheno/bin/python'
assert (L/'03_actual_exit_code').read_text().strip()=='0'
J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());inputs=[];artifacts=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def textout(p,s):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),str(p);b=s.encode();p.write_bytes(b);assert p.read_bytes()==b;artifacts.append({'path':str(p),'bytes':len(b),'sha256':sha(p)})
def dump(p,j):textout(p,json.dumps(j,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n')
def load(p,cols):
 pf=pq.ParquetFile(p,pre_buffer=False);assert set(cols)<=set(pf.schema_arrow.names),str(set(cols)-set(pf.schema_arrow.names));assert pf.metadata.num_rows<200000
 budget=sum(pf.metadata.row_group(i).column(j).total_uncompressed_size for i in range(pf.metadata.num_row_groups) for j in range(pf.metadata.row_group(i).num_columns) if pf.metadata.row_group(i).column(j).path_in_schema in cols);assert budget<128*1024**2
 inputs.append({'path':str(p),'rows':pf.metadata.num_rows,'columns':cols,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'projected_uncompressed_bytes':budget})
 return pd.concat([b.to_pandas(ignore_metadata=True) for b in pf.iter_batches(columns=cols,batch_size=2048,use_threads=False)],ignore_index=True)
def keymask(d):
 m=pd.Series(True,index=d.index)
 for c in K:
  s=d[c].astype('string');m&=s.notna()&~s.str.lower().isin(BAD)&s.eq(s.str.strip())
 v=pd.to_numeric(d.participant_id,errors='coerce');return (m&v.gt(0)&v.eq(np.floor(v))).fillna(False)
def stable(d):
 d=d.copy();d['participant_id']=pd.to_numeric(d.participant_id).astype('int64');d['cohort']=d.cohort.astype(str);d['research_stage']=d.research_stage.astype(str);return d
# Verify sources still match observed preflight, not legacy cache.
for s in J:
 p=Path(s['path']);assert p.stat().st_size==s['bytes'] and p.stat().st_mtime_ns==s['mtime_ns'];assert pq.ParquetFile(p).metadata.num_rows==s['rows']
# User's latest rules are deployed as a distinct snapshot by the transfer bundle.
U=P/'config/w10_user_rules_20261028';policy=json.loads((U/'visit_and_sampling_policy.json').read_text());six=json.loads((U/'liver_six_mean_spec.json').read_text());assert six['phenotype_count']==6
specs=[{'id':'total_fat_mass','domain':'body_totalfat','field':'body_comp_total_fat_mass','unit':'g','source':'body_composition/body_composition.parquet','valid':'finite_gt_zero','direction':'higher total fat mass; not automatically worse independent of body size','role':'candidate_totalfat_main_field_protocol_freeze_W18'}, {'id':'vat_volume','domain':'body_vat','field':'total_scan_vat_volume','unit':'cm^3','source':'body_composition/body_composition.parquet','valid':'finite_ge_zero','direction':'higher DXA-estimated visceral adipose volume; zero flagged, not proven absence','role':'candidate_VAT_main_field_protocol_freeze_W18'}, {'id':'bmi','domain':'anthropometrics','field':'bmi','unit':'kg/m^2','source':'anthropometrics/anthropometrics.parquet','valid':'finite_gt_zero','direction':'higher body mass relative to height squared; no monotonic health assumption','role':'secondary_candidate_not_promoted'}]
for f in six['fields']:specs.append({'id':'liver_'+f['phenotype_id'],'domain':'liver','field':f['raw_field'],'unit':f['unit'],'source':'liver_ultrasound/liver_ultrasound.parquet','valid':'finite_gt_zero','direction':{'attenuation':'higher ultrasound attenuation, fat-related proxy; not histological fat fraction','elasticity':'higher elasticity/stiffness, not liver fat','velocity':'higher shear-wave velocity, not liver fat','viscosity':'higher viscosity, no validated healthy direction assigned','dispersion':'higher dispersion, no validated healthy direction assigned','speed_of_sound':'higher sound speed, no universal healthy direction assigned'}[f['phenotype_id']],'role':'one_of_six_user_defined_liver_phenotypes_no_extra_endpoints','old_aggregate':f['matching_existing_aggregate_candidate']})
# Exact dictionary units must be present; percentage scale issue is audited separately.
for s in specs:
 e=next(x for x in J if str(x['path']).endswith(s['source']));u={r['row'].get('units') for r in e['dictionary_rows'] if r['row'].get('tabular_field_name')==s['field']};expected={'g':{'gram','g'},'cm^3':{'cm^3'},'kg/m^2':{'kg/m2','kg/m^2'},'Pa·s':{'Pa·s','Pa.s','Pa s'}}.get(s['unit'],{s['unit']});assert bool(u&expected),(s['field'],u,expected);s['source_dictionary_units']=sorted(u)
frames={};meta={};detailqc=[]
for rel in sorted({s['source'] for s in specs}):
 selected=[s['field'] for s in specs if s['source']==rel];extras=['collection_date','collection_timestamp','array_index'];available=pq.ParquetFile(R/rel).schema_arrow.names
 if rel.startswith('body_composition'):extras+=['body_comp_total_total_mass','body_comp_total_tissue_mass','body_comp_total_region_percent_fat','body_comp_total_tissue_percent_fat','total_scan_vat_mass','total_scan_vat_area']
 if rel.startswith('anthropometrics'):extras+=['height','weight']
 cols=list(dict.fromkeys(K+selected+[c for c in extras if c in available]));d=load(R/rel,cols);km=keymask(d);meta[rel]={'source_rows':len(d),'invalid_key_rows':int((~km).sum()),'valid_key_rows':int(km.sum()),'source_QC_fields':[c for c in available if any(t in c.lower() for t in ['quality','qc_','valid','reject','status','device'])]};d=stable(d.loc[km]);d['source_row_ordinal']=np.flatnonzero(km)
 # Baseline only in counts; all explicitly valid stages are retained in registry.
 meta[rel]['duplicate_key_groups']=int(d.groupby(K).size().gt(1).sum());meta[rel]['baseline_source_people']=int(d.loc[d.cohort.eq('10k')&d.research_stage.eq('00_00_visit'),'participant_id'].nunique());frames[rel]=d
 if not rel.startswith('liver'):assert not d.duplicated(K).any(),'Non-liver source duplicate keys require explicit review'
 # Quality raw measurements stay TRE, independent of all other phenotypes.
 for s in [z for z in specs if z['source']==rel]:
  v=pd.to_numeric(d[s['field']],errors='coerce');finite=np.isfinite(v);ok=finite&(v.ge(0) if s['valid']=='finite_ge_zero' else v.gt(0));reason=pd.Series('valid',index=d.index);reason.loc[d[s['field']].isna()]='source_missing';reason.loc[d[s['field']].notna()&v.isna()]='not_numeric';reason.loc[v.notna()&~finite]='nonfinite';reason.loc[finite&~ok]='outside_mathematical_domain'
  z=d[K+['source_row_ordinal']].copy();z['outcome_id']=s['id'];z['source_value']=v;z['measurement_eligible']=ok;z['reason']=reason;z['zero_value_flag']=finite&v.eq(0);detailqc.append(z)
# Minimal domain plausibility does not invent manufacturer thresholds or percentile trimming.
qc=pd.concat(detailqc,ignore_index=True);registry=[]
for s in specs:
 d=frames[s['source']];z=qc.loc[qc.outcome_id.eq(s['id'])].copy();assert len(z)==len(d)
 z['valid_value']=z.source_value.where(z.measurement_eligible)
 a=z.groupby(K,sort=True,observed=True).agg(value=('valid_value','mean'),n_source_measurements=('valid_value','size'),n_valid_measurements=('valid_value','count'),zero_measurement_count=('zero_value_flag','sum')).reset_index();a['outcome_id']=s['id'];a['domain']=s['domain'];a['unit']=s['unit'];a['phenotype_eligible']=a.n_valid_measurements.gt(0);a['status']=np.where(a.phenotype_eligible,'source_value_QC_pass','no_valid_measurement');a['manufacturer_QC_available']=False;a['date_admission_rule']='same_explicit_person_cohort_research_stage_no_lag_or_timestamp_gate'
 if 'collection_date' in d:
  dates=pd.to_datetime(d.collection_date,errors='coerce');t=d[K].copy();t['date']=dates.dt.strftime('%Y-%m-%d');t=t.groupby(K,sort=True,observed=True).agg(first_collection_date=('date','min'),last_collection_date=('date','max')).reset_index();a=a.merge(t,on=K,validate='one_to_one')
 else:a['first_collection_date']=None;a['last_collection_date']=None
 registry.append(a)
reg=pd.concat(registry,ignore_index=True);assert not reg.duplicated(K+['outcome_id']).any();assert reg.loc[~reg.phenotype_eligible,'value'].isna().all()
# CGM is a recipe plus a separate availability interface. It is NOT whole-period Y.
cgpath=P/'analysis/shared/cgm_daily_TRE_ONLY.parquet';cgnames=pq.ParquetFile(cgpath).schema_arrow.names;cg=load(cgpath,K+['date','cgm_day_eligible','glucose_mean_mg_dL']);assert not cg.duplicated(K+['date']).any();cga=cg.loc[cg.cgm_day_eligible.eq(True)].groupby(K,observed=True).agg(n_qualified_cgm_days=('date','size'),first_cgm_day=('date','min'),last_cgm_day=('date','max')).reset_index();cga['outcome_value_materialized']=False;cga['status']='W14_must_recompute_equal_day_mean_on_module_specific_shared_days';cga['minimum_days_frozen']=False
# Structural crosschecks are diagnostics, not automatic unrelated-variable exclusion.
b=frames['body_composition/body_composition.parquet'];fat=pd.to_numeric(b.body_comp_total_fat_mass,errors='coerce');bm=pd.to_numeric(b.body_comp_total_total_mass,errors='coerce');bt=pd.to_numeric(b.body_comp_total_tissue_mass,errors='coerce');diag={'total_fat_above_total_mass':int((fat.gt(bm)&fat.notna()&bm.notna()).sum()),'total_fat_above_tissue_mass':int((fat.gt(bt)&fat.notna()&bt.notna()).sum())}
for field,denom in [('body_comp_total_region_percent_fat',bm),('body_comp_total_tissue_percent_fat',bt)]:
 v=pd.to_numeric(b[field],errors='coerce');ratio=fat/denom;m=np.isfinite(v)&np.isfinite(ratio)&denom.gt(0);diag[field]={'source_dictionary_unit':'%','observed_max':float(v.max()),'stored_as_fraction_ratio_check_n':int(m.sum()),'max_abs_difference_vs_mass_ratio':float((v[m]-ratio[m]).abs().max()),'not_used_as_selected_outcome':True,'do_not_blindly_interpret_0_3_as_0_3_percent':True}
a=frames['anthropometrics/anthropometrics.parquet'];m=np.isfinite(a.bmi)&np.isfinite(a.weight)&np.isfinite(a.height)&a.height.gt(0);diff=(a.loc[m,'bmi']-a.loc[m,'weight']/(a.loc[m,'height']/100)**2).abs();diag['bmi_formula']={'comparable_rows':int(m.sum()),'max_absolute_error':float(diff.max()),'over_1e_minus_6':int(diff.gt(1e-6).sum()),'formula':'weight_kg/(height_cm/100)^2','not_recomputed_or_imputed':True}
# Compare all matched published means against consistent detail reconstruction.
aggcols=K+['array_index']+[s['old_aggregate'] for s in specs if s.get('old_aggregate')];ag=stable(load(R/'liver_ultrasound/liver_ultrasound_aggregated.parquet',aggcols));assert not ag.duplicated(K).any();comparison=[]
for s in specs:
 if not s.get('old_aggregate'):continue
 z=reg.loc[reg.outcome_id.eq(s['id'])].merge(ag[K+[s['old_aggregate']]],on=K,how='outer',validate='one_to_one',indicator=True);v=pd.to_numeric(z[s['old_aggregate']],errors='coerce');m=z.value.notna()&v.notna();err=(z.loc[m,'value']-v[m]).abs();comparison.append({'outcome_id':s['id'],'both_values':int(m.sum()),'max_absolute_difference':float(err.max()),'difference_gt_1e_minus_6':int(err.gt(1e-6).sum()),'detail_keys_without_aggregate':int(z['_merge'].eq('left_only').sum()),'aggregate_keys_without_detail':int(z['_merge'].eq('right_only').sum())})
# Record valid baseline source outcomes and optional diet-availability joins only; not final E/D.
food=load(P/'analysis/shared/food_daily_TRE_ONLY.parquet',K+['date','energy_eligible']);foods=food[K].drop_duplicates();en=food.loc[food.energy_eligible.eq(True),K].drop_duplicates();rows=[]
for s in specs:
 a=reg.loc[reg.outcome_id.eq(s['id'])&reg.cohort.eq('10k')&reg.research_stage.eq('00_00_visit')];v=a.loc[a.phenotype_eligible];q=qc.loc[qc.outcome_id.eq(s['id'])&qc.cohort.eq('10k')&qc.research_stage.eq('00_00_visit')];row={'outcome_id':s['id'],'unit':s['unit'],'baseline_source_people':len(a),'baseline_eligible_people':len(v),'baseline_missing_or_invalid_people':int((~a.phenotype_eligible).sum()),'baseline_with_indexed_food_same_visit':len(v.merge(foods,on=K,validate='one_to_one')),'baseline_with_energy_food_same_visit':len(v.merge(en,on=K,validate='one_to_one')),'baseline_zero_measurement_rows':int(q.zero_value_flag.sum()),'baseline_reason_counts':q.reason.value_counts().astype(int).to_dict(),'all_stage_eligible_visits':int(reg.loc[reg.outcome_id.eq(s['id']),'phenotype_eligible'].sum()),'not_final_microbiome_or_covariate_N':True};rows.append(row);print('W10_COUNT',json.dumps(row),flush=True)
def parquet(d,name,dest):
 src=L/name;assert not src.exists() and not dest.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),src,compression='zstd');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest) and pq.ParquetFile(dest).metadata.num_rows==len(d);artifacts.append({'path':str(dest),'bytes':dest.stat().st_size,'rows':len(d),'sha256':sha(dest),'runtime_source':str(src)})
parquet(qc,'outcome_measurement_QC_TRE_ONLY.parquet',Q/'outcome_measurement_QC_TRE_ONLY.parquet');parquet(reg,'outcome_registry_TRE_ONLY.parquet',P/'analysis/shared/outcome_registry_TRE_ONLY.parquet');parquet(cga,'cgm_outcome_availability_TRE_ONLY.parquet',Q/'cgm_outcome_availability_TRE_ONLY.parquet')
dictionary={'version':'W10_20261028_v1','status':'source_identity_unit_basic_QC_registered_not_final_model_protocol','person_visit_key':K,'independent_per_outcome_eligibility':True,'static_outcomes':specs,'liver_same_visit_aggregation':six['aggregation'],'body_BMI_duplicate_rule':'asserted_unique_person_cohort_stage_in_current_source; no arbitrary_dedup','dates':'descriptive_only_no_numeric_timestamp_unit_inference_no_date_gate','missing':'not_zero; no_imputation; all_missing_visit_value_missing','minimum_QC':'finite and mathematical_domain only; source device/manufacturer flags absent; no invented clinical ranges or winsorization','VAT_zero':'retained with zero flag; do not claim biological absence; optional zero-exclusion sensitivity before protocol freeze; no log applied','CGM':{'outcome_id':'glucose_common_day_mean','unit':'mg/dL','formula':'arithmetic_mean_of_daily_glucose_mean_over_module_specific_qualified_shared_calendar_dates','point_daily_contract':str(P/'config/cgm_point_daily_rules_W09_20261028.json'),'availability_table':str(Q/'cgm_outcome_availability_TRE_ONLY.parquet'),'values_materialized_W10':False,'materialize_in':'W14_separately_for_exposure_and_pair_common_window','do_not_substitute_iglu_full_period_mean':True,'minimum_days_not_frozen':True},'hypothesis_budget_not_frozen':True,'no_association_or_prediction_run':True,'nonselected_percentage_scale_diagnostic':diag,'source_QC_field_inventory':meta}
dump(P/'config/outcome_dictionary.yaml',dictionary);dump(Q/'source_input_manifest_TRE_ONLY.json',inputs);dump(Q/'source_QC_and_structural_diagnostics.json',{'source_meta':meta,'structural_diagnostics':diag,'old_liver_aggregate_comparison':comparison});dump(Q/'summary_TRE_ONLY.json',{'baseline_outcome_counts':rows,'registry_rows':len(reg),'measurement_QC_rows':len(qc),'CGM_availability_visits':len(cga),'CGM_baseline_available_people':int(cga.loc[cga.cohort.eq('10k')&cga.research_stage.eq('00_00_visit'),'participant_id'].nunique()),'static_outcome_count':len(specs),'source_meta':meta,'not_final_analysis_N':True})
# Human-readable CSV retains counts only; individual registry stays TRE.
import io
f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);textout(P/'data_qc/outcome_field_QC.csv',f.getvalue())
dump(Q/'build_artifact_manifest_TRE_ONLY.json',artifacts)
print('W10_BUILD_DONE',len(reg),len(qc),'peak_RSS_MiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True)
print('STRUCTURAL_DIAGNOSTICS',json.dumps(diag),flush=True);print('LIVER_MEAN_COMPARISON',json.dumps(comparison),flush=True)
