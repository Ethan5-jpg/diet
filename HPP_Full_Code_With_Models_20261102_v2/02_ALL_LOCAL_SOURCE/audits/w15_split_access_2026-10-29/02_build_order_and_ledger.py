# -*- coding: utf-8 -*-
"""W15 global family-aware order, historical contact ledger, fail-closed workflow guard; no Y projection."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from collections import defaultdict
import json,hashlib,resource,sys
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/w15_split_access_20261029');Q=P/'data_qc/w15_split_access_20261029';B=L/'artifacts';B.mkdir(exist_ok=False);K=['participant_id','cohort','research_stage'];PK=['cohort','participant_id'];resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(240,240));assert (L/'01_actual_exit_code').read_text().strip()=='0'
pre=json.loads((Q/'preflight.json').read_text());policy=json.loads((Q/'precommit_split_policy.json').read_text());history=json.loads((Q/'user_history_declaration.json').read_text());art=[];reads=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(role,cols):
 m=pre['inputs'][role];p=Path(m['path']);assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes'];assert set(cols)<=set(m['columns']);assert not any(c in cols for c in ['Y_value','value','glucose_mean_mg_dL','fiber_equal_day_mean','UPF_lower_pp']);f=pq.ParquetFile(p,pre_buffer=False);d=pd.concat([b.to_pandas(ignore_metadata=True) for b in f.iter_batches(columns=cols,batch_size=1024,use_threads=False)],ignore_index=True);reads.append({'role':role,'columns':cols,'rows':len(d),'sha256':m['sha256']});return d

def save(d,name):
 p=B/name;assert not p.exists();pq.write_table(pa.Table.from_pandas(d,preserve_index=False),p,compression='zstd');assert pq.ParquetFile(p).metadata.num_rows==len(d);art.append({'runtime':str(p),'name':name,'rows':len(d),'bytes':p.stat().st_size,'sha256':sha(p)})
def save_text(text,name):
 p=B/name;assert not p.exists();p.write_text(text);assert p.read_text()==text;art.append({'runtime':str(p),'name':name,'rows':None,'bytes':p.stat().st_size,'sha256':sha(p)})
assert sha(Q/'precommit_split_policy.json')==pre['precommit_sha256'];assert sha(Q/'user_history_declaration.json')==pre['history_sha256']
U=load('union',K+['C_observed_related_group_id','C_relatedness_fully_ascertained']);C=load('covariates',K+['observed_related_group_id','observed_related_component_size','relatedness_fully_ascertained']);E=load('edges',['cohort','participant_id','related_participant_id','link_kind']);A=load('eligibility',K+['analysis_id','base_eligible']);assert len(U)==6305 and not U.duplicated(PK).any();assert U.cohort.eq('10k').all() and U.research_stage.eq('00_00_visit').all();assert not U.C_relatedness_fully_ascertained.fillna(False).any()
# Build components on full known graph, including non-candidate bridge nodes, before restricting to the union.
parent={}
def node(cohort,pid):return str(cohort),int(pid)
def find(n):
 parent.setdefault(n,n)
 while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
 return n
def link(a,b):
 x,y=find(a),find(b)
 if x!=y:parent[max(x,y)]=min(x,y)
for cohort,pid in C[PK].drop_duplicates().itertuples(index=False,name=None):find(node(cohort,pid))
for r in E.itertuples(index=False):link(node(r.cohort,r.participant_id),node(r.cohort,r.related_participant_id))
for cohort,pid in U[PK].itertuples(index=False,name=None):find(node(cohort,pid))
groups=defaultdict(list)
for n in list(parent):groups[find(n)].append(n)
components={};priority={}
for root,members in groups.items():
 members=sorted(members);cid=hashlib.sha256(json.dumps(members,separators=(',',':')).encode()).hexdigest();pr=hashlib.sha256((policy['seed']+'|'+cid).encode()).hexdigest()
 for n in members:components[n]=(cid,len(members));priority[cid]=pr
# Existing W11 component codes must encode the same partition, never arbitrary per-visit group IDs.
comp_to_old=defaultdict(set);old_to_comp=defaultdict(set)
for r in C.itertuples(index=False):
 n=node(r.cohort,r.participant_id);cid,sz=components[n]
 if pd.notna(r.observed_related_group_id):comp_to_old[cid].add(str(r.observed_related_group_id));old_to_comp[str(r.observed_related_group_id)].add(cid)
 assert int(r.observed_related_component_size)==sz
assert all(len(v)==1 for v in comp_to_old.values()) and all(len(v)==1 for v in old_to_comp.values())
S=U.rename(columns={'C_observed_related_group_id':'W11_observed_related_group_id','C_relatedness_fully_ascertained':'relatedness_fully_ascertained'}).copy();S['split_unit_id']=[components[node(r.cohort,r.participant_id)][0] for r in S.itertuples(index=False)];S['known_full_component_size']=[components[node(r.cohort,r.participant_id)][1] for r in S.itertuples(index=False)];S['unit_priority_sha256']=S.split_unit_id.map(priority);order=sorted(set(S.split_unit_id),key=lambda x:(priority[x],x));rank={x:i+1 for i,x in enumerate(order)};S['unit_order']=S.split_unit_id.map(rank).astype('int64');S['within_unit_person_priority']=[hashlib.sha256((policy['seed']+'|person|'+str(r.cohort)+'|'+str(int(r.participant_id))).encode()).hexdigest() for r in S.itertuples(index=False)];S=S.sort_values(['unit_order','within_unit_person_priority','cohort','participant_id'],kind='stable').reset_index(drop=True);S['person_order']=np.arange(1,len(S)+1);S['unit_candidate_members']=S.groupby('split_unit_id').split_unit_id.transform('size');S['initial_role']='unopened_pending_design_freeze';S['final_role']=pd.array([None]*len(S),dtype='string');S['design_core_selected']=False;S['new_scientific_Y_access_opened']=False;S['history_exact_overlap_status']='unknown_not_proven_unexposed';S['order_version']=policy['version']
assert not S.duplicated(PK).any() and S.groupby('split_unit_id').unit_order.nunique().eq(1).all()
M=A.loc[A.base_eligible,K+['analysis_id']].merge(S[PK+['split_unit_id','unit_order','person_order','initial_role']],on=PK,how='left',validate='many_to_one');assert M.unit_order.notna().all() and not M.duplicated(K+['analysis_id']).any();assert set(M.participant_id)==set(S.participant_id);assert len(M)==54151
# All known visits and linked non-candidate people inherit the same unit if becoming eligible later.
relevant=set(S.split_unit_id);candidate_nodes=set(zip(S.cohort,S.participant_id));G=pd.DataFrame([{'cohort':n[0],'participant_id':n[1],'split_unit_id':cid,'unit_priority_sha256':priority[cid],'unit_order':rank[cid],'known_full_component_size':sz,'currently_eligible_candidate':n in candidate_nodes} for n,(cid,sz) in components.items() if cid in relevant]);assert not G.duplicated(PK).any()
H=S[PK+['split_unit_id','person_order']].copy();H['user_study_level_history']='known';H['user_exact_participant_contact']='unknown';H['colleague_history']='unknown_pending_self_confirmation';H['AI_study_level_history']='known_used_in_plan_discussion';H['AI_exact_historical_participant_contact']='unknown';H['logged_W14_AI_technical_processing']='known_current_candidate_metadata_and_Y_construction_not_association';H['other_sessions_or_manual_runs']='unverified';H['new_W15_association_access']=False;H['historically_unexposed_validation_established']=False
# Study-level events never fabricate individual historical access IDs.
ledger=[]
for actor in history['actors']:
 ledger.append({'prior_study_id':'early_HPP_exploration_or_other_unverified_sessions','source_version':'historical_report_as_logged_W04_20261027','participant_id':None,'family_id':None,'visit':'exact_historical_visit_unknown','sample_version':'unknown','exposure_id':'NOVA4_fiber_related_scope_uncertain','outcome_domain':'CGM_related_scope_uncertain' if actor['actor_id']!='other_sessions_or_manual_runs' else 'unknown','access_type':actor['historical_results_contact'],'who_accessed':actor['actor_id'],'access_time':'historical_time_unknown_recorded_2026-10-29','selection_use':actor['itemized_scope'],'overlap_status':'exact_ID_overlap_unknown','evidence_path':str(Q/'user_history_declaration.json'),'scope':'study_level_declaration_not_individual_ID_proof','evidence_note':actor['evidence']})
for section,exposure,domain in [('16','NOVA4_classified_energy_and_fiber_density','CGM'),('17','NOVA4_total_energy_comparison_equal_person_weight','between_and_within_person_CGM')]:
 ledger.append({'prior_study_id':'HPP_early_diet_gut_CGM','source_version':'report_2026-10-13_section_'+section,'participant_id':None,'family_id':None,'visit':'unknown','sample_version':'unknown','exposure_id':exposure,'outcome_domain':domain,'access_type':'known_study_level_result_contact','who_accessed':'AI_analysis_assistants','access_time':'prior_to_W15_exact_date_uncertain','selection_use':'plan_discussion_and_feature_overlap_review','overlap_status':'exact_ID_overlap_unknown','evidence_path':'local_config/w04_followup_20261027/historical_contact_ledger_draft.json; user_history_declaration.json','scope':'study_level_not_individual_ID_proof','evidence_note':'no new historical effect table opened in W15'})
ledger.append({'prior_study_id':'W14_current_technical_delivery','source_version':'w14_eligibility_20261029','participant_id':None,'family_id':None,'visit':'10k_00_00_visit','sample_version':'W13_locked_sample_version','exposure_id':'UPF_bounds_and_fiber_density','outcome_domain':'module_specific_according_to_W14_membership','access_type':'technical_Y_processing_no_model_or_association','who_accessed':'current_Codex_AI','access_time':'logged_W14_before_W15','selection_use':'QC_and_cohort_construction_only','overlap_status':'current_6305_union_known_by_membership','evidence_path':str(P/'data_qc/w14_eligibility_20261029/final_delivery_verification.json'),'scope':'participant_mapping_via_module_membership_TRE_ONLY_not_all_Y_for_every_person','evidence_note':'does not certify other sessions or human activity'})
ledger.append({'prior_study_id':'previous_NC_Diet_oral_diet_and_HPP_literature','source_version':'existing_W04_overlap_review_20261027','participant_id':None,'family_id':None,'visit':'unknown','sample_version':'unknown','exposure_id':'inherited_diet_definitions_and_NOVA_catalogue','outcome_domain':'oral_diet_and_gut_liver_literature','access_type':'methods_and_literature_context_known_specific_result_scope_uncertain','who_accessed':'research_team_and_AI_scope_uncertain','access_time':'prior_to_W15_exact_date_uncertain','selection_use':'definitions_and_plan_context','overlap_status':'exact_ID_overlap_unknown','evidence_path':'local_config/w04_diet_features_20261026/prior_work_overlap.md','scope':'study_level_not_individual_ID_proof','evidence_note':'no external validation claim from a new split alone'})
# No design core or final cutoffs activated: future opening requires a new pre-access receipt.
receipt={'version':policy['version'],'activation_status':'not_activated_optional_design_core','selected_people':0,'selected_units':0,'new_Y_access_opened':0,'eligible_unit_order_source':'freeze/split_order_TRE_ONLY.parquet','activation_rule':'whole-unit prefix selected and receipt sealed before any design-core Y read; exact count and purpose preregistered before opening, no choice by effect','if_activated_role':'discovery_only_never_confirmation','confirmation_and_real_supervised_training_currently_blocked':True,'final_fraction':None}
# The workflow guard cannot revoke existing filesystem access; it is the default route for subsequent scripts.
guard='''from pathlib import Path
import json
ALLOWED_COLUMNS=frozenset(['participant_id','cohort','research_stage','analysis_id','split_unit_id','unit_order','person_order','initial_role'])
ALLOWED_PURPOSE='technical_membership_only'
def authorize_request(columns,purpose):
    if purpose!=ALLOWED_PURPOSE or not set(columns)<=ALLOWED_COLUMNS:
        raise PermissionError('W15 locked: real Y, associations, losses, selective plots and real supervised fitting require later sealed access receipts')
    return True
def validate_transition(old,new):
    allowed={'unopened_pending_design_freeze':{'design_discovery','discovery','confirmation_reserved'},'design_discovery':{'discovery'},'discovery':set(),'confirmation_reserved':{'confirmation_open'}}
    if new not in allowed.get(old,set()):raise PermissionError('illegal role transition: opened/trained subjects cannot return to confirmation')
    return True
def require_later_freeze(receipt):
    raise PermissionError('W15 reader never authorizes real Y; a later separately verified W18/W22 reader and sealed receipts are required')
def load_blind_membership(analysis_id,columns=None):
    import pyarrow.parquet as pq
    columns=list(columns or sorted(ALLOWED_COLUMNS))
    authorize_request(columns,'technical_membership_only')
    project=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
    p=project/'freeze/module_membership_TRE_ONLY.parquet'
    keep=list(dict.fromkeys(columns+['analysis_id']))
    table=pq.ParquetFile(p,pre_buffer=False).read(columns=keep,use_threads=False).to_pandas(ignore_metadata=True)
    if analysis_id not in set(table.analysis_id):raise KeyError('unknown or deferred analysis_id')
    return table.loc[table.analysis_id.eq(analysis_id),columns].copy()
'''
ns={};exec(compile(guard,'w15_access_guard.py','exec'),ns);assert ns['authorize_request'](K,'technical_membership_only')
checks=1
for cols,purpose in [(['Y_value'],'technical_membership_only'),(['value'],'technical_membership_only'),(['participant_id'],'association'),(['participant_id'],'loss'),(['participant_id'],'selective_plot'),(['participant_id'],'real_supervised_training')]:
 try:ns['authorize_request'](cols,purpose)
 except PermissionError:checks+=1
 else:raise AssertionError('access not fail closed')
for old,new in [('discovery','confirmation_reserved'),('design_discovery','confirmation_reserved'),('confirmation_open','discovery')]:
 try:ns['validate_transition'](old,new)
 except PermissionError:checks+=1
 else:raise AssertionError('leakage transition accepted')
try:ns['require_later_freeze'](None)
except PermissionError:checks+=1
else:raise AssertionError('missing receipt accepted')
# Stress deterministic family consistency for many hypothetical future splits/folds, without assigning actual roles.
for cutoff in [0,1,len(order)//10,len(order)//5,len(order)//2,len(order)]:
 z=S.unit_order.le(cutoff);assert S.assign(test_role=z).groupby('split_unit_id').test_role.nunique().eq(1).all();checks+=1
for folds in [3,5]:assert S.assign(test_fold=S.unit_order.mod(folds)).groupby('split_unit_id').test_fold.nunique().eq(1).all();checks+=1
save(S,'split_order_TRE_ONLY.parquet');save(M,'module_membership_TRE_ONLY.parquet');save(G,'known_related_membership_TRE_ONLY.parquet');save(H,'person_access_status_TRE_ONLY.parquet');save_text(pd.DataFrame(ledger).to_csv(index=False),'access_ledger_TRE_ONLY.csv');save_text(json.dumps(policy,indent=2)+'\n','access_policy.yaml');save_text(json.dumps(receipt,indent=2)+'\n','design_core_receipt.json');save_text(guard,'w15_access_guard.py');save_text(json.dumps(history,indent=2)+'\n','user_history_declaration.json');save_text(json.dumps({'reads':reads,'scientific_Y_or_exposure_value_columns_read':[],'W15_new_scientific_result_reads':0},indent=2)+'\n','read_projection_manifest.json')
summary={'candidate_people':len(S),'candidate_units':len(order),'module_membership_rows':len(M),'modules':int(M.analysis_id.nunique()),'family_household_source_edges':len(E),'full_graph_nodes':len(components),'relevant_full_graph_nodes':len(G),'candidate_multi_person_units':int(S.loc[S.unit_candidate_members.gt(1),'split_unit_id'].nunique()),'people_in_multi_person_candidate_units':int(S.unit_candidate_members.gt(1).sum()),'people_with_known_full_component_size_gt1':int(S.known_full_component_size.gt(1).sum()),'candidate_singleton_units':int(S.loc[S.unit_candidate_members.eq(1),'split_unit_id'].nunique()),'largest_candidate_unit':int(S.unit_candidate_members.max()),'history_event_rows':len(ledger),'actor_declarations':len(history['actors']),'historically_unexposed_validation_established_people':0,'design_core_selected_people':0,'new_scientific_Y_access_opened_people':0,'actual_final_roles_assigned_people':0,'final_split_fraction':None,'guard_and_family_tests':checks,'source_Y_value_columns_read':0,'artifacts':len(art),'seed':policy['seed'],'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(Q/'build_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(Q/'build_artifact_manifest.json').write_text(json.dumps(art,indent=2)+'\n');print('W15_ORDER_LEDGER_BUILD_PASS',json.dumps(summary),flush=True)
