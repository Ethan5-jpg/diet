"""Independent BFS reconstruction and fail-closed access checks. No scientific values."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
from pathlib import Path
from collections import defaultdict,deque
import json,hashlib,resource,random,csv,io
import pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/w15_split_access_20261029');Q=P/'data_qc/w15_split_access_20261029';B=L/'artifacts';K=['participant_id','cohort','research_stage'];PK=['cohort','participant_id']
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(240,240))
assert (L/'02_actual_exit_code').read_text().strip()=='0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=json.loads((Q/'preflight.json').read_text());pol=json.loads((Q/'precommit_split_policy.json').read_text());art=json.loads((Q/'build_artifact_manifest.json').read_text());tests=[]
def ok(name,condition):
 assert condition,name
 tests.append(name)
def read(p,cols=None):
 f=pq.ParquetFile(p,pre_buffer=False)
 return pd.concat([b.to_pandas(ignore_metadata=True) for b in f.iter_batches(batch_size=127,columns=cols,use_threads=False)],ignore_index=True)
def src(role,cols):
 m=pre['inputs'][role];ok('source_'+role,sha(m['path'])==m['sha256']);return read(m['path'],cols)
for m in art:ok('artifact_'+m['name'],sha(m['runtime'])==m['sha256'] and Path(m['runtime']).stat().st_size==m['bytes'])
U=src('union',K);C=src('covariates',PK+['observed_related_group_id','observed_related_component_size']);E=src('edges',['cohort','participant_id','related_participant_id']);A=src('eligibility',K+['analysis_id','base_eligible'])
S=read(B/'split_order_TRE_ONLY.parquet');M=read(B/'module_membership_TRE_ONLY.parquet');G=read(B/'known_related_membership_TRE_ONLY.parquet');H=read(B/'person_access_status_TRE_ONLY.parquet')
def node(c,p):return str(c),int(p)
def bfs(nodes,edges):
 adj={n:set() for n in nodes}
 for a,b in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
 seen=set();out={}
 for start in sorted(adj):
  if start in seen:continue
  queue=deque([start]);seen.add(start);members=[]
  while queue:
   u=queue.popleft();members.append(u)
   for v in sorted(adj[u]):
    if v not in seen:seen.add(v);queue.append(v)
  members=sorted(members);cid=hashlib.sha256(json.dumps(members,separators=(',',':')).encode()).hexdigest()
  for n in members:out[n]=(cid,len(members))
 return out
nodes=set(node(c,p) for c,p in C[PK].itertuples(index=False,name=None))|set(node(c,p) for c,p in U[PK].itertuples(index=False,name=None));edges=[(node(r.cohort,r.participant_id),node(r.cohort,r.related_participant_id)) for r in E.itertuples(index=False)];components=bfs(nodes,edges)
ok('full_graph_BFS_count',len(components)==11269)
ok('candidate_set_exact',set(S[PK].itertuples(index=False,name=None))==set(U[PK].itertuples(index=False,name=None)) and len(S)==6305 and not S.duplicated(PK).any())
ok('BFS_candidate_component_and_size',all(components[node(r.cohort,r.participant_id)]==(r.split_unit_id,int(r.known_full_component_size)) for r in S.itertuples(index=False)))
ok('BFS_all_relevant_nodes',all(components[node(r.cohort,r.participant_id)]==(r.split_unit_id,int(r.known_full_component_size)) for r in G.itertuples(index=False)) and len(G)==6726)
relevant=set(S.split_unit_id);ok('relevant_node_set_exact',set(G[PK].itertuples(index=False,name=None))=={n for n,(cid,_) in components.items() if cid in relevant})
old_to_new=defaultdict(set);new_to_old=defaultdict(set)
for r in C.itertuples(index=False):
 cid,sz=components[node(r.cohort,r.participant_id)];old_to_new[str(r.observed_related_group_id)].add(cid);new_to_old[cid].add(str(r.observed_related_group_id));assert int(r.observed_related_component_size)==sz
ok('W11_full_partition_equivalent',all(len(v)==1 for v in old_to_new.values()) and all(len(v)==1 for v in new_to_old.values()))
ok('all_observed_edges_contained',all(components[a][0]==components[b][0] for a,b in edges))
prior={cid:hashlib.sha256((pol['seed']+'|'+cid).encode()).hexdigest() for cid in relevant};rank={cid:i+1 for i,cid in enumerate(sorted(relevant,key=lambda x:(prior[x],x)))}
ok('seed_priority_and_unit_order_exact',all(r.unit_priority_sha256==prior[r.split_unit_id] and r.unit_order==rank[r.split_unit_id] for r in S.itertuples(index=False)))
ok('within_person_order_exact',all(r.within_unit_person_priority==hashlib.sha256((pol['seed']+'|person|'+str(r.cohort)+'|'+str(int(r.participant_id))).encode()).hexdigest() for r in S.itertuples(index=False)))
expected=S.sort_values(['unit_order','within_unit_person_priority','cohort','participant_id'],kind='stable')
ok('person_order_exact',expected.person_order.tolist()==list(range(1,6306)))
shuffled=edges.copy();random.Random(991).shuffle(shuffled);ok('shuffled_reversed_duplicate_edges_invariant',bfs(nodes,[(b,a) for a,b in shuffled]+edges[:20])==components)
toy=bfs({('a',1),('a',2),('b',1)},[(('a',1),('a',99)),(('a',99),('a',2))]);ok('synthetic_non_candidate_bridge_retained',toy[('a',1)]==toy[('a',2)] and toy[('a',1)][1]==3);ok('synthetic_cross_cohort_namespace',toy[('a',1)][0]!=toy[('b',1)][0])
ok('singleton_not_claimed_unrelated',not S.relatedness_fully_ascertained.fillna(False).any())
expectedM=A.loc[A.base_eligible,K+['analysis_id']].copy();key=lambda d:set(d[K+['analysis_id']].itertuples(index=False,name=None));ok('all_module_membership_exact',len(M)==54151 and M.analysis_id.nunique()==11 and key(M)==key(expectedM) and not M.duplicated(K+['analysis_id']).any())
joined=M.merge(S[PK+['split_unit_id','unit_order','person_order','initial_role']],on=PK,validate='many_to_one',suffixes=('_m','_s'))
ok('all_modules_share_person_roles_and_order',all(joined[c+'_m'].equals(joined[c+'_s']) for c in ['split_unit_id','unit_order','person_order','initial_role']))
ok('all_actual_roles_unopened',S.initial_role.eq('unopened_pending_design_freeze').all() and S.final_role.isna().all() and not S.design_core_selected.any() and not S.new_scientific_Y_access_opened.any())
for n in [0,1,569,1139,2849,5699]:ok('hypothetical_cutoff_'+str(n),S.assign(role=S.unit_order.le(n)).groupby('split_unit_id').role.nunique().eq(1).all())
for n in [3,5,10]:ok('hypothetical_fold_'+str(n),S.assign(fold=S.unit_order.mod(n)).groupby('split_unit_id').fold.nunique().eq(1).all())
rows=list(csv.DictReader(io.StringIO((B/'access_ledger_TRE_ONLY.csv').read_text())));ok('history_event_count',len(rows)==8);ok('history_no_fabricated_individual_IDs',all(r['participant_id']=='' and r['family_id']=='' for r in rows));ok('AI_result_use_registered',any(r['who_accessed']=='AI_analysis_assistants' and r['selection_use']=='plan_discussion_and_feature_overlap_review' for r in rows));ok('colleague_status_unknown',H.colleague_history.eq('unknown_pending_self_confirmation').all());ok('person_history_status_coverage',len(H)==6305 and not H.duplicated(PK).any() and set(H[PK].itertuples(index=False,name=None))==set(U[PK].itertuples(index=False,name=None)));ok('no_historically_unexposed_validation_claim',not H.historically_unexposed_validation_established.any() and H.user_exact_participant_contact.eq('unknown').all() and H.AI_exact_historical_participant_contact.eq('unknown').all() and not H.new_W15_association_access.any())
receipt=json.loads((B/'design_core_receipt.json').read_text());ok('design_core_unactivated',receipt['selected_people']==receipt['selected_units']==receipt['new_Y_access_opened']==0 and receipt['final_fraction'] is None)
ns={};exec(compile((B/'w15_access_guard.py').read_text(),'independent_guard','exec'),ns);ok('allowed_metadata',ns['authorize_request'](K,'technical_membership_only'))
for cols,purpose in [(['Y_value'],'technical_membership_only'),(['glucose_mean_mg_dL'],'technical_membership_only'),(['UPF_lower_pp'],'technical_membership_only'),(K,'association'),(K,'loss'),(K,'selective_plot'),(K,'real_supervised_training')]:
 try:ns['authorize_request'](cols,purpose)
 except PermissionError:tests.append('denied_'+str(cols)+'_'+purpose)
 else:raise AssertionError('guard allowed forbidden request')
for old,new in [('discovery','confirmation_reserved'),('design_discovery','confirmation_reserved'),('confirmation_open','confirmation_reserved')]:
 try:ns['validate_transition'](old,new)
 except PermissionError:tests.append('denied_transition_'+old)
 else:raise AssertionError('invalid role transition')
try:ns['require_later_freeze']({'fake':True})
except PermissionError:tests.append('denied_fake_receipt')
else:raise AssertionError('fake receipt allowed')
projection=json.loads((B/'read_projection_manifest.json').read_text());white={'union':set(K+['C_observed_related_group_id','C_relatedness_fully_ascertained']),'covariates':set(K+['observed_related_group_id','observed_related_component_size','relatedness_fully_ascertained']),'edges':set(['cohort','participant_id','related_participant_id','link_kind']),'eligibility':set(K+['analysis_id','base_eligible'])}
ok('source_column_projection_no_scientific_values',len(projection['reads'])==4 and all(set(r['columns'])==white[r['role']] for r in projection['reads']) and not projection['scientific_Y_or_exposure_value_columns_read'])
ok('precommit_unchanged',sha(Q/'precommit_split_policy.json')==pre['precommit_sha256'] and sha(Q/'user_history_declaration.json')==pre['history_sha256'])
result={'status':'passed','independent_method':'BFS full graph; batch 127; full key-set comparison; no scientific values','checks':len(tests),'check_names':tests,'candidate_people':6305,'candidate_units':5699,'module_membership_rows':54151,'modules':11,'full_graph_nodes':11269,'relevant_full_graph_nodes':6726,'no_new_Y_values_read':True,'no_final_roles_or_fraction_assigned':True,'history_exact_ID_overlap':'unknown','source_fingerprints_unchanged':True,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(Q/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('W15_INDEPENDENT_VERIFY_PASS',json.dumps({k:v for k,v in result.items() if k!='check_names'}),flush=True)
