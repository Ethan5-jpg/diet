# -*- coding: utf-8 -*-
"""Persist W11/W12 reviewed scope, actual exit records, hash manifests and code-to-result memos."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w11_w12_covariates_samples_20261028';L=Path('/home/ec2-user/multisystem_runtime_20261021/w11_w12_covariates_samples_20261028')
actual={f'{i:02}':int((L/(f'{i:02}_actual_exit_code')).read_text()) for i in range(1,11)};assert all(v==0 for v in actual.values());V=json.loads((Q/'independent_verification_TRE_ONLY.json').read_text());assert V['passed'];A=json.loads((Q/'w11_summary_TRE_ONLY.json').read_text());B=json.loads((Q/'w12_summary_TRE_ONLY.json').read_text());OH=json.loads((Q/'oral_health_completion_TRE_ONLY.json').read_text());files=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(Path(OH['source_path']))==OH['source_sha256'] and sha(Path(OH['coding_source_path']))==OH['coding_source_sha256']
for file in ['w11_current_artifact_manifest_TRE_ONLY.json','w12_artifact_manifest_TRE_ONLY.json']:
 for e in json.loads((Q/file).read_text()):assert sha(Path(e['path']))==e['sha256'];files.append(e)
for e in json.loads((Q/'w11_artifact_revisions_TRE_ONLY.json').read_text()):assert sha(Path(e['archived_path']))==e['before_sha256']
code=P/'scripts/w11_w12_covariates_samples_20261028';steps='''| Code | Function | Actual exit |
|---|---|---|
|01_preflight.py|W10 gate hashes/exit0; bounded schemas/dictionaries for W11/W12|0|
|02_read_contracts.py|Field/time/sample metadata and coding-file locations|0|
|03_compact_contract.py|Focused schema/codebook inspection|0|
|04_covariate_value_contract.py|Projected covariate missingness, levels, duplicate-key checks|0|
|05_sample_identity_probe.py|Complete selected identity projections; gut cardinality and date absence|0|
|06_focused_values_and_codes.py|Actual sex/smoking/family codebook, source start-date fields, candidate summaries|0|
|07_build_covariates.py|Union registry; calendar-year age, source sex, smoking/alcohol/activity/sleep, medication start intervals and ATC records, family/household edges|0|
|08_build_sample_candidates.py|All primary abundance and index-only candidates, source versions, full oral bridge, gut held reasons, descriptive lags and frozen random-priority rule|0|
|09_complete_oral_health.py|Use actual source spelling oralhealth_antibiotics_for_ginvitis and code021_02, integrate decoded frequency; archive four prior artifacts and publish current manifest|0|
|10_independent_verify.py|Independent source key/value reconstruction, whole registry union, medication codes/start rows, family graph, all candidate links and brute-force lags, hashes and binary mirrors|0|
|11_finalize_and_memos.py|Final readback, code/result/MEMO index and execution-state update|external wrapper records exit after completion|
'''
common=f'''# W11/W12 execution memo — started2026-10-28, completed2026-10-29

Environment: {sys.executable}. Authorized HPP Server Skill; Chrome → WorkSpaces only. No SSH/API, original files read-only, no private individual table/ID/date download, no model fitting, no imputation or learned preprocessing. Local copies are authored code and manually transcribed aggregates only.

Code: {code}
Result: {Q}
Logs/runtime: {L}

{steps}

## W11 outputs and interpretation

Shared: P/analysis/shared/covariate_registry_TRE_ONLY.parquet; P/config/covariate_roles.yaml; P/data_qc/covariate_missingness.csv. P={P}.

The registry is a UNION of explicitly valid person/cohort/stage keys, not all-domain complete cases. {A['registry_person_visits']} person-visits / {A['registry_people']} unique source participants across stages; baseline registry {A['baseline_registry_people']}. Baseline age+sex10417, age+sex+smoking8219; not final diet/O/G/outcome N. Birth year supports calendar-year age only; no exact birth date invented. Explicit source sex code9 Female0/Male1; gender is not silently substituted. Source refusals and repeat conflicts remain unknown, not No. Activity/sleep/alcohol sources remain sensitivity candidates pending target/adjustment-set freeze.

Medication source has start_year/start_month/start_date, but no supplied stopping/continuation field. Preserve start year/month uncertainty intervals, source date conflicts and start-before/within/after-food statuses. Collection date is recording time, not drug start or proof of active use. ATC positive/not-listed/unknown flags describe supplied coding, not actual absence at the food window. They are sensitivity inputs until medication-time and target definitions are frozen. Missing start/stop cannot be repaired by imputing a date. No unvalidated diabetes/general diagnosis variable was substituted from the diabetes-eye questionnaire item; existing medical-condition source requires a separately justified disease/time definition before use.

Actual oral-health source field is spelled oralhealth_antibiotics_for_ginvitis. Source code021_02 has Never/Rarely/Often, not a generic binary recent-antibiotic question. Script09 corrected the exact-source-name lookup, preserved previous derived versions, and kept {OH.get('baseline_available')} baseline decoded values. It is not silently relabeled all systemic/recent antibiotic exposure.

Family/household components use explicit positive self-reported participant links across visits within cohort. Complete source edge graph rebuilt independently; absence of a link does not prove unrelatedness. This publishes grouping evidence for later split construction, not actual discovery/confirmation assignment. No genetic relatedness completeness claimed. Other outcomes/BMI/weight are not automatic covariates; module energy/calendar/device/microbiome batch interfaces remain module-specific and unlearned.

## W12 outputs and interpretation

Shared: P/analysis/shared/sample_candidates_TRE_ONLY.parquet; P/config/sample_selection_rules.yaml; P/data_qc/sample_identity_reasons.csv.

All20,129 source aggregate abundance rows (oral8834+gut11295) represented once, plus418 oral index-only rows =20,547 candidates. Primary oral index only; alternate UUID version inventoried, not pooled. Rebuilt oral exact person/source-date/unique-primary-index evidence for visit repair, compared all8834 rows to historical bridge. Resolved8398 across stages,8291 baseline;436 unresolved raw abundance rows are held, NOT labeled baseline simply from participant identity.

Gut unique visit-level index/profile links10646 across stages; baseline8676. Of599 unmatched source person-visits,185 have only another stage's same-person/cohort index;414 participants absent current primary index.50 repeated-visit profile rows (24 groups including22 both-many groups) held for specimen linkage evidence; equality of array_index is recorded as a proposal, not proof of shared biological sample namespace. No Cartesian expansion, arbitrary dedup or random unresolved join. A versioned aggregate row locator is not claimed a biological sample ID. Sample_name has no repeated values in current primary index, but biological vs technical identity must still be checked where required.

W12 is candidate/identity work, not W13 matrix QC. Baseline identity-ready oral8291/gut8676/joint7214, not final microbial QC or model N. Missing date/multiple dates/long lag are descriptive, not exclusions; no fixed30/60/90-day or before-sampling gate. Food nearest actual-day signed median oral+68, gut+9 among computable ready baseline rows; the two denominators differ, no long-term stability/cause claim.

User rule overrides older quality ranking: fixed source version, required identity/technical QC first, then uniformly seeded one eligible BIOLOGICAL sample per person/cohort/stage/site; seed20261027, immutable hash priority. No outcome, phenotype, effect or nearest-date ranking. Technical remeasurements and uncertain repeats require evidence before this biological random choice. Final selection is performed and locked by W13; selected count here0, all technical_QC_status=not_evaluated_W13. The same final sample must be reused across outcome modules.

## Verification / scope

```json
{json.dumps(V,indent=2)}
```

Current W11 manifest records actual final hashes after oral-health completion; original W11 artifact manifest plus archived prior files retains the complete first production version. Three W11 binary mirrors plus one W12 mirror independently verified. No team human review or fresh Astra review claimed. Actual exits01–10 all0; finalizer11 actual exit is recorded externally after it returns. W13 and final model protocol remain pending.

## Baseline covariate counts

```json
{json.dumps(A['baseline_counts'],indent=2)}
```

## Candidate sample counts and held reasons

```json
{json.dumps(B,indent=2)}
```
'''
for name in ['w11_w12_covariates_samples_20261028','w11_covariates_20261028','w12_sample_candidates_20261028']:
 p=P/'memos'/name/'MEMO.md';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(common);assert p.read_text()==common;files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
for p in sorted(code.glob('*.py')):files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
delivery={'passed':True,'W11_source_coding_time_registry_delivered':True,'W12_all_candidates_and_selection_rules_delivered':True,'actual_exits_before_finalizer':actual,'independent_verification':V,'baseline_covariate_counts':A['baseline_counts'],'baseline_denominator':A['baseline_registry_people'],'baseline_candidate_identity_counts':B['per_site'],'joint_baseline_identity_ready_not_final':B['baseline_joint_identity_ready_people_not_final_QC'],'candidate_rows':B['candidate_rows'],'W13_technical_QC_done':False,'final_primary_samples_selected':0,'model_protocol_frozen':False,'raw_data_exported':False,'team_human_review_claimed':False,'files':files};out=Q/'delivery_TRE_ONLY.json';assert not out.exists();out.write_text(json.dumps(delivery,indent=2));assert json.loads(out.read_text())['passed']
statepath=P/'PROJECT_EXECUTION_STATE.json';state=json.loads(statepath.read_text()) if statepath.exists() else {};state['W11']={'status':'source_coding_time_registry_independently_verified_adjustment_protocol_pending','code_directory':str(code),'result_directory':str(Q),'memo':str(P/'memos/w11_covariates_20261028/MEMO.md'),'not_final_model_N':True};state['W12']={'status':'all_candidates_and_user_random_selection_contract_verified_final_QC_and_selection_W13_pending','code_directory':str(code),'result_directory':str(Q),'memo':str(P/'memos/w12_sample_candidates_20261028/MEMO.md'),'selected_primary_samples':0};statepath.write_text(json.dumps(state,indent=2));assert json.loads(statepath.read_text())['W12']==state['W12']
print('FINAL_W11_W12_VERIFICATION',json.dumps({k:v for k,v in V.items() if k!='source_key_counts'}),flush=True);print('FINAL_W11_SUMMARY',A['registry_person_visits'],A['registry_people'],A['baseline_registry_people'],'EDGES',A['family_edges'],'BASELINE_OBSERVED_GROUP_GT1',A['baseline_in_observed_related_group_size_gt_one'],flush=True);print('FINAL_W12_SUMMARY',B['candidate_rows'],B['baseline_joint_identity_ready_people_not_final_QC'],'SELECTED',0,flush=True);print('MED_START_STATUS',json.dumps(A['medication_start_precision_counts']),flush=True);print('MED_START_FOOD_TIME',json.dumps(A['medication_start_vs_food_counts']),flush=True);print('DELIVERY_AND_THREE_MEMOS_READBACK_PASS',flush=True)
