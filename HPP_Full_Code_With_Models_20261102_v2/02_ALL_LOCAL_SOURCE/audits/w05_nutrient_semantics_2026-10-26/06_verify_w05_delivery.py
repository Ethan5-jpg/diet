"""Read back W05 outputs and actual exit status; print aggregate-only pages."""
import hashlib,json,sys
from pathlib import Path
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
L=Path('/home/ec2-user/multisystem_runtime_20261021/w05_launch_20261026_v3')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
x=read(P/'W05_LATEST_RUN.json');assert x['status']=='completed',x
run=x['run'];q=Path(x['result']);s=read(q/'summary.json');v=read(q/'verification.json');inp=read(P/'manifests'/run/'inputs.json')
assert read(L/'exit_code.json')['exit_code']==0
assert v['passed'] and set(v['actual_worker_exit_codes'].values())=={0} and v['independent_example_readbacks']==12
assert h(P/'scripts'/run/'05_validate_nutrients.py')=='3d6bd89f84cf17eefbc5f1a13599fbb4f4b3b1b77bdcc48ddffa87effa2835ce'
assert s['total_event_rows_scanned']==9125981 and len(s['source_summaries'])==2
assert pq.read_metadata(q/'nutrient_conversion_audit.parquet').num_rows==28
for a in v['artifacts']:
    p=Path(a['path']);assert p.stat().st_size==a['bytes'] and h(p)==a['sha256']
for source in inp['sources']:
    p=Path(source['path']);st=p.stat();assert st.st_size==source['bytes'] and st.st_mtime_ns==source['mtime_ns']
for d in inp['dictionaries'].values():assert h(d['path'])==d['sha256']
config=P/'config'/run/'nutrient_semantics.yaml';assert read(config)==read(P/'config'/'nutrient_semantics.yaml')
assert all(f['multiplier']==1 and not f['weight_rescaling'] for f in read(config)['fields'].values())
assert (P/'memos'/run/'MEMO.md').read_bytes()==(q/'MEMO.md').read_bytes()
page=int(sys.argv[1]) if len(sys.argv)>1 else 1
print('\n'*35+'W05_FINAL_READBACK_PASS page',page,run)
print('ACTUAL_EXIT main=0 processed=0 raw=0; independent_examples=12; audit_rows=28')
if page==1:
    for ss in s['source_summaries']:
        print('SOURCE',ss['source'],'FULL_ROWS',ss['rows_scanned'],'BASE_ROWS',ss['quality_flags']['baseline_10k_rows']['rows'],'RSS_KiB',ss['peak_rss_kib'])
        for scope in ['all_rows','baseline_10k_rows']:
            for f in ['weight_g','calories_kcal','dietary_fiber_g','alcohol_g']:
                z=ss['fields'][scope][f]
                print(scope,f,'NA',z['missing'],'NEG',z['negative'],'INF',z['nonfinite_nonmissing'],'ZERO',z['zero'],'MIN',round(z['minimum'],6) if z['minimum'] is not None else None,'MAX',round(z['maximum'],6) if z['maximum'] is not None else None)
        print('QUALITY',json.dumps(ss['quality_flags'],separators=(',',':')))
    print('RESULT',q)
else:
    for ss in s['source_summaries']:
        print('SOURCE',ss['source'])
        print('PORTION',json.dumps(ss['portion_consistency'],separators=(',',':')))
        for scope in ['all_rows','baseline_10k_rows']:
            for name,z in ss['energy_diagnostics'][scope].items():
                print('ENERGY',scope,name,'N',z['n'],'MAE',round(z['sum_absolute_difference']/z['n'],4) if z['n'] else None,'GT1',z['absolute_difference_gt_1kcal'],'GT10',z['absolute_difference_gt_10kcal'])
    print('RULES: event intake; multiplier=1; source energy preserved; no inferred alcohol; missing remains NA')
    print('HELD: W06 clean edits/missing/extremes; W07 NOVA mapping; fiber-energy/recipe conventions unknown')
    print('CONFIG',config)
receipt={'passed':True,'run':run,'main_actual_exit':0,'worker_actual_exit_codes':v['actual_worker_exit_codes'],'independent_examples':12,'audited_event_rows':s['total_event_rows_scanned'],'artifact_hashes_rechecked':len(v['artifacts']),'code_sha256':h(P/'scripts'/run/'05_validate_nutrients.py'),'config_sha256':h(config),'memo_sha256':h(q/'MEMO.md'),'source_fingerprints_and_dictionary_hashes_rechecked':True,'no_outcome_or_microbiome_reads':True,'participant_features_or_models_computed':False}
(q/'delivery_readback.json').write_text(json.dumps(receipt,indent=2)+'\n')
code=P/'scripts'/run/'06_verify_w05_delivery.py'
if Path(__file__).resolve()!=code.resolve():code.write_bytes(Path(__file__).read_bytes())
print('DELIVERY_RECEIPT',q/'delivery_readback.json');print('W05_READBACK_END')
