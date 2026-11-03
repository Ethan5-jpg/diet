from pathlib import Path
import json,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
a=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());r=json.loads(Path(a['analysis_register']).read_text())
print('REGISTER_KEYS',list(r))
for m in r['modules']:
    if m['analysis_id'].startswith('D_') or m['analysis_id']=='Daily_CGM':
        print('MODULE_FIELDS',m['analysis_id'],json.dumps({k:v for k,v in m.items() if k in ['outcome_id','outcome_ids','outcome_field','unit','minimum_valid_days_candidate','diet_view','base_eligibility_id','required_sources','output_count','execution_status']}))
print('SHARED_FILENAMES',','.join(p.name for p in sorted((P/'analysis/shared').iterdir()) if p.is_file()))
for p in sorted((P/'analysis/shared').glob('*.parquet')):
    if any(t in p.name for t in ['candidate','sample_registry']):
        f=pq.ParquetFile(p,pre_buffer=False)
        print('SOURCE_COLUMNS',p.name,f.metadata.num_rows,','.join(f.schema_arrow.names))
for p in sorted(Q.glob('food_qc_v1/*windows*.parquet')):
    f=pq.ParquetFile(p,pre_buffer=False);print('NEW_FOOD_WINDOW_COLUMNS',p.name,','.join(f.schema_arrow.names))
for p in sorted((P/'config').glob('*w13*.json')):
    v=json.loads(p.read_text());print('W13_CONFIG_KEYS',p.name,list(v))
for root in [P/'data_qc/w14_eligibility_20261029',P/'data_qc/w13_microbiome_qc_20261029']:
    print('ARTIFACT_DIRECTORY',str(root),[p.name for p in sorted(root.iterdir()) if p.is_file()] if root.exists() else 'not_found')
groups={}
for run in ['w13_microbiome_qc_20261029','w14_eligibility_20261029']:
    manifest=json.loads((P/'data_qc'/run/'final_artifact_manifest.json').read_text())
    for item in manifest:
        p=Path(item['path'])
        if p.suffix!='.parquet':continue
        f=pq.ParquetFile(p,pre_buffer=False);columns=tuple(f.schema_arrow.names)
        groups.setdefault(columns,[]).append(str(p.relative_to(P)))
for columns,files in groups.items():print('PUBLISHED_PARQUET_SCHEMA_GROUP',json.dumps({'files':files,'columns':columns}))
print('SOURCE_CATALOG_COMPLETE_NO_Y_VALUES')
