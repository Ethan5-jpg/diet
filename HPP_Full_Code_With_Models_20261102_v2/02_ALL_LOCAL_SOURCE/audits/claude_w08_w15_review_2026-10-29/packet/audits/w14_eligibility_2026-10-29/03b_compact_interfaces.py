from pathlib import Path
import json,pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');Q=P/'data_qc/w14_eligibility_20261029';J=json.loads((Q/'preflight_TRE_ONLY.json').read_text());A=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());R=json.loads(Path(A['analysis_register']).read_text())
for m in R['modules']:print('MODULE',m['analysis_id'],json.dumps({k:v for k,v in m.items() if any(t in k for t in ['outcome','phenotype','enabled'])}))
for role in ['outcomes','food_cgm_daily']:
 f=pq.ParquetFile(J['inputs'][role]['path'],pre_buffer=False);print('FIELDS',role,json.dumps(f.schema_arrow.names))
O=pq.ParquetFile(J['inputs']['outcomes']['path']);print('OUTCOME_IDS',json.dumps(sorted(set(O.read(columns=['outcome_id']).column(0).to_pylist()))))
print('COMPACT_PASS')
