import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');R=Path('/home/ec2-user/studies/hpp_datasets/cgm');Q=P/'data_qc/w09_cgm_point_qc_20261028'
s=json.loads((Q/'source_contract_TRE_ONLY.json').read_text())
for m in s['source_metadata']:print('TABLE',Path(m['file']).name,m['rows'],m['schema'].split('-- schema metadata --')[0])
for item in s['dictionaries']:
 for r in item['rows']:
  if r.get('tabular_field_name') in ['glucose','cgm_device_type','timezone','cgm_time_series','connection_id']:print('UNIT_REF',json.dumps(r))
d=pq.read_table(R/'cgm.parquet').to_pandas();print('INDEX_COLUMNS',list(d.columns),'INDEX_NAMES',d.index.names);d=d.reset_index()
for c in ['cgm_device_type','timezone','research_stage','cohort']:print('VALUES',c,d[c].value_counts(dropna=False).to_dict() if c in d else 'absent')
for c in ['participant_id','connection_id','cgm_time_series']:print('KEY',c,'missing',int(d[c].isna().sum()),'nunique',d[c].nunique())
print('MULTI_VISITS',int(d.groupby(['participant_id','cohort','research_stage']).size().gt(1).sum()))
print('PROBE_EXIT',(Path('/home/ec2-user/multisystem_runtime_20261021/w09_cgm_point_qc_20261028/01_actual_exit_code')).read_text());print('CONTRACT_2_DONE')
