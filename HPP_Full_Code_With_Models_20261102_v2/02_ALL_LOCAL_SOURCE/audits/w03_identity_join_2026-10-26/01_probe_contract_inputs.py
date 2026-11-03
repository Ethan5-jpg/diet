"""W03 inspect existing manifest only; no participant rows or values printed."""
import json,re,socket,sys
from pathlib import Path
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
M=P/'manifests/w02_sources_20261026T130908Z/source_manifest.json'
v=json.loads(M.read_text())
print('\n'*30+'W03 INPUT SCHEMA PROBE',socket.gethostname(),sys.executable)
for e in v['sources']:
 if not e['required']: continue
 cols=[(f['name'],f['type']) for f in e.get('columns',[]) if re.search(r'^(participant_id|cohort|research_stage|sample_id|uuid|connection_id|collection_date|collection_timestamp|local_timestamp|logging_day|.*visit.*|.*timezone.*|.*time_zone.*|.*_id|__index_level_\d+__)$',f['name'],re.I) or re.search('date|timestamp',f['name'],re.I)]
 print(e['role'],'rows',e.get('rows_metadata'),'keys',json.dumps(cols,separators=(',',':')))
print('CGM_REPRESENTATIVES')
for e in v['cgm_timeseries']['representatives']:
 cols=[(f['name'],f['type']) for f in e.get('columns',[]) if re.search('participant|cohort|stage|uuid|connection|date|time|index',f['name'],re.I)]
 print('representative_schema',json.dumps(cols,separators=(',',':')))
print('PROBE_DONE')
