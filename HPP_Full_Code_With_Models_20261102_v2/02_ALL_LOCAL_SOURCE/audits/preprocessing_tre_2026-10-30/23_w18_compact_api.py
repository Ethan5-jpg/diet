from pathlib import Path
import ast,json,hashlib
import pyarrow.parquet as pq
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
a=json.loads((P/'config/ACTIVE_PROTOCOL.json').read_text());r=json.loads(Path(a['analysis_register']).read_text())
print('ACTIVE',json.dumps(a))
for m in r['modules']:print('MODULE',json.dumps(m))
manifest=json.loads((P/'data_qc/w15_split_access_20261029/final_artifact_manifest.json').read_text())
for x in manifest:
    p=Path(x['path'])
    if p.suffix=='.parquet':
        f=pq.ParquetFile(p,pre_buffer=False);print('W15_FILE',str(p),f.metadata.num_rows,','.join(f.schema_arrow.names))
    if p.name=='w15_access_guard.py':
        s=p.read_text();print('GUARD_CODE_BYTES_SHA',len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
        tree=ast.parse(s)
        for n in tree.body:
            if isinstance(n,(ast.Assign,ast.AnnAssign)):print('GUARD_ASSIGN',ast.unparse(n))
            if isinstance(n,ast.FunctionDef):print('GUARD_FUNCTION',n.name,ast.unparse(n.args))
print('W18_COMPACT_API_NO_Y_VALUES')
