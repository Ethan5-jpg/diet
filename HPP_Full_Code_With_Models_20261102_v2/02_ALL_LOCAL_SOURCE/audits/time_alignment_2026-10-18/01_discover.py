"""Read-only bounded HPP module discovery; no participant data are printed."""
from pathlib import Path
import os, sys, json, datetime, hashlib
import pyarrow.parquet as pq

ROOT = Path('/home/ec2-user/studies/hpp_datasets')
PROJECT = Path('/home/ec2-user/studies/diet_gut_cgm')
OUT = PROJECT / ('time_alignment_audit_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
storage_error = None
try:
    OUT.mkdir(exist_ok=False)
except PermissionError as exc:
    storage_error = repr(exc)
    OUT = Path('/home/ec2-user') / ('hpp_' + OUT.name)
    OUT.mkdir(exist_ok=False)
    print('CLOUD_WRITE_BLOCKED', storage_error, '\nTEMPORARY_TRE_WORKDIR', OUT, flush=True)
source = Path(__file__).read_bytes()
(OUT / '01_discover.py').write_bytes(source)
records = []
for module in ['diet_logging', 'cgm', 'oral_microbiome', 'gut_microbiome']:
    base = ROOT / module
    print('\nMODULE', base, flush=True)
    for entry in sorted(base.iterdir()):
        print('ENTRY', entry.name, 'DIR' if entry.is_dir() else entry.stat().st_size)
        candidates = [entry] if entry.is_file() else list(sorted(entry.iterdir()))[:40]
        for p in candidates:
            rec = dict(module=module, path=str(p), is_dir=p.is_dir())
            if p.is_file():
                rec['bytes'] = p.stat().st_size
                if p.suffix == '.parquet':
                    try:
                        f = pq.ParquetFile(p, pre_buffer=False)
                        rec.update(rows=f.metadata.num_rows, row_groups=f.metadata.num_row_groups,
                                   schema=[dict(name=x.name, type=str(x.type)) for x in f.schema_arrow])
                        print('PARQUET', p.relative_to(ROOT), 'ROWS', rec['rows'], 'COLS', len(rec['schema']))
                        print('KEY_TIME_FIELDS', [(x['name'],x['type']) for x in rec['schema'] if any(t in x['name'].lower() for t in ['participant','cohort','stage','date','time','sample','connection','source','day'])])
                    except Exception as e:
                        rec['error'] = repr(e)
                        print('ERROR', p.name, repr(e))
                elif entry.is_dir():
                    print('CHILD', p.relative_to(ROOT), rec['bytes'])
            records.append(rec)
manifest = dict(status='DISCOVERY_COMPLETE_BOUNDED', python=sys.executable,
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root=str(ROOT),
                output=str(OUT), source_sha256=hashlib.sha256(source).hexdigest(),
                cloud_write_error=storage_error, intended_project=str(PROJECT),
                scope='Module top level plus first 40 children per directory. Metadata only, no data rows.',
                records=records)
(OUT / 'discovery.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
(OUT / 'MEMO.md').write_text('# Time alignment audit memo\n\n01_discover.py: bounded read-only module discovery and Parquet schema/row-count inspection.\nSource and metadata are saved in this directory. No abundance matrices or participant rows were read.\nNext: select source tables and audit actual dates with explicit visit/connection rules.\n')
(OUT / '01_exit_code').write_text('0\n')
print('\nAUDIT_DIR', OUT, '\nDISCOVERY_EXIT_CODE 0', flush=True)
