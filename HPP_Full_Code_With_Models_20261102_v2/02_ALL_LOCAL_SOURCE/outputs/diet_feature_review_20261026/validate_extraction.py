"""Cross-check OOXML extraction against an independent, read-only XLSX reader."""
import hashlib
import json
import pathlib
from collections import Counter
import openpyxl

ROOT = pathlib.Path(__file__).resolve().parent
d = json.loads((ROOT / 'feature_tables_extracted.json').read_text())
sources = d['manifest']['sources']
source_hashes_unchanged = all(hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest() == x['sha256'] for x in sources)
book = openpyxl.load_workbook(sources[0]['path'], read_only=True, data_only=False)
checks = []
for sheet in d['xlsx_sheets']:
    actual = {c.coordinate: str(c.value) for row in book[sheet['sheet']].iter_rows() for c in row if c.value is not None}
    extracted = {c['cell']: str(c['value']) for c in sheet['cells']}
    checks.append({'sheet': sheet['sheet'], 'exact_nonempty_cells_match': actual == extracted, 'nonempty_cells': len(actual)})
rows = d['csv_rows']
binary_columns = [c for c in d['csv_summary']['columns'] if c not in ('Food', 'NOVA')]
binary_domains_valid = all(r[c] in ('0', '1') for r in rows for c in binary_columns)
nova_domain_valid = all(r['NOVA'] in ('', '1', '2', '3', '4') for r in rows)
out = {'date': '2026-10-26', 'source_hashes_unchanged': source_hashes_unchanged, 'independent_reader': 'openpyxl ' + openpyxl.__version__,
       'sheet_checks': checks, 'binary_domains_valid': binary_domains_valid, 'nova_domain_valid': nova_domain_valid,
       'csv_rows': len(rows), 'csv_unique_names': len(set(r['Food'] for r in rows)),
       'nova_missing_foods': [r['Food'] for r in rows if not r['NOVA']],
       'formula_cell_count': sum(c['formula'] is not None for s in d['xlsx_sheets'] for c in s['cells']),
       'participant_data_present_in_supplied_tables': False,
       'interpretation': 'Definition/mapping consistency only; no verification of TRE intake units, mapping coverage, distributions, predictive value or phenotype associations.'}
out['passed'] = source_hashes_unchanged and all(c['exact_nonempty_cells_match'] for c in checks) and binary_domains_valid and nova_domain_valid
(ROOT / 'extraction_validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
if not out['passed']:
    raise SystemExit('Extraction check failed')
