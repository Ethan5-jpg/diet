"""Read-only result audit plus aggregate household-fold diagnostics, within TRE."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

p = Path(sys.argv[1]).resolve()
receipt = json.loads((p/'deployment.json').read_text())
manifest = json.loads((p/'code/source_sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((p/'code'/name).read_bytes()).hexdigest() == expected, name
files = {x.name: x.read_text() for x in (p/'code').iterdir() if x.is_file() and x.suffix in ('.md','.py','.json')}
assert hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == receipt['payload_sha256']
assert (p/'logs/actual_exit_code.txt').read_text().strip() == '0'
s = json.loads((p/'results/summary.json').read_text())
assert s['status'] == 'completed_and_readback_verified'
assert not s['confirmation_Y_accessed']
assert s['n_people'] == 3343 and s['n_families'] == 3059
assert len(list((p/'results/fits_TRE_ONLY').glob('*.npz'))) == 420
held = pq.ParquetFile(p/'results/held_predictions_and_contrasts_TRE_ONLY.parquet', pre_buffer=False).read().to_pandas(ignore_metadata=True)
held = held.loc[held.Y_processed.notna()].copy()
assert len(held) == 3343 and held.split_unit_id.nunique() == 3059
rows = []
for name in ['ridge_PC10_gut', 'ridge_PC10_oral_gut', 'nonlinear_PC5_gut']:
    values = []
    for fold, d in held.groupby('cv_fold'):
        y = d.Y_processed.to_numpy(float)
        base = d.matched_null_prediction.to_numpy(float)
        pred = d[name+'_prediction'].to_numpy(float)
        delta = float((np.mean((y-base)**2)-np.mean((y-pred)**2))/np.var(y))
        values.append(dict(fold=int(fold), n=len(d), delta_R2=delta))
    rows.append(dict(model=name, positive_folds=sum(x['delta_R2']>0 for x in values), folds=values))
record = dict(status='PASS', source_payload_sha256=receipt['payload_sha256'], actual_exit_code=0,
              expected_fits=420, n_people=3343, n_families=3059, fold_diagnostics=rows,
              confirmation_Y_accessed=False,
              github_sync=dict(status='blocked', reason='GitHub integration 403 Resource not accessible by integration',
                               user_authorized=True, commit=None, repository='Ethan5-jpg/diet'),
              limitations=['same CV used to select hyperparameters', 'conditional intervals omit selection uncertainty',
                           'association heterogeneity, not confirmed individual causal benefit'])
with (p/'postrun_audit.json').open('x') as f:
    json.dump(record, f, indent=2); f.write('\n')
print('POSTRUN_AUDIT', json.dumps(record))
