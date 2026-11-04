"""Paired replay from saved discovery OOF predictions; all outputs stay in TRE."""
from pathlib import Path
import io
import json

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / 'diet_nested_stability_20261103_v4'


def record(section, message):
    # Object-backed mount: replace by a complete sequential write, no append.
    path = ROOT / 'STAGES.md'
    previous = path.read_text() if path.exists() else '# Discovery stages\n'
    path.write_text(previous + '\n## ' + section + '\n' + message + '\n')


def main():
    out = V4 / 'results'
    assert json.loads((out/'audit.json').read_text())['status'] == 'PASS'
    index = pq.ParquetFile(ROOT.parents[1]/'inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet').read(columns=['input_row','final_role','split_unit_id']).to_pandas()
    assert index.final_role.eq('discovery').all()
    np.testing.assert_array_equal(index.input_row, np.arange(len(index)))
    counts = json.loads((out/'fold_counts.json').read_text())
    diag = json.loads((out/'diagnostics.json').read_text())
    existing = json.loads((out/'comparisons.json').read_text())
    detail, messages = [], []
    with np.load(io.BytesIO((out/'TRE_ONLY/final_predictions.npz').read_bytes())) as a:
        for r in range(2):
            y, folds = a['y'][r], a['outer'][r]
            mask = np.isfinite(y)
            g = index.split_unit_id.to_numpy()
            headline = f'Round {r+1}: {mask.sum()} people, {len(set(g[mask]))} households.'
            for old, new in [('host','oral'), ('oral','oral_gut')]:
                p, q = [a['hist_boost_'+k][r] for k in (old,new)]
                for v in (p,q):
                    np.testing.assert_array_equal(np.isfinite(v), mask)
                errors = []
                for f in range(5):
                    k = mask & (folds==f)
                    c = next(z for z in counts if (z['repeat'],z['fold'])==(r,f))
                    assert c['validation_n']==k.sum() and c['validation_households']==len(set(g[k]))
                    assert not (set(g[folds!=f]) & set(g[folds==f]))
                    row = dict(repeat=r, fold=f, reference=old, model=new, n=int(k.sum()), households=len(set(g[k])))
                    for name,pred in [(old,p),(new,q)]:
                        mse = float(np.mean((pred[k]-y[k])**2))
                        d = next(d for d in diag if d['stage']=='outer' and (d['repeat'],d['fold'],d['model'])==(r,f,'hist_boost_'+name))
                        np.testing.assert_allclose(mse,d['MSE'],atol=1e-9,rtol=0)
                        row[name+'_MSE'] = mse
                    errors.append(row[old+'_MSE']-row[new+'_MSE'])
                    detail.append(row)
                m0,m1 = [np.mean((v[mask]-y[mask])**2) for v in (p,q)]
                gain=(m0-m1)/np.var(y[mask])
                d=next(z for z in existing if (z['repeat'],z['model'],z['reference'])==(r,'hist_boost_'+new,'hist_boost_'+old))
                np.testing.assert_allclose([gain,1-m1/m0], [d['delta_R2'],d['MSE_reduction']])
                headline+=f' {new} vs {old}: delta R2 {gain:+.6f}, MSE reduction {m0-m1:+.5f} ({1-m1/m0:+.3%}), {sum(e>0 for e in errors)}/5 better.'
            messages.append(headline)
    (ROOT/'paired_v4_TRE_ONLY.json').write_text(json.dumps(detail,indent=2))
    record('Existing v4 replay', '\n'.join(messages)+'\nFold detail: paired_v4_TRE_ONLY.json')
    print('\n'.join(messages))


if __name__ == '__main__':
    main()
