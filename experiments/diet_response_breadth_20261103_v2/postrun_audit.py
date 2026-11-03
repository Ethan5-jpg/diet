"""Independent readback of prediction artifacts; only aggregate output is printed."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd


def run(root):
    root = Path(root)
    results = root/'results'
    assert (root/'logs/actual_exit_code.txt').read_text().strip() == '0'
    for name, sha in json.loads((root/'source_sha256.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
    summary = json.loads((results/'summary.json').read_text())
    registration = json.loads((results/'registration.json').read_text())
    diagnostics = json.loads((results/'diagnostics.json').read_text())
    scores = json.loads((results/'grid_scores.json').read_text())
    held = pd.read_parquet(io.BytesIO((results/'predictions_TRE_ONLY.parquet').read_bytes()))
    with np.load(io.BytesIO((results/'all_candidate_predictions_TRE_ONLY.npz').read_bytes()), allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    y, baseline, folds = arrays.pop('y'), arrays.pop('baseline'), arrays.pop('folds')
    common = np.isfinite(y)
    assert common.sum() == 3343 and held.split_unit_id.nunique() == 3059
    assert held.groupby('split_unit_id').cv_fold.nunique().max() == 1
    np.testing.assert_allclose(y[common], held.Y_processed, rtol=0, atol=0)
    variance = np.var(y[common])
    null_r2 = 1-np.mean((y[common]-baseline[common])**2)/variance
    assert abs(null_r2-summary['baseline']['R2']) < 1e-12
    assert len(diagnostics) == 630 and len(arrays) == 54
    assert len(list((results/'fits_TRE_ONLY').glob('*.pkl'))) == 495
    assert len(list((results/'fits_TRE_ONLY').glob('*.npz'))) == 180
    for model in registration['models']:
        name = model['name']
        pred = arrays[name]
        assert pred.shape == (4427, len(model['grid']))
        assert np.isfinite(pred[common]).all()
        mse = np.array([[np.mean((y[folds==k]-pred[folds==k,j])**2) for k in range(5)]
                        for j in range(len(model['grid']))])
        chosen = mse.mean(1).argmin()
        for j in range(len(model['grid'])):
            saved = next(s for s in scores if s['model']==name and s['grid']==j)
            np.testing.assert_allclose(mse[j], saved['fold_MSE'], rtol=0, atol=1e-10)
        selected = next(s for s in summary['models'] if s['model']==name)
        assert selected['selected_setting'] == model['grid'][chosen]
        np.testing.assert_allclose(pred[common, chosen], held[name+'_prediction'], rtol=0, atol=0)
        r2 = 1-np.mean((y[common]-pred[common,chosen])**2)/variance
        assert abs(r2-selected['R2']) < 1e-12
        if model['track'] == 'varying':
            mr2 = 1-np.mean((y[common]-held[name+'_matched_null'].to_numpy())**2)/variance
            assert abs(r2-mr2-selected['interaction_delta_R2']) < 1e-12
    warning_fits = [x for x in diagnostics if x['warnings']]
    table = pd.DataFrame(summary['models'])
    best = table[table.track.eq('prediction')].sort_values('R2', ascending=False)
    best['algorithm'] = best.model.str.replace(r'_(PC10|CLR)_(host|oral|gut|oral_gut)$', '', regex=True)
    best = best.drop_duplicates('algorithm')
    receipt = dict(status='PASS', n_people=3343, n_households=3059,
        families=54, candidate_fits=630, source_hashes_verified=True,
        all_candidate_losses_recomputed=True, all_selected_predictions_recomputed=True,
        fits_with_warnings=len(warning_fits), warning_models=sorted(set(x['model'] for x in warning_fits)),
        elapsed_seconds=summary['elapsed_seconds'], baseline_R2=null_r2,
        best_per_prediction_algorithm=best.to_dict(orient='records'),
        varying_models=table[table.track.eq('varying')].sort_values('R2',ascending=False).to_dict(orient='records'))
    # pandas records have NaN for columns not applicable to a track.
    receipt = json.loads(json.dumps(receipt), parse_constant=lambda _: None)
    with (root/'postrun_audit.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('BREADTH AUDIT PASS | 54 families | 630 fits | 3343 people | 3059 households')
    print('SOURCE HASHES PASS | ALL PREDICTION SCORES RECOMPUTED | EXIT 0')
    print('BASELINE R2',round(null_r2,6),'WARNING FITS',len(warning_fits),'SECONDS',round(summary['elapsed_seconds']))
    print('BEST PER PREDICTION ALGORITHM:')
    print(best[['model','R2','delta_R2_vs_v1_null','folds_improved_vs_v1_null']].to_string(index=False, float_format=lambda x:f'{x:.6f}'))
    print('EXPLICIT VARYING COEFFICIENT MODELS:')
    print(table[table.track.eq('varying')].sort_values('R2',ascending=False)[['model','R2','interaction_delta_R2','contrast_SD']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('WARNINGS:',json.dumps(receipt['warning_models']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment')
    run(parser.parse_args().experiment)
