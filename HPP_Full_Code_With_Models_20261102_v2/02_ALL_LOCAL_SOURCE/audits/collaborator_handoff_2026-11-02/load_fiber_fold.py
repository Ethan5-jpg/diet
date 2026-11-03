"""Read the existing fiber/Glu fold interface; never fit a new prediction model."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import sys, io, json
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'preprocessing' / 'code'))
from preprocessing_v1_20261030 import PhenotypeOutliers

def read(path):
    return pq.ParquetFile(path, pre_buffer=False).read().to_pandas(ignore_metadata=True)

def load_fold(fold=0):
    """Return aligned arrays and household-isolated masks for a single fold."""
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    B = ROOT / 'inputs' / 'discovery_v2'
    X = read(B / 'discovery_covariates_food_TRE_ONLY.parquet')
    assert X.final_role.eq('discovery').all() and X.participant_id.is_unique
    assert np.array_equal(X.input_row.to_numpy(), np.arange(len(X)))
    assert X.groupby('split_unit_id').cv_fold.nunique().max() == 1
    folder = ROOT / 'preprocessing' / 'folds' / ('fold' + str(fold))
    with np.load(io.BytesIO((folder / 'transformed_discovery_TRE_ONLY.npz').read_bytes()), allow_pickle=False) as f:
        a = {name: f[name] for name in f.files}
    training = a['training'].astype(bool)
    np.testing.assert_array_equal(training, X.cv_fold.ne(fold).to_numpy())
    Yd = read(B / 'D_Glu_raw_Y_TRE_ONLY.parquet')
    rows = Yd.input_row.to_numpy(int)
    assert len(set(rows)) == len(rows)
    raw_y = Yd.Y_value.to_numpy(float)
    keep = training[rows] & np.isfinite(raw_y)
    converter = PhenotypeOutliers().fit(raw_y[keep], rows[keep].tolist(), endpoint='D_Glu')
    values = converter.transform(raw_y, endpoint='D_Glu')['values']
    state = converter.state
    y = np.full(len(X), np.nan)
    y[rows] = values
    diet = json.loads((folder / 'diet_basis.json').read_text())['fiber_density']
    diet.update(u=a['fiber_u'], h=a['fiber_h'])
    E = X.mean_daily_energy_fiber.to_numpy(float)
    eligible = np.asarray(X.n_fiber_days.ge(3).to_numpy() & a['supported'] &
        np.isfinite(X.age_calendar_years.to_numpy(float)) & X.sex_code.isin([0, 1]).to_numpy() &
        np.isfinite(E) & np.isfinite(diet['u']) & np.isfinite(diet['h']) & np.isfinite(y))
    tr = training & eligible
    va = X.cv_fold.eq(fold).to_numpy() & eligible
    assert not set(X.loc[tr, 'split_unit_id']) & set(X.loc[va, 'split_unit_id'])
    result = dict(index=X, covariates=X[['age_calendar_years', 'sex_code', 'smoking_status', 'mean_daily_energy_fiber']].copy(),
        oral_z=a['oral_z'], gut_z=a['gut_z'], oral_PC=a['oral_PC'], gut_PC=a['gut_PC'],
        u=diet['u'], h=diet['h'], y=y, train=tr, valid=va, supported=a['supported'],
        Y_unit=Yd.Y_unit.dropna().unique().tolist(), phenotype_transform=state)
    assert np.isfinite(result['oral_z'][tr | va]).all()
    assert np.isfinite(result['gut_z'][tr | va]).all()
    return result

def check():
    rows = []
    seen = set()
    for fold in range(5):
        d = load_fold(fold)
        X, tr, va = d['index'], d['train'], d['valid']
        assert not seen & set(X.loc[va, 'input_row'])
        seen.update(X.loc[va, 'input_row'])
        rows.append(dict(fold=fold, training_n=int(tr.sum()), validation_n=int(va.sum()),
            training_families=int(X.loc[tr, 'split_unit_id'].nunique()),
            oral_features=d['oral_z'].shape[1], gut_features=d['gut_z'].shape[1]))
    assert len(seen) == 3343
    assert [r['validation_n'] for r in rows] == [661, 677, 668, 665, 672]
    return dict(status='PASS', python=sys.executable, model_fitted=False, total_CV_n=len(seen), folds=rows)

if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
