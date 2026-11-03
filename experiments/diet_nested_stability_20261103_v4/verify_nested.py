"""Meaningful synthetic leakage, grouping, prediction and archive checks."""
import argparse
import copy
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pandas as pd
import run_nested as run


def verify(args):
    run.dependencies(Path(args.package), Path(args.pilot_code), Path(args.v3_code))
    rng = np.random.default_rng(91)
    n = 360
    x = pd.DataFrame(dict(participant_id=np.arange(n), split_unit_id=np.repeat(np.arange(n//2), 2),
        n_fiber_days=4, fiber_density=rng.uniform(5, 25, n), age_calendar_years=rng.uniform(30, 70, n),
        sex_code=rng.integers(0, 2, n), mean_daily_energy_fiber=rng.uniform(1500, 2500, n), smoking_status='never'))
    groups = x.split_unit_id.astype(str).to_numpy()
    outer = run.folds(groups, 3, 5)
    tr, va = outer != 0, outer == 0
    raw = {s: rng.dirichlet(np.ones(12), n) for s in ('oral', 'gut')}
    for a in raw.values():
        a[::3, 0] = 0
        a /= a.sum(axis=1, keepdims=True)
    taxa = {s: [str(i) for i in range(12)] for s in raw}
    y = 90+0.1*x.age_calendar_years.to_numpy()+rng.normal(0, 4, n)
    original = run.context(x, raw, taxa, y, tr, va)
    x2, raw2, y2 = x.copy(), copy.deepcopy(raw), y.copy()
    x2.loc[va, 'fiber_density'] *= 100
    x2.loc[va, 'age_calendar_years'] += 100
    x2.loc[va, 'smoking_status'] = 'unseen_validation_level'
    y2[va] += 1000
    for s in raw:
        raw2[s][va] = rng.dirichlet(np.ones(12)*.1, va.sum())
    # Keep some validation outcomes valid so the context's usability guard applies.
    y2[np.flatnonzero(va)[:50]] = y[np.flatnonzero(va)[:50]]
    altered = run.context(x2, raw2, taxa, y2, tr, va)
    assert pickle.dumps(original[4]) == pickle.dumps(altered[4]), 'Held-out rows affected fitted preprocessing'
    np.testing.assert_array_equal(original[2], altered[2])
    for site in run.SITES:
        np.testing.assert_array_equal(original[0][site][tr], altered[0][site][tr])
    np.testing.assert_array_equal(original[1][tr], altered[1][tr])
    for seed in (run.SEED, run.SEED+1):
        assignment = run.folds(groups, 5, seed)
        order = rng.permutation(n)
        np.testing.assert_array_equal(run.folds(groups[order], 5, seed), assignment[order])
        for f in range(5):
            train, valid = assignment != f, assignment == f
            run.separated(groups, train, valid)
            inner = run.folds(groups[train], 3, seed+f)
            for k in range(3):
                run.separated(groups[train], inner != k, inner == k)
    assert not np.array_equal(run.folds(groups, 5, run.SEED), run.folds(groups, 5, run.SEED+1))
    d, yy, train, valid, _ = original
    for algorithm in run.ALGORITHMS:
        for setting in run.shared.GRIDS[algorithm]:
            pp, fitted, diagnostic = run.fit_predict(d['oral'], yy, train, valid, algorithm, setting)
            assert len(pp) == valid.sum() and np.isfinite(pp).all()
            np.testing.assert_allclose(diagnostic['MSE'], np.mean((yy[valid]-pp)**2))
            if algorithm == 'ridge':
                # Independent closed-form standardized ridge prediction.
                a = d['oral'][train]
                mean, scale = a.mean(0), a.std(0)
                scale[scale < 1e-12] = 1
                z = (a-mean)/scale
                target = (yy[train]-fitted['y_mean'])/fitted['y_sd']
                beta = np.linalg.solve(z.T@z+setting['alpha']*np.eye(z.shape[1]), z.T@target)
                expected = ((d['oral'][valid]-mean)/scale)@beta*fitted['y_sd']+fitted['y_mean']
                np.testing.assert_allclose(pp, expected, atol=1e-8, rtol=0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)/'test.npz'
        run.shared.save_arrays(p, x=original[1])
        try:
            run.shared.save_arrays(p, x=original[1])
        except FileExistsError:
            pass
        else:
            raise AssertionError('Archive overwrite not rejected')
    assert 2*5*(3*3*sum(len(run.shared.GRIDS[a]) for a in run.ALGORITHMS)+9) == 1170
    print('SYNTHETIC_NESTED_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('package', 'pilot-code', 'v3-code'):
        parser.add_argument('--'+key, required=True)
    verify(parser.parse_args())
