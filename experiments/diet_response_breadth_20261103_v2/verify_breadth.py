"""Synthetic checks without private data; run before deploying to TRE."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from run_breadth import estimator, feature_map, shifted_host, specs


def run(pilot):
    sys.path.insert(0, str(Path(pilot).resolve()))
    from core import Background, rcs
    from run_pilot import host_design
    rng = np.random.default_rng(42)
    n = 240
    x = rng.normal(size=(n, 12))
    y = .3*x[:, 0]+.5*x[:, 1]*x[:, 2]+rng.normal(size=n)*.1
    tr = np.arange(n) < 180
    checks = {}
    for m in specs():
        if m['track'] != 'prediction' or m['site'] != 'host':
            continue
        for j, setting in enumerate(m['grid']):
            est = estimator(m['algorithm'], setting, x.shape[1])
            with warnings.catch_warnings(record=True):
                est.fit(x[tr], y[tr])
            pred = np.asarray(est.predict(x[~tr])).ravel()
            assert pred.shape == (60,) and np.isfinite(pred).all()
        checks[m['algorithm']] = 'finite_fit_predict'
    for kind in ('spline_PC10', 'rbf_PC10', 'rbf_CLR'):
        phi, _ = feature_map(x, tr, kind)
        changed = x.copy()
        changed[~tr] += 100
        phi_changed, _ = feature_map(changed, tr, kind)
        np.testing.assert_allclose(phi[tr], phi_changed[tr], atol=1e-12)
        np.testing.assert_allclose(phi[tr].mean(0), 0, atol=1e-10)
        np.testing.assert_allclose(phi[tr].std(0), 1, atol=1e-10)
        checks[kind] = 'heldout_X_does_not_change_training_map'
    index = pd.DataFrame(dict(age_calendar_years=rng.uniform(20, 70, n),
        sex_code=rng.integers(0, 2, n), mean_daily_energy_fiber=rng.uniform(1400, 2800, n),
        smoking_status=rng.choice(['never', 'current'], n)))
    u = rng.normal(size=n)
    h = rcs(u, [-1., 0., 1.])
    host, encoding = host_design(index, tr, u, h)
    shifted = shifted_host(host, encoding, index, .3, float(rcs(.3, [-1., 0., 1.])))
    labels = encoding['labels']
    for j, label in enumerate(labels):
        if not label.startswith('diet_'):
            np.testing.assert_array_equal(shifted[:, j], host[:, j])
    np.testing.assert_allclose(shifted[:, labels.index('diet_linear')], .3)
    checks['contrast'] = 'same_host_and_microbiome_only_diet_changes'
    # Independently solve the expanded penalized normal equations used by VC.
    phi, _ = feature_map(x[:, :3], tr, 'rbf_PC10')
    base = np.column_stack([np.ones(n), u])
    main = np.column_stack([x[:, :3], phi])
    interaction = u[:, None]*phi
    yy = 1+u+2*u*np.sin(x[:, 0])+rng.normal(size=n)*.1
    bg = Background(base, main, yy, tr, 1.)
    block = bg.interaction(interaction)
    fit = block.ridge(.1)
    design = np.column_stack([base, main/bg.z_scale, interaction/block.scale])
    penalty = np.r_[np.zeros(2), np.ones(main.shape[1]), np.full(phi.shape[1], .1)]
    solution = np.linalg.solve(design[tr].T@design[tr]/tr.sum()+np.diag(penalty),
                               design[tr].T@yy[tr]/tr.sum())
    np.testing.assert_allclose(fit['prediction'], design@solution, rtol=0, atol=2e-9)
    yy2 = yy.copy()
    yy2[~tr] += 500
    fit2 = Background(base, main, yy2, tr, 1.).interaction(interaction).ridge(.1)
    np.testing.assert_allclose(fit['prediction'], fit2['prediction'], atol=1e-12)
    checks['varying_coefficient'] = 'joint_solution_and_heldout_Y_invariance'
    total = sum(len(m['grid']) for m in specs())*5
    assert len(specs()) == 54 and total == 630
    return dict(status='PASS', synthetic_only=True, families=len(specs()), candidate_fits=total,
                checks=checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-code', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pilot_code), indent=2))
