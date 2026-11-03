"""Synthetic leakage, algebra and replay checks; no participant data required."""
import json
import warnings
import numpy as np
from run_screen import GRIDS, design, estimator, pca_map, specs


def run():
    rng = np.random.default_rng(41)
    n = 220
    z = rng.normal(size=(n,70))
    tr = np.arange(n)<160
    mapped, fitted = pca_map(z,tr)
    changed = z.copy()
    changed[~tr] += 100
    mapped2, fitted2 = pca_map(changed,tr)
    np.testing.assert_allclose(fitted.components_,fitted2.components_,atol=1e-12,rtol=0)
    np.testing.assert_allclose(mapped[tr],mapped2[tr],atol=1e-12,rtol=0)
    host = np.column_stack([np.ones(n),rng.normal(size=(n,3))])
    clr, pcs = dict(oral=z,gut=z*2),dict(oral=mapped,gut=mapped*2)
    x10 = design(host,clr,pcs,'oral_gut','PCA10')
    x50 = design(host,clr,pcs,'oral_gut','PCA50')
    np.testing.assert_array_equal(x10[:,:14],x50[:,:14])
    np.testing.assert_array_equal(x10[:,14:],x50[:,54:64])
    np.testing.assert_array_equal(design(host,clr,pcs,'oral','CLR')[:,4:],z)
    y = host[:,1]+.3*z[:,0]*z[:,1]+rng.normal(size=n)*.1
    for algorithm, grid in GRIDS.items():
        for setting in grid:
            e = estimator(algorithm,setting)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                e.fit(x10[tr],y[tr])
            pp = e.predict(x10[~tr])
            assert pp.shape==(60,) and np.isfinite(pp).all()
    # Check ridge against an independent centered penalized normal-equation solution.
    e = estimator('ridge',dict(alpha=100.)).fit(x10[tr],y[tr])
    scaler = e.steps[0][1]
    scaled = scaler.transform(x10)
    yy = y[tr]-y[tr].mean()
    beta = np.linalg.solve(scaled[tr].T@scaled[tr]+100*np.eye(x10.shape[1]),scaled[tr].T@yy)
    np.testing.assert_allclose(e.predict(x10),scaled@beta+y[tr].mean(),atol=1e-10,rtol=0)
    assert len(specs())==52
    assert 5*sum(len(m['grid']) for m in specs())==910
    return dict(status='PASS',checks=['heldout_X_invariant_PCA','nested_sitewise_PCA_prefixes',
        'unchanged_CLR_columns','all_grids_finite_without_warnings','independent_ridge_solution'],
        families=52,candidate_fits=910)


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
