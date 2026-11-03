"""Independent numerical checks; synthetic data only, never TRE inputs."""
import json
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import tempfile
from pathlib import Path
import numpy as np
from scipy import optimize
from core import Background, rcs, save_npz


def verify():
    rng = np.random.default_rng(20261103)
    n, p = 220, 9
    z = rng.normal(size=(n, p))
    u = rng.normal(size=n)
    base = np.column_stack([np.ones(n), u, rng.normal(size=(n, 2))])
    y = base@np.array([2., .4, -.2, .5])+z[:, 0]+u*z[:, 2]*1.5+rng.normal(size=n)*.3
    training = np.arange(n) < 160
    bg = Background(base, z, y, training, .7)
    raw = u[:, None]*z
    interaction = bg.interaction(raw)
    fitted = interaction.ridge(.3)
    design = np.column_stack([base, z/bg.z_scale, raw/interaction.scale])
    penalty = np.diag([0.]*base.shape[1]+[.7]*p+[.3]*p)*training.sum()
    direct = np.linalg.solve(design[training].T@design[training]+penalty,
                             design[training].T@y[training])
    ridge_error = float(np.max(np.abs(fitted["prediction"]-design@direct)))
    np.testing.assert_allclose(fitted["prediction"], design@direct, atol=1e-10)
    null_design = design[:, :base.shape[1]+p]
    null_penalty = penalty[:null_design.shape[1], :null_design.shape[1]]
    null_coef = np.linalg.solve(null_design[training].T@null_design[training]+null_penalty,
                                null_design[training].T@y[training])
    np.testing.assert_allclose(bg.null()["prediction"], null_design@null_coef, atol=1e-10)
    # An independent constrained smooth optimization checks the nonsmooth EN solution.
    en = interaction.elastic_net(.8, .1)
    lam, alpha = en["penalty"], en["alpha"]
    gram, rhs = interaction.gram, interaction.rhs
    def objective(v):
        c = v[:p]-v[p:]
        return .5*c@gram@c-rhs@c+.5*lam*(1-alpha)*(c@c)+lam*alpha*v.sum()
    def gradient(v):
        c = v[:p]-v[p:]
        g = gram@c-rhs+lam*(1-alpha)*c
        return np.concatenate([g, -g])+lam*alpha
    reference = optimize.minimize(objective, np.zeros(2*p), jac=gradient,
                                  bounds=[(0, None)]*(2*p), method="L-BFGS-B",
                                  options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 10000})
    if not reference.success:
        raise AssertionError(reference.message)
    expected = reference.x[:p]-reference.x[p:]
    np.testing.assert_allclose(en["interaction_coef"]*interaction.scale, expected, atol=2e-6)
    en_error = float(np.max(np.abs(en["interaction_coef"]*interaction.scale-expected)))
    strong = interaction.elastic_net(.2, 1.)
    np.testing.assert_array_equal(strong["standardized_coef"], np.zeros(p))
    assert strong["penalty"] > 0
    np.testing.assert_allclose(strong["prediction"], bg.null()["prediction"], atol=1e-10)
    assert not np.any(strong["interaction_coef"])
    # Perturbing held-out outcomes cannot change training parameters.
    changed = y.copy()
    changed[~training] += 100000
    refit = Background(base, z, changed, training, .7).interaction(raw).ridge(.3)
    for name in ("base_coef", "main_coef", "interaction_coef"):
        np.testing.assert_array_equal(fitted[name], refit[name])
    reconstructed = base@fitted["base_coef"]+z@fitted["main_coef"]+raw@fitted["interaction_coef"]
    np.testing.assert_allclose(reconstructed, fitted["prediction"], atol=1e-12)
    # Duplicate unpenalized columns must preserve fitted predictions.
    duplicate = Background(np.column_stack([base, base[:, 2]]), z, y, training, .7)
    np.testing.assert_allclose(duplicate.interaction(raw).ridge(.3)["prediction"],
                               fitted["prediction"], atol=1e-10)
    # Fixed diet contrast: analytical interaction contrast equals two predictions.
    knots = [-1., 0., 1.]
    low, high = -.5, .5
    c1, c2 = rng.normal(size=(2, p))
    nonlinear = lambda d: d*(z@c1)+rcs(d, knots)*(z@c2)
    contrast = (high-low)*(z@c1)+(rcs(high, knots)-rcs(low, knots))*(z@c2)
    np.testing.assert_allclose(contrast, nonlinear(high)-nonlinear(low), atol=1e-12)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"safe.npz"
        save_npz(path, coefficients=fitted["interaction_coef"])
        try:
            save_npz(path, coefficients=fitted["interaction_coef"])
            raise AssertionError("Overwrite was not rejected")
        except FileExistsError:
            pass
    return dict(status="PASS", seed=20261103, synthetic_only=True,
                ridge_vs_joint_normal_equations_max_abs_error=ridge_error,
                elastic_net_vs_independent_constrained_optimizer_max_abs_error=en_error,
                checks=["matched_ridge_null", "joint_block_ridge_equivalence",
                        "elastic_net_independent_optimizer", "lambda_max_is_null",
                        "heldout_Y_invariance", "predictor_round_trip",
                        "rank_deficient_background", "nonlinear_contrast_identity",
                        "safe_sequential_npz_and_no_overwrite"])


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
