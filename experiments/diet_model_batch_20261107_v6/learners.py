"""Training-only screening, boosting, supervised interaction axes and fusion."""

import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor


def screen(host, microbes, target, count):
    if count == 0 or count >= microbes.shape[1]:
        return np.arange(microbes.shape[1])
    if count < 0:
        raise ValueError("Feature count must be nonnegative")
    residual = target - host @ np.linalg.lstsq(host, target, rcond=None)[0]
    centered = microbes - microbes.mean(0)
    norm = np.sqrt(np.sum(centered**2, axis=0))
    scores = np.divide(
        abs(centered.T @ residual), norm, out=np.zeros_like(norm), where=norm > 1e-12
    )
    return np.sort(np.argsort(-scores, kind="stable")[:count])


def tree_fit(algorithm, ht, hv, zt, zv, target, case, seed, threads, keep_model):
    columns = screen(ht, zt, target, case["top_k"])
    train, valid = (
        np.column_stack([ht, zt[:, columns]]),
        np.column_stack([hv, zv[:, columns]]),
    )
    common = dict(random_state=seed)
    settings = case["settings"]
    if algorithm == "hgb":
        estimator = HistGradientBoostingRegressor(
            **settings, early_stopping=False, **common
        )
    elif algorithm == "catboost":
        from catboost import CatBoostRegressor

        estimator = CatBoostRegressor(
            **settings,
            loss_function="RMSE",
            learning_rate=0.03,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.5,
            random_seed=seed,
            thread_count=threads,
            allow_writing_files=False,
            verbose=False,
            use_best_model=False,
        )
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            **settings,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.5,
            max_bin=128,
            verbosity=0,
            n_jobs=threads,
            **common,
        )
    else:
        raise ValueError("Unknown algorithm")
    mean, scale = float(target.mean()), float(target.std())
    if scale <= 0:
        raise ValueError("Constant training target")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(train, (target - mean) / scale)
        prediction = estimator.predict(valid) * scale + mean
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite tree prediction")
    result = dict(
        prediction=prediction, warnings=[str(item.message) for item in caught]
    )
    if keep_model:
        result.update(
            model=estimator, columns=columns, target_mean=mean, target_scale=scale
        )
    return result


def supervised_basis(z, diet, residual, components):
    """Fit PLS to diet×gut columns and training null residuals, not to validation outcomes."""
    mean = z.mean(0)
    u = diet[:, 0] - diet[:, 0].mean()
    u /= max(float(u.std()), 1e-12)
    matrix = u[:, None] * (z - mean)
    count = min(components, matrix.shape[1], len(matrix) - 1)
    if np.std(residual) < 1e-12:
        return dict(
            mean=mean,
            rotation=np.zeros((z.shape[1], count)),
            scale=np.ones(count),
            warnings=["Constant null residual: zero supervised interaction basis"],
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pls = PLSRegression(n_components=count, scale=False, max_iter=500, tol=1e-7)
        pls.fit(matrix, residual)
    rotation = pls.x_rotations_
    if not np.isfinite(rotation).all():
        raise ValueError("Nonfinite supervised projection")
    scale = ((z - mean) @ rotation).std(0)
    scale[scale < 1e-12] = 1.0
    return dict(
        mean=mean,
        rotation=rotation,
        scale=scale,
        warnings=[str(item.message) for item in caught],
    )


def apply_basis(z, state):
    return ((z - state["mean"]) @ state["rotation"]) / state["scale"]


def blend_weights(predictions, target):
    """Convex fusion fitted only to outer-training, inner-OOF predictions."""
    predictions, target = np.asarray(predictions, float), np.asarray(target, float)
    if not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite fusion inputs")
    center, scale = target.mean(), max(float(target.std()), 1e-12)
    x, y = (predictions - center) / scale, (target - center) / scale
    uniform = np.full(x.shape[1], 1.0 / x.shape[1])

    def objective(weight):
        error = x @ weight - y
        return np.mean(error**2) + 0.001 * np.sum((weight - uniform) ** 2)

    def gradient(weight):
        return 2 * x.T @ (x @ weight - y) / len(y) + 0.002 * (weight - uniform)

    result = minimize(
        objective,
        uniform,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * x.shape[1],
        constraints=[
            dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones_like(w))
        ],
        options=dict(maxiter=200, ftol=1e-10),
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Fusion optimisation failed: " + result.message)
    weight = np.maximum(result.x, 0)
    return weight / weight.sum()
