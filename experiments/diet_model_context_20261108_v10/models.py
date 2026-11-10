"""Native-unit regressors and zero-preserving matched fiber corrections."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from estimators import AdditiveKernel, BlockRidge


@dataclass
class Fitted:
    state: dict
    model: object
    mean: float
    scale: float

    def predict(self, role, training=False):
        x, _ = role.replay(self.state, training)
        return self.mean + self.scale * self.model.predict(x)


def atomic(role, spec, case, config, threads):
    design = role.design(
        spec["site"], spec["representation"], case, config["completion_iterations"], spec["context"]
    )
    y, family, seed = role.split.yt, spec["learner"], config["model_seed"]
    mean, scale = float(y.mean()), float(y.std())
    if scale <= 0:
        raise ValueError("Constant training outcome")
    if family == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=case["iterations"],
            max_leaf_nodes=case["leaves"],
            min_samples_leaf=case["leaf"],
            l2_regularization=case["l2"],
            learning_rate=0.04,
            early_stopping=False,
            random_state=seed,
        )
    elif family == "cat":
        model = CatBoostRegressor(
            iterations=case["iterations"],
            depth=case["depth"],
            learning_rate=0.035,
            l2_leaf_reg=case["l2"],
            loss_function="RMSE",
            border_count=64,
            rsm=0.8,
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
    elif family in ("block", "spline"):
        model = BlockRidge(design["widths"], design["labels"], case, spline=family == "spline")
    elif family == "svr":
        model = make_pipeline(StandardScaler(), SVR(C=case["C"], epsilon=0.1, gamma="scale"))
    elif family == "extra":
        model = ExtraTreesRegressor(
            n_estimators=case["iterations"],
            min_samples_leaf=case["leaf"],
            max_features=case["features"],
            n_jobs=threads,
            random_state=seed,
        )
    elif family == "kernel":
        model = AdditiveKernel(design["widths"], case)
    else:
        raise ValueError("Unknown V10 learner")
    model.fit(design["xt"], (y - mean) / scale)
    warnings = sorted({w for s in design["state"]["microbes"] for w in s["warnings"]})
    return Fitted(design["state"], model, mean, scale), dict(warnings=warnings)


@dataclass
class Correction:
    base: Fitted
    weight: float
    modifier_state: object = None
    nonlinear: bool = False
    projection: object = None
    scale: object = None
    model: object = None

    def product(self, z, diet):
        d = diet if self.nonlinear else diet[:, :1]
        return (z[:, :, None] * d[:, None, :]).reshape(len(z), -1)

    def predict(self, role, training=False):
        value = self.base.predict(role, training)
        if self.weight == 0:
            return value
        x, _ = role.replay(self.base.state, training)
        z = role.replay_modifier(self.modifier_state, training)
        diet = role.split.dt if training else role.split.dv
        raw = self.product(z, diet) - x @ self.projection
        return value + self.base.scale * self.model.predict(raw / self.scale)

    def modifier(self, role, training=False):
        count = len(role.split.yt) if training else len(role.split.yv)
        if self.weight == 0:
            return np.zeros(count)
        x, _ = role.replay(self.base.state, training)
        z = role.replay_modifier(self.modifier_state, training)
        delta = np.broadcast_to(role.split.delta, (count, 2))
        raw = self.product(z, delta)
        dh = role.host_at(role.split.delta, training) - role.host_at([0, 0], training)
        dx = np.column_stack([dh, np.zeros((count, x.shape[1] - dh.shape[1]))])
        return self.base.scale * self.model.predict((raw - dx @ self.projection) / self.scale)


def interaction(role, spec, case, base, config):
    if case["weight"] == 0:
        return Correction(base, 0.0), dict(converged=True)
    k, source, nonlinear = case["components"], spec["source"], spec["nonlinear"]
    key = ("correction", id(base), source, k, nonlinear)
    if key not in role.cache:
        z, _, state = role.modifier_basis(source, k)
        x, _ = role.replay(base.state, True)
        fit = Correction(base, 1.0, state, nonlinear)
        raw = fit.product(z, role.split.dt)
        # Regularized residualization remains defined when the background is wide.
        projection = (
            Ridge(alpha=len(x) * config["projection_penalty"], fit_intercept=False).fit(x, raw).coef_.T
        )
        residual = raw - x @ projection
        scale = residual.std(0)
        scale[scale < 1e-10] = 1.0
        target = (role.split.yt - base.predict(role, True)) / base.scale
        role.cache[key] = (base, state, projection, scale, residual / scale, target)
    _, state, projection, scale, design, target = role.cache[key]
    model = Ridge(alpha=len(design) * case["penalty"], fit_intercept=False).fit(design, target)
    fit = Correction(base, 1.0, state, nonlinear, projection, scale, model)
    factor_warnings = [w for f in state.get("factors", []) for w in f["factor"]["warnings"]]
    return fit, dict(
        converged=True,
        warnings=sorted(set(factor_warnings)),
        source=source,
        requested_components=k,
        modifier_columns=design.shape[1] // (2 if nonlinear else 1),
        regularized_background_projection=config["projection_penalty"],
    )
