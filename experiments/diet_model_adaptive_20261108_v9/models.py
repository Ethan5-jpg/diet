"""Atomic learners, native-unit target inversion and matched interaction corrections."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

from estimators import DeepRVFL, ResidualMLP


@dataclass
class Fitted:
    state: dict
    model: object
    mean: float
    scale: float
    transformer: object = None

    def predict(self, role, training=False):
        x, _ = role.replay(self.state, training)
        prediction = self.mean + self.scale * self.model.predict(x)
        if self.transformer is not None:
            prediction = self.transformer.inverse_transform(prediction[:, None])[:, 0]
        return prediction


def hgb(case, seed):
    return HistGradientBoostingRegressor(
        max_iter=case["iterations"],
        max_leaf_nodes=case["leaves"],
        min_samples_leaf=case.get("leaf", 40),
        l2_regularization=case.get("l2", 30.0),
        learning_rate=0.04,
        early_stopping=False,
        random_state=seed,
    )


def atomic(role, spec, case, config, threads):
    design = role.design(
        spec["site"], spec["representation"], case, config["completion_iterations"], spec.get("daily", False)
    )
    y, family, seed = role.split.yt, spec["learner"], config["model_seed"]
    mean, scale, transform = float(y.mean()), float(y.std()), None
    if scale <= 0:
        raise ValueError("Constant training target")
    target = (y - mean) / scale
    if family == "gauss_cat":
        transform = QuantileTransformer(
            n_quantiles=min(512, len(y)),
            output_distribution="normal",
            subsample=max(10000, len(y)),
            random_state=seed,
        ).fit(y[:, None])
        target, mean, scale = transform.transform(y[:, None])[:, 0], 0.0, 1.0
    if family == "tweedie":
        if (y <= 0).any():
            raise ValueError("Tweedie glucose fitting requires positive native outcomes")
        mean, scale = 0.0, float(y.mean())
        target = y / scale
    if family == "hgb":
        model = hgb(case, seed)
    elif family in ("cat", "ordered_cat", "leafwise_cat", "gauss_cat"):
        options = dict(
            iterations=case["iterations"],
            depth=case["depth"],
            learning_rate=0.035,
            l2_leaf_reg=case.get("l2", 30.0),
            loss_function="RMSE",
            border_count=64,
            rsm=0.8,
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
        if family == "ordered_cat":
            options["boosting_type"] = "Ordered"
        elif family == "leafwise_cat":
            options.update(
                grow_policy="Lossguide",
                max_leaves=case["leaves"],
                min_data_in_leaf=40,
                score_function="L2",
                bootstrap_type="Bernoulli",
                subsample=0.8,
            )
        model = CatBoostRegressor(**options)
    elif family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=len(y) * case["penalty"]))
    elif family == "enet":
        model = make_pipeline(
            StandardScaler(), ElasticNet(alpha=case["penalty"], l1_ratio=0.8, max_iter=6000, tol=1e-5)
        )
    elif family == "svr":
        model = make_pipeline(StandardScaler(), SVR(C=case["C"], epsilon=0.1, gamma="scale"))
    elif family == "tweedie":
        model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=case["power"],
            n_estimators=case["iterations"],
            num_leaves=case["leaves"],
            min_child_samples=40,
            learning_rate=0.035,
            reg_lambda=30.0,
            max_bin=63,
            max_delta_step=0.5,
            verbosity=-1,
            n_jobs=threads,
            deterministic=True,
            force_col_wise=True,
            random_state=seed,
        )
    elif family == "residual_mlp":
        model = ResidualMLP(case, config["neural_seeds"], seed)
    elif family == "deep_rvfl":
        model = DeepRVFL(case, seed)
    else:
        raise ValueError("Unknown V9 learner")
    model.fit(design["xt"], target)
    note = dict(getattr(model, "diagnostic", {}))
    state = design["state"]
    states = state["blocks"] if state.get("format") == "v9" else list(state.values())
    note["warnings"] = sorted(set(w for saved in states for w in saved.get("warnings", [])))
    return Fitted(state, model, mean, scale, transform), note


@dataclass
class Correction:
    base: Fitted
    weight: float
    projection: object = None
    scale: object = None
    model: object = None

    def design(self, role, training=False):
        x, blocks = role.replay(self.base.state, training)
        z = np.column_stack(blocks[1:])
        d = role.split.dt if training else role.split.dv
        product = (z[:, :, None] * d[:, None, :]).reshape(len(x), -1)
        return (product - x @ self.projection) / self.scale

    def predict(self, role, training=False):
        value = self.base.predict(role, training)
        if self.weight == 0:
            return value
        return value + self.base.scale * self.model.predict(self.design(role, training))

    def modifier(self, role, training=False):
        s = role.split
        if self.weight == 0:
            return np.zeros(len(s.yt) if training else len(s.yv))
        x, blocks = role.replay(self.base.state, training)
        z = np.column_stack(blocks[1:])
        raw = (z[:, :, None] * s.delta[None, None, :]).reshape(len(x), -1)
        dh = role.host_at(s.delta, training) - role.host_at([0, 0], training)
        dx = np.column_stack([dh, np.zeros((len(x), x.shape[1] - dh.shape[1]))])
        return self.base.scale * self.model.predict((raw - dx @ self.projection) / self.scale)


def interaction(role, case, base):
    if case["weight"] == 0:
        return Correction(base, 0.0), dict(converged=True)
    x, blocks = role.replay(base.state, True)
    z = np.column_stack(blocks[1:])
    raw = (z[:, :, None] * role.split.dt[:, None, :]).reshape(len(x), -1)
    projection = np.linalg.lstsq(x, raw, rcond=1e-8)[0]
    residual = raw - x @ projection
    scale = residual.std(0)
    scale[scale < 1e-10] = 1.0
    target = (role.split.yt - base.predict(role, True)) / base.scale
    model = Ridge(alpha=len(x) * case["penalty"], fit_intercept=False).fit(residual / scale, target)
    return Correction(base, 1.0, projection, scale, model), dict(converged=True)
