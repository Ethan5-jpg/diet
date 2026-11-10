"""Native-unit regressors, grouped residual prediction and fixed fiber contrasts."""

import copy
from dataclasses import dataclass
import warnings

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from estimators import AdaptiveENet, BlockRidge


def warning_text(values):
    return [str(v.message).encode("ascii", "backslashreplace").decode() for v in values]


def make_model(spec, case, widths, labels, n, seed, threads):
    family = spec["learner"]
    if family == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=case["iterations"],
            max_leaf_nodes=case["leaves"],
            min_samples_leaf=case["leaf"],
            l2_regularization=case["l2"],
            learning_rate=0.04,
            early_stopping=False,
            random_state=seed,
        )
    if family in ("cat", "posterior_cat", "huber_cat"):
        params = dict(
            iterations=case["iterations"],
            depth=case["depth"],
            learning_rate=0.035,
            l2_leaf_reg=case["l2"],
            loss_function="Huber:delta=1.0" if family == "huber_cat" else "RMSE",
            border_count=64,
            rsm=0.8,
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
        if family == "posterior_cat":
            params["posterior_sampling"] = True
        return CatBoostRegressor(**params)
    if family in ("linear_tree", "dart"):
        params = dict(
            n_estimators=case["iterations"],
            num_leaves=case["leaves"],
            min_child_samples=case["leaf"],
            objective="regression",
            learning_rate=0.035,
            reg_lambda=30.0,
            max_bin=63,
            verbosity=-1,
            n_jobs=threads,
            deterministic=True,
            force_col_wise=True,
            random_state=seed,
        )
        if family == "linear_tree":
            params.update(linear_tree=True, linear_lambda=case["linear_lambda"])
        else:
            params.update(boosting_type="dart", drop_rate=0.1, skip_drop=0.5, drop_seed=seed)
        return make_pipeline(StandardScaler(), LGBMRegressor(**params))
    if family == "block":
        return BlockRidge(widths, labels, case)
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=n * case["penalty"]))
    if family == "bayes":
        return make_pipeline(StandardScaler(), BayesianRidge(max_iter=500, tol=1e-5))
    if family == "ard":
        return make_pipeline(
            StandardScaler(), ARDRegression(max_iter=500, tol=1e-4, threshold_lambda=case["threshold"])
        )
    if family == "adaptive_enet":
        return AdaptiveENet(case)
    raise ValueError("Unknown V11 learner: " + family)


@dataclass
class Fitted:
    state: dict
    model: object
    mean: float
    scale: float
    spec: dict
    case: dict
    widths: list
    labels: list
    seed: int

    def predict(self, role, training=False):
        x, _ = role.replay(self.state, training)
        return self.mean + self.scale * self.model.predict(x)


def fit_standardized(model, x, y):
    mean, scale = float(y.mean()), float(y.std())
    if scale < 1e-12:
        model = DummyRegressor(strategy="constant", constant=0.0).fit(x, np.zeros(len(y)))
        return model, mean, 1.0
    model.fit(x, (y - mean) / scale)
    return model, mean, scale


def atomic(role, spec, case, config, threads, target=None):
    design = role.design(
        spec["site"], spec["representation"], case, config["completion_iterations"], spec["context"]
    )
    y = role.split.yt if target is None else target
    seed = config["anchor_seed"] if spec.get("anchor") else config["model_seed"]
    model = make_model(spec, case, design["widths"], design["labels"], len(y), seed, threads)
    model, mean, scale = fit_standardized(model, design["xt"], y)
    minimal = {k: v for k, v in spec.items() if k not in ("grid",)}
    fit = Fitted(
        design["state"], model, mean, scale, minimal, dict(case), design["widths"], design["labels"], seed
    )
    notes = sorted({w for s in design["state"]["microbes"] for w in s["warnings"]})
    last = model[-1] if hasattr(model, "steps") else model
    if spec["learner"] in ("ard", "bayes") and getattr(last, "n_iter_", 0) >= 500:
        notes.append("Bayesian iteration limit reached")
    return fit, dict(warnings=notes)


def crossfit_background(role, base, config, threads):
    """Grouped CF conditional on this role's frozen design; not an extra validation score."""
    key = ("crossfit", id(base))
    if key not in role.cache:
        x, _ = role.replay(base.state, True)
        y = role.split.yt
        groups = role.index.iloc[role.split.train_rows].split_unit_id.astype(str).to_numpy()
        prediction = np.full(len(y), np.nan)
        notes = []
        for train, valid in GroupKFold(config["crossfit_folds"]).split(x, y, groups):
            if set(groups[train]) & set(groups[valid]):
                raise ValueError("Crossfit household leakage")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = make_model(
                    base.spec, base.case, base.widths, base.labels, len(train), base.seed, threads
                )
                model, mean, scale = fit_standardized(model, x[train], y[train])
                prediction[valid] = mean + scale * model.predict(x[valid])
            notes += warning_text(caught)
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete grouped residual targets")
        role.cache[key] = (base, prediction, sorted(set(notes)))
    return role.cache[key][1:]


@dataclass
class ResidualFit:
    base: Fitted
    weight: float
    stage: object = None

    def predict(self, role, training=False):
        value = self.base.predict(role, training)
        return value if self.weight == 0 else value + self.weight * self.stage.predict(role, training)


def residual(role, spec, case, base, config, threads):
    if case["weight"] == 0:
        return ResidualFit(base, 0.0), dict(warnings=[])
    stage_case = {k: v for k, v in case.items() if k != "weight"}
    key = ("residual_stage", id(base), spec["name"], tuple(sorted(stage_case.items())))
    if key not in role.cache:
        prediction, notes = crossfit_background(role, base, config, threads)
        stage_role = copy.copy(role)
        stage_role.split = copy.copy(role.split)
        stage_role.split.yt = role.split.yt - prediction
        stage_role.cache = {}
        fit, note = atomic(stage_role, spec, stage_case, config, threads)
        role.cache[key] = (base, fit, dict(warnings=sorted(set(notes + note.get("warnings", [])))))
    _, fit, note = role.cache[key]
    return ResidualFit(base, case["weight"], fit), note


@dataclass
class FiberFit:
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
        return value + self.base.scale * self.model.predict(
            (self.product(z, diet) - x @ self.projection) / self.scale
        )

    def modifier(self, role, training=False):
        n = len(role.split.yt) if training else len(role.split.yv)
        if self.weight == 0:
            return np.zeros(n)
        x, _ = role.replay(self.base.state, training)
        z = role.replay_modifier(self.modifier_state, training)
        raw = self.product(z, np.broadcast_to(role.split.delta, (n, 2)))
        dh = role.host_at(role.split.delta, training) - role.host_at([0, 0], training)
        dx = np.column_stack([dh, np.zeros((n, x.shape[1] - dh.shape[1]))])
        return self.base.scale * self.model.predict((raw - dx @ self.projection) / self.scale)


def interaction(role, spec, case, base, config, threads):
    if case["weight"] == 0:
        return FiberFit(base, 0.0), dict(warnings=[])
    key = ("fiber_stage", id(base), spec["source"], case["components"], spec["nonlinear"])
    if key not in role.cache:
        z, _, state = role.modifier_basis(spec["source"], case["components"])
        x, _ = role.replay(base.state, True)
        raw = FiberFit(base, 1.0, state, spec["nonlinear"]).product(z, role.split.dt)
        projection = (
            Ridge(alpha=len(x) * config["projection_penalty"], fit_intercept=False).fit(x, raw).coef_.T
        )
        values = raw - x @ projection
        scale = values.std(0)
        scale[scale < 1e-10] = 1.0
        oof, notes = crossfit_background(role, base, config, threads)
        notes = sorted(
            set(notes + [w for factor in state.get("factors", []) for w in factor["factor"]["warnings"]])
        )
        role.cache[key] = (
            base,
            state,
            projection,
            scale,
            values / scale,
            (role.split.yt - oof) / base.scale,
            notes,
        )
    _, state, projection, scale, design, target, notes = role.cache[key]
    model = Ridge(alpha=len(design) * case["penalty"], fit_intercept=False).fit(design, target)
    return FiberFit(base, 1.0, state, spec["nonlinear"], projection, scale, model), dict(
        warnings=notes,
        source=spec["source"],
        requested_components=case["components"],
        modifier_columns=design.shape[1] // (2 if spec["nonlinear"] else 1),
    )
