"""Replayable atomic learners and strictly matched explicit interaction corrections."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from estimators import AdditiveBoost, GroupLasso, GuardedFM, LeafRules, ResidualNeighbors, RotationBoost, hgb


@dataclass
class Fitted:
    state: dict
    model: object
    mean: float
    scale: float
    blocks: bool = False

    def predict(self, role, training=False):
        x, blocks = role.replay(self.state, training)
        return self.mean + self.scale * self.model.predict(blocks if self.blocks else x)


def atomic(role, spec, case, config, threads):
    design = role.design(
        spec["site"], spec["representation"], case, config["nmf_iterations"], spec.get("joint", False)
    )
    s, family = role.split, spec["learner"]
    mean, scale = float(s.yt.mean()), float(s.yt.std())
    if scale <= 0:
        raise ValueError("Constant training target")
    y = (s.yt - mean) / scale
    seed, blocks = config["model_seed"], False
    if family == "hgb":
        model = hgb(case, seed)
    elif family in ("cat", "robust_cat"):
        model = CatBoostRegressor(
            iterations=case["iterations"],
            depth=case["depth"],
            learning_rate=0.035,
            l2_leaf_reg=case.get("l2", 30.0),
            loss_function="Huber:delta=1.0" if family == "robust_cat" else "RMSE",
            random_seed=seed,
            thread_count=threads,
            border_count=64,
            rsm=0.8,
            verbose=False,
            allow_writing_files=False,
        )
    elif family == "rules":
        model = LeafRules(case, threads, seed)
    elif family == "linear_leaf":
        model = make_pipeline(
            StandardScaler(),
            LGBMRegressor(
                linear_tree=True,
                linear_lambda=case["linear_l2"],
                n_estimators=case["iterations"],
                num_leaves=case["leaves"],
                learning_rate=0.035,
                min_child_samples=case["leaf"],
                reg_lambda=30.0,
                max_bin=63,
                verbosity=-1,
                n_jobs=threads,
                force_col_wise=True,
                deterministic=True,
                random_state=seed,
            ),
        )
    elif family == "rotation":
        model = RotationBoost(s.ht.shape[1], case, seed)
    elif family == "neighbors":
        model = ResidualNeighbors(s.ht.shape[1], case, threads)
    elif family == "fm":
        model = GuardedFM(case["rank"], case["penalty"], config["fm_iterations"], seed)
    elif family == "additive":
        model, blocks = AdditiveBoost(case, seed), True
    elif family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=len(y) * case["penalty"]))
    else:
        raise ValueError("Unknown learner: " + family)
    model.fit(design["blocks_t"] if blocks else design["xt"], y)
    note = dict(getattr(model, "diagnostic", {}))
    factor_quality = {}
    for site, saved in design["state"].items():
        if "factor" not in saved:
            continue
        factor = saved["factor"]
        quality = dict(method=factor["method"], warnings=factor["warnings"])
        if factor["method"] == "nmf":
            fitted = factor["model"]
            quality.update(iterations=int(fitted.n_iter_), converged=bool(fitted.n_iter_ < fitted.max_iter))
        factor_quality[site] = quality
    if factor_quality:
        note["factor_quality"] = factor_quality
        note["warnings"] = sorted(set(w for v in factor_quality.values() for w in v["warnings"]))
        if any(v.get("converged") is False for v in factor_quality.values()):
            note["converged"] = False
    return Fitted(design["state"], model, mean, scale, blocks), note


@dataclass
class Correction:
    base: Fitted
    weight: float
    projection: object = None
    scale: object = None
    model: object = None
    columns: object = None

    def design(self, role, training=False):
        x, blocks = role.replay(self.base.state, training)
        z = np.column_stack(blocks[1:])[:, self.columns]
        d = role.split.dt if training else role.split.dv
        product = (z[:, :, None] * d[:, None, :]).reshape(len(x), -1)
        return (product - x @ self.projection) / self.scale

    def predict(self, role, training=False):
        base = self.base.predict(role, training)
        if self.weight == 0:
            return base
        return base + self.weight * self.base.scale * self.model.predict(self.design(role, training))

    def modifier(self, role, training=False):
        s = role.split
        if self.weight == 0:
            return np.zeros(len(s.yt) if training else len(s.yv))
        x, blocks = role.replay(self.base.state, training)
        z = np.column_stack(blocks[1:])[:, self.columns]
        product_delta = (z[:, :, None] * s.delta[None, None, :]).reshape(len(x), -1)
        h = blocks[0]
        delta_h = role.host_at(s.delta, training) - role.host_at([0.0, 0.0], training)
        delta_x = np.column_stack([delta_h, np.zeros((len(x), x.shape[1] - h.shape[1]))])
        contrast = (product_delta - delta_x @ self.projection) / self.scale
        return self.weight * self.base.scale * self.model.predict(contrast)


def interaction(role, spec, case, base, config):
    if case["weight"] == 0:
        return Correction(base, 0.0), dict(converged=True, active_groups=0)
    x, blocks = role.replay(base.state, True)
    z = np.column_stack(blocks[1:])
    columns = []
    offset = 0
    for block, saved in zip(blocks[1:], base.state.values()):
        width = (
            saved["factor"]["model"].n_components if spec["learner"] == "nmf_interaction" else block.shape[1]
        )
        columns.extend(range(offset, offset + width))
        offset += block.shape[1]
    columns = np.asarray(columns, int)
    z = z[:, columns]
    raw = (z[:, :, None] * role.split.dt[:, None, :]).reshape(len(x), -1)
    projection = np.linalg.lstsq(x, raw, rcond=1e-8)[0]
    centered = raw - x @ projection
    scale = centered.std(0)
    scale[scale < 1e-10] = 1.0
    a = centered / scale
    target = (role.split.yt - base.predict(role, True)) / base.scale
    if spec["learner"] == "sparse_interaction":
        model = GroupLasso(case["fraction"], ridge=0.02, iterations=config["sparse_iterations"]).fit(
            a, target
        )
        note = model.diagnostic
    else:
        model = Ridge(alpha=len(x) * case["penalty"], fit_intercept=False).fit(a, target)
        note = dict(converged=True)
    return Correction(base, case["weight"], projection, scale, model, columns), dict(note)
