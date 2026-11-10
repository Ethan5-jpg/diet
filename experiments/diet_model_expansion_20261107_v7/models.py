"""Shared prediction wrappers, new learners and matched interaction extensions."""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomTreesEmbedding
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from estimators import PLSSVR, AdditiveBoost, Factorization, MixtureExperts, ResidualTrees
from features import selected


@dataclass
class Fitted:
    kind: str
    model: object
    design_state: dict
    mean: float
    scale: float
    extra: dict = field(default_factory=dict)

    def predict(self, role, training=False):
        x, blocks = role.replay(self.design_state, training)
        if self.kind == "additive":
            value = self.model.predict(blocks)
        elif self.kind == "mixture":
            z = role.view("gut", "clr")[0 if training else 1]
            g = self.extra["pc"].transform(z) / self.extra["pc_scale"]
            value = self.model.predict(x, g)
        else:
            value = self.model.predict(x)
        return np.asarray(value).reshape(-1) * self.scale + self.mean


def atomic(role, spec, case, config, threads):
    s, family = role.split, spec["family"]
    representation = {
        "hgb_hellinger": "hellinger",
        "nystrom": "hellinger",
        "hgb_presence": "presence",
    }.get(family, "clr")
    hybrid = family in ("additive", "mixture")
    design = role.design(spec["site"], representation, case.get("count", 64), hybrid)
    x = design.xt
    h = s.ht.shape[1]
    mean, scale = float(s.yt.mean()), float(s.yt.std())
    if scale <= 0:
        raise ValueError("Constant training target")
    y = (s.yt - mean) / scale
    seed = config["model_seed"]
    extra = {}
    if family == "lgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=case["iterations"],
            learning_rate=0.03,
            num_leaves=case["leaves"],
            min_child_samples=30,
            reg_lambda=case["l2"],
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            n_jobs=threads,
            random_state=seed,
        )
    elif family == "ebm":
        from interpret.glassbox import ExplainableBoostingRegressor

        model = ExplainableBoostingRegressor(
            interactions=0,
            max_bins=64,
            max_interaction_bins=16,
            max_rounds=case["iterations"],
            learning_rate=0.03,
            min_samples_leaf=30,
            validation_size=0,
            outer_bags=1,
            inner_bags=0,
            early_stopping_rounds=0,
            n_jobs=threads,
            random_state=seed,
        )
    elif family in ("hgb_hellinger", "hgb_presence"):
        model = HistGradientBoostingRegressor(
            max_iter=case["iterations"],
            max_leaf_nodes=case["leaves"],
            l2_regularization=30.0,
            learning_rate=0.04,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=seed,
        )
    elif family == "gam":
        mapping = ColumnTransformer(
            [
                (
                    "microbes",
                    SplineTransformer(
                        n_knots=4,
                        degree=3,
                        knots="quantile",
                        include_bias=False,
                        extrapolation="constant",
                    ),
                    list(range(h, x.shape[1])),
                ),
                ("host", "passthrough", list(range(h))),
            ]
        )
        model = make_pipeline(mapping, StandardScaler(), Ridge(alpha=len(x) * case["penalty"]))
    elif family == "tree_embed":
        mapping = ColumnTransformer(
            [
                (
                    "microbes",
                    RandomTreesEmbedding(
                        n_estimators=64,
                        max_depth=case["depth"],
                        min_samples_leaf=20,
                        n_jobs=threads,
                        random_state=seed,
                    ),
                    list(range(h, x.shape[1])),
                ),
                ("host", "passthrough", list(range(h))),
            ],
            sparse_threshold=1.0,
        )
        model = make_pipeline(mapping, Ridge(alpha=len(x) * case["penalty"], solver="lsqr"))
    elif family == "pls_svr":
        model = PLSSVR(h, case["components"], case["C"], seed)
    elif family == "factorization":
        model = Factorization(case["rank"], case["penalty"], config["fm_iterations"], seed)
    elif family == "host_residual":
        model = ResidualTrees(h, case["leaf"], config["residual_trees"], threads, seed)
    elif family == "nystrom":
        sample = x[: min(len(x), 256), h:]
        distances = pdist(sample, "sqeuclidean")
        positive = distances[distances > 1e-12]
        gamma = 1.0 / float(np.median(positive)) if len(positive) else 1.0
        mapping = ColumnTransformer(
            [
                (
                    "microbes",
                    Nystroem(
                        kernel="rbf",
                        gamma=gamma,
                        n_components=min(case["landmarks"], len(x)),
                        random_state=seed,
                        n_jobs=threads,
                    ),
                    list(range(h, x.shape[1])),
                ),
                ("host", "passthrough", list(range(h))),
            ]
        )
        model = make_pipeline(mapping, StandardScaler(), Ridge(alpha=len(x) * case["penalty"]))
    elif family == "mixture":
        g, _, pc, pc_scale = role.pcs()
        extra = dict(pc=pc, pc_scale=pc_scale)
        model = MixtureExperts(case["clusters"], case["penalty"], seed)
    elif family == "additive":
        model = AdditiveBoost(case["cycles"], case["iterations"], case["leaves"], seed)
    else:
        raise ValueError("Unknown model family")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "mixture":
            model.fit(x, g, y)
        elif family == "additive":
            model.fit(design.blocks_t, y)
        else:
            model.fit(x, y)
    fit = Fitted(family, model, design.state, mean, scale, extra)
    diagnostic = dict(warnings=[str(w.message) for w in caught])
    if isinstance(model, Factorization):
        diagnostic.update(converged=model.converged, message=model.message)
    return fit, diagnostic


@dataclass
class Interaction:
    kind: str
    base: Fitted
    weight: float
    extra: dict = field(default_factory=dict)

    def matrix(self, role, training=False, contrast=False):
        position = 0 if training else 1
        z = role.view("gut", "clr")[position]
        g = self.extra["pc"].transform(z) / self.extra["pc_scale"]
        if "gate" in self.extra:
            g = self.extra["gate"].predict_proba(g)
        g = (g - self.extra["g_mean"]) / self.extra["g_scale"]
        if contrast:
            d = role.split.delta / self.extra["d_scale"]
            return (d[None, :, None] * g[:, None, :]).reshape(len(g), -1) / self.extra[
                "product_scale"
            ]
        d = role.split.dt if training else role.split.dv
        d = (d - self.extra["d_mean"]) / self.extra["d_scale"]
        return (
            (d[:, :, None] * g[:, None, :]).reshape(len(g), -1) - self.extra["product_mean"]
        ) / self.extra["product_scale"]

    def predict(self, role, training=False):
        base = self.base.predict(role, training)
        if self.weight == 0:
            return base
        if self.kind == "ebm_interaction":
            return base + self.weight * (self.extra["full"].predict(role, training) - base)
        return base + self.extra["ridge"].predict(self.matrix(role, training))

    def modifier(self, role, training=False):
        n = len(role.split.yt) if training else len(role.split.yv)
        if self.weight == 0:
            return np.zeros(n)
        if self.kind == "ebm_interaction":
            full = self.extra["full"]
            x, _ = role.replay(full.design_state, training)
            low, high = role.low_high(x, self.extra["contrast"])
            terms = self.extra["terms"]
            a = full.model.eval_terms(low)
            b = full.model.eval_terms(high)
            return self.weight * full.scale * (b[:, terms] - a[:, terms]).sum(1)
        return self.matrix(role, training, True) @ self.extra["ridge"].coef_


def interaction(role, spec, case, base, config, threads, remember):
    kind = spec["family"]
    if case["weight"] == 0:
        return Interaction(kind, base, 0.0), dict(warnings=[])
    if kind == "ebm_interaction":

        def fit_full():
            from copy import deepcopy

            full = deepcopy(base)
            x, _ = role.replay(full.design_state, True)
            j = role.split.preprocessing["host"]["labels"].index("diet_linear")
            h = role.split.ht.shape[1]
            gut_state = full.design_state["gut"]
            oral_width = len(full.design_state["oral"]["indices"])
            start = h + oral_width
            picked = selected(
                role.split.ht, x[:, start : start + len(gut_state["indices"])], role.split.yt, 8
            )
            pairs = [(j, start + int(k)) for k in picked]
            full.model.set_params(interactions=pairs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                full.model.fit(x, (role.split.yt - full.mean) / full.scale)
            return full, [str(w.message) for w in caught]

        full, caught = remember("ebm_full", fit_full)
        terms = [j for j, t in enumerate(full.model.term_features_) if len(t) == 2]
        fit = Interaction(
            kind, base, case["weight"], dict(full=full, terms=terms, contrast=config["contrast"])
        )
        return fit, dict(warnings=caught)

    def fit_map():
        g, _, pc, scale = role.pcs()
        extra = dict(pc=pc, pc_scale=scale)
        diagnostic = dict(warnings=[])
        if kind == "gate_interaction":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gate = GaussianMixture(
                    n_components=case["clusters"],
                    covariance_type="diag",
                    reg_covar=0.01,
                    n_init=1,
                    max_iter=100,
                    random_state=config["model_seed"],
                ).fit(g)
            diagnostic.update(
                warnings=[str(w.message) for w in caught], converged=bool(gate.converged_)
            )
            g = gate.predict_proba(g)
            extra["gate"] = gate
        d = role.split.dt
        extra.update(g_mean=g.mean(0), g_scale=g.std(0), d_mean=d.mean(0), d_scale=d.std(0))
        for name in ("g_scale", "d_scale"):
            extra[name][extra[name] < 1e-12] = 1.0
        a = (d - extra["d_mean"]) / extra["d_scale"]
        b = (g - extra["g_mean"]) / extra["g_scale"]
        product = (a[:, :, None] * b[:, None, :]).reshape(len(g), -1)
        extra.update(product_mean=product.mean(0), product_scale=product.std(0))
        extra["product_scale"][extra["product_scale"] < 1e-12] = 1.0
        return extra, diagnostic

    extra, diagnostic = remember(kind + str(case.get("clusters", 0)), fit_map)
    fit = Interaction(kind, base, 1.0, dict(extra))
    x = fit.matrix(role, True)
    fit.extra["ridge"] = Ridge(alpha=len(x) * case["penalty"], fit_intercept=False).fit(
        x, role.split.yt - base.predict(role, True)
    )
    return fit, diagnostic
