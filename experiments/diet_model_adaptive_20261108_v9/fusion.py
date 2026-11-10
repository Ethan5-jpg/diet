"""Current-inner-OOF fixed and conditional fusion; no historical prediction input."""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from estimators import ConditionalGate
from registry import pools


def blend_weights(matrix, target):
    scale, count = max(float(np.var(target)), 1e-12), matrix.shape[1]

    def objective(w):
        error = matrix @ w - target
        return float(np.mean(error**2) / scale), 2 * matrix.T @ error / (len(target) * scale)

    fit = minimize(
        objective,
        np.full(count, 1 / count),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(count)},
        options=dict(maxiter=1000, ftol=1e-12),
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Convex fusion failed: " + str(fit.message))
    w = np.clip(fit.x, 0, 1)
    return w / w.sum()


@dataclass
class FusionFit:
    members: list
    kind: str
    model: object = None
    weights: object = None
    diagnostic: dict = field(default_factory=dict)

    def predict(self, predictions, gates):
        matrix = np.column_stack([predictions[n] for n in self.members])
        if self.kind == "convex":
            return matrix @ self.weights
        if self.kind == "gate":
            return self.model.predict(gates, matrix)
        if self.kind == "tree":
            matrix = np.column_stack([matrix, gates])
        return self.model.predict(matrix)

    def record(self, gates):
        if self.kind == "convex":
            weights, contribution, kind = self.weights, self.weights, "convex_weights"
        elif self.kind == "ridge":
            contribution = self.model[-1].coef_
            weights, kind = contribution / self.model[0].scale_, "ridge_raw_coefficients"
        elif self.kind == "gate":
            values = self.model.weights(gates)
            weights, contribution, kind = values.mean(0), values.max(0), "mean_conditional_convex_weights"
        else:
            weights, contribution, kind = (
                np.zeros(len(self.members)),
                np.ones(len(self.members)),
                "nonlinear_inputs",
            )
        return dict(
            members=self.members,
            weights=weights.tolist(),
            contribution=contribution.tolist(),
            weight_kind=kind,
        )


def meta_fit(config, predictions, target, gates, losses, choices):
    fits = {}
    for name, members in pools(config).items():
        matrix = np.column_stack([predictions[n] for n in members])
        fits["convex_" + name] = FusionFit(members, "convex", weights=blend_weights(matrix, target))
        model = make_pipeline(StandardScaler(), Ridge(alpha=len(target) * config["stack_penalty"]))
        model.fit(matrix, target)
        fits["ridge_" + name] = FusionFit(members, "ridge", model=model)
    members = pools(config)["core"]
    matrix = np.column_stack([predictions[n] for n in members])
    for name, penalty in zip(("gated_soft", "gated_medium", "gated_strong"), config["gate_penalties"]):
        model = ConditionalGate(penalty).fit(gates, matrix, target, fits["convex_core"].weights)
        fits[name] = FusionFit(members, "gate", model=model, diagnostic=model.diagnostic)
    tree = HistGradientBoostingRegressor(
        max_iter=config["meta_tree_iterations"],
        max_leaf_nodes=3,
        min_samples_leaf=80,
        l2_regularization=50.0,
        learning_rate=0.025,
        early_stopping=False,
        random_state=config["model_seed"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree.fit(np.column_stack([matrix, gates]), target)
        poly = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=len(target) * 0.5),
        )
        poly.fit(matrix, target)
    notes = dict(
        warnings=sorted(set(str(v.message).encode("ascii", "backslashreplace").decode() for v in caught))
    )
    fits["tree_meta"], fits["polynomial_stack"] = (
        FusionFit(members, "tree", model=tree, diagnostic=notes),
        FusionFit(members, "polynomial", model=poly, diagnostic=notes),
    )
    winner = min(pools(config)["wide"], key=lambda n: losses[n][choices[n]])
    fits["inner_selector"] = FusionFit([winner], "convex", weights=np.ones(1))
    return fits
