"""Current-inner-OOF convex and regularized linear fusion; no prior OOF inputs."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

    def predict(self, predictions):
        matrix = np.column_stack([predictions[n] for n in self.members])
        return matrix @ self.weights if self.kind == "convex" else self.model.predict(matrix)

    def record(self):
        if self.kind == "convex":
            weights, contribution, kind = self.weights, self.weights, "convex_weights"
        else:
            contribution = self.model[-1].coef_
            weights, kind = contribution / self.model[0].scale_, "ridge_raw_coefficients"
        return dict(
            members=self.members,
            weights=weights.tolist(),
            contribution=contribution.tolist(),
            weight_kind=kind,
        )


def meta_fit(config, predictions, target):
    fits = {}
    for name, members in pools(config).items():
        matrix = np.column_stack([predictions[n] for n in members])
        fits["convex_" + name] = FusionFit(members, "convex", weights=blend_weights(matrix, target))
        model = make_pipeline(StandardScaler(), Ridge(alpha=len(target) * config["stack_penalty"]))
        model.fit(matrix, target)
        fits["ridge_" + name] = FusionFit(members, "ridge", model=model)
    return fits
