"""Household-tuned fusion inside outer training, including a fixed V10 replay."""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
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
class LinearFit:
    scaler: object
    model: object
    mean: float
    scale: float

    def predict(self, x):
        return self.mean + self.scale * self.model.predict(self.scaler.transform(x))


def linear_fit(x, y, kind, penalty, replay=False):
    scaler = StandardScaler().fit(x)
    mean, scale = (0.0, 1.0) if replay else (float(y.mean()), max(float(y.std()), 1e-12))
    if kind == "sparse":
        model = ElasticNet(alpha=penalty, l1_ratio=0.7, max_iter=20000, tol=1e-6)
    else:
        model = Ridge(alpha=len(y) * penalty, positive=kind == "positive", tol=1e-8)
    model.fit(scaler.transform(x), (y - mean) / scale)
    return LinearFit(scaler, model, mean, scale)


@dataclass
class FusionFit:
    members: list
    kind: str
    model: object = None
    weights: object = None
    diagnostic: dict = field(default_factory=dict)

    def predict(self, predictions):
        x = np.column_stack([predictions[n] for n in self.members])
        return x @ self.weights if self.kind == "convex" else self.model.predict(x)

    def record(self):
        if self.kind == "convex":
            weights, contribution = self.weights, self.weights
        else:
            contribution = self.model.scale * self.model.model.coef_
            weights = contribution / self.model.scaler.scale_
        return dict(
            members=self.members,
            weights=weights.tolist(),
            contribution=contribution.tolist(),
            weight_kind="convex_weights" if self.kind == "convex" else "native_linear_coefficients",
            **self.diagnostic,
        )


def meta_fit(config, predictions, target, groups):
    """Meta CV is a training heuristic on inner OOF features, not an independent score."""
    result = {}
    folds = list(GroupKFold(config["inner_folds"]).split(target, target, groups))
    if any(set(groups[t]) & set(groups[v]) for t, v in folds):
        raise ValueError("Meta household leakage")
    for pool, members in pools(config).items():
        matrix = np.column_stack([predictions[n] for n in members])
        result["convex_" + pool] = FusionFit(members, "convex", weights=blend_weights(matrix, target))
        for kind in ("ridge", "positive", "sparse"):
            if kind == "sparse" and pool not in ("new", "combined"):
                continue
            penalties = config["sparse_penalties" if kind == "sparse" else "meta_penalties"]
            losses, notes = [], []
            for penalty in penalties:
                p, note = np.full(len(target), np.nan), []
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for t, v in folds:
                        p[v] = linear_fit(matrix[t], target[t], kind, penalty).predict(matrix[v])
                note += [str(w.message).encode("ascii", "backslashreplace").decode() for w in caught]
                losses.append(float(np.mean((target - p) ** 2)))
                notes.append(note)
            chosen = int(np.argmin(losses))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = linear_fit(matrix, target, kind, penalties[chosen])
            note = notes[chosen] + [
                str(w.message).encode("ascii", "backslashreplace").decode() for w in caught
            ]
            result[kind + "_" + pool] = FusionFit(
                members,
                kind,
                model=model,
                diagnostic=dict(
                    penalty=penalties[chosen], training_cv_losses=losses, warnings=sorted(set(note))
                ),
            )
    # This fixed reconstruction is checked against completed V10 OUTER predictions only after fitting.
    members = ["old__" + n for n in config["legacy_members"]]
    matrix = np.column_stack([predictions[n] for n in members])
    result["v10_replay"] = FusionFit(
        members,
        "ridge",
        model=linear_fit(matrix, target, "ridge", config["legacy_stack_penalty"], replay=True),
    )
    return result
