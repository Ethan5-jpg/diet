"""Household-tuned fusion inside outer training, including the frozen V11 replay."""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from registry import pools
from models import regressor
from projections import StablePCA


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


def selected_linear(members, matrix, target, groups, kind, penalties):
    """Household meta-CV is internal training selection, not an external score."""
    folds = list(GroupKFold(3).split(matrix, target, groups))
    losses, notes = [], []
    for penalty in penalties:
        p = np.full(len(target), np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for train, valid in folds:
                if set(groups[train]) & set(groups[valid]):
                    raise ValueError("Meta household leakage")
                p[valid] = linear_fit(matrix[train], target[train], kind, penalty).predict(matrix[valid])
        losses.append(float(np.mean((target - p) ** 2)))
        notes.append([str(w.message).encode("ascii", "backslashreplace").decode() for w in caught])
    chosen = int(np.argmin(losses))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = linear_fit(matrix, target, kind, penalties[chosen])
    note = notes[chosen] + [str(w.message).encode("ascii", "backslashreplace").decode() for w in caught]
    return FusionFit(
        members,
        kind,
        model=model,
        diagnostic=dict(penalty=penalties[chosen], training_cv_losses=losses, warnings=sorted(set(note))),
    )


def meta_fit(config, predictions, target, groups, replay):
    result = {}
    for pool, members in pools(config).items():
        matrix = np.column_stack([predictions[n] for n in members])
        result["convex_" + pool] = FusionFit(members, "convex", weights=blend_weights(matrix, target))
        for kind in ("ridge", "positive", "sparse"):
            if kind == "sparse" and pool not in ("new", "combined"):
                continue
            penalties = config["sparse_penalties" if kind == "sparse" else "meta_penalties"]
            result[kind + "_" + pool] = selected_linear(members, matrix, target, groups, kind, penalties)
    for pool in ("bank", "combined"):
        members = pools(config)[pool]
        matrix = np.column_stack([predictions[n] for n in members])
        for family in ("local", "ntk", "rbf_ard"):
            result[family + "_stack_" + pool] = selected_nonlinear(
                config, members, matrix, target, groups, family
            )
    result.update(replay)
    return result


@dataclass
class NonlinearFit:
    members: list
    kind: str
    scaler: object
    projection: object
    model: object
    mean: float
    scale: float
    diagnostic: dict = field(default_factory=dict)

    def predict(self, predictions):
        x = np.column_stack([predictions[n] for n in self.members])
        return self.mean + self.scale * self.model.predict(
            self.projection.transform(self.scaler.transform(x))
        )

    def record(self):
        return dict(
            members=self.members,
            weights=None,
            contribution=[1.0] * len(self.members),
            weight_kind="nonlinear_all_inputs_for_conservative_warning_propagation",
            **self.diagnostic,
        )


def nonlinear_fit(members, matrix, target, family, case, seed):
    scaler = StandardScaler().fit(matrix)
    projection = StablePCA(
        n_components=max(1, min(case["meta_components"], len(matrix) - 1, matrix.shape[1])), svd_solver="full"
    ).fit(scaler.transform(matrix))
    z = projection.transform(scaler.transform(matrix))
    mean, scale = float(target.mean()), max(float(target.std()), 1e-12)
    model = regressor(family, case, seed, 1).fit(z, (target - mean) / scale)
    return NonlinearFit(members, family, scaler, projection, model, mean, scale)


def selected_nonlinear(config, members, matrix, target, groups, family):
    cases = config["nonlinear_meta_grid"][family]
    folds = list(GroupKFold(3).split(matrix, target, groups))
    losses, notes = [], []
    for case in cases:
        p = np.full(len(target), np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for train, valid in folds:
                if set(groups[train]) & set(groups[valid]):
                    raise ValueError("Nonlinear meta household leakage")
                fit = nonlinear_fit(members, matrix[train], target[train], family, case, config["model_seed"])
                p[valid] = fit.predict({n: matrix[valid, j] for j, n in enumerate(members)})
        losses.append(float(np.mean((target - p) ** 2)))
        notes.append([str(w.message).encode("ascii", "backslashreplace").decode() for w in caught])
    chosen = int(np.argmin(losses))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = nonlinear_fit(members, matrix, target, family, cases[chosen], config["model_seed"])
    fit.diagnostic = dict(
        selected=chosen,
        training_cv_losses=losses,
        warnings=sorted(
            set(notes[chosen] + [str(w.message).encode("ascii", "backslashreplace").decode() for w in caught])
        ),
    )
    if family == "rbf_ard" and fit.model.model.n_iter_ >= cases[chosen]["max_iter"]:
        fit.diagnostic["warnings"].append("Sparse Bayesian stack reached iteration limit")
    return fit
