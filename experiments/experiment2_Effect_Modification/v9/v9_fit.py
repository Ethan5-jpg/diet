"""Matched fits using a shared background and an exact per-recipe Schur complement."""

import numpy as np
from scipy import linalg
from sklearn.linear_model import lasso_path

import v7_models as legacy
import v9_features
import v9_grid as grid


class InteractionBlock:
    """Profile out the paired main effects; solve in weighted product coordinates."""

    def __init__(self, profile, mains, products, weights):
        ix = np.ix_
        q = profile.gram[ix(products, products)]
        c = profile.rhs[products]
        p = profile.projected_valid[:, products]
        self.off = profile.null
        if len(mains):
            g = profile.gram[ix(mains, mains)]
            g.flat[:: len(g) + 1] += legacy.MAIN_PENALTY
            link = profile.gram[ix(mains, products)]
            factor = linalg.cho_factor(g, lower=True, check_finite=False)
            # Keep the off solve independent even of the number of on columns.
            main = linalg.cho_solve(factor, profile.rhs[mains], check_finite=False)
            change = linalg.cho_solve(factor, link, check_finite=False)
            pv = profile.projected_valid[:, mains]
            self.off = self.off + pv @ main
            q, c, p = q - link.T @ change, c - link.T @ main, p - pv @ change
        scale = 1.0 / np.sqrt(weights)
        self.q = ((q + q.T) * 0.5) * scale[:, None] * scale
        self.c = c * scale
        self.p = p * scale
        eigen, self.vectors = linalg.eigh(self.q, check_finite=False)
        if eigen[0] < -1e-8 * max(1.0, eigen[-1]):
            raise ArithmeticError("Indefinite interaction Gram matrix")
        self.eigen = np.maximum(eigen, 0.0)
        self.signal = self.vectors.T @ self.c

    def ridge(self, options):
        coefficients = self.signal[:, None] / (self.eigen[:, None] + np.asarray(options))
        return self.off[:, None] + (self.p @ self.vectors) @ coefficients

    def pcr(self, options):
        columns = []
        for k in options:
            keep = np.flatnonzero(self.eigen > 1e-10 * max(1.0, self.eigen[-1]))[-k:]
            columns.append(
                self.vectors[:, keep] @ (self.signal[keep] / (self.eigen[keep] + grid.ALT_RIDGE))
            )
        return self.off[:, None] + self.p @ np.column_stack(columns)

    def krylov(self, options):
        matrix = self.q + grid.ALT_RIDGE * np.eye(len(self.c))
        b = np.zeros_like(self.c)
        residual = self.c.copy()
        direction = residual.copy()
        norm2 = residual @ residual
        threshold = max(norm2 * 1e-24, 1e-30)
        columns = []
        for step in range(1, max(options) + 1):
            if norm2 > threshold:
                product = matrix @ direction
                length = norm2 / (direction @ product)
                b += length * direction
                residual -= length * product
                new_norm2 = residual @ residual
                direction = residual + (new_norm2 / norm2) * direction
                norm2 = new_norm2
            if step in options:
                columns.append(b.copy())
        return self.off[:, None] + self.p @ np.column_stack(columns)

    def elastic(self, options):
        # Cholesky encodes 1/2 b'(Q+rho I)b - c'b exactly as least squares.
        # sklearn's path divides squared loss by the number of pseudo-rows.
        dimension = len(self.c)
        penalty = np.max(np.abs(self.c)) * np.asarray(options)
        if penalty[0] == 0:
            return np.repeat(self.off[:, None], len(options), axis=1)
        gram = self.q + grid.ALT_RIDGE * np.eye(dimension)
        a = linalg.cholesky(gram, lower=False)
        target = linalg.solve_triangular(a.T, self.c, lower=True)
        _, coef, _ = lasso_path(a, target, alphas=penalty / dimension, tol=1e-10, max_iter=30000)
        # Numerically verify the optimality conditions of the declared objective.
        gradient = gram @ coef - self.c[:, None]
        error = np.where(
            coef != 0.0, abs(gradient + penalty * np.sign(coef)), np.maximum(abs(gradient) - penalty, 0.0)
        )
        if np.max(error) > 2e-6 * max(1.0, np.max(np.abs(self.c))):
            raise ArithmeticError("Elastic net failed its optimality check")
        return self.off[:, None] + self.p @ coef


def evaluate_split(source, split):
    if np.intersect1d(source.groups[split.train_rows], source.groups[split.rows]).size:
        raise ValueError("Household crosses training and evaluation")
    bank, bases, doses = v9_features.build(source, split)
    mains = list(dict.fromkeys(m for model in grid.RECIPES.values() for m in model.mains))
    terms = list(dict.fromkeys(t for model in grid.RECIPES.values() for t in model.terms))
    mi = {m: j for j, m in enumerate(mains)}
    ti = {t: len(mains) + j for j, t in enumerate(terms)}
    columns = [
        np.column_stack(
            [*(bank[m][side] for m in mains), *(doses[side][d] * bank[m][side] for d, m in terms)]
        )
        for side in (0, 1)
    ]
    profile = legacy._Profile(
        *bases,
        np.column_stack((split.ot, split.gt)),
        np.column_stack((split.ov, split.gv)),
        split.yt,
        *columns,
    )
    pairs = {}

    def record(name, off, on):
        if on.shape != (len(off), len(grid.SETTINGS[name])) or not np.isfinite(on).all():
            raise ArithmeticError("Invalid prediction: " + name)
        for j, p in enumerate(grid.SETTINGS[name]):
            pairs[legacy.setting_key(name, p)] = off, on[:, j]

    for model in grid.RECIPES.values():
        block = InteractionBlock(
            profile, [mi[m] for m in model.mains], [ti[t] for t in model.terms], model.weights
        )
        record(model.name, block.off, block.ridge(grid.PENALTIES))
        for method, options in grid.OPTIONS.items():
            name = method + "__" + model.name
            if name in grid.ALTERNATIVES:
                record(name, block.off, getattr(block, method)(options))
    for name, parents in grid.BLENDS.items():
        for penalty in grid.PENALTIES:
            members = [pairs[legacy.setting_key(p, penalty)] for p in parents]
            pairs[legacy.setting_key(name, penalty)] = tuple(
                np.mean(arm, axis=0) for arm in zip(*members, strict=True)
            )
    for name, (parent, fraction) in grid.ATTENUATION.items():
        for penalty in grid.SETTINGS[name]:
            off, on = pairs[legacy.setting_key(parent, penalty)]
            pairs[legacy.setting_key(name, penalty)] = off, off + fraction * (on - off)
    return pairs
