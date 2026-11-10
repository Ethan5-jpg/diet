"""Nested household CV, fixed-background interaction tests and honest outer scoring."""

import time

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import Role
from models import atomic, interaction
from registry import INTERACTIONS, REFERENCES, specifications


def blend_weights(matrix, target):
    scale = max(float(np.var(target)), 1e-12)
    count = matrix.shape[1]

    def objective(weight):
        error = matrix @ weight - target
        return float(np.mean(error**2) / scale), 2 * matrix.T @ error / (len(target) * scale)

    fit = minimize(
        objective,
        np.full(count, 1.0 / count),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(count)},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Convex fusion optimizer failed: " + fit.message)
    weight = np.clip(fit.x, 0, 1)
    return weight / weight.sum()


class Engine:
    def __init__(self, config, memo, threads, reference, log=print):
        self.config, self.memo, self.threads = config, memo, threads
        self.reference, self.log = reference, log
        self.specs = specifications(config)

    def fit(self, role, spec, case, bases, cache):
        def remember(key, compute):
            if key not in cache:
                cache[key] = compute()
            return cache[key]

        if spec["name"] in INTERACTIONS:
            return interaction(
                role,
                spec,
                case,
                bases[INTERACTIONS[spec["name"]]],
                self.config,
                self.threads,
                remember,
            )
        return atomic(role, spec, case, self.config, self.threads)

    @staticmethod
    def release(role):
        # Keep only small PCA state; raw CLR matrices belong to the frozen split.
        role.cache = {k: v for k, v in role.cache.items() if k[0] == "pc"}

    def outer(self, dataset, repeat, fold):
        cfg, x = self.config, dataset.index
        groups = x.split_unit_id.astype(str).to_numpy()
        assignment = dataset.reference["outer"][repeat]
        train, valid = assignment != fold, assignment == fold
        dataset.api.separated(groups, train, valid)
        inner = np.full(len(x), -1)
        inner[train] = dataset.api.folds(
            groups[train], cfg["inner_folds"], cfg["outer_seed"] + 100 * repeat + fold + 1
        )
        tag = "r%d_f%d" % (repeat, fold)
        roles = [
            Role(
                dataset.split(train & (inner != k), train & (inner == k), cfg["contrast"]),
                dataset.raw,
                cfg["model_seed"],
            )
            for k in range(cfg["inner_folds"])
        ]
        target = np.concatenate([r.split.yv for r in roles])
        bases, caches = [{} for _ in roles], [{} for _ in roles]
        choices, losses, oof, diagnostics = {}, {}, {}, {}
        nulls = set(INTERACTIONS.values())
        for spec in self.specs:
            name, started = spec["name"], time.monotonic()
            self.log("START %s %s candidates=%d" % (tag, name, len(spec["grid"])))
            matrices, notes = [], []
            for k, role in enumerate(roles):
                columns = []
                for j, case in enumerate(spec["grid"]):

                    def compute(role=role, case=case, spec=spec, base=bases[k], cache=caches[k]):
                        fitted, diagnostic = self.fit(role, spec, case, base, cache)
                        return dict(prediction=fitted.predict(role), diagnostic=diagnostic)

                    unit = self.memo.get(tag + "_i%d_%s_c%d" % (k, name, j), compute)
                    columns.append(unit["prediction"])
                    notes.append(dict(inner_fold=k, candidate=j, **unit["diagnostic"]))
                matrices.append(np.column_stack(columns))
            matrix = np.concatenate(matrices)
            if matrix.shape != (len(target), len(spec["grid"])) or not np.isfinite(matrix).all():
                raise ValueError("Invalid inner predictions: " + name)
            loss = np.mean((target[:, None] - matrix) ** 2, axis=0)
            chosen = int(loss.argmin())
            choices[name], losses[name] = chosen, loss.tolist()
            oof[name], diagnostics[name] = matrix[:, chosen], notes
            if name in nulls:
                for k, role in enumerate(roles):
                    # Reuse exactly the same selected null in every interaction candidate.
                    fitted = self.memo.get(
                        tag + "_i%d_%s_selected" % (k, name),
                        lambda role=role, spec=spec, chosen=chosen, base=bases[k], cache=caches[k]: (
                            self.fit(role, spec, spec["grid"][chosen], base, cache)[0]
                        ),
                    )
                    bases[k][name] = fitted
            for role in roles:
                self.release(role)
            self.log(
                "SELECT %s %s case=%d elapsed=%.1fs"
                % (tag, name, chosen, time.monotonic() - started)
            )

        # No outer validation feature or label has entered model/ensemble selection.
        winner = min(choices, key=lambda name: losses[name][choices[name]])
        members = cfg["blend_members"]
        matrix = np.column_stack([oof[name] for name in members])
        weights = blend_weights(matrix, target)
        stack = make_pipeline(StandardScaler(), Ridge(alpha=len(target) * cfg["stack_penalty"]))
        stack.fit(matrix, target)
        del roles, bases, caches, oof, matrix, target

        split = dataset.split(train, valid, cfg["contrast"])
        rows = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(split.rows, rows) or not np.array_equal(
            split.yv, dataset.reference["y"][repeat, rows]
        ):
            raise ValueError("Outer rows or processed targets differ from frozen V4")
        role, bases, cache = Role(split, dataset.raw, cfg["model_seed"]), {}, {}
        self.memo.get(tag + "_outer_inputs", split.snapshot)
        predictions, modifiers, active, outer_notes = {}, {}, {}, {}
        for spec in self.specs:
            name = spec["name"]

            def compute(spec=spec, name=name):
                fitted, diagnostic = self.fit(role, spec, spec["grid"][choices[name]], bases, cache)
                value = dict(prediction=fitted.predict(role), diagnostic=diagnostic, fitted=fitted)
                if name in nulls:
                    value["fitted_null"] = fitted
                if name in INTERACTIONS:
                    axis = fitted.modifier(role)
                    value.update(
                        modifier=axis - fitted.modifier(role, True).mean(),
                        active=bool(fitted.weight > 0 and np.std(axis) > 1e-12),
                    )
                return value

            value = self.memo.get(tag + "_o_" + name, compute)
            predictions[name], outer_notes[name] = value["prediction"], value["diagnostic"]
            if name in nulls:
                bases[name] = value["fitted_null"]
            if name in INTERACTIONS:
                modifiers[name], active[name] = value["modifier"], value["active"]
            self.release(role)
        matrix = np.column_stack([predictions[name] for name in members])
        predictions.update(
            inner_selector=predictions[winner].copy(),
            convex_blend=matrix @ weights,
            ridge_stack=stack.predict(matrix),
        )
        for name in REFERENCES[:2]:
            predictions[name] = dataset.reference[name][repeat, rows]
        # V6 labels/predictions are used only for matched evaluation after all choices.
        predictions["v6_fusion"] = self.reference.get(repeat, fold, rows, split.yv)
        if not all(
            p.shape == split.yv.shape and np.isfinite(p).all() for p in predictions.values()
        ):
            raise ValueError("Nonfinite or misaligned outer prediction")
        return dict(
            rows=rows,
            y=split.yv,
            groups=groups[rows],
            predictions=predictions,
            modifiers=modifiers,
            active=active,
            selected=choices,
            inner_losses=losses,
            inner_selector=winner,
            blend_weights=weights.tolist(),
            stack=stack,
            inner_diagnostics=diagnostics,
            outer_diagnostics=outer_notes,
        )
