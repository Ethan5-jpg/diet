"""Nested household selection; current inner OOF fusion; evaluation-only references."""

import time
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import Role
from models import atomic, interaction
from registry import INTERACTIONS, pools, specifications


def flagged(note):
    return bool(note.get("warnings") or note.get("converged") is False)


def blend_weights(matrix, target):
    scale, count = max(float(np.var(target)), 1e-12), matrix.shape[1]

    def objective(weight):
        error = matrix @ weight - target
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
    weight = np.clip(fit.x, 0, 1)
    return weight / weight.sum()


def meta_fit(config, predictions, target, losses, choices):
    models = {}
    for pool_name, members in pools(config).items():
        matrix = np.column_stack([predictions[n] for n in members])
        name = "convex_" + pool_name
        models[name] = dict(members=members, weights=blend_weights(matrix, target))
        if pool_name != "with_fm":
            model = make_pipeline(StandardScaler(), Ridge(alpha=len(target) * config["stack_penalty"]))
            model.fit(matrix, target)
            models["ridge_" + pool_name] = dict(members=members, model=model)
    for name, pool_name in (("inner_selector_core", "wide"), ("inner_selector_joint", "joint")):
        winner = min(pools(config)[pool_name], key=lambda n: losses[n][choices[n]])
        models[name] = dict(members=[winner], weights=np.ones(1))
    return models


class Engine:
    def __init__(self, config, memo, threads, reference, log=print):
        self.config, self.memo, self.threads, self.reference = config, memo, threads, reference
        self.log, self.specs = log, specifications(config)

    def fit(self, role, spec, case, bases):
        if spec["name"] in INTERACTIONS:
            return interaction(role, spec, case, bases[INTERACTIONS[spec["name"]]], self.config)
        return atomic(role, spec, case, self.config, self.threads)

    def evaluate(self, role, spec, case, bases, keep=False):
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, note = self.fit(role, spec, case, bases)
            value = dict(prediction=model.predict(role))
            if spec["name"] in INTERACTIONS:
                axis = model.modifier(role)
                value.update(
                    modifier=axis - model.modifier(role, True).mean(),
                    active=bool(model.weight > 0 and np.std(axis) > 1e-12),
                )
        note = dict(
            note,
            warnings=sorted(
                set(
                    note.get("warnings", [])
                    + [str(w.message).encode("ascii", "backslashreplace").decode() for w in caught]
                )
            ),
            seconds=time.monotonic() - started,
        )
        value["diagnostic"] = note
        if keep:
            value["fitted"] = model
        return value

    @staticmethod
    def release(role):
        role.cache = {k: v for k, v in role.cache.items() if k[0] != "design"}

    def outer(self, dataset, repeat, fold):
        cfg, x = self.config, dataset.index
        groups, assignment = x.split_unit_id.astype(str).to_numpy(), dataset.reference["outer"][repeat]
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
                dataset.index,
                cfg["model_seed"],
            )
            for k in range(cfg["inner_folds"])
        ]
        target = np.concatenate([r.split.yv for r in roles])
        bases, choices, losses, oof, diagnostics = [{} for _ in roles], {}, {}, {}, {}
        nulls = set(INTERACTIONS.values())
        for spec in self.specs:
            name, started = spec["name"], time.monotonic()
            self.log("START %s %s candidates=%d" % (tag, name, len(spec["grid"])))
            matrices, notes = [], []
            for k, role in enumerate(roles):
                columns = []
                for j, case in enumerate(spec["grid"]):
                    unit = self.memo.get(
                        tag + "_i%d_%s_c%d" % (k, name, j),
                        lambda role=role, spec=spec, case=case, base=bases[k]: self.evaluate(
                            role, spec, case, base
                        ),
                    )
                    columns.append(unit["prediction"])
                    notes.append(dict(inner_fold=k, candidate=j, **unit["diagnostic"]))
                matrices.append(np.column_stack(columns))
            matrix = np.concatenate(matrices)
            if matrix.shape != (len(target), len(spec["grid"])) or not np.isfinite(matrix).all():
                raise ValueError("Invalid inner predictions: " + name)
            loss = np.mean((target[:, None] - matrix) ** 2, axis=0)
            chosen = int(loss.argmin())
            choices[name], losses[name], oof[name], diagnostics[name] = (
                chosen,
                loss.tolist(),
                matrix[:, chosen],
                notes,
            )
            if name in nulls:
                for k, role in enumerate(roles):
                    selected = self.memo.get(
                        tag + "_i%d_%s_selected" % (k, name),
                        lambda role=role, spec=spec, chosen=chosen, base=bases[k]: self.evaluate(
                            role, spec, spec["grid"][chosen], base, True
                        ),
                    )
                    bases[k][name] = selected["fitted"]
            for role in roles:
                self.release(role)
            self.log("SELECT %s %s case=%d elapsed=%.1fs" % (tag, name, chosen, time.monotonic() - started))

        # The entire meta training set is inside the current outer training families.
        meta = meta_fit(cfg, oof, target, losses, choices)
        selected_inner_flags = {
            n: any(flagged(v) for v in notes if v["candidate"] == choices[n])
            for n, notes in diagnostics.items()
        }
        del roles, bases, oof, target, matrix
        split = dataset.split(train, valid, cfg["contrast"])
        rows = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(split.rows, rows) or not np.array_equal(
            split.yv, dataset.reference["y"][repeat, rows]
        ):
            raise ValueError("Rows or processed targets differ from frozen V4")
        role, bases = Role(split, dataset.raw, dataset.index, cfg["model_seed"]), {}
        self.memo.get(tag + "_outer_inputs", split.snapshot)
        predictions, modifiers, active, outer_notes = {}, {}, {}, {}
        for spec in self.specs:
            name = spec["name"]
            value = self.memo.get(
                tag + "_o_" + name,
                lambda spec=spec, name=name: self.evaluate(
                    role, spec, spec["grid"][choices[name]], bases, True
                ),
            )
            predictions[name], outer_notes[name] = value["prediction"], dict(value["diagnostic"])
            if selected_inner_flags[name]:
                outer_notes[name]["selected_inner_warning"] = True
            if name in nulls:
                bases[name] = value["fitted"]
            if name in INTERACTIONS:
                modifiers[name], active[name] = value["modifier"], value["active"]
                base_name = INTERACTIONS[name]
                if flagged(outer_notes[base_name]) or outer_notes[base_name].get("selected_inner_warning"):
                    outer_notes[name]["background_warning"] = base_name
            self.release(role)
        meta_records = {}
        for name, fit in meta.items():
            members = fit["members"]
            matrix = np.column_stack([predictions[n] for n in members])
            if "weights" in fit:
                weights = fit["weights"]
                predictions[name] = matrix @ weights
                contributions = weights
                kind = "selector" if name.startswith("inner_selector") else "convex_weights"
            else:
                predictions[name] = fit["model"].predict(matrix)
                weights = fit["model"][-1].coef_ / fit["model"][0].scale_
                contributions = fit["model"][-1].coef_
                kind = "ridge_raw_coefficients"
            contributors = [n for n, w in zip(members, contributions) if abs(w) > 1e-8]
            affected = [
                n
                for n in contributors
                if flagged(outer_notes[n])
                or outer_notes[n].get("selected_inner_warning")
                or outer_notes[n].get("background_warning")
            ]
            outer_notes[name] = dict(warnings=[], contributing_fit_warnings=affected)
            meta_records[name] = dict(
                members=members,
                weights=weights.tolist(),
                weight_kind=kind,
                standardized_contributions=contributions.tolist(),
            )
        for name in ("v4_hgb_oral", "v4_hgb_oral_gut"):
            predictions[name] = dataset.reference[name][repeat, rows]
        # Historical references are read only after every choice and prediction.
        predictions.update(self.reference.get(repeat, fold, rows, split.yv))
        if not all(p.shape == split.yv.shape and np.isfinite(p).all() for p in predictions.values()):
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
            meta=meta_records,
            meta_fitted=meta,
            inner_diagnostics=diagnostics,
            outer_diagnostics=outer_notes,
        )
