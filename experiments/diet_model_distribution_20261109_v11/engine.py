"""Nested household selection; current inner OOF fusion; evaluation-only references."""

import time
import warnings

import numpy as np

from features import Role
from fusion import meta_fit
from models import atomic, interaction, residual
from registry import CORRECTIONS, INTERACTIONS, REFERENCES, names, specifications


def flagged(note):
    return bool(note.get("warnings") or note.get("converged") is False)


class Engine:
    def __init__(self, config, memo, threads, reference, log=print):
        self.config, self.memo, self.threads, self.reference = config, memo, threads, reference
        self.log, self.specs = log, specifications(config)

    def fit(self, role, spec, case, bases):
        if spec["name"] in INTERACTIONS:
            return interaction(role, spec, case, bases[INTERACTIONS[spec["name"]]], self.config, self.threads)
        if spec["name"] in CORRECTIONS:
            return residual(role, spec, case, bases[CORRECTIONS[spec["name"]]], self.config, self.threads)
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
        role.cache = {
            k: v
            for k, v in role.cache.items()
            if k[0] not in ("design", "block", "residual_stage", "fiber_stage")
        }

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
                cfg["anchor_seed"],
                taxa=dataset.taxa,
                extra=dataset.extra,
                diet_parameters=cfg["diet_parameters"],
            )
            for k in range(cfg["inner_folds"])
        ]
        target = np.concatenate([r.split.yv for r in roles])
        legacy_oof, legacy_choices = self.reference.inner(repeat, fold, roles)
        meta_groups = np.concatenate([groups[r.split.rows] for r in roles])
        bases, choices, losses, oof, diagnostics = [{} for _ in roles], {}, {}, {}, {}
        nulls = set(INTERACTIONS.values()) | set(CORRECTIONS.values())
        for spec in self.specs:
            name, started = spec["name"], time.monotonic()
            progress = started
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
                    if time.monotonic() - progress >= 30:
                        self.log("PROGRESS %s %s inner=%d candidate=%d" % (tag, name, k, j))
                        progress = time.monotonic()
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
        meta = self.memo.get(
            tag + "_meta_fit",
            lambda legacy_oof=legacy_oof, oof=oof, target=target: meta_fit(
                cfg, dict(legacy_oof, **oof), target, meta_groups
            ),
        )
        selected_inner_flags = {
            n: any(flagged(v) for v in notes if v["candidate"] == choices[n])
            for n, notes in diagnostics.items()
        }
        del roles, bases, oof, legacy_oof, target, matrix
        split = dataset.split(train, valid, cfg["contrast"])
        rows = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(split.rows, rows) or not np.array_equal(
            split.yv, dataset.reference["y"][repeat, rows]
        ):
            raise ValueError("Rows or processed targets differ from frozen V4")
        role, bases = (
            Role(
                split,
                dataset.raw,
                dataset.index,
                cfg["anchor_seed"],
                taxa=dataset.taxa,
                extra=dataset.extra,
                diet_parameters=cfg["diet_parameters"],
            ),
            {},
        )
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
            if name in INTERACTIONS or name in CORRECTIONS:
                base_name = INTERACTIONS.get(name, CORRECTIONS.get(name))
                if flagged(outer_notes[base_name]) or outer_notes[base_name].get("selected_inner_warning"):
                    outer_notes[name]["background_warning"] = base_name
            self.release(role)
        # Prior outer payloads are opened only after every training choice and meta fit is fixed.
        old_predictions, old_notes = self.reference.get(repeat, fold, rows, split.yv, legacy_choices, role)
        predictions.update(old_predictions)
        outer_notes.update(old_notes)
        meta_records = {}
        for name, fit in meta.items():
            predictions[name] = fit.predict(predictions)
            record = fit.record()
            contributors = [n for n, w in zip(record["members"], record["contribution"]) if abs(w) > 1e-8]
            affected = [
                n
                for n in contributors
                if flagged(outer_notes[n])
                or outer_notes[n].get("selected_inner_warning")
                or outer_notes[n].get("background_warning")
            ]
            outer_notes[name] = dict(fit.diagnostic, contributing_fit_warnings=affected)
            meta_records[name] = record
        replay_error = float(np.max(np.abs(predictions["v10_replay"] - predictions["v10_ridge_diverse"])))
        if replay_error > 1e-8:
            raise ValueError("V10 inner-bank replay differs from its completed ridge fusion")
        kept = names(cfg) + list(REFERENCES)
        predictions = {n: predictions[n] for n in kept}
        outer_notes = {n: outer_notes.get(n, {}) for n in kept}
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
            inner_diagnostics=diagnostics,
            outer_diagnostics=outer_notes,
            legacy_selected=legacy_choices,
            v10_replay_max_error=replay_error,
        )
