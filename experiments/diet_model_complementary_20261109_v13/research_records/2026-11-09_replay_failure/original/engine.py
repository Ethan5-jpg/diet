"""Nested family selection, current inner fusion and inference-only legacy arrays."""

import time
import warnings

import numpy as np

from features import Role
from fusion import meta_fit
from models import atomic
from registry import REFERENCES, names, specifications


def flagged(note):
    return bool(
        note.get("warnings")
        or note.get("selected_inner_warning")
        or note.get("background_warning")
        or note.get("contributing_fit_warnings")
    )


class Engine:
    def __init__(self, config, memo, threads, reference, log=print):
        self.config, self.memo, self.threads, self.reference = config, memo, threads, reference
        self.log, self.specs = log, specifications(config)

    def fit(self, role, spec, case):
        return atomic(role, spec, case, self.config, self.threads)

    def evaluate(self, role, spec, case, keep=False):
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, note = self.fit(role, spec, case)
            value = dict(prediction=model.predict(role))
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
        role.cache = {k: v for k, v in role.cache.items() if k[0] not in ("design", "block")}

    def outer(self, dataset, repeat, fold):
        cfg, x = self.config, dataset.index
        groups = x.split_unit_id.astype(str).to_numpy()
        assignment = dataset.reference["outer"][repeat]
        train, valid = assignment != fold, assignment == fold
        dataset.api.separated(groups, train, valid)
        inner = np.full(len(x), -1)
        inner[train] = dataset.api.folds(groups[train], 3, cfg["outer_seed"] + 100 * repeat + fold + 1)

        def role(t, v):
            return Role(
                dataset.split(t, v, cfg["contrast"]),
                dataset.raw,
                x,
                cfg["model_seed"],
                taxa=dataset.taxa,
                extra=dataset.extra,
            )

        roles = [role(train & (inner != k), train & (inner == k)) for k in range(3)]
        target = np.concatenate([r.split.yv for r in roles])
        meta_groups = np.concatenate([groups[r.split.rows] for r in roles])
        tag = "r%d_f%d" % (repeat, fold)
        prior = self.memo.get(
            tag + "_prior_inner", lambda roles=roles: self.reference.inner(repeat, fold, roles)
        )
        choices, losses, oof, diagnostics = {}, {}, {}, {}
        for spec in self.specs:
            name = spec["name"]
            start = time.monotonic()
            progress = start
            self.log("START %s %s candidates=%d" % (tag, name, len(spec["grid"])))
            pieces, notes = [], []
            for k, r in enumerate(roles):
                columns = []
                for j, case in enumerate(spec["grid"]):
                    unit = self.memo.get(
                        tag + "_i%d_%s_c%d" % (k, name, j),
                        lambda r=r, spec=spec, case=case: self.evaluate(r, spec, case),
                    )
                    columns.append(unit["prediction"])
                    notes.append(dict(inner_fold=k, candidate=j, **unit["diagnostic"]))
                    if time.monotonic() - progress >= 30:
                        self.log("PROGRESS %s %s inner=%d candidate=%d" % (tag, name, k, j))
                        progress = time.monotonic()
                pieces.append(np.column_stack(columns))
            matrix = np.concatenate(pieces)
            if matrix.shape != (len(target), len(spec["grid"])) or not np.isfinite(matrix).all():
                raise ValueError("Invalid inner prediction")
            loss = np.mean((target[:, None] - matrix) ** 2, axis=0)
            chosen = int(loss.argmin())
            choices[name], losses[name], oof[name], diagnostics[name] = (
                chosen,
                loss.tolist(),
                matrix[:, chosen],
                notes,
            )
            for r in roles:
                self.release(r)
            self.log("SELECT %s %s case=%d elapsed=%.1fs" % (tag, name, chosen, time.monotonic() - start))
        meta = self.memo.get(
            tag + "_meta_fit",
            lambda oof=oof, target=target: meta_fit(
                cfg, dict(prior["predictions"], **oof), target, meta_groups, prior["replay"]
            ),
        )
        selected_flags = {
            n: any(flagged(v) for v in vals if v["candidate"] == choices[n])
            for n, vals in diagnostics.items()
        }
        del roles, oof, target, matrix
        r = role(train, valid)
        split = r.split
        rows = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(split.rows, rows) or not np.array_equal(
            split.yv, dataset.reference["y"][repeat, rows]
        ):
            raise ValueError("Frozen V4 rows/processed target differs")
        self.memo.get(tag + "_outer_inputs", split.snapshot)
        predictions, outer_notes = {}, {}
        for spec in self.specs:
            name = spec["name"]
            unit = self.memo.get(
                tag + "_o_" + name,
                lambda spec=spec, name=name: self.evaluate(r, spec, spec["grid"][choices[name]], True),
            )
            predictions[name], outer_notes[name] = unit["prediction"], dict(unit["diagnostic"])
            if selected_flags[name]:
                outer_notes[name]["selected_inner_warning"] = True
            self.release(r)
        # Training, all candidate choices and all fusion fits are fixed before this call.
        old, notes = self.reference.get(repeat, fold, rows, split.yv, prior["choices"], r)
        predictions.update(old)
        outer_notes.update(notes)
        records = {}
        for name, fit in meta.items():
            predictions[name] = fit.predict(predictions)
            record = fit.record()
            records[name] = record
            affected = [
                n
                for n, w in zip(record["members"], record["contribution"])
                if abs(w) > 1e-8 and flagged(outer_notes.get(n, {}))
            ]
            outer_notes[name] = dict(fit.diagnostic, contributing_fit_warnings=affected)
        error = float(np.max(abs(predictions["v11_replay"] - predictions["v11_positive_combined"])))
        if error > 1e-8:
            raise ValueError("V11 positive fusion replay differs")
        kept = names(cfg) + list(REFERENCES)
        predictions = {n: predictions[n] for n in kept}
        if not all(p.shape == split.yv.shape and np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Invalid outer predictions")
        return dict(
            rows=rows,
            y=split.yv,
            groups=groups[rows],
            predictions=predictions,
            selected=choices,
            inner_losses=losses,
            inner_diagnostics=diagnostics,
            outer_diagnostics={n: outer_notes.get(n, {}) for n in kept},
            meta=records,
            legacy_selected=prior["choices"],
            v11_replay_max_error=error,
        )
