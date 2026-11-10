"""Nested household prediction selection and outcome-blind ecological targets."""

import time
import warnings

import numpy as np

from features import Design, Role, partial_out
from models import fit_augmented, fit_main, specifications


class Engine:
    def __init__(self, config, memo, log=print):
        self.config, self.memo, self.log = config, memo, log

    def design(self, dataset, train, valid):
        s = dataset.split(train, valid, self.config["contrast"])
        role = Role(s, dataset.raw, dataset.index, self.config["model_seed"], taxa=dataset.taxa, extra=dataset.extra)
        design = Design(role, self.config)
        # Everything needed for replay is now local to the design.
        role.raw, role.extra, role.cache = {}, {}, {}
        for name in ("ot", "ov", "gt", "gv"):
            setattr(s, name, np.empty((0, 0)))
        return design

    def outer(self, dataset, repeat, fold):
        cfg, index = self.config, dataset.index
        tag = "r%d_f%d" % (repeat, fold)
        groups = index.split_unit_id.astype(str).to_numpy()
        assignment = dataset.reference["outer"][repeat]
        train, valid = assignment != fold, assignment == fold
        dataset.api.separated(groups, train, valid)
        inner = np.full(len(index), -1)
        inner[train] = dataset.api.folds(groups[train], cfg["inner_folds"], cfg["outer_seed"] + 100 * repeat + fold + 1)
        self.log("START %s inner feature construction" % tag)
        designs = [self.design(dataset, train & (inner != k), train & (inner == k)) for k in range(cfg["inner_folds"])]
        target = np.concatenate([d.role.split.yv for d in designs])
        base_inner = []
        for k, design in enumerate(designs):
            key = tag + "_inner%d_main" % k
            base_inner.append(self.memo.get(key, lambda design=design: np.column_stack([
                fit_main(design, penalty).predict() for penalty in cfg["main_penalties"]
            ])))
        base_matrix = np.concatenate(base_inner)
        losses = np.mean((target[:, None] - base_matrix) ** 2, axis=0)
        chosen = int(losses.argmin())
        main_penalty = cfg["main_penalties"][chosen]
        choices, inner_losses = {}, {"main": losses.tolist()}
        self.log("SELECT %s main penalty=%g" % (tag, main_penalty))
        candidates = [None] + cfg["interaction_penalties"]
        for spec in specifications():
            matrices = []
            for k, design in enumerate(designs):
                def compute(design=design, k=k, spec=spec):
                    columns = [base_inner[k][:, chosen]]
                    for penalty in cfg["interaction_penalties"]:
                        columns.append(fit_augmented(design, spec, main_penalty, penalty).predict())
                    return np.column_stack(columns)
                matrices.append(self.memo.get(tag + "_inner%d_%s" % (k, spec["name"]), compute))
            loss = np.mean((target[:, None] - np.concatenate(matrices)) ** 2, axis=0)
            choices[spec["name"]] = candidates[int(loss.argmin())]
            inner_losses[spec["name"]] = loss.tolist()
        del designs, base_inner, base_matrix, target, matrices
        self.log("FIT %s outer models; all selection fixed" % tag)
        design = self.design(dataset, train, valid)
        s = design.role.split
        expected = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(s.rows, expected) or not np.array_equal(s.yv, dataset.reference["y"][repeat, expected]):
            raise ValueError("Outer evaluation rows/outcomes differ from the frozen reference")
        # Inference is an ordinary regression with fold-specific nuisance coefficients.
        # This separate use of held-out Y never feeds prediction fitting/selection.
        values = np.column_stack([s.yv, dataset.y[s.rows], design.targets_v])
        residuals, inference_q, rank = partial_out(design.xv, values)
        base = fit_main(design, main_penalty)
        predictions, contrasts, components, fitted = {}, {}, {}, {"main": base}
        predictions["main"] = base.predict()
        contrasts["main"], components["main"] = base.contrasts()
        for spec in specifications():
            model = fit_augmented(design, spec, main_penalty, choices[spec["name"]], base)
            predictions[spec["name"]] = model.predict()
            contrasts[spec["name"]], components[spec["name"]] = model.contrasts()
            fitted[spec["name"]] = model
        support = design.support()
        result = dict(
            rows=s.rows, y=s.yv, raw_y=dataset.y[s.rows], prediction=predictions,
            total_contrast=contrasts, interaction_component=components,
            y_residual=residuals[:, 0], raw_y_residual=residuals[:, 1],
            target_residual=residuals[:, 2:], inference_main_q=inference_q,
            target_raw=design.targets_v, iqr=design.iqr, support=support,
            main_penalty=main_penalty, choices=choices, inner_losses=inner_losses,
            nuisance_rank=rank, main_columns=design.xt.shape[1],
            loadings=design.loadings(), models=fitted,
        )
        if not all(np.isfinite(v).all() for v in predictions.values()):
            raise ValueError("Nonfinite outer predictions")
        design.slim()
        self.log("FINISHED %s people=%d support=%.1f%%" % (tag, len(s.rows), 100 * support.mean()))
        return result

    def checkpoint(self, dataset, repeat, fold):
        def compute():
            start = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = self.outer(dataset, repeat, fold)
            value["warnings"] = sorted(set(str(w.message).encode("ascii", "backslashreplace").decode() for w in caught))
            value["seconds"] = time.monotonic() - start
            return value
        return self.memo.get("r%d_f%d_COMPLETE" % (repeat, fold), compute)
