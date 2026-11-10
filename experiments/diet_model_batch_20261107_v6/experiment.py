"""One shared nested-CV engine for six method groups."""

import itertools

import numpy as np

from learners import apply_basis, blend_weights, supervised_basis, tree_fit
from numerics import (
    KernelProfile,
    interaction_contrast,
    interaction_kernel,
    site_kernel,
)

INTERACTIONS = ("kernel_linear", "kernel_spline", "kernel_supervised")
REFERENCES = ("v4_hgb_oral", "v4_hgb_oral_gut")


def model_names(config):
    trees = [
        algorithm + "_" + site
        for algorithm in config["tree_grids"]
        for site in ("oral", "oral_gut")
    ]
    return ["kernel_main"] + trees + list(INTERACTIONS) + ["fusion"]


def cases(name, config):
    if name == "kernel_main":
        return [
            dict(kind=k, gut_weight=w, penalty=p)
            for k, w, p in itertools.product(
                config["kernels"], config["gut_weights"], config["main_penalties"]
            )
        ]
    if name in INTERACTIONS:
        if name == "kernel_supervised":
            return [dict(components=0, weight=0.0)] + [
                dict(components=k, weight=w)
                for k in config["pls_components"]
                for w in config["interaction_weights"]
                if w > 0
            ]
        return [
            dict(form=name.split("_", 1)[1], weight=w)
            for w in config["interaction_weights"]
        ]
    algorithm, _ = name.split("_", 1)
    return [
        dict(top_k=k, settings=setting)
        for k in config["top_k"]
        for setting in config["tree_grids"][algorithm]
    ]


class Engine:
    def __init__(self, config, memo, threads):
        self.config, self.memo, self.threads = config, memo, threads
        self.local = {}

    def remember(self, key, compute):
        if key not in self.local:
            self.local[key] = self.memo.get(key, compute)
        return self.local[key]

    def profile(self, split, kt, kv):
        return KernelProfile(split.ht, split.hv, kt, kv, split.yt)

    def main_fit(self, split, tag, index, outer):
        case = cases("kernel_main", self.config)[index]

        def compute():
            oral, gut = split.kernels(case["kind"], self.config["model_seed"])
            solver = self.profile(
                split,
                oral[0] + case["gut_weight"] * gut[0],
                oral[1] + case["gut_weight"] * gut[1],
            )
            return (
                solver.fit(case["penalty"])
                if outer
                else solver.path(self.config["main_penalties"])
            )

        if outer:
            result = self.remember(tag + "_main_%d" % index, compute)
        else:
            count = len(self.config["main_penalties"])
            group, within = divmod(index, count)
            result = self.remember(
                tag + "_main_group%d" % group,
                compute,
            )[within]
        return result

    def set_base(self, split, tag, index, outer):
        case = cases("kernel_main", self.config)[index]
        fitted = self.main_fit(split, tag, index, outer)
        oral, gut = split.kernels(case["kind"], self.config["model_seed"])
        split.base = dict(
            case=case,
            fitted=fitted,
            oral_state=oral[2],
            gut_state=gut[2],
            kt=oral[0] + case["gut_weight"] * gut[0],
            kv=oral[1] + case["gut_weight"] * gut[1],
            kg=gut[0],
            kgv=gut[1],
        )
        split.cache = {case["kind"]: split.cache[case["kind"]]}

    def inter_fit(self, split, tag, case, name, outer):
        base, weight = split.base, case["weight"]
        if weight == 0:
            return dict(
                prediction=base["fitted"]["prediction"],
                modifier=np.zeros(len(split.yv)),
                effective_weight=0.0,
            )
        kg, kgv, form, basis = base["kg"], base["kgv"], case.get("form", "linear"), None
        if name == "kernel_supervised":
            components = case["components"]
            fitted = base["fitted"]
            residual = (
                split.yt - split.ht @ fitted["host_coef"] - base["kt"] @ fitted["alpha"]
            )
            basis = self.remember(
                tag + "_basis%d" % components,
                lambda: supervised_basis(split.gt, split.dt, residual, components),
            )
            cache_key = ("guided", components)
            if cache_key not in split.cache:
                split.cache[cache_key] = site_kernel(
                    apply_basis(split.gt, basis),
                    apply_basis(split.gv, basis),
                    "linear",
                    self.config["model_seed"],
                )
            kg, kgv, _ = split.cache[cache_key]
        cache_key = ("interaction", name, case.get("components", 0))
        if cache_key not in split.cache:
            split.cache[cache_key] = interaction_kernel(
                kg, kgv, split.dt, split.dv, form
            )
        ki, kiv, state = split.cache[cache_key]
        if np.trace(ki) < 1e-12:
            return dict(
                prediction=base["fitted"]["prediction"],
                modifier=np.zeros(len(split.yv)),
                effective_weight=0.0,
            )
        fit = self.profile(
            split, base["kt"] + weight * ki, base["kv"] + weight * kiv
        ).fit(base["case"]["penalty"])
        if not outer:
            return dict(prediction=fit["prediction"])
        training_axis = interaction_contrast(
            kg, state, split.delta, fit["alpha"], weight
        )
        axis = (
            interaction_contrast(kgv, state, split.delta, fit["alpha"], weight)
            - training_axis.mean()
        )
        return dict(
            **fit,
            modifier=axis,
            effective_weight=weight,
            basis=basis,
            interaction_state=state,
        )

    def fit(self, name, split, tag, index, outer=False):
        if name == "kernel_main":
            return self.main_fit(split, tag, index, outer)
        case = cases(name, self.config)[index]
        key = tag + "_" + name + "_%d" % index
        if name in INTERACTIONS:
            return self.remember(
                key, lambda: self.inter_fit(split, tag, case, name, outer)
            )
        algorithm, site = name.split("_", 1)
        zt, zv = split.microbes(site)
        return self.remember(
            key,
            lambda: tree_fit(
                algorithm,
                split.ht,
                split.hv,
                zt,
                zv,
                split.yt,
                case,
                self.config["model_seed"],
                self.threads,
                outer,
            ),
        )

    def outer(self, dataset, repeat, fold):
        cfg, x = self.config, dataset.index
        groups = x.split_unit_id.astype(str).to_numpy()
        assignment = dataset.reference["outer"][repeat]
        train, valid = assignment != fold, assignment == fold
        dataset.api.separated(groups, train, valid)
        inner = np.full(len(x), -1)
        inner[train] = dataset.api.folds(
            groups[train],
            cfg["inner_folds"],
            cfg["outer_seed"] + 100 * repeat + fold + 1,
        )
        tag = "r%d_f%d" % (repeat, fold)
        splits = [
            dataset.split(train & (inner != k), train & (inner == k), cfg["contrast"])
            for k in range(cfg["inner_folds"])
        ]
        selected, oof, losses = {}, {}, {}
        for name in model_names(cfg)[:-1]:
            predictions, errors = [], []
            for k, split in enumerate(splits):
                fitted = [
                    self.fit(name, split, tag + "_i%d" % k, j)
                    for j in range(len(cases(name, cfg)))
                ]
                matrix = np.column_stack([fit["prediction"] for fit in fitted])
                if not np.isfinite(matrix).all():
                    raise ValueError("Nonfinite inner prediction")
                predictions.append(matrix)
                errors.append(np.mean((split.yv[:, None] - matrix) ** 2, 0))
            loss = np.mean(errors, axis=0)
            index = int(loss.argmin())
            selected[name], losses[name] = index, loss.tolist()
            oof[name] = np.concatenate([pred[:, index] for pred in predictions])
            if name == "kernel_main":
                for k, split in enumerate(splits):
                    self.set_base(split, tag + "_i%d" % k, index, False)
            for split in splits:
                for key in list(split.cache):
                    if isinstance(key, tuple):
                        del split.cache[key]
        target = np.concatenate([split.yv for split in splits])
        members = cfg["blend_members"]
        weight = self.remember(
            tag + "_fusion",
            lambda: blend_weights(np.column_stack([oof[m] for m in members]), target),
        )
        # Only after all choices are fixed do we prepare the outer validation role.
        del splits
        outer = dataset.split(train, valid, cfg["contrast"])
        rows = np.flatnonzero(valid & np.isfinite(dataset.reference["y"][repeat]))
        if not np.array_equal(outer.rows, rows) or not np.array_equal(
            outer.yv, dataset.reference["y"][repeat, rows]
        ):
            raise ValueError(
                "Outer targets or eligible rows differ from the matched v4 reference"
            )
        self.remember(tag + "_outer_inputs", outer.snapshot)
        self.set_base(outer, tag + "_o", selected["kernel_main"], True)
        predictions, axes, effective = {}, {}, {}
        for name, index in selected.items():
            fitted = self.fit(name, outer, tag + "_o", index, True)
            predictions[name] = fitted["prediction"]
            if name in INTERACTIONS:
                axes[name], effective[name] = (
                    fitted["modifier"],
                    fitted["effective_weight"],
                )
        predictions["fusion"] = (
            np.column_stack([predictions[m] for m in members]) @ weight
        )
        predictions.update(
            {name: dataset.reference[name][repeat, rows] for name in REFERENCES}
        )
        if not all(np.isfinite(value).all() for value in predictions.values()):
            raise ValueError("Nonfinite outer prediction")
        return dict(
            rows=rows,
            y=outer.yv,
            predictions=predictions,
            modifiers=axes,
            effective_weights=effective,
            selected=selected,
            inner_losses=losses,
            fusion_weights=weight.tolist(),
            validation_households=len(set(groups[rows])),
        )
