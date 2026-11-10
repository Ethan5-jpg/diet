"""Read-only V11/V10 inner-bank reuse; historical outer arrays are inference-only."""

import json
from pathlib import Path

import numpy as np

from bridge import V6, V8, V10, V11, V11Bank, imported, sha
from fusion import selected_linear


class Reference:
    def __init__(self, package, config, dataset, log=lambda _: None):
        self.source = Path(package) / "experiments" / V11.name
        self.root = self.source / "runs" / config["v11_reference_run"]
        registration, summary = self.root / "registration.json", self.root / "summary.json"
        stored = json.loads(registration.read_text())
        self.config = stored["config"]
        for key in (
            "outcome",
            "exposure",
            "outer_seed",
            "outer_repeats",
            "outer_folds",
            "inner_folds",
            "contrast",
            "confirmation_outcomes_read",
        ):
            if self.config.get(key) != config.get(key):
                raise ValueError("V11 scope/folds differ: " + key)
        if json.loads(summary.read_text()).get("status") != "COMPLETE":
            raise ValueError("V11 must be complete")
        required = [
            self.source / n
            for n in (
                "config.json",
                "bridge.py",
                "context.py",
                "food_features.py",
                "features.py",
                "models.py",
                "estimators.py",
                "engine.py",
                "registry.py",
                "fusion.py",
                "legacy.py",
            )
        ]
        required += (
            list(dataset.paths)
            + [V6 / n for n in ("data.py", "numerics.py", "storage.py")]
            + [V8 / "features.py", V10 / "context.py", V10 / "features.py"]
        )
        for path in dict.fromkeys(required):
            if str(path) not in stored["hashes"] or sha(path) != stored["hashes"][str(path)]:
                raise ValueError("V11 source/data fingerprint differs: " + str(path))
        registry = imported("hpp_v11_cache_registry", self.source / "registry.py")
        self.specs = {s["name"]: s for s in registry.specifications(self.config)}
        self.members = registry.pools(self.config)["combined"]
        if self.members != config["legacy_members"]:
            raise ValueError("V11 combined-bank recipe differs")
        self.old = V11Bank(package, self.config, dataset, log)
        self.paths = list(dict.fromkeys([registration, summary] + required + self.old.paths))
        self.units = {}
        for r in range(2):
            for f in range(5):
                tag = "r%d_f%d" % (r, f)
                self.units[tag + "_result"] = self.receipt(tag + "_result")
                for name, spec in self.specs.items():
                    for k in range(3):
                        for j in range(len(spec["grid"])):
                            key = tag + "_i%d_%s_c%d" % (k, name, j)
                            self.units[key] = self.receipt(key)
                log("V11_CACHE_CHECK %s completed" % tag)

    receipt = V11Bank.receipt
    read = V11Bank.read

    def inner(self, repeat, fold, roles):
        original, old_choices = self.old.inner(repeat, fold, roles)
        target = np.concatenate([r.split.yv for r in roles])
        groups = np.concatenate(
            [r.index.iloc[r.split.rows].split_unit_id.astype(str).to_numpy() for r in roles]
        )
        choices = {}
        tag = "r%d_f%d" % (repeat, fold)
        for name, spec in self.specs.items():
            pieces = []
            for k, role in enumerate(roles):
                cols = [
                    np.asarray(self.read(tag + "_i%d_%s_c%d" % (k, name, j))["prediction"])
                    for j in range(len(spec["grid"]))
                ]
                if any(p.shape != role.split.yv.shape or not np.isfinite(p).all() for p in cols):
                    raise ValueError("V11 inner rows differ: " + name)
                pieces.append(np.column_stack(cols))
            matrix = np.concatenate(pieces)
            chosen = int(np.mean((matrix - target[:, None]) ** 2, axis=0).argmin())
            choices[name] = chosen
            original[name] = matrix[:, chosen]
        members = ["bank__" + n for n in self.members]
        predictions = {"bank__" + n: original[n] for n in self.members}
        replay = selected_linear(
            members,
            np.column_stack([predictions[n] for n in members]),
            target,
            groups,
            "positive",
            self.config["meta_penalties"],
        )
        return dict(predictions=predictions, choices=dict(v10=old_choices, v11=choices), replay=replay)

    def get(self, repeat, fold, rows, y, choices, role=None):
        original, notes = self.old.get(repeat, fold, rows, y, choices["v10"])
        value = self.read("r%d_f%d_result" % (repeat, fold))
        if not np.array_equal(value["rows"], rows) or not np.array_equal(value["y"], y):
            raise ValueError("V11 outer rows/processed targets differ")
        if any(value["selected"].get(n) != j for n, j in choices["v11"].items()):
            raise ValueError("Reconstructed V11 inner choices differ")
        original.update({n: value["predictions"][n] for n in self.specs})
        notes.update({n: value["outer_diagnostics"].get(n, {}) for n in self.specs})
        predictions = {"bank__" + n: original[n] for n in self.members}
        predictions.update(
            v10_ridge_diverse=value["predictions"]["v10_ridge_diverse"],
            v11_positive_combined=value["predictions"]["positive_combined"],
            v11_sparse_combined=value["predictions"]["sparse_combined"],
        )
        if not all(np.asarray(p).shape == y.shape and np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Invalid prior outer prediction")
        return predictions, {"bank__" + n: notes.get(n, {}) for n in self.members}
