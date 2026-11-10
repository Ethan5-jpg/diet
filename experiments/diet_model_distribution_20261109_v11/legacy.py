"""Read-only, provenance-checked reuse of V10 fold-matched INNER predictions."""

import json
import pickle
from pathlib import Path

import numpy as np

from bridge import V6, V8, V10, imported, sha


class Reference:
    def __init__(self, package, config, dataset, log=lambda _: None):
        self.source = Path(package) / "experiments" / V10.name
        self.root = self.source / "runs" / config["v10_reference_run"]
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
                raise ValueError("V10 cache scope/folds differ: " + key)
        if json.loads(summary.read_text()).get("status") != "COMPLETE":
            raise ValueError("V10 cache must belong to a completed run")
        required = [
            self.source / name
            for name in (
                "config.json",
                "bridge.py",
                "context.py",
                "features.py",
                "models.py",
                "estimators.py",
                "engine.py",
                "registry.py",
                "fusion.py",
            )
        ]
        required += list(dataset.paths)
        required += [V6 / name for name in ("data.py", "numerics.py", "storage.py")]
        required += [V8 / "features.py"]
        for path in dict.fromkeys(required):
            if str(path) not in stored["hashes"] or sha(path) != stored["hashes"][str(path)]:
                raise ValueError("V10 cache source/data fingerprint differs: " + str(path))
        registry = imported("hpp_v10_cache_registry", self.source / "registry.py")
        self.members = registry.pools(self.config)["diverse"]
        self.specs = {s["name"]: s for s in registry.specifications(self.config)}
        if (
            self.members != config["legacy_members"]
            or self.config["stack_penalty"] != config["legacy_stack_penalty"]
        ):
            raise ValueError("V10 bank recipe differs from the declared replay")
        self.paths = list(dict.fromkeys([registration, summary] + required))
        self.units = {}
        for r in range(2):
            for f in range(5):
                tag = "r%d_f%d" % (r, f)
                self.units[tag + "_result"] = self.receipt(tag + "_result")
                for name in self.members:
                    for k in range(3):
                        for j in range(len(self.specs[name]["grid"])):
                            key = tag + "_i%d_%s_c%d" % (k, name, j)
                            self.units[key] = self.receipt(key)
                log("V10_CACHE_CHECK %s completed" % tag)

    def receipt(self, key):
        folder = self.root / "TRE_ONLY" / key
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Required V10 cache unit missing: " + key)
        valid = []
        for receipt in sorted(folder.glob("done_*.json")):
            try:
                record = json.loads(receipt.read_text())
            except json.JSONDecodeError:
                continue
            path = folder / record["file"]
            if (
                Path(record["file"]).name != record["file"]
                or path.is_symlink()
                or sha(path) != record["sha256"]
            ):
                raise ValueError("Invalid V10 cache receipt: " + key)
            self.paths += [receipt, path]
            valid.append((path, record["sha256"]))
        if len(valid) != 1:
            raise ValueError("Expected one completed V10 cache payload: " + key)
        return valid[0]

    def read(self, key):
        path, digest = self.units[key]
        payload = path.read_bytes()
        import hashlib

        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("V10 cache changed during use")
        return pickle.loads(payload)

    def inner(self, repeat, fold, roles):
        """No outer result is unpickled here; order follows the frozen V10 engine."""
        target = np.concatenate([role.split.yv for role in roles])
        tag, out, choices = "r%d_f%d" % (repeat, fold), {}, {}
        for name in self.members:
            matrices = []
            for k, role in enumerate(roles):
                predictions = []
                for j in range(len(self.specs[name]["grid"])):
                    value = np.asarray(self.read(tag + "_i%d_%s_c%d" % (k, name, j))["prediction"])
                    if value.shape != role.split.yv.shape or not np.isfinite(value).all():
                        raise ValueError("V10 inner prediction rows differ: " + name)
                    predictions.append(value)
                matrices.append(np.column_stack(predictions))
            matrix = np.concatenate(matrices)
            chosen = int(np.mean((matrix - target[:, None]) ** 2, axis=0).argmin())
            out["old__" + name], choices[name] = matrix[:, chosen], chosen
        return out, choices

    def get(self, repeat, fold, rows, y, choices, role=None):
        """Called only after all new choices and fusion fits are fixed."""
        value = self.read("r%d_f%d_result" % (repeat, fold))
        if not np.array_equal(value["rows"], rows) or not np.array_equal(value["y"], y):
            raise ValueError("V10 outer rows/processed targets differ")
        if any(value["selected"].get(n) != j for n, j in choices.items()):
            raise ValueError("Reconstructed V10 inner selections do not match the completed run")
        predictions = {"old__" + n: value["predictions"][n] for n in self.members}
        predictions.update(
            v8_ridge_wide=value["predictions"]["v8_ridge_wide"],
            v10_ridge_diverse=value["predictions"]["ridge_diverse"],
        )
        if not all(np.asarray(p).shape == y.shape and np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Invalid V10 outer predictor")
        notes = {"old__" + n: value["outer_diagnostics"].get(n, {}) for n in self.members}
        return predictions, notes
