"""Read-only V13 INNER snapshots and original saved V11/V13 meta-model replay."""

import copy
import json
from pathlib import Path

import numpy as np

from bridge import V13, V13Reference, imported, sha


class Reference:
    receipt = V13Reference.receipt
    read = V13Reference.read

    def __init__(self, package, config, dataset, log=lambda _: None):
        self.old = V13Reference(package, config, dataset, log)
        self.source = Path(package) / "experiments" / V13.name
        self.root = self.source / "runs" / config["v13_reference_run"]
        if (self.root / "RUNNING.lock").exists():
            raise ValueError("V13 reference run still has a lock")
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
            "legacy_members",
        ):
            if self.config.get(key) != config.get(key):
                raise ValueError("V13 reference scope/recipe differs: " + key)
        if json.loads(summary.read_text()).get("status") != "COMPLETE":
            raise ValueError("V13 reference must be complete")
        required = sorted(self.source.glob("*.py")) + [
            self.source / "config.json",
            self.source / "requirements.txt",
        ]
        required += list(dataset.paths)
        for path in dict.fromkeys(required):
            if sha(path) != stored["hashes"].get(str(path)):
                raise ValueError("V13 source/data fingerprint differs: " + str(path))
        registry = imported("hpp_v13_cache_registry", self.source / "registry.py")
        self.specs = {s["name"]: s for s in registry.specifications(self.config)}
        if list(self.specs) != config["prior_members"]:
            raise ValueError("V13 selected INNER member recipe differs")
        # Checked again by normal identity() before fitting, including old registrations/receipts.
        inherited = list(dict.fromkeys(required + self.old.paths))
        if any(str(path) not in stored["hashes"] for path in inherited):
            raise ValueError("V13 registration omitted a required historical dependency")
        self.expected_hashes = {str(path): stored["hashes"][str(path)] for path in inherited}
        self.paths = [registration, summary] + inherited
        self.units = {}
        for r in range(2):
            for f in range(5):
                tag = "r%d_f%d" % (r, f)
                for suffix in ("_prior_inner", "_meta_fit", "_result"):
                    self.units[tag + suffix] = self.receipt(tag + suffix)
                for name, spec in self.specs.items():
                    for k in range(3):
                        for j in range(len(spec["grid"])):
                            key = tag + "_i%d_%s_c%d" % (k, name, j)
                            self.units[key] = self.receipt(key)
                log("V13_CACHE_CHECK %s completed" % tag)
        self.paths = list(dict.fromkeys(self.paths))

    def inner(self, repeat, fold, roles):
        tag = "r%d_f%d" % (repeat, fold)
        prior = self.read(tag + "_prior_inner")
        target = np.concatenate([r.split.yv for r in roles])
        old_names = ["bank__" + n for n in self.config["legacy_members"]]
        if list(prior["predictions"]) != old_names or prior["replay"].members != old_names:
            raise ValueError("Saved V11 INNER/replay member order differs")
        predictions = {n: np.asarray(prior["predictions"][n]) for n in old_names}
        if any(p.shape != target.shape or not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Saved prior INNER rows differ")
        choices = {}
        for name, spec in self.specs.items():
            pieces = []
            for k, role in enumerate(roles):
                cols = [
                    np.asarray(self.read(tag + "_i%d_%s_c%d" % (k, name, j))["prediction"])
                    for j in range(len(spec["grid"]))
                ]
                if any(p.shape != role.split.yv.shape or not np.isfinite(p).all() for p in cols):
                    raise ValueError("V13 INNER rows differ: " + name)
                pieces.append(np.column_stack(cols))
            matrix = np.concatenate(pieces)
            choices[name] = int(np.mean((matrix - target[:, None]) ** 2, axis=0).argmin())
            predictions["prior__" + name] = matrix[:, choices[name]]
        replay = self.read(tag + "_meta_fit")["positive_combined"]
        if replay.members != old_names + list(self.specs):
            raise ValueError("Saved V13 positive fusion recipe differs")
        replay = copy.deepcopy(replay)
        replay.members = [n if n.startswith("bank__") else "prior__" + n for n in replay.members]
        return dict(
            predictions=predictions,
            choices=dict(v11=prior["choices"], v13=choices),
            replay=dict(v11_replay=prior["replay"], v13_replay=replay),
        )

    def get(self, repeat, fold, rows, y, choices, role=None):
        predictions, notes = self.old.get(repeat, fold, rows, y, choices["v11"], role)
        value = self.read("r%d_f%d_result" % (repeat, fold))
        if not np.array_equal(value["rows"], rows) or not np.array_equal(value["y"], y):
            raise ValueError("V13 outer rows/processed targets differ")
        if value["selected"] != choices["v13"]:
            raise ValueError("Reconstructed V13 inner choices differ")
        for name in self.specs:
            predictions["prior__" + name] = value["predictions"][name]
            notes["prior__" + name] = value["outer_diagnostics"].get(name, {})
        predictions["v13_positive_combined"] = value["predictions"]["positive_combined"]
        if any(np.asarray(p).shape != y.shape or not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("Invalid historical outer prediction")
        return predictions, notes
