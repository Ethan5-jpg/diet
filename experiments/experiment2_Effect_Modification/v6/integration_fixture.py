"""Generated, non-HPP person-level sources for V6 integration checks.

This module creates no files and reads no native participant data.  It mirrors
the frozen discovery adapter's household roles so that preparation, nested
selection and resume checks can run locally.  Synthetic numbers must never be
described as an HPP model result.
"""

from copy import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd


PANEL_TAXA = (
    "k__Bacteria|g__Ruminococcus|s__Ruminococcus_bromii",
    "k__Bacteria|g__Bifidobacterium|s__Bifidobacterium_adolescentis",
    "k__Bacteria|g__Eubacterium|s__Eubacterium_rectale",
)


class Generated:
    """Small PersonSource-compatible, reproducible two-repeat discovery source.

    ``effect`` sets a planted fiber-by-gut-latent interaction.  It is not a
    tuning parameter of the delivered native experiment.  Every household has
    two people; the two outer splits use independently shuffled households.
    ``with_y`` replaces outcomes while preserving predictors and all folds.
    """

    scope = "generated data only"

    def __init__(self, n=180, seed=20261118, effect=5.0, noise=0.6):
        if (not isinstance(n, int) or n < 90 or n % 2
                or not np.isfinite(effect) or not np.isfinite(noise) or noise < 0):
            raise ValueError("INVALID_GENERATED_SOURCE_CONFIG")
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.groups = np.array([f"synthetic_household_{j // 2:04d}" for j in range(n)])
        self.index = pd.DataFrame({
            "participant_id": np.arange(900000, 900000 + n, dtype=np.int64),
            "split_unit_id": self.groups,
        })
        self.fiber = rng.uniform(7.0, 24.0, n)
        self.age = rng.uniform(20.0, 75.0, n)
        self.sex = rng.integers(0, 2, n).astype(float)
        latent = rng.normal(size=n)
        self.latent = latent
        self.raw, self.taxa = {}, {}
        for site, width in (("oral", 48), ("gut", 64)):
            logits = rng.normal(0.0, 0.6, (n, width))
            if site == "gut":
                loading = np.linspace(-0.8, 0.8, width)
                logits += latent[:, None] * loading
            else:
                logits += rng.normal(size=(n, 1)) * np.linspace(-0.4, 0.4, width)
            mass = np.exp(logits - logits.max(axis=1, keepdims=True))
            self.raw[site] = mass / mass.sum(axis=1, keepdims=True)
            labels = [f"s__synthetic_{site}_{j:03d}" for j in range(width)]
            if site == "gut":
                labels[:3] = PANEL_TAXA
            self.taxa[site] = labels
        u = (self.fiber - 12.5) / 5.0
        age = (self.age - 45.0) / 15.0
        self.y = (100.0 + 2.0 * u + 0.5 * u ** 2 + 1.2 * age
                  + 0.3 * self.sex + 2.0 * latent + effect * u * latent
                  + rng.normal(0.0, noise, n))
        outer = np.empty((2, n), dtype=int)
        for repeat in range(2):
            household_assignment = np.empty(n // 2, dtype=int)
            order = np.random.default_rng(seed + repeat + 1).permutation(n // 2)
            household_assignment[order] = np.arange(n // 2) % 5
            outer[repeat] = np.repeat(household_assignment, 2)
        self.reference = {"y": np.tile(self.y, (2, 1)), "outer": outer}
        self.extra = None
        self.source_hashes = {}
        self.historical_scores = (0.13, 0.14)  # Shape-only synthetic reference.
        self.verify_calls = 0

    def verify_sources(self):
        """The generated fixture has no on-disk sources to re-hash."""
        self.verify_calls += 1

    def with_y(self, outcome):
        """Return a predictor/fold-identical source with replaced finite outcomes."""
        outcome = np.asarray(outcome, dtype=float)
        if outcome.shape != self.y.shape or not np.isfinite(outcome).all():
            raise ValueError("INVALID_GENERATED_OUTCOME")
        result = copy(self)
        result.y = outcome.copy()
        result.reference = {"y": np.tile(outcome, (2, 1)),
                            "outer": self.reference["outer"].copy()}
        return result

    def split(self, train, valid, contrast=(10.0, 15.0)):
        """Train-fit numeric transforms without looking at validation targets."""
        train, valid = np.asarray(train), np.asarray(valid)
        if (train.shape != self.y.shape or valid.shape != self.y.shape
                or train.dtype != bool or valid.dtype != bool
                or not train.any() or not valid.any() or (train & valid).any()
                or np.intersect1d(self.groups[train], self.groups[valid]).size):
            raise ValueError("INVALID_GENERATED_HOUSEHOLD_ROLE")
        t, v = np.flatnonzero(train), np.flatnonzero(valid)
        basis = np.column_stack((self.fiber, self.fiber ** 2))
        center, scale = basis[t].mean(axis=0), basis[t].std(axis=0)
        dose = (basis - center) / scale
        age = (self.age - self.age[t].mean()) / self.age[t].std()
        host = np.column_stack((dose, age, self.sex))
        low, high = contrast
        delta = (np.array([high, high ** 2]) - np.array([low, low ** 2])) / scale
        matrices = {}
        for site in ("oral", "gut"):
            log = np.log(self.raw[site])
            clr = log - log.mean(axis=1, keepdims=True)
            location = clr[t].mean(axis=0)
            deviation = clr[t].std(axis=0)
            deviation = np.where(deviation > 1e-10, deviation, 1.0)
            matrices[site] = ((clr[t] - location) / deviation,
                              (clr[v] - location) / deviation)
        return SimpleNamespace(
            train_rows=t, rows=v, yt=self.y[t].copy(), yv=self.y[v].copy(),
            ht=host[t], hv=host[v], dt=dose[t], dv=dose[v], delta=delta,
            ot=matrices["oral"][0], ov=matrices["oral"][1],
            gt=matrices["gut"][0], gv=matrices["gut"][1],
            preprocessing={"host": {"labels": ["fiber_density", "fiber_density_sq", "age", "sex"]}},
        )

    def outer_split(self, repeat, fold):
        if repeat not in range(2) or fold not in range(5):
            raise ValueError("INVALID_GENERATED_OUTER_FOLD")
        valid = self.reference["outer"][repeat] == fold
        return self.split(~valid, valid)

    def roles(self, repeat, fold):
        outer = self.outer_split(repeat, fold)
        train = self.reference["outer"][repeat] != fold
        groups = np.unique(self.groups[train])
        order = np.random.default_rng(self.seed + 100 * repeat + fold + 11).permutation(len(groups))
        assignments = np.empty(len(groups), dtype=int)
        assignments[order] = np.arange(len(groups)) % 3
        mapping = dict(zip(groups, assignments, strict=True))
        inner = np.array([mapping.get(group, -1) for group in self.groups])
        roles = tuple(self.split(train & (inner != k), train & (inner == k)) for k in range(3))
        return roles, outer


def make_generated(**kwargs):
    """Named entry point for preflight; does not construct native HPP paths."""
    return Generated(**kwargs)


def make_native_layout(package, archive=None):
    """Build a fake TRE layout using archived loader code and generated data.

    This optional local development check needs the full local code archive;
    it is deliberately excluded from the upload's default preflight.  It
    copies archived Python files into ``package`` and creates synthetic
    parquet/NPZ inputs.  Nothing is copied from native participant datasets.
    The returned path is the fake package's V6 directory.
    """
    from adapters import person_inputs

    package = Path(package).resolve()
    archive = Path(archive).resolve() if archive is not None else Path(__file__).resolve().parents[2]
    original_v6 = Path(__file__).resolve().parent
    original_v1 = archive / "experiment2_Effect_Modification/v1"
    script = original_v1 / "exp2_native_fixture.py"
    if not script.is_file():
        raise FileNotFoundError("LOCAL_ARCHIVED_FIXTURE_UNAVAILABLE")
    before_path = sys.path[:]
    names = ("exp2_inputs", "exp2_store")
    before_modules = {name: sys.modules.pop(name, None) for name in names}
    try:
        sys.path.insert(0, str(original_v1))
        spec = importlib.util.spec_from_file_location("hpp_v6_generated_native_fixture", script)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        # The legacy fixture copies its own directory.  Keep this local-only
        # source staging small: archived screenshots are neither dependencies
        # nor test inputs, and may be cloud placeholders on a developer Mac.
        with tempfile.TemporaryDirectory(prefix="hpp_v6_fixture_code_") as staged:
            staged_root = Path(staged) / "experiment2_Effect_Modification/v1"
            staged_root.mkdir(parents=True)
            # make() only needs this file in its ROOT; its copied experiment
            # is unused because V6 itself is copied below.  Avoid downloading
            # unrelated cloud-backed archived V1 source modules.
            source = original_v1 / "dependency_manifest.json"
            manifest = json.loads(source.read_text(encoding="utf-8"))
            needed = {
                f"{person_inputs.V4}/run_nested.py",
                f"{person_inputs.V3}/run_screen.py",
                f"{person_inputs.V6}/data.py",
                f"{person_inputs.V6}/numerics.py",
                f"{person_inputs.V10}/context.py",
            }
            manifest["experiments"] = {name: value for name, value in manifest["experiments"].items()
                                       if name in needed}
            (staged_root / source.name).write_text(json.dumps(manifest), encoding="utf-8")
            fixture.ROOT = staged_root
            context = person_inputs.imported("hpp_v6_fixture_context_only",
                                             archive / "experiments" / person_inputs.V10 / "context.py")
            fixture.adapters = lambda: (SimpleNamespace(imported=person_inputs.imported, Context=context), None)
            fixture.make(package)
    finally:
        sys.path[:] = before_path
        for name, previous in before_modules.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous

    for relative in (
        f"experiments/{person_inputs.V6}/data.py",
        f"experiments/{person_inputs.V6}/numerics.py",
        f"experiments/{person_inputs.V10}/context.py",
    ):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive / relative, destination)
    taxa_path = package / "inputs/discovery_v2/gut_taxa.json"
    taxa = json.loads(taxa_path.read_text(encoding="utf-8"))
    taxa[:3] = PANEL_TAXA
    taxa_path.write_text(json.dumps(taxa, ensure_ascii=False), encoding="utf-8")
    fake_v6 = package / "experiment2_Effect_Modification/v6"
    shutil.copytree(original_v6, fake_v6, ignore=shutil.ignore_patterns(
        "__pycache__", "runs_TRE_ONLY", "runs", "*.log", "*.zip"))
    raw, context, helpers = person_inputs.registered_sources(package)
    config = dict(version="v11_20261109", exposure="fiber_density", outcome="D_Glu",
                  confirmation_outcomes_read=False, outer_seed=person_inputs.OUTER_SEED,
                  outer_repeats=2, outer_folds=5, inner_folds=3, contrast=[10, 15])
    registration = dict(config=config, hashes={str(p): person_inputs.digest(p)
                                               for p in (*raw, *context, *helpers)},
                        fixture_scope="generated participants and outcomes only")
    report = dict(status="COMPLETE", confirmation_outcomes_read=False,
                  scores=[dict(model="positive_combined", repeat=r, N=3343, R2=value)
                          for r, value in enumerate((0.13, 0.14))],
                  fixture_scope="generated participants and outcomes only")
    run = package / "experiments" / person_inputs.V11 / "runs" / person_inputs.V11_RUN
    run.mkdir(parents=True)
    for name, content in (("registration.json", registration), ("summary.json", report)):
        (run / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return fake_v6
