"""Refit the registered FoodID calibration on each inner training household set."""

from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

import protocol as p
from source import check, keys, numbers


LOWER = "UPF_lower_equalday_fraction"
UPPER = "UPF_upper_equalday_fraction"
FOOD_FILES = ("discovery_covariates_food_TRE_ONLY.parquet", "reference_food_presence_TRE_ONLY.parquet",
              "reference_food_labels_TRE_ONLY.parquet", "unknown_food_categories_TRE_ONLY.parquet",
              "unknown_weights_TRE_ONLY.npz")


def spline(x, knots):
    a, b, c = knots
    check(a < b < c, "V6_UPF_NONDISTINCT_KNOTS")
    return (np.maximum(x - a, 0.)**3 - (c - a)/(c - b)*np.maximum(x - b, 0.)**3
            + (b - a)/(c - b)*np.maximum(x - c, 0.)**3)/(c - a)**2


@dataclass(frozen=True)
class Calibration:
    lower: np.ndarray
    upper: np.ndarray
    qualified: np.ndarray
    weights: object
    presence: object
    reference: object
    unknown: object
    group_draws: object

    def refit(self, training, seed):
        used = set(self.presence.loc[self.presence.input_row.isin(np.flatnonzero(training)), "food_id"])
        ref = self.reference.loc[self.reference.food_id.isin(used)]
        check(len(ref) > 50, "V6_UPF_INSUFFICIENT_REFERENCE_LABELS")
        importance = 1. / ref.groupby("label_group").food_id.transform("size")
        q, _ = self.group_draws(ref.label_group, ref.category, ref.y, importance, self.unknown.category,
                                draws=1024, seed=seed, shrinkage=10)
        labels = np.random.default_rng(seed + 1).random(q.shape) < q
        lo = np.nan_to_num(self.lower, nan=0.)
        latent = lo[None, :] + (self.weights @ labels.T.astype(float)).T
        expected = lo + self.weights @ q.mean(axis=0)
        valid = self.qualified
        check(np.all(expected[valid] >= lo[valid] - 1e-8)
              and np.all(expected[valid] <= self.upper[valid] + 1e-8), "V6_UPF_EXPECTATION_BOUNDS")
        knots = np.quantile(latent[:, valid & training], [.1, .5, .9])
        center = float(knots[1])
        scale = .1
        normalized = (knots - center) / scale
        values = np.column_stack(((expected - center) / scale, spline((latent - center) / scale, normalized).mean(axis=0)))
        return values, {"center": center, "scale": scale, "knots": normalized.tolist()}, len(ref)

    def inner(self, context, fold, seed):
        old = context.backgrounds[fold]
        values, calibration, n_foods = self.refit(old.role, seed)
        context.backgrounds[fold] = replace(old, archived={**old.archived, "UPF": values}, upf=calibration)
        return context, n_foods


def load(root, study):
    home = Path(root).parent
    directory = home / "data_qc/w20_simplified_20261101_v2"
    code = home / "scripts/preprocessing_20261030_v1/upf_catalogue_calibration_v2.py"
    spec = importlib.util.spec_from_file_location("full_pheno_upf_catalogue", code)
    module = importlib.util.module_from_spec(spec)
    # The original file has a sibling import alongside the calibrated implementation.
    import sys
    sys.path.insert(0, str(code.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    paths = tuple(directory / name for name in FOOD_FILES)
    index = pq.read_table(paths[0], columns=[*p.KEY, "input_row", "final_role", LOWER, UPPER])
    check(all(np.array_equal(a, b) for a, b in zip(keys(index), study.keys, strict=True))
          and np.array_equal(index["input_row"].to_numpy(), np.arange(study.n))
          and set(index["final_role"].to_pylist()) == {"discovery"}, "V6_UPF_REGISTRY_ORDER")
    lower, upper = numbers(index, LOWER), numbers(index, UPPER)
    with np.load(paths[-1], allow_pickle=False) as data:
        weights = csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
    support = study.numeric["n_energy_days"] >= 3
    check(weights.shape[0] == study.n and np.isfinite(weights.data).all() and np.all(weights.data >= 0.)
          and np.allclose(np.asarray(weights.sum(axis=1)).ravel()[support], (upper-lower)[support], atol=1e-8),
          "V6_UPF_GEOMETRY_MISMATCH")
    presence, references, unknown = (pq.read_table(path).to_pandas(ignore_metadata=True) for path in paths[1:-1])
    check(presence.input_row.between(0, study.n-1).all() and references.food_id.is_unique
          and len(unknown) == weights.shape[1], "V6_UPF_CATALOGUE_MISMATCH")
    return Calibration(lower, upper, support, weights, presence, references, unknown,
                       module.grouped_probability_draws), (*paths, code, code.with_name("upf_catalogue_calibration.py"))
