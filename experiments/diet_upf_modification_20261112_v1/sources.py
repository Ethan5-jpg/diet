"""Verified discovery inputs and training-only FoodID probability calibration."""

from dataclasses import dataclass
import importlib
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse


LOWER = "UPF_lower_equalday_fraction"
UPPER = "UPF_upper_equalday_fraction"


def checked_import(directory, digests, names, sha):
    for filename, digest in digests.items():
        if sha(directory / filename) != digest:
            raise ValueError(f"Dependency fingerprint differs: {filename}")
    sys.path.insert(0, str(directory))
    try:
        modules = [importlib.import_module(name) for name in names]
    finally:
        sys.path.pop(0)
    if any(directory.resolve() != Path(module.__file__).resolve().parent for module in modules):
        raise ImportError("Unexpected dependency")
    return modules


def parquet(path, columns=None):
    return pq.ParquetFile(path, pre_buffer=False).read(columns=columns, use_threads=False)


def as_numpy(path, key):
    with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        return archive[key]


@dataclass
class Cohort:
    index: object
    rows: np.ndarray
    y: np.ndarray
    folds: np.ndarray
    raw: dict
    taxa: dict
    lower: np.ndarray
    upper: np.ndarray
    weights: object
    presence: object
    reference: object
    unknown: object
    context: np.ndarray
    provenance: dict


def read(package, inputs, context, food_code, base_code):
    check = inputs.require
    h = package / "inputs/discovery_v2"
    legacy = package.parent / "data_qc/w20_simplified_20261101_v2"
    out = package / "experiment2_Effect_Modification/v1/runs_TRE_ONLY" / inputs.EXP2
    receipt = out / "registration.json"
    complete = inputs.json_read(out / "COMPLETE.json")
    check(inputs.digest(receipt) == complete["registration"], "Experiment 2 registration hash differs")
    anchors = inputs.json_read(receipt)["sources"]
    x_path = h / "discovery_covariates_food_TRE_ONLY.parquet"
    y_path = h / "D_Glu_raw_Y_TRE_ONLY.parquet"
    split_path = package / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    raw_files = {s: h / f"{s}_raw_TRE_ONLY.npz" for s in ("gut", "oral")}
    tax_files = {s: h / f"{s}_taxa.json" for s in raw_files}
    frozen_paths = [x_path, y_path, split_path, *raw_files.values(), *tax_files.values()]
    source_hashes = {str(path): inputs.digest(path) for path in frozen_paths}
    check(all(anchors.get(path) == sha for path, sha in source_hashes.items()), "Frozen input fingerprint differs")
    cols = inputs.INDEX + [LOWER, UPPER, "n_energy_days", "n_fiber_days", "fiber_density",
                           "mean_daily_energy_UPF", "smoking_status", "matrix_row_ordinal"]
    index = inputs.frame(parquet(x_path, cols))
    inputs.unique(index, inputs.KEY)
    check(len(index) == 4427 and index.final_role.eq("discovery").all()
          and np.array_equal(index.input_row, np.arange(len(index))), "Unexpected discovery index")
    with np.load(BytesIO(split_path.read_bytes()), allow_pickle=False) as data:
        eligible, folds = np.isfinite(data["y"]), data["outer"]
    check(eligible.shape == folds.shape == (2, len(index)) and np.array_equal(eligible[0], eligible[1]),
          "Unexpected two-repeat partition")
    check(eligible[0].sum() == 3343 and index.loc[eligible[0], "split_unit_id"].nunique() == 3059,
          "Frozen comparison cohort changed")
    lower, upper = (index[n].to_numpy(float) for n in (LOWER, UPPER))
    keep = eligible[0] & index.n_energy_days.ge(3).to_numpy() & np.isfinite(lower) & np.isfinite(upper)
    keep &= np.isfinite(index.mean_daily_energy_UPF.to_numpy(float))
    rows = np.flatnonzero(keep)
    check(len(rows) > 0 and (lower[keep] >= 0).all() and (upper[keep] <= 1 + 1e-8).all()
          and (upper[keep] >= lower[keep] - 1e-8).all(), "Invalid UPF interval")
    check(index.loc[keep, "sex_code"].isin([0, 1]).all()
          and np.isfinite(index.loc[keep, "age_calendar_years"].to_numpy(float)).all(),
          "Unexpected host fields")
    for rep in folds:
        check(set(rep[keep]) == set(range(5))
              and index.assign(fold=rep).groupby("split_unit_id").fold.nunique().eq(1).all(),
              "Household overlaps across folds")
    target = inputs.frame(parquet(y_path, inputs.KEY + ["input_row", "Y_unit", "Y_value"]))
    inputs.unique(target, inputs.KEY)
    check(set(target.Y_unit.dropna()) == {"mg/dL"}, "Unexpected glucose unit")
    y = index.loc[keep, inputs.KEY + ["input_row"]].merge(
        target, on=inputs.KEY + ["input_row"], how="left", validate="one_to_one")["Y_value"].to_numpy(float)
    check(np.isfinite(y).all(), "Unexpected missing raw glucose")
    old_x = legacy / x_path.name
    old = inputs.frame(parquet(old_x, inputs.KEY + ["input_row", "final_role", "matrix_row_ordinal", LOWER, UPPER]))
    check(old.final_role.eq("discovery").all()
          and old[inputs.KEY + ["input_row"]].equals(index[inputs.KEY + ["input_row"]])
          and np.allclose(old[[LOWER, UPPER, "matrix_row_ordinal"]],
                          index[[LOWER, UPPER, "matrix_row_ordinal"]], equal_nan=True),
          "UPF geometry and discovery rows disagree")
    w_path = legacy / "unknown_weights_TRE_ONLY.npz"
    with np.load(BytesIO(w_path.read_bytes()), allow_pickle=False) as a:
        matrix = sparse.csr_matrix((a["data"], a["indices"], a["indptr"]), shape=tuple(a["shape"]))
    check(matrix.shape[0] == len(index) and np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
          and np.allclose(np.asarray(matrix[rows].sum(axis=1)).ravel(), upper[keep] - lower[keep], atol=1e-8),
          "Unknown FoodID energy does not reconstruct the UPF interval")
    path_presence = legacy / "reference_food_presence_TRE_ONLY.parquet"
    path_ref = legacy / "reference_food_labels_TRE_ONLY.parquet"
    path_unknown = legacy / "unknown_food_categories_TRE_ONLY.parquet"
    presence = parquet(path_presence).to_pandas(ignore_metadata=True)
    reference = parquet(path_ref).to_pandas(ignore_metadata=True)
    unknown = parquet(path_unknown).to_pandas(ignore_metadata=True)
    check(presence.input_row.between(0, len(index) - 1).all() and reference.food_id.is_unique
          and len(unknown) == matrix.shape[1], "Unexpected food catalogue inputs")
    extra, extra_paths = context.read_context(package, index, eligible[0])
    # Preserve the source nutrient densities and add fiber in its original scale.
    additional = np.column_stack([index.fiber_density.to_numpy(float),
                                  extra["blocks"]["full"]])[rows]
    raw = {s: as_numpy(path, "abundance") for s, path in raw_files.items()}
    taxa = {s: inputs.json_read(path) for s, path in tax_files.items()}
    source_hashes.update({str(path): inputs.digest(path) for path in
                          [receipt, old_x, w_path, path_presence, path_ref, path_unknown, *extra_paths,
                           *food_code, *base_code]})
    return Cohort(index, rows, y, folds, raw, taxa, lower[keep], upper[keep], matrix[rows],
                  presence, reference, unknown, additional, source_hashes)


def integrate(lower, upper, weights, probabilities, seed):
    if (probabilities.ndim != 2 or probabilities.shape[1] != weights.shape[1]
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Invalid FoodID classification draws")
    labels = np.random.default_rng(seed).random(probabilities.shape) < probabilities
    draws = lower[None, :] + np.asarray(weights @ labels.T.astype(float)).T
    expected = lower + np.asarray(weights @ probabilities.mean(axis=0)).ravel()
    if (draws < lower[None, :] - 1e-8).any() or (draws > upper[None, :] + 1e-8).any():
        raise ValueError("Latent UPF draws outside documented bounds")
    return expected, draws


def calibration(data, training, group_draws, config, seed):
    used = set(data.presence.loc[data.presence.input_row.isin(np.flatnonzero(training)), "food_id"])
    ref = data.reference.loc[data.reference.food_id.isin(used)].copy()
    if len(ref) <= 50:
        raise ValueError("Insufficient training-food reference catalogue")
    weight = 1 / ref.groupby("label_group").food_id.transform("size")
    q, meta = group_draws(ref.label_group, ref.category, ref.y, weight, data.unknown.category,
                          draws=config["draws"], seed=seed, shrinkage=10)
    expected, draws = integrate(data.lower, data.upper, data.weights, q, seed + 1)
    return expected, draws, meta
