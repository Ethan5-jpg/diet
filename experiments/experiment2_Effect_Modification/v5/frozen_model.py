"""Freeze the discovery-fitted v4 fermentation model before reserved outcome access."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve
from sklearn.decomposition import PCA

from fermentation_score import build_score, resolve_species
from interactions import _product, _smooth_markers, fit_corrections
from models import BASE
from moderation_models import SEED, SITES, ecology, fit_base, residuals


ROUTE = "fermentation_smooth_curve@0.1"
PENALTY = 0.1
HOST_FIELDS = ("age_calendar_years", "sex_code", "mean_daily_energy_fiber")


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray

    @classmethod
    def fit(cls, values, *, retain_constant=False):
        mean, scale = values.mean(axis=0), values.std(axis=0)
        keep = scale > 1e-10
        if retain_constant:
            return cls(mean, np.where(keep, scale, 1.0), np.ones(len(scale), dtype=bool))
        return cls(mean[keep], scale[keep], keep)

    def apply(self, values):
        return (values[:, self.keep] - self.mean) / self.scale


@dataclass(frozen=True)
class SmoothCorrection:
    dose_center: np.ndarray
    delta: np.ndarray
    knots: np.ndarray
    smooth_center: np.ndarray
    marker_center: np.ndarray
    product_keep: np.ndarray
    product_scale: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class FrozenModel:
    training_rows: int
    training_households: int
    clr: dict
    dose: dict
    target: dict
    host: dict
    alcohol_levels: tuple
    pca: dict
    pc_scale: dict
    eco_scale: Standardization
    panel_taxa: tuple
    panel_scale: Standardization
    panel_constant: bool
    baseline: object
    correction: SmoothCorrection

    @property
    def delta(self):
        return self.correction.delta


def _diet_basis(fiber, state):
    # Exact archived core.rcs evaluation order and scale.
    u = (np.asarray(fiber, dtype=float) - state["center"]) / state["scale"]
    a, b, c = np.asarray(state["knots"], dtype=float)
    if not a < b < c:
        raise ValueError("INVALID_FROZEN_DIET_KNOTS")
    h = (np.maximum(u - a, 0) ** 3 - (c - a) / (c - b) * np.maximum(u - b, 0) ** 3
         + (b - a) / (c - b) * np.maximum(u - c, 0) ** 3) / (c - a) ** 2
    return np.column_stack((u, h))


def _target_values(target, y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or target.get("endpoint") != "D_Glu" or target.get("kind") != "hpp_endpoint_shortest95_8sd_5sd_v1":
        raise ValueError("INVALID_FROZEN_TARGET")
    limits = np.asarray([target.get(key, np.nan) for key in ("remove_low", "clip_low", "clip_high", "remove_high")])
    if not np.isfinite(limits).all() or np.any(np.diff(limits) <= 0):
        raise ValueError("INVALID_FROZEN_TARGET_LIMITS")
    remove_low, clip_low, clip_high, remove_high = limits
    allowed = np.isfinite(y) & (y >= remove_low) & (y <= remove_high)
    out = np.full(y.shape, np.nan)
    out[allowed] = np.clip(y[allowed], clip_low, clip_high)
    return out


def transform_target(state, y):
    """Apply discovery-fitted endpoint limits from a model or receipt dictionary."""
    return _target_values(state if isinstance(state, dict) else state.target, y)


def _fit_preprocessing(source, train):
    index = source.index
    y = np.asarray(source.y, dtype=float)
    train = np.ones(len(index), dtype=bool) if train is None else np.asarray(train)
    if train.shape != (len(index),) or train.dtype != np.dtype(bool) or not train.any() or y.shape != train.shape:
        raise ValueError("INVALID_FROZEN_DISCOVERY_MASK")
    # These three fitting sets match the archived V4 order and may differ.
    fiber = index.fiber_density.to_numpy(dtype=float, na_value=np.nan)
    diet_rows = train & index.n_fiber_days.ge(3).fillna(False).to_numpy(dtype=bool) & np.isfinite(fiber)
    raw_knots = np.quantile(fiber[diet_rows], (.1, .5, .9))
    center, scale = float(raw_knots[1]), 5.0
    dose_state = {"center": center, "scale": scale, "knots": (raw_knots - center) / scale}
    dose = _diet_basis(fiber, dose_state)

    finite_y = train & np.isfinite(y)
    target = source.api.PhenotypeOutliers().fit(y[finite_y], np.flatnonzero(finite_y).tolist(), endpoint="D_Glu")
    processed = target.transform(y, endpoint="D_Glu")["values"]
    supported = np.ones(len(index), dtype=bool)
    clr_states, z = {}, {}
    for site in SITES:
        fit = source.api.MicrobiomeCLR().fit(
            source.raw[site][train], source.taxa[site],
            index.loc[train, "participant_id"].tolist(), site=site,
        )
        transformed = fit.transform(source.raw[site], source.taxa[site], site=site)
        supported &= transformed["supported"]
        clr_states[site], z[site] = fit.state, transformed["values"]

    eligible = train & supported & np.isfinite(processed) & _host_eligible(index, dose)
    if np.count_nonzero(eligible) <= 100:
        raise ValueError("INSUFFICIENT_FROZEN_DISCOVERY_ROWS")
    return eligible, dose_state, dose, target.state, processed, clr_states, z


def training_rows(source, train=None):
    """Return the exact V4 eligible discovery positions for context rehydration."""
    return np.flatnonzero(_fit_preprocessing(source, train)[0])


def _host_eligible(index, dose):
    return (index.n_fiber_days.ge(3).fillna(False).to_numpy(dtype=bool)
            & np.isfinite(index[HOST_FIELDS[0]].to_numpy(dtype=float, na_value=np.nan))
            & index[HOST_FIELDS[1]].isin([0, 1]).to_numpy(dtype=bool)
            & np.isfinite(index[HOST_FIELDS[2]].to_numpy(dtype=float, na_value=np.nan))
            & np.isfinite(dose).all(axis=1))


def _replay_host(index, dose, state):
    numeric = index[list(HOST_FIELDS)].to_numpy(float)
    numeric = (numeric - state["numeric_means"]) / state["numeric_scales"]
    levels = state["smoking_levels"]
    category = index.smoking_status.astype(str).to_numpy()
    category = np.where(np.isin(category, levels), category, "__UNKNOWN__")
    u, h = dose.T
    columns = [np.ones(len(index)), numeric]
    columns.extend((category == level).astype(float) for level in levels[1:])
    columns.extend((u, h, u * numeric[:, 0], u * numeric[:, 1]))
    return np.column_stack(columns)[:, state["keep"]]


def _replay_clr(raw, state):
    selected = raw[:, state["reference_indices"]]
    supported = selected.sum(axis=1) > 0
    selected = selected[supported]
    positive = np.where(selected == 0, state["pseudocounts"], selected)
    logs = np.log(positive / positive.sum(axis=1, keepdims=True))
    clr = logs - logs.mean(axis=1, keepdims=True)
    values = (clr[:, state["output_indices"]] - state["means"]) / state["scales"]
    aligned = np.full((len(raw), values.shape[1]), np.nan)
    aligned[supported] = values
    return aligned, supported


def _context_levels(extra, rows):
    if extra is None:
        return ()
    alcohol = np.asarray(extra["alcohol"], dtype=object)
    return tuple(sorted({value for value in alcohol[rows] if isinstance(value, str) and value.strip()}))


def _context(extra, rows, levels):
    if extra is None:
        return np.empty((len(rows), 0))
    alcohol = np.asarray(extra["alcohol"], dtype=object)[rows]
    return np.column_stack([
        extra["blocks"]["full"][rows], extra["blocks"]["distribution"][rows],
        *((alcohol == level).astype(float) for level in levels),
    ])


def _fit_correction(dose, delta, marker, residual):
    knots = np.quantile(marker, (0.1, 0.5, 0.9), axis=0)[:, 0]
    if np.any(np.diff(knots) <= 1e-8):
        knots = np.empty(0)
    expanded, _ = _smooth_markers(marker, marker)
    smooth_center = _spline_tail(marker, knots).mean(axis=0)
    marker_center = expanded.mean(axis=0)
    # Match both historical dose centerings, including their floating-point order.
    dose_center = np.vstack((dose.mean(axis=0), (dose - dose.mean(axis=0)).mean(axis=0)))
    train = _product((dose - dose_center[0]) - dose_center[1], expanded - marker_center)
    scale = np.sqrt(np.mean(train ** 2, axis=0))
    keep = scale > 1e-10
    train = train[:, keep] / scale[keep]
    if keep.any():
        gram = train.T @ train / len(train)
        cross = train.T @ residual / len(train)
        beta = solve(gram + PENALTY * np.eye(len(cross)), cross, assume_a="pos")
    else:
        beta = np.empty(0)
    return SmoothCorrection(dose_center, delta, knots, smooth_center, marker_center, keep, scale[keep], beta)


def _spline_tail(marker, knots):
    if not len(knots):
        return np.empty((len(marker), 0))
    lo, mid, hi = knots
    u = (marker - lo) / (hi - lo)
    knot = (mid - lo) / (hi - lo)
    bounded = np.clip(u, 0, 1)
    curve = bounded ** 3 - np.maximum(bounded - knot, 0) ** 3 / (1 - knot)
    return curve + 3 * knot * np.maximum(u - 1, 0)


def _apply_correction(state, dose, marker):
    smooth = _spline_tail(marker, state.knots) - state.smooth_center
    basis = np.column_stack((marker, smooth)) - state.marker_center
    products = _product((dose - state.dose_center[0]) - state.dose_center[1], basis)
    contrasts = (state.delta[None, :, None] * basis[:, None, :]).reshape(products.shape)
    return tuple(part[:, state.product_keep] / state.product_scale @ state.beta
                 for part in (products, contrasts))


def fit_model(source, train=None):
    """Fit the fixed v4 baseline and @0.1 correction on discovery only.

    Predictor context stays as supplied on source.extra. The caller may rebuild
    it for the exact training_rows(source, train) before fitting.
    """
    mask, dose_state, dose, target, y, clr, z = _fit_preprocessing(source, train)
    rows = np.flatnonzero(mask)
    groups = np.asarray(source.groups)
    if groups.shape != mask.shape:
        raise ValueError("INVALID_FROZEN_DISCOVERY_HOUSEHOLDS")
    host, host_state = source.api.host_design(source.index, mask, *dose.T)
    levels = _context_levels(source.extra, rows)
    left = np.column_stack((host[rows], _context(source.extra, rows, levels)))

    pca, pc_scale, pcs = {}, {}, []
    for site in SITES:
        x = z[site][rows]
        k = min(12, x.shape[1], len(x) - 1)
        if k < 1:
            raise ValueError("INVALID_FROZEN_PCA_RANK")
        pca[site] = PCA(n_components=k, svd_solver="randomized", random_state=SEED).fit(x)
        projected = pca[site].transform(x)
        pc_scale[site] = Standardization.fit(projected)
        pcs.append(pc_scale[site].apply(projected))
    raw_eco = np.column_stack([ecology(source.raw[site][rows]) for site in SITES])
    eco_scale = Standardization.fit(raw_eco, retain_constant=True)
    score, matches, _ = build_score(source.raw["gut"][rows], source.taxa["gut"])
    if not np.isfinite(score).all():
        raise ValueError("INCOMPLETE_FROZEN_DISCOVERY_PANEL")
    score = score[:, None]
    panel_scale = Standardization.fit(score, retain_constant=True)
    panel_constant = bool(np.std(score) <= 1e-10)
    panel = np.zeros_like(score) if panel_constant else panel_scale.apply(score)
    design = np.column_stack((left, *pcs, eco_scale.apply(raw_eco), panel))
    shared = tuple(j for j, label in enumerate(host_state["labels"]) if label in ("age", "sex"))
    if np.isinf(design).any():
        raise ValueError("NONFINITE_FROZEN_DISCOVERY_DESIGN")
    baseline = fit_base(design, y[rows], left.shape[1], BASE, shared_columns=shared)
    role = SimpleNamespace(design=(design,), y=(y[rows],), groups=groups[rows], width=left.shape[1],
                           shared_columns=shared)
    delta = np.diff(_diet_basis((10.0, 15.0), dose_state), axis=0)[0]
    correction = _fit_correction(dose[rows], delta, panel, residuals(role, BASE))
    return FrozenModel(len(rows), int(np.unique(groups[rows]).size), clr, dose_state, target, host_state, levels,
                       pca, pc_scale, eco_scale, tuple(matches.items()), panel_scale, panel_constant,
                       baseline, correction)


def predict_model(state, source):
    """Predict baseline, added correction and added dose contrast, without outcomes."""
    index = source.index
    dose = _diet_basis(index.fiber_density.to_numpy(dtype=float, na_value=np.nan), state.dose)
    eligible = _host_eligible(index, dose)
    technical = np.asarray(getattr(source, "predictor_eligible", np.ones(len(index), dtype=bool)))
    if technical.shape != eligible.shape or technical.dtype != np.dtype(bool):
        raise ValueError("INVALID_RESERVED_PREDICTOR_MASK")
    eligible &= technical
    if tuple(resolve_species(source.taxa["gut"]).items()) != state.panel_taxa:
        raise ValueError("FROZEN_PANEL_MAPPING_CHANGED")
    profiles = {}
    for site in SITES:
        if list(source.taxa[site]) != state.clr[site]["source_features"]:
            raise ValueError("FROZEN_TAXON_DICTIONARY_CHANGED")
        raw = np.asarray(source.raw[site], dtype=float)
        if raw.shape != (len(index), len(source.taxa[site])):
            raise ValueError("INVALID_RESERVED_PROFILE_ALIGNMENT")
        complete = np.isfinite(raw).all(axis=1)
        totals = raw[complete].sum(axis=1)
        if (np.isinf(raw).any() or np.any(raw[np.isfinite(raw)] < 0)
                or not np.allclose(totals[totals > 0], 1, rtol=0, atol=1e-3)):
            raise ValueError("INVALID_RESERVED_PROFILE_VALUES")
        good = np.zeros(len(index), dtype=bool)
        good[complete] = totals > 0
        eligible &= good
        profiles[site] = raw
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return (rows, *(np.empty(0) for _ in range(3)))

    z, supported = {}, np.ones(len(rows), dtype=bool)
    for site in SITES:
        z[site], local = _replay_clr(profiles[site][rows], state.clr[site])
        supported &= local
    z = {site: z[site][supported] for site in SITES}
    rows = rows[supported]
    if not len(rows):
        return (rows, *(np.empty(0) for _ in range(3)))

    host = _replay_host(index.iloc[rows], dose[rows], state.host)
    pcs = [state.pc_scale[site].apply(state.pca[site].transform(z[site])) for site in SITES]
    raw_eco = np.column_stack([ecology(profiles[site][rows]) for site in SITES])
    score, _, _ = build_score(profiles["gut"][rows], source.taxa["gut"])
    score = score[:, None]
    panel = np.zeros_like(score) if state.panel_constant else state.panel_scale.apply(score)
    design = np.column_stack((host, _context(source.extra, rows, state.alcohol_levels), *pcs,
                              state.eco_scale.apply(raw_eco), panel))
    if not np.isfinite(host).all() or np.isinf(design).any() or not np.isfinite(panel).all():
        raise ValueError("NONFINITE_RESERVED_DESIGN")
    baseline = state.baseline.predict(design)
    correction, contrast = _apply_correction(state.correction, dose[rows], panel)
    if not all(np.isfinite(part).all() for part in (baseline, correction, contrast)):
        raise ValueError("NONFINITE_RESERVED_PREDICTION")
    return rows, baseline, correction, contrast


def model_selftest():
    """Verify the archived correction and portable target on generated inputs."""
    rng = np.random.default_rng(6320)
    dose = rng.normal(size=(150, 2))
    valid_dose = rng.normal(size=(30, 2))
    marker = rng.normal(size=(150, 1))
    valid_marker = rng.normal(size=(30, 1))
    diet = {"center": 11.8, "scale": 5., "knots": [-1.3, 0., 1.6]}
    fixed = _diet_basis((10., 15.), diet)
    delta = np.diff(fixed, axis=0)[0]
    error = dose[:, 0] * marker[:, 0] + rng.normal(size=len(dose))
    for constant in (False, True):
        training = np.zeros_like(marker) if constant else marker
        held = np.zeros_like(valid_marker) if constant else valid_marker
        role = SimpleNamespace(rows=np.arange(len(held)), dose=(dose - dose.mean(0), valid_dose - dose.mean(0)),
                               delta=delta, markers={"fermentation": (training, held)})
        expected = fit_corrections(role, error, (ROUTE,))[ROUTE]
        state = _fit_correction(dose, delta, training, error)
        got = _apply_correction(state, valid_dose, held)
        for a, b in zip(got, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        low, high = (_apply_correction(state, np.broadcast_to(x, valid_dose.shape), held)[0] for x in fixed)
        np.testing.assert_allclose(high - low, got[1], atol=1e-12, rtol=0)
    receipt = {"kind": "hpp_endpoint_shortest95_8sd_5sd_v1", "endpoint": "D_Glu",
               "remove_low": -8., "clip_low": -5., "clip_high": 5., "remove_high": 8.}
    np.testing.assert_array_equal(transform_target(receipt, [-9, -8, -5, 0, 6, 8, 9, np.nan]),
                                  [np.nan, -5, -5, 0, 5, 5, np.nan, np.nan])
