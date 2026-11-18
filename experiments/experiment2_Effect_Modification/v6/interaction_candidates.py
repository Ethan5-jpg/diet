"""Training-only, zero-at-zero-dose microbial interaction corrections.

The caller supplies a fixed baseline and household-cross-fitted training
residuals. This module never receives validation outcomes. Its predictors are
products of dose basis values and a train-fitted microbial representation;
there is no correction intercept or independent microbial main-effect term.
For ``both`` routes the caller must take k PCs from *each* site and concatenate
them. This module uses all supplied microbial columns, rather than discarding
the second site by truncating a concatenated matrix to k columns.
"""

from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
from scipy.linalg import solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor


SEED = 20261118
KINDS = ("ridge", "pls", "elasticnet", "rbf", "tensor", "vc_boost")
SITES = ("gut", "oral", "both")
VIEWS = ("pc", "balance", "taxa", "two_part")


@dataclass(frozen=True)
class Candidate:
    site: str
    n_components: int
    kind: str
    strength: float
    n_dose: int = 1
    latent_rank: int = 3
    l1_ratio: float = 0.5
    rff_features: int = 48
    seed: int = SEED
    view: str = "pc"

    def __post_init__(self):
        if (self.site not in SITES or self.kind not in KINDS or self.view not in VIEWS
                or not isinstance(self.n_components, int) or self.n_components < 1
                or not isinstance(self.n_dose, int) or self.n_dose not in (1, 2)
                or not isinstance(self.latent_rank, int) or self.latent_rank < 1
                or not isinstance(self.rff_features, int) or self.rff_features < 1
                or not isinstance(self.seed, int) or self.seed < 0
                or not math.isfinite(self.strength) or self.strength <= 0
                or not math.isfinite(self.l1_ratio) or not 0 < self.l1_ratio <= 1):
            raise ValueError("INVALID_INTERACTION_CANDIDATE")
        if self.kind == "vc_boost" and self.n_dose != 1:
            raise ValueError("VARYING_COEFFICIENT_REQUIRES_LINEAR_DOSE")
        if self.view != "pc" and self.kind not in ("ridge", "elasticnet"):
            raise ValueError("UNSUPPORTED_COMPOSITIONAL_INTERACTION")

    @property
    def route(self):
        return f"{self.site}_{self.kind}_{self.view}{self.n_components}_dose{self.n_dose}"

    def key(self):
        return f"{self.route}@{self.strength:g}"

    def to_dict(self):
        return asdict(self)


def options():
    """Return 66 fixed route families, each with three penalty settings.

    There are 22 routes per site: 6 ordinary linear ridge corrections,
    2 supervised low-rank PLS corrections, 2 elastic-net corrections, and
    2 nonlinear RBF-modifier corrections, 2 tensor corrections, 2 shallow-tree
    varying-coefficient corrections, and 6 compositional-view corrections.
    A zero correction is added by the caller to every inner-CV selection.
    """
    specifications = (
        *(("ridge", k, 1) for k in (5, 10, 20, 40)),
        ("ridge", 10, 2), ("ridge", 20, 2),
        ("pls", 10, 1), ("pls", 20, 1),
        ("elasticnet", 10, 1), ("elasticnet", 20, 1),
        ("rbf", 10, 1), ("rbf", 20, 1),
        ("tensor", 5, 2), ("tensor", 10, 2),
        ("vc_boost", 10, 1), ("vc_boost", 20, 1),
    )
    output = {}
    for site in SITES:
        for kind, k, n_dose in specifications:
            candidates = tuple(Candidate(site, k, kind, strength, n_dose=n_dose)
                               for strength in (1.0, 0.1, 0.01))
            output[candidates[0].route] = candidates
        for view in ("balance", "taxa", "two_part"):
            for kind in ("ridge", "elasticnet"):
                candidates = tuple(Candidate(site, 16, kind, strength, view=view)
                                   for strength in (1.0, 0.1, 0.01))
                output[candidates[0].route] = candidates
    return output


def _matrix(values, name, rows=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or not values.shape[1]
            or (rows is not None and len(values) != rows)
            or not np.isfinite(values).all()):
        raise ValueError("INVALID_" + name)
    return values


def _validate(z_train, z_valid, dose_train, dose_valid, delta_dose, residual, config):
    if not isinstance(config, Candidate):
        raise TypeError("CANDIDATE_REQUIRED")
    z_train = _matrix(z_train, "TRAIN_MICROBIOME")
    z_valid = _matrix(z_valid, "VALID_MICROBIOME")
    n, p = z_train.shape
    if n < 3 or z_valid.shape[1] != p:
        raise ValueError("INCONSISTENT_MICROBIOME_WIDTH")
    dose_train = _matrix(dose_train, "TRAIN_DOSE", n)
    dose_valid = _matrix(dose_valid, "VALID_DOSE", len(z_valid))
    if dose_train.shape[1] != dose_valid.shape[1] or dose_train.shape[1] < config.n_dose:
        raise ValueError("INCONSISTENT_DOSE_WIDTH")
    residual = np.asarray(residual, dtype=float)
    if residual.shape != (n,) or not np.isfinite(residual).all():
        raise ValueError("INVALID_TRAINING_RESIDUAL")
    delta_dose = np.asarray(delta_dose, dtype=float)
    if delta_dose.ndim == 1:
        if delta_dose.shape[0] < config.n_dose:
            raise ValueError("INVALID_DOSE_CONTRAST")
        delta_dose = np.broadcast_to(delta_dose[:config.n_dose], (len(z_valid), config.n_dose))
    elif delta_dose.ndim == 2:
        if len(delta_dose) != len(z_valid) or delta_dose.shape[1] < config.n_dose:
            raise ValueError("INVALID_DOSE_CONTRAST")
        delta_dose = delta_dose[:, :config.n_dose]
    else:
        raise ValueError("INVALID_DOSE_CONTRAST")
    if not np.isfinite(delta_dose).all():
        raise ValueError("INVALID_DOSE_CONTRAST")
    return (z_train, z_valid, dose_train[:, :config.n_dose],
            dose_valid[:, :config.n_dose], delta_dose, residual)


def _representation(train, valid, config):
    """Fit centering/random features using training predictors only."""
    center = train.mean(axis=0)
    train, valid = train - center, valid - center
    if config.kind == "tensor":
        knots = np.quantile(train, (0.25, 0.5, 0.75), axis=0)
        train = np.column_stack([train, *(np.maximum(train - knot, 0.0) for knot in knots)])
        valid = np.column_stack([valid, *(np.maximum(valid - knot, 0.0) for knot in knots)])
        center = train.mean(axis=0)
        return train - center, valid - center
    if config.kind != "rbf":
        return train, valid
    # A fixed gamma=1/p avoids searching validation distances/bandwidths.
    rng = np.random.default_rng(config.seed)
    omega = rng.normal(scale=np.sqrt(2.0 / train.shape[1]),
                       size=(train.shape[1], config.rff_features))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=config.rff_features)
    factor = np.sqrt(2.0 / config.rff_features)
    train = factor * np.cos(train @ omega + phase)
    valid = factor * np.cos(valid @ omega + phase)
    center = train.mean(axis=0)
    return train - center, valid - center


def prepare_modifier(z_train, z_valid, config):
    """Expose the exact train-fitted modifier for the matched main-effect model.

    The returned train/validation matrices can enter the shared baseline as
    main effects, and be supplied unchanged to ``fit_correction``. Penalty and
    dose dimension never change this representation. No target is accepted.
    """
    if not isinstance(config, Candidate):
        raise TypeError("CANDIDATE_REQUIRED")
    train = _matrix(z_train, "TRAIN_MICROBIOME")
    valid = _matrix(z_valid, "VALID_MICROBIOME")
    if len(train) < 3 or valid.shape[1] != train.shape[1]:
        raise ValueError("INCONSISTENT_MICROBIOME_WIDTH")
    input_columns = train.shape[1]
    train, valid = _representation(train, valid, config)
    return train, valid, dict(kind=config.kind, columns=train.shape[1],
                             seed=config.seed,
                             view=config.view,
                             tensor_quantiles=[0.25, 0.5, 0.75] if config.kind == "tensor" else None,
                             rff_gamma=1.0 / input_columns if config.kind == "rbf" else None)


def _products(dose, microbial):
    return (dose[:, :, None] * microbial[:, None, :]).reshape(len(dose), -1)


def _ridge(train, y, strength):
    gram = train.T @ train / len(train)
    return solve(gram + strength * np.eye(gram.shape[0]),
                 train.T @ y / len(train), assume_a="pos", check_finite=False)


def _pls_rotation(train, y, requested):
    """NIPALS PLS1 without implicit centering or a fitted intercept.

    sklearn PLSRegression centers X and y even when scale=False. Replaying that
    intercept on a zero product would break the pure interaction restriction.
    This equivalent no-intercept Krylov construction keeps every correction
    a linear combination of the original dose products.
    """
    x, target = train.copy(), y.copy()
    weights, loadings = [], []
    for _ in range(min(requested, train.shape[1], len(train) - 1)):
        weight = x.T @ target
        length = np.linalg.norm(weight)
        if length <= 1e-12 * max(1.0, np.linalg.norm(x) * np.linalg.norm(target)):
            break
        weight /= length
        score = x @ weight
        norm2 = score @ score
        if norm2 <= 1e-20:
            break
        loading = x.T @ score / norm2
        target_loading = score @ target / norm2
        weights.append(weight)
        loadings.append(loading)
        x -= score[:, None] * loading
        target -= score * target_loading
    if not weights:
        return np.empty((train.shape[1], 0))
    weights, loadings = np.column_stack(weights), np.column_stack(loadings)
    return weights @ np.linalg.pinv(loadings.T @ weights, rcond=1e-12)


def _varying_coefficient_boost(train, valid, dose, residual, strength, seed,
                               steps=40, learning_rate=0.15):
    """Fit d*g(Z) using shallow microbial-only partitions and centered updates.

    The weighted Newton pseudo-response is residual/d with weight d**2. Rows
    with numerically zero dose never enter this division; they are retained
    in centering and prediction. A ridge-regularized line search scales each
    centered tree direction. The current outcome residual supplies no validation
    targets, and the tree cannot split on dose. Centering every update gives
    mean_train(g)=0, excluding a separate constant dose slope from the repair.
    This is a predictive varying-coefficient algorithm, not a causal forest.
    """
    n = len(train)
    g_train, g_valid = np.zeros(n), np.zeros(len(valid))
    rms = float(np.sqrt(np.mean(dose ** 2)))
    usable = np.abs(dose) > max(1e-12, 1e-7 * rms)
    min_leaf = max(8, int(np.ceil(0.04 * n)))
    detail = dict(vc_stages=0, vc_max_stages=int(steps), vc_max_depth=2,
                  vc_min_samples_leaf=min_leaf, vc_learning_rate=float(learning_rate),
                  vc_training_dose_nonzero_rows=int(usable.sum()),
                  vc_centering="unweighted current training rows")
    if usable.sum() < 2 * min_leaf or rms <= 1e-12 or not np.any(residual):
        detail["vc_training_modifier_mean"] = 0.0
        return g_train, g_valid, detail
    sample_weight = dose[usable] ** 2
    for stage in range(steps):
        error = residual - dose * g_train
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=min_leaf,
                                     random_state=seed + stage)
        tree.fit(train[usable], error[usable] / dose[usable], sample_weight=sample_weight)
        h_train, h_valid = tree.predict(train), tree.predict(valid)
        center = float(h_train.mean())
        h_train, h_valid = h_train - center, h_valid - center
        direction = dose * h_train
        numerator = float(direction @ error)
        denominator = float(direction @ direction + strength * (h_train @ h_train))
        # Training centering may reverse a tree direction's covariance with
        # the current residual. A negative line-search coefficient remains a
        # valid descent step; only a numerically zero direction should stop.
        if abs(numerator) <= 1e-14 or denominator <= 1e-20:
            break
        coefficient = learning_rate * numerator / denominator
        g_train += coefficient * h_train
        g_valid += coefficient * h_valid
        detail["vc_stages"] += 1
    # Correct accumulated roundoff with the same training-only centering.
    center = float(g_train.mean())
    g_train, g_valid = g_train - center, g_valid - center
    detail["vc_training_modifier_mean"] = float(g_train.mean())
    if not np.isfinite(g_train).all() or not np.isfinite(g_valid).all():
        raise ArithmeticError("NONFINITE_VARYING_COEFFICIENT_FIT")
    return g_train, g_valid, detail


def fit_correction(z_train, z_valid, dose_train, dose_valid, delta_dose,
                   residual, config, prepared_modifier=None):
    """Return held-out correction, same-fit dose contrast, and fit metadata.

    ``delta_dose`` is the frozen basis difference for fiber 10 to 15, either
    a common basis vector or one row per held-out participant. The same fitted
    coefficients produce predictions and contrasts. Validation targets are
    absent from this API. All supplied Z columns are used (see module header).
    """
    z_train, z_valid, dose_train, dose_valid, delta, y = _validate(
        z_train, z_valid, dose_train, dose_valid, delta_dose, residual, config)
    if prepared_modifier is None:
        microbial_train, microbial_valid, _ = prepare_modifier(z_train, z_valid, config)
    else:
        if not isinstance(prepared_modifier, (tuple, list)) or len(prepared_modifier) not in (2, 3):
            raise ValueError("INVALID_PREPARED_MODIFIER")
        microbial_train = _matrix(prepared_modifier[0], "PREPARED_TRAIN_MODIFIER", len(z_train))
        microbial_valid = _matrix(prepared_modifier[1], "PREPARED_VALID_MODIFIER", len(z_valid))
        if microbial_train.shape[1] != microbial_valid.shape[1]:
            raise ValueError("INCONSISTENT_PREPARED_MODIFIER_WIDTH")
    train = _products(dose_train, microbial_train)
    valid = _products(dose_valid, microbial_valid)
    contrast = _products(delta, microbial_valid)
    rms = np.sqrt(np.mean(train ** 2, axis=0))
    active = rms > 1e-12
    metadata = dict(kind=config.kind, training_rows=len(train),
                    supplied_microbiome_columns=z_train.shape[1],
                    product_columns=train.shape[1], active_columns=int(active.sum()),
                    strength=float(config.strength), converged=True,
                    convergence_warning=False, pls_components_used=0,
                    modifier_prepared=prepared_modifier is not None,
                    seed=config.seed)
    if not active.any():
        return np.zeros(len(valid)), np.zeros(len(valid)), metadata
    if config.kind == "vc_boost":
        _, g_valid, detail = _varying_coefficient_boost(
            microbial_train, microbial_valid, dose_train[:, 0], y,
            config.strength, config.seed)
        metadata.update(detail)
        return dose_valid[:, 0] * g_valid, delta[:, 0] * g_valid, metadata
    train = train[:, active] / rms[active]
    valid = valid[:, active] / rms[active]
    contrast = contrast[:, active] / rms[active]

    if config.kind in ("ridge", "rbf", "tensor"):
        coefficient = _ridge(train, y, config.strength)
    elif config.kind == "pls":
        rotation = _pls_rotation(train, y, config.latent_rank)
        metadata["pls_components_used"] = rotation.shape[1]
        if not rotation.shape[1]:
            coefficient = np.zeros(train.shape[1])
        else:
            latent = train @ rotation
            latent_rms = np.sqrt(np.mean(latent ** 2, axis=0))
            use = latent_rms > 1e-12
            rotation, latent_rms = rotation[:, use], latent_rms[use]
            metadata["pls_components_used"] = int(use.sum())
            coefficient = (rotation @ (_ridge((train @ rotation) / latent_rms,
                                              y, config.strength) / latent_rms)
                           if use.any() else np.zeros(train.shape[1]))
    else:
        alpha_max = np.max(np.abs(train.T @ y)) / (len(train) * config.l1_ratio)
        metadata["alpha"] = float(config.strength * alpha_max)
        if alpha_max <= 1e-15:
            coefficient = np.zeros(train.shape[1])
        else:
            learner = ElasticNet(alpha=config.strength * alpha_max,
                                 l1_ratio=config.l1_ratio, fit_intercept=False,
                                 max_iter=5000, tol=1e-6, selection="cyclic")
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                learner.fit(train, y)
            metadata["convergence_warning"] = any(
                issubclass(item.category, ConvergenceWarning) for item in recorded)
            metadata["converged"] = not metadata["convergence_warning"]
            metadata["iterations"] = int(learner.n_iter_)
            coefficient = learner.coef_
    prediction, dose_contrast = valid @ coefficient, contrast @ coefficient
    if not (np.isfinite(coefficient).all() and np.isfinite(prediction).all()
            and np.isfinite(dose_contrast).all()):
        raise ArithmeticError("NONFINITE_INTERACTION_FIT")
    return prediction, dose_contrast, metadata
