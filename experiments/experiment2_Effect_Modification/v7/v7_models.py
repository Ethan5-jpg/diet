"""Paired discovery fits anchored to the historical gut PC10 profiled ridge.

All models jointly refit host, main and interaction coefficients. Off sets only
the fiber-product block to zero. Both arms use the same rows, CLR main effects,
host/dose terms, additional modifier main effects, scaling and main penalties.
Native PCA uses all site-supported training-role profiles before joint eligibility,
replaying the archived W19 centering, eigensolver, orientation and PC scales.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import linalg

import v7_modifier_views as views


ANCHOR = "gut_pc10_ridge"
FIXED_ANCHOR_SETTING = ANCHOR + "@0.1"
PENALTIES = (100., 10., 1., .1, .01)
MAIN_PENALTY = 1.0


@dataclass(frozen=True)
class Route:
    name: str
    modifiers: tuple
    # 0 = original linear dose, 1 = original restricted cubic spline basis.
    doses: tuple = (0,)
    penalty_shape: str = "uniform"


def pc_list(site, count):
    return tuple(f"{site}{j}" for j in range(1, count + 1))


def definitions():
    gut = pc_list("g", 10)
    routes = [Route(ANCHOR, gut)]
    for k in (4, 6, 8, 12, 16, 20, 32):
        routes.append(Route(f"gut_pc{k}_ridge", pc_list("g", k)))
    for k in (6, 10, 16, 24):
        routes.append(Route(f"gut_pc{k}_dose_spline", pc_list("g", k), (0, 1)))
    for k in (10, 16, 24, 32):
        routes.append(Route(f"gut_pc{k}_spectral_taper", pc_list("g", k),
                            penalty_shape="taper"))
    routes.extend([
        Route("gut_pc10_dose_spline_shrink", gut, (0, 1), "shrink_curve"),
        Route("gut_pc16_dose_spline_taper", pc_list("g", 16), (0, 1), "taper"),
    ])
    for k in (4, 10):
        pc = pc_list("g", k)
        routes.extend([
            Route(f"gut_pc{k}_quadratic", pc + tuple("sq_" + p for p in pc)),
            Route(f"gut_pc{k}_hinge", pc + tuple("hinge_" + p for p in pc)),
        ])
    pairs = tuple(f"pair_{a}_{b}" for a, b in combinations(gut[:4], 2))
    routes.append(Route("gut_pc10_community_pairs", gut + pairs))
    for oral_k in (2, 5, 10):
        joint = gut + pc_list("o", oral_k)
        routes.extend([
            Route(f"gut_pc10_oral_pc{oral_k}", joint),
            Route(f"gut_pc10_oral_pc{oral_k}_shrink", joint, penalty_shape="shrink_oral"),
        ])
    for group, additions in [("shannon", ("shannon",)), ("fermentation", ("fermentation",)),
                             ("ecology_both", ("shannon", "fermentation"))]:
        routes.append(Route(f"gut_pc10_{group}", gut + additions))
    for band in views.RFF_BANDS:
        features = views.labels("rff_" + band, 2 * views.N_FREQUENCIES)
        routes.extend([Route("gut_rff_" + band, features),
                       Route("gut_pc10_plus_rff_" + band, gut + features)])
    for view, width in views.COMPOSITION_WIDTHS.items():
        features = views.labels(view, width)
        routes.extend([Route("gut_taxa_" + view, features),
                       Route("gut_pc10_plus_taxa_" + view, gut + features)])
    for name, markers in (("shannon", ("shannon",)), ("fermentation", ("fermentation",)),
                          ("both", ("shannon", "fermentation"))):
        gates = tuple(f"v7_gate_{marker}_g{j}" for marker in markers for j in range(1, 5))
        routes.append(Route("gut_pc10_marker_gate_" + name, gut + markers + gates))
    return tuple(routes)


ROUTES = definitions()

# Equal, predeclared weights at the same interaction penalty for every parent.
# Both predictions of the composite model must use exactly the same weights.
BLENDS = {
    "blend_gut_rank": ("gut_pc6_ridge", ANCHOR, "gut_pc16_ridge"),
    "blend_gut_shape": (ANCHOR, "gut_pc10_dose_spline", "gut_pc10_hinge"),
    "blend_gut_ecology": (ANCHOR, "gut_pc10_shannon", "gut_pc10_fermentation"),
    "blend_gut_rff": tuple("gut_pc10_plus_rff_" + name for name in views.RFF_BANDS),
    "blend_gut_composition": tuple("gut_pc10_plus_taxa_" + view for view in views.COMPOSITION_WIDTHS),
}


def setting_key(route, penalty):
    return f"{route}@{penalty:g}"


def families():
    """Ordered settings from stronger to weaker interaction shrinkage."""
    names = [r.name for r in ROUTES] + list(BLENDS)
    return {name: [setting_key(name, p) for p in PENALTIES] for name in names}


def _column_sd(matrix):
    sd = matrix.std(axis=0)
    return np.where(sd > 1e-12, sd, 1.0)


def _pca_reference(source, split, site, eligible_train):
    """Replay registered CLR on each site's full training role before eligibility."""
    if "CLR" not in split.preprocessing:
        if hasattr(source, "api"):
            raise ValueError("NATIVE_SPLIT_MISSING_CLR")
        return eligible_train  # Generated fixture has no archived CLR transform.
    rows = np.asarray(split.pca_training_rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or (
            len(rows) == 0 or rows[0] < 0 or rows[-1] >= len(source.groups)
            or np.any(rows[1:] <= rows[:-1])):
        raise ValueError("MALFORMED_PCA_TRAINING_ROWS")
    if np.intersect1d(source.groups[rows], source.groups[split.rows]).size:
        raise ValueError("HOUSEHOLD_CROSSES_PCA_ROLE")
    transform = source.api.MicrobiomeCLR()
    transform.state = state = split.preprocessing["CLR"][site]
    if len(rows) != state["training_participants"]:
        raise ValueError("PCA_TRAINING_ROLE_SIZE_MISMATCH")
    values = transform.transform(source.raw[site][rows], source.taxa[site], site=site)
    positions = np.searchsorted(rows, split.train_rows)
    if np.any(positions >= len(rows)) or not np.array_equal(rows[positions], split.train_rows):
        raise ValueError("ELIGIBLE_TRAIN_ROW_ABSENT_FROM_PCA_ROLE")
    if not np.all(values["supported"][positions]) or not np.allclose(
            values["values"][positions], eligible_train, rtol=0, atol=1e-10):
        raise ValueError("TRAINING_CLR_REPLAY_MISMATCH")
    if int(values["supported"].sum()) != state["training_supported_participants"]:
        raise ValueError("PCA_SITE_SUPPORT_SIZE_MISMATCH")
    return values["values"][values["supported"]]


def _pcs(train, valid, reference, n):
    """W19 pca_fit applied to an independent reference matrix of training rows."""
    k = min(n, reference.shape[1], len(reference) - 1)
    if k < n:
        raise ValueError(f"PC_WIDTH_INSUFFICIENT: need {n} have {k}")
    center = reference.mean(axis=0)
    a = reference - center
    width = reference.shape[1]
    values, vectors = linalg.eigh(a.T @ a, subset_by_index=[width - n, width - 1],
                                  driver="evr", check_finite=True)
    values, vectors = values[::-1], vectors[:, ::-1]
    if values[-1] <= max(values[0], 1.0) * 1e-10:
        raise ValueError("MICROBIAL_PC_RANK_BELOW_REQUESTED_GRID")
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(n)])
    vectors = vectors * signs
    scales = (a @ vectors).std(axis=0)
    return (train - center) @ vectors / scales, (valid - center) @ vectors / scales


def _extra_name(label):
    return label.startswith(("hinge_", "sq_", "pair_", "v7_")) or label in ("fermentation", "shannon")


def _feature_bank(source, split, view_audit=None):
    from adapters.fermentation_score import build_score

    bank = {}
    for site, a, b, n in (("g", split.gt, split.gv, 32), ("o", split.ot, split.ov, 10)):
        reference = _pca_reference(source, split, {"g": "gut", "o": "oral"}[site], a)
        train, valid = _pcs(a, b, reference, n)
        del reference
        for j, label in enumerate(pc_list(site, n)):
            bank[label] = (train[:, j], valid[:, j])
    for label in pc_list("g", 10):
        train, valid = bank[label]
        median = np.median(train)
        # Keeping the raw PC as well as its centered square makes the joint
        # feature span stable against a sign convention change.
        bank["sq_" + label] = (train**2, valid**2)
        bank["hinge_" + label] = (np.maximum(train - median, 0), np.maximum(valid - median, 0))
    for a, b in combinations(pc_list("g", 4), 2):
        bank[f"pair_{a}_{b}"] = tuple(x * y for x, y in zip(bank[a], bank[b]))
    shannon, fermentation = [], []
    for rows in (split.train_rows, split.rows):
        profile = np.asarray(source.raw["gut"][rows], dtype=float)
        score, _, _ = build_score(profile, source.taxa["gut"])
        if not np.isfinite(score).all():
            raise ValueError("NONFINITE_FERMENTATION_IN_FIXED_ROLE")
        normalized = profile / profile.sum(axis=1, keepdims=True)
        shannon.append(-(normalized * np.log(np.maximum(normalized, 1e-300))).sum(axis=1))
        fermentation.append(score)
    bank["shannon"], bank["fermentation"] = tuple(shannon), tuple(fermentation)
    # Remain zero-mean in the training modifier for derived nonlinear axes.
    # Linear PC values retain the legacy PCA center and dose coding.
    for label in bank:
        if _extra_name(label):
            train, valid = bank[label]
            bank[label] = (train - train.mean(), valid - train.mean())
    views.add_views(bank, source, split, audit=view_audit)
    return bank


class _Profile:
    """Profile out unpenalized B and ridge main Z once; reuse all candidate columns."""

    def __init__(self, b, bv, z, zv, y, features, fv):
        n = len(y)
        if not all(np.isfinite(a).all() for a in (b, bv, z, zv, y, features, fv)):
            raise ValueError("NONFINITE_MATCHED_FIT")
        _, r, piv = linalg.qr(b, mode="economic", pivoting=True, check_finite=False)
        diag = np.abs(np.diag(r))
        rank = int(np.sum(diag > max(diag.max(initial=0), 1.0) * 1e-10))
        keep = np.sort(piv[:rank])
        q, r = linalg.qr(b[:, keep], mode="economic", check_finite=False)
        # bv @ pinv(b) applied through the small triangular R.
        vq = linalg.solve_triangular(r.T, bv[:, keep].T, lower=True, check_finite=False).T
        z_sd, f_sd = _column_sd(z), _column_sd(features)
        z, zv = z / z_sd, zv / z_sd
        features, fv = features / f_sd, fv / f_sd
        t = np.column_stack((y, features))
        qt, qz = q.T @ t, q.T @ z
        residual = t - q @ qt
        residual_z = z - q @ qz
        residual_zv = zv - vq @ qz
        if residual_z.shape[1] <= n:
            gram = residual_z.T @ residual_z
            gram.flat[::len(gram) + 1] += n * MAIN_PENALTY
            coef = linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=False),
                                    residual_z.T @ residual, check_finite=False)
            adjusted = residual - residual_z @ coef
        else:
            gram = residual_z @ residual_z.T
            gram.flat[::n + 1] += n * MAIN_PENALTY
            dual = linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=False),
                                    residual, check_finite=False)
            adjusted = n * MAIN_PENALTY * dual
            coef = residual_z.T @ dual
        fitted_v = vq @ qt + residual_zv @ coef
        self.null = fitted_v[:, 0]
        self.projected_valid = fv - fitted_v[:, 1:]
        moment = residual[:, 1:].T @ adjusted / n
        self.rhs = moment[:, 0]
        self.gram = (moment[:, 1:] + moment[:, 1:].T) * .5

    def prediction(self, indices, penalty):
        if not len(indices):
            return self.null
        gram = self.gram[np.ix_(indices, indices)]
        gram.flat[::len(gram) + 1] += penalty
        coef = linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=False),
                                self.rhs[indices], check_finite=False)
        return self.null + self.projected_valid[:, indices] @ coef


def _weights(route):
    weights = []
    for dose in route.doses:
        for label in route.modifiers:
            weight = 1.0
            if route.penalty_shape == "taper":
                weight = 1 + ((int(label[1:]) - 1) / 5.0) ** 2
            elif route.penalty_shape == "shrink_oral" and label.startswith("o"):
                weight = 4.0
            elif route.penalty_shape == "shrink_curve" and dose == 1:
                weight = 4.0
            weights.append(weight)
    return np.array(weights)


def evaluate_split(source, split, *, view_audit=None):
    """Return setting -> (interaction-off, interaction-on) validation vectors.

    Training rows determine PCA loadings, transformation parameters and every
    fitted coefficient. Validation targets are deliberately never accessed.
    """
    if np.intersect1d(source.groups[split.train_rows], source.groups[split.rows]).size:
        raise ValueError("HOUSEHOLD_CROSSES_MODEL_ROLE")
    bank = _feature_bank(source, split, view_audit)
    extras = list(dict.fromkeys(label for r in ROUTES for label in r.modifiers if _extra_name(label)))
    products = list(dict.fromkeys((d, label) for r in ROUTES for d in r.doses for label in r.modifiers))
    extra_index = {label: i for i, label in enumerate(extras)}
    product_index = {key: len(extras) + i for i, key in enumerate(products)}
    feature_blocks, bases = [], []
    for side, (host, dose) in enumerate(((split.ht, split.dt), (split.hv, split.dv))):
        main_pcs = [bank[label][side] for label in (*pc_list("o", 5), *pc_list("g", 5))]
        bases.append(np.column_stack((np.ones(len(host)), host, dose, *main_pcs)))
        feature_blocks.append(np.column_stack([*(bank[label][side] for label in extras),
            *(dose[:, d] * bank[label][side] for d, label in products)]))
    profiled = _Profile(*bases, np.column_stack((split.ot, split.gt)),
                        np.column_stack((split.ov, split.gv)), split.yt, *feature_blocks)
    off_predictions = {}
    prediction = {}
    for r in ROUTES:
        extra = tuple(extra_index[m] for m in r.modifiers if m in extra_index)
        if extra not in off_predictions:
            off_predictions[extra] = profiled.prediction(list(extra), MAIN_PENALTY)
        indices = [*extra, *(product_index[(d, m)] for d in r.doses for m in r.modifiers)]
        weights = _weights(r)
        for penalty in PENALTIES:
            penalties = np.r_[np.full(len(extra), MAIN_PENALTY), penalty * weights]
            on = profiled.prediction(indices, penalties)
            if not np.isfinite(on).all() or not np.isfinite(off_predictions[extra]).all():
                raise ArithmeticError("NONFINITE_PAIRED_PREDICTION")
            prediction[setting_key(r.name, penalty)] = off_predictions[extra], on
    for name, members in BLENDS.items():
        for penalty in PENALTIES:
            parents = [prediction[setting_key(member, penalty)] for member in members]
            prediction[setting_key(name, penalty)] = tuple(
                np.mean(np.stack(arm), axis=0) for arm in zip(*parents))
    return prediction
