"""Training-only feature maps and replayable main/interaction decomposition."""

from copy import copy
from dataclasses import dataclass, replace

import numpy as np
from scipy import linalg

from exp2_expansion import detection_features, soft_profiles
from exp2_followup import balance_profiles, local_dose_features, two_part_profiles


def sites_for(site):
    return ("oral", "gut") if site == "both" else (site,)


def standardized(train, valid, clip=None):
    center, sd = train.mean(0), train.std(0)
    take = sd > 1e-9
    result = tuple((a[:, take] - center[take]) / sd[take] for a in (train, valid))
    return result if clip is None else tuple(np.clip(a, -clip, clip) for a in result)


def sqdist(x, y):
    # Matrix multiplication avoids a separate loop across every sample/taxon pair.
    result = -2.0 * (x @ y.T)
    result += np.einsum("ij,ij->i", x, x)[:, None]
    result += np.einsum("ij,ij->i", y, y)[None, :]
    return np.maximum(result, 0, out=result)


def bandwidth(train):
    # Capped training-only sample for pairwise bandwidth estimation.
    sub = train[np.linspace(0, len(train) - 1, min(256, len(train)), dtype=int)]
    d2 = sqdist(sub, sub)
    nonzero = d2[np.triu_indices(len(sub), 1)]
    nonzero = nonzero[nonzero > 1e-12]
    return float(np.median(nonzero)) if len(nonzero) else 1.0


def fourier(train, valid, seed):
    """Fixed random Fourier dictionary at two training-relative RBF scales."""
    rng = np.random.default_rng(seed)
    d2 = bandwidth(train)
    maps = []
    for scale in (0.5, 2.0):
        weight = rng.normal(size=(train.shape[1], 16)) / np.sqrt(d2 * scale)
        phase = rng.uniform(0, 2 * np.pi, 16)
        maps.append(tuple(np.cos(a @ weight + phase) for a in (train, valid)))
    return standardized(*(np.column_stack([m[k] for m in maps]) for k in (0, 1)), clip=8)


def nystrom(train, valid, seed):
    """Training landmarks for an RBF kernel in Hellinger coordinates."""
    rng = np.random.default_rng(seed)
    anchors = train[rng.choice(len(train), min(32, len(train)), replace=False)]
    d2 = bandwidth(train)

    def kernel(x, y):
        return np.exp(-sqdist(x, y) / d2)

    eig, vec = linalg.eigh(kernel(anchors, anchors), check_finite=False)
    take = eig > max(1e-8, eig[-1] * 1e-6)
    basis = vec[:, take] / np.sqrt(eig[take])
    return standardized(kernel(train, anchors) @ basis, kernel(valid, anchors) @ basis, clip=8)


def prevalent_clr(split, site, raw, limit):
    # Output CLR columns index into the retained reference, not the raw profile.
    state = split.preprocessing["CLR"][site]
    reference = np.asarray(state.get("reference_indices", np.arange(raw.shape[1])), int)
    source = reference[np.asarray(state["output_indices"], int)]
    clr = (split.ot, split.ov) if site == "oral" else (split.gt, split.gv)
    if len(source) != clr[0].shape[1] or len(source) != clr[1].shape[1]:
        raise ValueError("CLR feature identity mismatch")
    present = (raw[:, source] > 0).mean(axis=0)
    spread = raw[:, source].std(axis=0)
    take = np.lexsort((source, -spread, -present))[:limit]
    return tuple(np.clip(a[:, take], -8, 8) for a in clr)


def dose_hinges(diet, knots):
    return np.maximum(diet[:, :1] - knots, 0.0)


def anchored_curve(diets, h_at_center):
    """Anchor at zero exposure and remove training-only linear collinearity."""
    train = diets[0]
    linear = train[:, 0]
    h = train[:, 1] - h_at_center
    norm = linear @ linear
    slope = (linear @ h) / norm if norm > 1e-12 else 0.0
    curve = h - slope * linear
    scale = np.sqrt(np.mean(curve ** 2))
    return tuple(np.column_stack([
        d[:, 0],
        (d[:, 1] - h_at_center - slope * d[:, 0]) / max(scale, 0.1)
        if scale > 1e-9 else np.zeros(len(d)),
    ]) for d in diets)


@dataclass
class Projection:
    """Orthogonalize on well-supported main directions and replay in held roles."""

    q: object
    q_target: object
    directions: object
    replay: tuple
    condition: float

    @classmethod
    def fit(cls, xt, xv, rcond):
        u, singular, vt = linalg.svd(xt, full_matrices=False, check_finite=False)
        rank = int((singular > rcond * singular[0]).sum())
        if rank < 1 or len(xt) - rank < 20:
            raise ValueError("Insufficient training residual degrees after projection")
        q = u[:, :rank]
        # All numerically identifiable directions can be removed from the
        # training target. This uses orthonormal vectors only, never inverses.
        q_target = u[:, singular > 1e-10 * singular[0]]
        # Map held covariates to the same orthonormal training coordinates.
        # Inverting tiny singular values amplifies minor differences in correlated bases.
        directions = vt[:rank].T / singular[:rank]
        return cls(q, q_target, directions, (q, xv @ directions),
                   float(singular[0] / singular[rank - 1]))

    def residual(self, train):
        return train - self.q @ (self.q.T @ train)

    def target_residual(self, outcome):
        return outcome - self.q_target @ (self.q_target.T @ outcome)

    def coefficients(self, train):
        return self.q.T @ train

    def transform(self, values, coefficients, pos=1, x=None):
        design = self.replay[pos] if x is None else x @ self.directions
        return values - design @ coefficients


class RoleDesign:
    """One fold. Extra nonlinear microbial columns are shared by every baseline."""

    def __init__(self, dataset, train, valid, cfg, features, seed):
        split = self.split = dataset.split(train, valid, cfg["contrast"])
        self.seed = seed
        self.project_rcond = cfg["project_rcond"]
        self.screen_keep = cfg["screen_keep"]
        role = features.Role(split, dataset.raw, dataset.index, cfg["model_seed"],
                             taxa=dataset.taxa, extra=dataset.extra)
        self.design = features.Design(role, cfg)
        self.groups = dataset.index.split_unit_id.astype(str).to_numpy()[split.train_rows]
        self.diet = (split.dt, split.dv)
        self.lo, self.hi = (self.design.diet_basis(False, value) for value in cfg["contrast"])
        center = split.preprocessing["diet"]["center"]
        center_curve = self.design.diet_basis(True, center)[0, 1]
        self.local_curve_diet = anchored_curve((*self.diet, self.lo, self.hi), center_curve)
        if "local_dose_curve" in cfg["grids"]:
            self.dose_bins = local_dose_features((*self.diet, self.lo, self.hi))
        self.pcs, self.ecos, self.features = {}, {}, {}
        pc_splines = {}
        hell, taxa, screened_taxa, detected = {}, {}, {}, {}
        profile_maps = {"balance_curve": balance_profiles, "two_part_curve": two_part_profiles}
        added = {name: {} for name in profile_maps if name in cfg["grids"]}
        detection_limit = cfg.get("detection_taxa_per_site") if "detection_slope" in cfg["grids"] else None
        for site in ("oral", "gut"):
            self.pcs[site] = tuple(np.clip(a, -8, 8) for a in self.design.pcs[site])
            pcs = tuple(p[:, :cfg["curved_components_per_site"]] for p in self.design.pcs[site])
            curves = tuple(self.design.pc_curves[site].transform(p) for p in pcs)
            pc_splines[site] = standardized(
                *(np.column_stack([a, b]) for a, b in zip(pcs, curves)), clip=8,
            )
            self.ecos[site] = standardized(*self.design.ecos[site], clip=8)
            raw = tuple(dataset.raw[site][rows] for rows in (split.train_rows, split.rows))
            taxa[site] = prevalent_clr(split, site, raw[0], cfg["taxa_per_site"])
            screened_taxa[site] = prevalent_clr(split, site, raw[0], cfg["screen_taxa_per_site"])
            if detection_limit is not None:
                detected[site] = standardized(*detection_features(*raw, detection_limit), clip=8)
            for family, by_site in added.items():
                by_site[site] = standardized(*profile_maps[family](*raw), clip=8)
            hell[site] = tuple(np.sqrt(a / a.sum(1, keepdims=True)) for a in raw)
        # Every dictionary used by a fixed nonlinear kernel gets its standalone
        # microbial features in both models. Tree/NN marginal coverage is finite.
        knots = np.quantile(self.diet[0][:, 0], [1 / 3, 2 / 3])
        self.dose_hinge = tuple(np.column_stack([d[:, 0], dose_hinges(d, knots)])
                                for d in (*self.diet, self.lo, self.hi))
        train_hinge = self.dose_hinge[0]
        hmean, hsd = train_hinge[:, 1:].mean(0), train_hinge[:, 1:].std(0)
        hsd[hsd < 1e-9] = 1.0
        hinge_main = tuple((d[:, 1:] - hmean) / hsd for d in self.dose_hinge)
        paired = standardized(*(
            (self.pcs["oral"][k][:, :4, None] * self.pcs["gut"][k][:, None, :4]).reshape(len(d), -1)
            for k, d in enumerate(self.diet)
        ), clip=8)
        self.features[("oral_gut_tensor", "both")] = paired
        marginal = ([paired[0]], [paired[1]])
        for j, site in enumerate(cfg["sites"]):
            which = sites_for(site)
            wide = tuple(np.column_stack([self.pcs[s][k] for s in which]) for k in (0, 1))
            narrow = tuple(np.column_stack([self.pcs[s][k][:, :10] for s in which]) for k in (0, 1))
            ecology = tuple(np.column_stack([self.ecos[s][k] for s in which]) for k in (0, 1))
            tree = tuple(np.column_stack([a, b]) for a, b in zip(narrow, ecology))
            # Elastic interaction features reuse PC/eco linear main effects plus
            # a small nonlinear block present in the common main design.
            extra = tuple(np.maximum(a[:, :min(6, a.shape[1])], 0) for a in narrow)
            hinges = standardized(*extra, clip=8)
            elastic = tuple(np.column_stack([a, b, c]) for a, b, c in zip(wide, ecology, hinges))
            nys = nystrom(*(np.column_stack([hell[s][k] for s in which]) / np.sqrt(len(which))
                            for k in (0, 1)), seed + 701 + j)
            rff = fourier(*tree, seed + 1001 + j)
            taxa_site = tuple(np.column_stack([taxa[s][k] for s in which]) for k in (0, 1))
            screening = tuple(np.column_stack([screened_taxa[s][k] for s in which]) for k in (0, 1))
            smooth = tuple(np.column_stack([pc_splines[s][k] for s in which]) for k in (0, 1))
            if site == "both":
                self.features[("oral_gut_plus", site)] = tuple(
                    np.column_stack([a, b]) for a, b in zip(narrow, paired)
                )
                self.joint_linear_width = narrow[0].shape[1]
            for family, data in (("ridge_pc10", narrow), ("ridge_pc30", wide), ("ridge_spline", narrow),
                                 ("elastic", elastic), ("fourier", rff), ("hellinger", nys),
                                 ("forest", tree), ("boost", tree), ("neural", tree),
                                 ("ridge_ecology", ecology), ("taxa_sparse", taxa_site),
                                 ("pls", wide), ("fourier_curve", rff),
                                 ("dose_hinge", narrow), ("local_slope", tree),
                                 ("spline_modifier", smooth), ("huber_curve", narrow),
                                 ("local_curve", tree), ("taxa_screen", screening),
                                 ("group_curve", narrow), ("boost_curve", tree)):
                self.features[(family, site)] = data
            if "soft_cluster" in cfg["grids"]:
                self.features[("soft_cluster", site)] = standardized(
                    *soft_profiles(*tree, seed + 2001 + j), clip=8)
            if detection_limit is not None:
                self.features[("detection_slope", site)] = tuple(
                    np.column_stack([detected[s][k] for s in which]) for k in (0, 1))
            for family, source in added.items():
                self.features[(family, site)] = tuple(
                    np.column_stack([source[s][k] for s in which]) for k in (0, 1))
            for k in (0, 1):
                marginal[k].extend((hinges[k], rff[k], nys[k]))
                if site != "both":
                    marginal[k].append(taxa_site[k])
        self.marginal = tuple(np.column_stack(parts) for parts in marginal)
        self.x = tuple(np.column_stack([self.design.main(k == 0), self.marginal[k], hinge_main[k]]) for k in (0, 1))
        if not all(np.isfinite(a).all() for a in self.x):
            raise ValueError("Nonfinite common main design")
        self.projection = Projection.fit(*self.x, cfg["project_rcond"])
        # Replay main columns at 10/15 while holding microbial/other inputs fixed.
        self.x_lo, self.x_hi = tuple(np.column_stack([self.design.main(False, value), self.marginal[1], hinge_main[k + 2]])
                                    for k, value in enumerate(cfg["contrast"]))
        role.raw, role.extra, role.cache = {}, {}, {}
        split.ot = split.ov = split.gt = split.gv = np.empty((0, 0))

    def contrast(self, raw_lo, raw_hi, coefficients):
        return self.projection.transform(raw_hi - raw_lo, coefficients, x=self.x_hi - self.x_lo)

    def at_validation_dose(self, high=False):
        """Replay the original training role with held doses set to one contrast endpoint."""
        view = copy(self)
        view.diet = (self.diet[0], self.hi if high else self.lo)
        view.x = (self.x[0], self.x_hi if high else self.x_lo)
        view.projection = replace(self.projection,
                                  replay=(self.projection.q, view.x[1] @ self.projection.directions))
        for name in ("dose_hinge", "local_curve_diet", "dose_bins"):
            if hasattr(self, name):
                train, _, lo, hi = getattr(self, name)
                setattr(view, name, (train, hi if high else lo, lo, hi))
        return view
