"""Outcome-blind modules and replayable, training-fitted additive designs."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bridge import BaseRole

ECOLOGY = ("log_richness", "shannon", "simpson_diversity", "dominance", "evenness")
TARGET_NAMES = ("oral_shannon", "oral_dominance", "gut_shannon", "gut_dominance")


def ecology(raw):
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid ecological profile")
    total = raw.sum(1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("Empty ecological profile")
    p = raw / total
    richness = (p > 0).sum(1)
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    return np.column_stack([
        np.log1p(richness), entropy, 1 - (p * p).sum(1), p.max(1),
        entropy / np.maximum(np.log(np.maximum(richness, 1)), 1e-12),
    ])


def fixed_targets(diet, markers):
    """Stable raw coordinates: per 5 g/1,000 kcal, per raw ecological unit."""
    return ((np.asarray(diet, float) - 10.0) / 5.0)[:, None] * markers


def rcs(values, knots):
    a, b, c = np.asarray(knots, float)
    if not a < b < c:
        return np.zeros_like(np.asarray(values, float))
    x = np.asarray(values, float)
    return (np.maximum(x - a, 0) ** 3 - (c - a) / (c - b) * np.maximum(x - b, 0) ** 3
            + (b - a) / (c - b) * np.maximum(x - c, 0) ** 3) / ((c - a) ** 2)


class Curves:
    """One training-quantile RCS main-effect term per nonconstant column."""

    def fit(self, x):
        x = np.asarray(x, float)
        self.center = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-10] = 1.0
        z = (x - self.center) / self.scale
        self.knots = np.quantile(z, [0.1, 0.5, 0.9], axis=0).T
        self.keep = [j for j, k in enumerate(self.knots) if np.diff(k).min() > 1e-7]
        return self

    def transform(self, x):
        z = (np.asarray(x, float) - self.center) / self.scale
        return np.column_stack([rcs(z[:, j], self.knots[j]) for j in self.keep]) if self.keep else np.empty((len(z), 0))


class ContextEncoder:
    def fit(self, extra, rows):
        a = extra["blocks"]["full"][rows]
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in a.T])
        self.categories = sorted({v for v in extra["alcohol"][rows] if v is not None})
        self.scaler = StandardScaler().fit(self.raw(extra, rows))
        return self

    def raw(self, extra, rows):
        a, cat = extra["blocks"]["full"][rows], extra["alcohol"][rows]
        categories = np.column_stack([cat == v for v in self.categories]) if self.categories else np.empty((len(rows), 0))
        return np.column_stack([
            np.where(np.isfinite(a), a, self.medians), ~np.isfinite(a), categories,
            cat == None,  # noqa: E711
            [v is not None and v not in self.categories for v in cat],
        ]).astype(float)

    def transform(self, extra, rows):
        return self.scaler.transform(self.raw(extra, rows))


@dataclass
class InteractionMap:
    projection: object
    keep: object
    scaler: object

    def transform(self, design, spec, training=False, value=None):
        raw = design.interaction_raw(spec, training, value)
        residual = raw - design.main(training, value) @ self.projection
        if not len(self.keep):
            return np.empty((len(raw), 0))
        return self.scaler.transform(residual[:, self.keep])

    def product_difference(self, design, spec, lo, hi):
        raw = design.interaction_raw(spec, False, hi) - design.interaction_raw(spec, False, lo)
        return raw[:, self.keep] / self.scaler.scale_ if len(self.keep) else np.empty((len(raw), 0))


@dataclass
class Role(BaseRole):
    taxa: object = None
    extra: object = None


class Design:
    """A single matched main-effect design used by all twelve augmented models."""

    def __init__(self, role, config):
        self.role, self.config = role, config
        s = role.split
        self.context = ContextEncoder().fit(role.extra, s.train_rows)
        self.ct, self.cv = [self.context.transform(role.extra, rows) for rows in (s.train_rows, s.rows)]
        self.ut, self.uv, self.upf_state = role.dietary()
        labels = s.preprocessing["host"]["labels"]
        self.background_columns = [j for j, name in enumerate(labels) if name != "intercept" and not name.startswith("diet_")]
        self.background_t = np.column_stack([s.ht[:, self.background_columns], self.ct, self.ut])
        self.background_v = np.column_stack([s.hv[:, self.background_columns], self.cv, self.uv])
        self.background_curve = Curves().fit(self.background_t)
        self.pcs, self.ecos, self.pc_states = {}, {}, {}
        for site in ("oral", "gut"):
            # No marginal or interaction screen; both PCA and ecology are outcome-blind.
            a, b = role.view(site, "clr")
            count = min(config["main_components_per_site"], a.shape[1], len(a) - 1)
            pc = PCA(n_components=count, svd_solver="randomized", random_state=role.seed).fit(a)
            scaler = StandardScaler().fit(pc.transform(a))
            self.pcs[site] = (scaler.transform(pc.transform(a)), scaler.transform(pc.transform(b)))
            self.pc_states[site] = dict(model=pc, scaler=scaler)
            self.ecos[site] = tuple(ecology(role.raw[site][rows]) for rows in (s.train_rows, s.rows))
        self.markers = tuple(np.column_stack([self.ecos[site][pos][:, [1, 3]] for site in ("oral", "gut")]) for pos in (0, 1))
        self.eco_curve = Curves().fit(np.column_stack([self.ecos[site][0] for site in ("oral", "gut")]))
        self.pc_curves = {site: Curves().fit(self.pcs[site][0][:, :config["curved_components_per_site"]]) for site in ("oral", "gut")}
        self.main_scaler = StandardScaler().fit(self._main_raw(True))
        self.xt, self.xv = self.main(True), self.main(False)
        self.interaction_states = {}
        self.targets_t = fixed_targets(self.diet(True), self.markers[0])
        self.targets_v = fixed_targets(self.diet(False), self.markers[1])
        self.iqr = np.quantile(self.markers[0], 0.75, axis=0) - np.quantile(self.markers[0], 0.25, axis=0)

    def diet(self, training):
        rows = self.role.split.train_rows if training else self.role.split.rows
        return self.role.index.fiber_density.to_numpy(float)[rows]

    def diet_basis(self, training, value=None):
        if value is None:
            return self.role.split.dt if training else self.role.split.dv
        state = self.role.split.preprocessing["diet"]
        u = (float(value) - state["center"]) / state["scale"]
        n = len(self.role.split.yt if training else self.role.split.yv)
        return np.tile([u, rcs(u, state["knots"])], (n, 1))

    def _main_raw(self, training, value=None):
        pos = 0 if training else 1
        s, role = self.role.split, self.role
        diet = self.diet_basis(training, value)
        host = (s.ht if training else s.hv) if value is None else role.host_at(diet[0], training)
        background = self.background_t if training else self.background_v
        all_eco = np.column_stack([self.ecos[site][pos] for site in ("oral", "gut")])
        pieces = [host, background, self.background_curve.transform(background), all_eco, self.eco_curve.transform(all_eco)]
        for site in ("oral", "gut"):
            pc = self.pcs[site][pos]
            pieces += [pc, self.pc_curves[site].transform(pc[:, :self.config["curved_components_per_site"]])]
        # Permit diet modification by measured nonmicrobial background in both models.
        pieces += [background * diet[:, j:j + 1] for j in range(2)]
        result = np.column_stack(pieces)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite matched main-effect design")
        return result

    def main(self, training, value=None):
        a = self.main_scaler.transform(self._main_raw(training, value))
        return np.column_stack([np.ones(len(a)), a])

    def interaction_raw(self, spec, training, value=None):
        pos = 0 if training else 1
        sites = ("oral", "gut") if spec["site"] == "both" else (spec["site"],)
        if spec["module"] == "ecology":
            markers = np.column_stack([self.ecos[site][pos][:, [1, 3]] for site in sites])
            reference = np.column_stack([self.ecos[site][0][:, [1, 3]] for site in sites]).mean(0)
        else:
            markers = np.column_stack([self.pcs[site][pos][:, :self.config["modifier_components_per_site"]] for site in sites])
            reference = np.column_stack([self.pcs[site][0][:, :self.config["modifier_components_per_site"]] for site in sites]).mean(0)
        markers = markers - reference
        count = 1 if spec["form"] == "linear" else 2
        diet = self.diet_basis(training, value)
        return np.column_stack([markers * diet[:, j:j + 1] for j in range(count)])

    def interactions(self, spec):
        key = spec["name"]
        if key not in self.interaction_states:
            raw = self.interaction_raw(spec, True)
            projection = linalg.lstsq(self.xt, raw, cond=1e-9, lapack_driver="gelsy", check_finite=False)[0]
            residual = raw - self.xt @ projection
            keep = np.flatnonzero(residual.std(0) > 1e-8 * np.maximum(raw.std(0), 1.0))
            scaler = StandardScaler().fit(residual[:, keep]) if len(keep) else None
            self.interaction_states[key] = InteractionMap(projection, keep, scaler)
        state = self.interaction_states[key]
        return state.transform(self, spec, True), state.transform(self, spec, False), state

    def support(self):
        """Outcome-free local range diagnostic; never an analysis exclusion."""
        lo, hi = self.config["contrast"]
        train = np.column_stack([self.background_t, *[self.pcs[site][0][:, :6] for site in ("oral", "gut")], self.markers[0]])
        valid = np.column_stack([self.background_v, *[self.pcs[site][1][:, :6] for site in ("oral", "gut")], self.markers[1]])
        scale = StandardScaler().fit(train)
        neighbors = min(self.config["support_neighbors"], len(train))
        nn = NearestNeighbors(n_neighbors=neighbors).fit(scale.transform(train))
        ids = nn.kneighbors(scale.transform(valid), return_distance=False)
        bounds = np.quantile(self.diet(True)[ids], self.config["support_quantiles"], axis=1)
        return (bounds[0] <= lo) & (bounds[1] >= hi)

    def loadings(self):
        result = {}
        for site, state in self.pc_states.items():
            # Frozen CLR output indices determine which original taxon each loading names.
            indices = np.asarray(self.role.split.preprocessing["CLR"][site]["output_indices"], int)
            taxa = np.asarray(self.role.taxa[site], object)[indices]
            pcs = []
            for row in state["model"].components_[:self.config["modifier_components_per_site"]]:
                keep = np.argsort(-abs(row), kind="stable")[:8]
                pcs.append([dict(taxon=str(taxa[j]), loading=float(row[j])) for j in keep])
            result[site] = dict(top_loadings=pcs, explained_variance_ratio=state["model"].explained_variance_ratio_.tolist())
        return result

    def slim(self):
        """Do not duplicate whole-cohort raw microbiomes in every saved fold model."""
        self.role.raw, self.role.extra, self.role.cache = {}, {}, {}
        for name in ("ot", "ov", "gt", "gv"):
            setattr(self.role.split, name, np.empty((0, 0)))


def partial_out(x, values):
    """Ordinary finite-basis partial regression, retaining Q for leverage adjustment."""
    q, r, _ = linalg.qr(x, mode="economic", pivoting=True, check_finite=False)
    diagonal = abs(np.diag(r))
    rank = int((diagonal > max(float(diagonal.max(initial=0)), 1) * 1e-9).sum())
    if rank == 0 or rank >= len(x) - 20:
        raise ValueError("Too few residual degrees of freedom for the fixed inference basis")
    q = q[:, :rank]
    return values - q @ (q.T @ values), q, rank
