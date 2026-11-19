"""Inner household visits and target-free, split-specific microbiome transforms."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

import protocol as p
from preprocessing_v1_20261030 import MicrobiomeCLR
from source import Background, check


@dataclass(frozen=True)
class Site:
    abundance: np.ndarray
    taxa: list


def read_sites(study):
    return {
        site: Site(study.snapshot.npz(f"{p.INPUTS}/{site}_raw_TRE_ONLY.npz", ("abundance",))["abundance"],
                   study.snapshot.json(f"{p.INPUTS}/{site}_taxa.json"))
        for site in ("oral", "gut")
    }


def fit_site(site, source, ids, training):
    """Refit archive-style CLR and the leading normalized PCs using training X."""
    model = MicrobiomeCLR().fit(source.abundance[training], source.taxa, ids[training], site=site)
    transformed = model.transform(source.abundance, source.taxa, site=site)
    support = transformed["supported"]
    use = training & support
    z = transformed["values"]
    check(use.sum() > p.N_PC and z.shape[1] >= p.N_PC, "V3_INNER_PCA_INSUFFICIENT " + site)
    mean = z[use].mean(axis=0)
    center = z[use] - mean
    cov = center.T @ center
    vals, vec = eigh(cov, subset_by_index=[len(cov)-p.N_PC, len(cov)-1], driver="evr")
    check(vals[0] > p.RANK_TOL * max(vals[-1], 1.), "V3_INNER_PCA_RANK " + site)
    vec = vec[:, ::-1]
    anchors = vec[np.argmax(abs(vec), axis=0), np.arange(p.N_PC)]
    vec = vec * np.where(anchors >= 0, 1., -1.)
    scores = np.zeros((len(z), p.N_PC))
    scores[support] = (z[support] - mean) @ vec
    scales = scores[use].std(axis=0)
    check(np.isfinite(scales).all() and np.all(scales > 0), "V3_INNER_PCA_SCALE " + site)
    return scores / scales, support


class InnerContexts:
    def __init__(self, study, sites):
        self.study = study
        self.sites = sites
        self.cache = {}

    def split(self, outer_fold, inner_fold):
        check(outer_fold in p.FOLDS and inner_fold in p.FOLDS and outer_fold != inner_fold,
              "V3_INVALID_INNER_ROLES")
        study = self.study
        outside = study.fold == outer_fold
        valid = study.fold == inner_fold
        train = ~(outside | valid)
        check(not np.intersect1d(study.family[train], study.family[valid | outside]).size
              and not np.intersect1d(study.family[valid], study.family[outside]).size,
              "V3_NESTED_HOUSEHOLD_OVERLAP")
        # The train set is identical when the two excluded folds swap roles.
        key = tuple(sorted((outer_fold, inner_fold)))
        if key not in self.cache:
            fits = [fit_site(s, self.sites[s], study.keys[0], train) for s in ("oral", "gut")]
            pcs = np.column_stack([fit[0] for fit in fits])
            support = fits[0][1] & fits[1][1]
            self.cache[key] = pcs, support
        pcs, support = self.cache[key]
        archived = study.backgrounds[outer_fold]
        background = Background(train, support, pcs, archived.archived, archived.upf)
        return InnerStudy(study, inner_fold, background, ~outside)


class InnerStudy:
    """Use the same predictor contract with training roles narrower than v1."""

    def __init__(self, study, fold, background, active):
        self.family, self.numeric = study.family, study.numeric
        self.markers, self.ecology, self.smoking = study.markers, study.ecology, study.smoking
        self.backgrounds = {fold: background}
        self.active = active

    def eligible(self, exposure, endpoint, fold, cleaned):
        bg = self.backgrounds[fold]
        days, energy, _ = p.EXPOSURES[exposure]
        kcal = self.numeric[energy]
        ok = (self.active & bg.supported & (self.numeric[days] >= 3)
              & np.isfinite(self.numeric["age_calendar_years"])
              & np.isin(self.numeric["sex_code"], (0, 1)) & np.isfinite(kcal)
              & np.isfinite(bg.archived[exposure]).all(axis=1) & np.isfinite(cleaned))
        check(np.all(kcal[ok] > 0), "V3_INNER_ENERGY")
        if exposure == "fiber_density":
            fiber = self.numeric[exposure][ok]
            check(np.isfinite(fiber).all() and np.all(fiber >= 0), "V3_INNER_FIBER")
        else:
            amount = bg.archived[exposure][ok, 0]*bg.upf["scale"] + bg.upf["center"]
            check(np.all((-1e-8 <= amount) & (amount <= 1. + 1e-8)), "V3_INNER_UPF_FRACTION")
        return ok
