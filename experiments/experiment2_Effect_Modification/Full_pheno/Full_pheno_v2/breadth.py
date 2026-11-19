"""Matched prediction, training-only cross-endpoint pooling and family inference."""

from dataclasses import dataclass

import numpy as np

import design as d
import protocol as p   # pinned Full_pheno v1 helpers
from models import create_design, fiber_basis, predict, products, standardize
from source import check, clean_outcome


@dataclass(frozen=True)
class Profile:
    z_valid: np.ndarray
    gram: np.ndarray
    rhs: np.ndarray
    scale: float

    def fit(self, penalty, prior=0.):
        beta = np.linalg.solve(self.gram + penalty * np.eye(len(self.rhs)), self.rhs + penalty * prior)
        return self.z_valid @ beta * self.scale


def fit_profile(columns, main, inverse, residual):
    n = len(residual)
    # Both the interaction projection and the normal equations use only training rows.
    z = columns - main @ (inverse @ columns[:n])
    a = z[:n]
    sd = float(np.linalg.norm(residual) / np.sqrt(n))
    scale = sd if sd > p.RANK_TOL else 1.
    return Profile(z[n:], a.T @ a / n, a.T @ (residual / scale) / n, scale)


def shared_coefficients(profiles):
    domains = [d.domain(e) for e in profiles]
    weights = {e: 1. / (len(set(domains)) * domains.count(d.domain(e))) for e in profiles}
    g = sum(weights[k] * prof.gram for k, prof in profiles.items())
    r = sum(weights[k] * prof.rhs for k, prof in profiles.items())
    return np.linalg.solve(g + d.SHARED_PENALTY * np.eye(len(r)), r)


def basis(study, exposure, fold):
    bg = study.backgrounds[fold]
    raw = (fiber_basis(study.numeric[exposure]) if exposure == "fiber_density"
           else bg.archived[exposure])
    use = bg.role & bg.supported & np.isfinite(raw).all(axis=1)
    check(int(use.sum()) >= 3, "TOO_FEW_TRAINING_ROWS_FOR_V2_BASIS")
    order = np.r_[np.flatnonzero(use), np.flatnonzero(~use)]
    n = int(use.sum())
    inv = np.argsort(order)
    linear, scales = standardize(study.markers[order], n)
    score = study.markers[order, 0]
    hinge = np.maximum(score - np.median(score[:n]), 0.) / scales[0]
    panel = np.column_stack((linear[:, 0], hinge - hinge[:n].mean()))[inv]
    shannon = linear[inv, 1:2]
    eco = standardize(study.ecology[order], n)[0][inv]
    # The PCs already have a fold-specific training scale, shared across endpoints.
    pcs = bg.pcs
    blocks = dict(panel=panel, shannon=shannon, panel_shannon=np.column_stack((panel, shannon)),
                  ecology=eco, oral_pc5=pcs[:, :p.N_PC], gut_pc5=pcs[:, p.N_PC:],
                  both_pc5=pcs, panel_shannon_both_pc5=np.column_stack((panel, shannon, pcs)))
    return raw, blocks


@dataclass
class Fold:
    rows: np.ndarray
    y: np.ndarray
    preds: dict
    counts: dict


def evaluate_split(study, exposure, outcomes, fold):
    """Fit all endpoints on the same frozen household partition."""
    raw, views = basis(study, exposure, fold)
    results, work = {}, {shape: {} for shape in d.SHARED_MODES}
    roles = study.backgrounds[fold].role
    for endpoint, outcome in outcomes.items():
        y = clean_outcome(outcome.y, roles)
        ok = study.eligible(exposure, endpoint, fold, y)
        tr, te = np.flatnonzero(roles & ok), np.flatnonzero(~roles & ok)
        tg, vg = np.unique(study.family[tr]), np.unique(study.family[te])
        check(len(tr) >= 3 and len(te) >= 2 and len(tg) >= 2 and len(vg) > 0,
              f"MISSING_FOLD_SUPPORT {exposure} {endpoint} {fold}")
        check(not np.intersect1d(tg, vg).size, "V2_FAMILY_OVERLAP")
        rows = np.r_[tr, te]
        n = len(tr)
        des = create_design(study, exposure, fold, rows, n)
        old = predict(des, y[tr])
        predictions = {name: old[:, j] for j, name in enumerate((d.BASE, *d.REFS))}
        x = des.main
        inverse = np.linalg.pinv(x[:n], rcond=p.RANK_TOL)
        residual = y[tr] - x[:n] @ (inverse @ y[tr])
        xdose = raw[rows]
        for view, marker in views.items():
            mat = products(xdose, marker[rows])
            for shape in d.SHARED_MODES:
                cols = mat[:, :marker.shape[1]] if shape == "linear" else mat
                prof = fit_profile(cols, x, inverse, residual)
                for penalty in d.PENALTIES:
                    name = d.Recipe(view, shape, penalty).name
                    predictions[name] = old[:, 0] + prof.fit(penalty)
                if view == "both_pc5":
                    work[shape][endpoint] = prof
        counts = dict(fold=fold, training_people=len(tr), held_people=len(te),
                      training_households=len(tg), held_households=len(vg),
                      held_outcome_winsorized=int((y[te] != outcome.y[te]).sum()),
                      held_outcome_8sd_excluded=int((~roles & np.isfinite(outcome.y) & ~np.isfinite(y)).sum()))
        results[endpoint] = Fold(te, y[te], predictions, counts)

    for shape, profiles in work.items():
        check(set(profiles) == set(outcomes), "INCOMPLETE_SHARED_TRAINING")
        common = shared_coefficients(profiles)
        for endpoint, prof in profiles.items():
            preds = results[endpoint].preds
            preds[f"joint_both_pc5/{shape}/pooled"] = preds[d.BASE] + (prof.z_valid @ common) * prof.scale
            preds[f"joint_both_pc5/{shape}/partial"] = preds[d.BASE] + prof.fit(d.SHARED_ADAPT, common)
    return results


@dataclass(frozen=True)
class Task:
    exposure: str
    endpoint: str
    people: int
    households: int
    folds: list
    # Per household: N, sum(centered y), sum(centered y squared), off SSE, and off minus on SSE.
    loss: np.ndarray


def collect(exposure, endpoint, parts, fold_ids, codes, n_families):
    check(tuple(part.counts["fold"] for part in parts) == tuple(p.FOLDS)
          and all(len(part.rows) == len(part.y) == part.counts["held_people"]
                  and np.all(fold_ids[part.rows] == part.counts["fold"]) for part in parts),
          "V2_HELD_FOLD_MISMATCH")
    rows = np.concatenate([part.rows for part in parts])
    y = np.concatenate([part.y for part in parts])
    check(len(np.unique(rows)) == len(rows), "V2_DUPLICATE_SCORED_PERSON")
    g = codes[rows]
    hh = len(np.unique(g))
    check(sum(part.counts["held_households"] for part in parts) == hh, "V2_DUPLICATE_SCORED_FAMILY")
    all_names = (d.BASE, *d.NAMES)
    check(all(set(part.preds) == set(all_names) for part in parts), "V2_MISSING_PREDICTION_COLUMN")
    preds = np.concatenate([np.column_stack([part.preds[name] for name in all_names]) for part in parts], axis=0)
    check(np.isfinite(y).all() and np.isfinite(preds).all(), "V2_NONFINITE_SCORE_INPUT")
    yc = y - y.mean()
    off = y - preds[:, 0]
    on = y[:, None] - preds[:, 1:]
    # Difference of squares, evaluated before household aggregation.
    gain = (preds[:, 1:] - preds[:, :1]) * (off[:, None] + on)
    cols = np.column_stack((np.ones(len(y)), yc, yc**2, off**2, gain))
    check(np.isfinite(cols).all(), "V2_NONFINITE_SCORE_MOMENTS")
    hh_loss = np.column_stack([np.bincount(g, weights=c, minlength=n_families) for c in cols.T])
    return Task(exposure, endpoint, len(rows), hh, [part.counts for part in parts], hh_loss)


def point_scores(task):
    n, s, s2, off_sse, *sse_gain = task.loss.sum(axis=0)
    sst = s2 - s**2/n
    check(np.isfinite(sst) and sst > 0, f"V2_DEGENERATE_ENDPOINT {task.exposure} {task.endpoint}")
    off_r2 = 1. - off_sse / sst
    delta = np.asarray(sse_gain) / sst
    return np.r_[off_r2, off_r2 + delta], delta


def replay(task, reference):
    check(reference["people"] == task.people and reference["households"] == task.households and
          reference["folds"] == task.folds, f"V1_REPLAY_COHORT_DIFFERS {task.exposure} {task.endpoint}")
    old = np.array([reference["score"]["models"][label]["r2"] for label in p.PREDICTORS])
    vals, _ = point_scores(task)
    err = float(np.max(abs(vals[:3] - old)))
    check(np.isfinite(err) and err <= d.REPLAY_ATOL,
          f"V1_REPLAY_R2_DIFFERS {task.exposure} {task.endpoint}")
    return err


def joint_bootstrap(tasks, draws, seed):
    """Use identical multinomial household weights for every endpoint and candidate."""
    packed = np.concatenate([t.loss for t in tasks], axis=1)
    rows = np.any(np.stack([t.loss[:, 0] > 0 for t in tasks], axis=1), axis=1)
    packed = packed[rows]
    ng = len(packed)
    check(ng >= 2 and draws >= 20, "V2_BOOTSTRAP_SUPPORT")
    ncols = tasks[0].loss.shape[1]
    rng = np.random.default_rng(seed)
    prob = np.full(ng, 1. / ng)
    delta = np.empty((draws, len(tasks), len(d.NAMES)))
    stable = np.ones((len(tasks), len(d.NAMES)), dtype=bool)
    for start in range(0, draws, 96):
        n_batch = min(96, draws - start)
        w = rng.multinomial(ng, prob, size=n_batch)
        t = (w @ packed).reshape(n_batch, len(tasks), ncols)
        n, sum_y, sum_yy = (t[:, :, j] for j in range(3))
        check(np.all(n > 0), "V2_BOOTSTRAP_EMPTY_ENDPOINT")
        sst = sum_yy - sum_y**2 / n
        check(np.all(np.isfinite(sst) & (sst > 0)), "V2_BOOTSTRAP_DEGENERATE_ENDPOINT")
        gains = t[:, :, 4:] / sst[:, :, None]
        off_r2 = 1. - t[:, :, 3:4] / sst[:, :, None]
        check(np.isfinite(gains).all() and np.isfinite(off_r2).all(), "V2_BOOTSTRAP_NONFINITE_SCORE")
        delta[start:start+n_batch] = gains
        stable &= np.all((off_r2 > p.R2_EPS) & (off_r2 + gains > p.R2_EPS), axis=0)
    return delta, stable


def make_summary(tasks, draws=p.BOOTSTRAP_DRAWS):
    vals, delta = map(np.stack, zip(*(point_scores(task) for task in tasks), strict=True))
    boot, stable = joint_bootstrap(tasks, draws, p.BOOTSTRAP_SEED)
    s = boot.std(axis=0, ddof=1)
    flat = s <= 1e-15
    deviation = abs(boot - delta)
    flat_radius = deviation[:, flat].max(axis=0)
    # Near-constant cells use their largest observed deviation and do not calibrate the max statistic.
    np.divide(deviation, s, out=deviation, where=~flat)
    deviation[:, flat] = 0.
    max_error = deviation.max(axis=(1, 2))
    crit = float(np.quantile(max_error, 1. - p.ALPHA, method="higher"))
    radius = crit * s
    radius[flat] = flat_radius
    lo, hi = delta - radius, delta + radius
    marg = np.quantile(boot, (.025, .975), axis=0)
    details = []
    for i, task in enumerate(tasks):
        for j, name in enumerate(d.NAMES):
            r0, r1 = (float(vals[i, k]) for k in (0, j+1))
            gain = float(delta[i, j])
            ratio_ok = stable[i, j] and r0 > p.R2_EPS and r1 > p.R2_EPS
            details.append(dict(exposure=task.exposure, endpoint=task.endpoint, domain=d.domain(task.endpoint),
                                people=task.people, households=task.households, recipe=name,
                                r2_off=r0, r2_on=r1, delta_r2_pp=gain*100,
                                point95_low_pp=float(marg[0, i, j]*100),
                                point95_high_pp=float(marg[1, i, j]*100),
                                simultaneous95_low_pp=float(lo[i, j]*100),
                                simultaneous95_high_pp=float(hi[i, j]*100),
                                share_pct=100*gain/r1 if ratio_ok else None))
    return dict(draws=draws, max_bias_scaled_critical=crit, simultaneous_cells=len(details),
                ranking=rank(details), details=details)


def rank(details):
    ranking = []
    eps = 1e-9  # percentage points, only to avoid zero/threshold rounding changing counts
    threshold = 100*d.MIN_GAIN
    for name in d.NAMES:
        arm = [row for row in details if row["recipe"] == name]
        check(len(arm) == len(p.MODES["all"]) and
              {(row["exposure"], row["endpoint"]) for row in arm} == set(p.MODES["all"]),
              "V2_INCOMPLETE_BREADTH_DENOMINATOR")
        delta = np.array([row["delta_r2_pp"] for row in arm])
        lower = np.array([row["simultaneous95_low_pp"] for row in arm])
        hits = delta >= threshold - eps
        per_diet = {diet: sum(bool(i) and row["exposure"] == diet for i, row in zip(hits, arm, strict=True))
                    for diet in p.EXPOSURES}
        per_domain = {domain: sum(bool(i) and row["domain"] == domain for i, row in zip(hits, arm, strict=True))
                      for domain in ("glucose", "adiposity", "liver")}
        domain_rates = [per_domain[key] / sum(row["domain"] == key for row in arm) for key in per_domain]
        ranking.append(dict(recipe=name, positive=int((delta > eps).sum()), at_least_010=int(hits.sum()),
                            at_least_025=int((delta >= 100*d.LARGER_GAIN-eps).sum()),
                            median_delta_r2_pp=float(np.median(delta)),
                            corrected_positive=int((lower > eps).sum()),
                            corrected_at_least_010=int((lower >= threshold-eps).sum()),
                            majority=bool(hits.sum() >= 10), corrected_majority=bool((lower >= threshold-eps).sum() >= 10),
                            fiber_hits=per_diet["fiber_density"], upf_hits=per_diet["UPF"],
                            glucose_hits=per_domain["glucose"], adiposity_hits=per_domain["adiposity"],
                            liver_hits=per_domain["liver"], domain_balanced_hit_rate=float(np.mean(domain_rates)),
                            at_least_010_both_diets=all(per_diet.values()),
                            at_least_010_all_domains=all(per_domain.values()),
                            material_loss=int((delta <= -threshold + eps).sum()),
                            worst_delta_r2_pp=float(delta.min())))
    return sorted(ranking, key=lambda r: (-r["at_least_010"], -min(r["fiber_hits"], r["upf_hits"]),
                                          -r["domain_balanced_hit_rate"], -r["median_delta_r2_pp"], r["recipe"]))
