"""Nested household evaluation of pure diet-by-microbiome corrections.

No held-out target is passed to a correction learner. The baseline and every
candidate use identical evaluated rows. Fixed microbial representations enter
the shared baseline, including exact RBF, hinge and compositional features.
Tree corrections use PCs present in the nonlinear baseline microbial block;
their outcome-learned tree leaves are not separately added as main effects.
"""
from dataclasses import dataclass, replace
from collections import Counter
import time

import numpy as np
from sklearn.decomposition import PCA

from adapters import moderation_models as mm
from adapters.fermentation_score import build_score
from interaction_candidates import Candidate, options, prepare_modifier, fit_correction
from ratio_metrics import select_correction, summarize_paired
from compositional_views import prepare_views

SEED = 20261118
FIXED = ("gut_shannon", "fermentation")


def grid():
    output = options()
    for name in FIXED:
        output[name] = tuple(Candidate("gut", 1, "ridge", p) for p in (1., .1, .01))
    return output


def settings():
    return [(name, c) for name, cs in grid().items() for c in cs]


def setting_key(name, c):
    return f"{name}@{c.strength:g}"


def route_names():
    return [*grid(), "nested_selector", "selector_gut", "selector_oral", "selector_both",
            "stack_balanced", "stack_gut", "stack_linear", "selector_original",
            "stack_compositional", "stack_expanded"]


@dataclass
class Prepared:
    role: object
    pcs: dict
    modifiers: dict
    fixed: dict
    views: dict
    tensors: dict
    feature_audit: dict


def microbial(pcs, site, k):
    sites = ("oral", "gut") if site == "both" else (site,)
    return tuple(np.column_stack([pcs[s][p][:, :k] for s in sites]) for p in (0, 1))


def prepare(source, split):
    role = mm.prepare(source, split, shared_host=True)
    pcs, ranks = {}, {}
    for name, train, valid in (("oral", split.ot, split.ov), ("gut", split.gt, split.gv)):
        k = min(40, train.shape[1], len(train) - 1)
        pca = PCA(n_components=k, svd_solver="randomized", random_state=SEED).fit(train)
        pcs[name] = mm.scale_columns(pca.transform(train), pca.transform(valid))
        ranks[name] = pcs[name][0].shape[1]
        if ranks[name] < 1:
            raise ValueError("NO_NONCONSTANT_MICROBIAL_PCS")
    score = getattr(source, "fermentation_score", None)
    if score is None:
        score, _, _ = build_score(source.raw["gut"], source.taxa["gut"])
    rows = (split.train_rows, split.rows)
    values = tuple(score[r, None] for r in rows)
    if not all(np.isfinite(x).all() for x in values):
        raise ValueError("FERMENTATION_SCORE_MISSING_IN_FIXED_COHORT")
    panel = mm.scale_columns(*values, allow_constant=True)
    fixed = {"gut_shannon": role.markers["gut_shannon"], "fermentation": panel}
    modifiers = {}
    tensors = {}
    views, composition_audit = {}, {}
    for site in ("oral", "gut"):
        views[site], composition_audit[site] = prepare_views(
            source.raw[site][split.train_rows], source.raw[site][split.rows], source.taxa[site])
    views["both"] = {name: tuple(np.column_stack([views[site][name][p] for site in ("oral", "gut")])
                                   for p in (0, 1)) for name in ("taxa", "balance", "two_part")}
    for site in ("gut", "oral", "both"):
        for k in (10, 20):
            c = Candidate(site, k, "rbf", 1.)
            modifiers[(site, k)] = prepare_modifier(*microbial(pcs, site, k), c)
        for k in (5, 10):
            c = Candidate(site, k, "tensor", 1., n_dose=2)
            tensors[(site, k, "tensor")] = prepare_modifier(*microbial(pcs, site, k), c)
    # The left block is the archived host/diet/context design. The right block
    # contains the fixed microbial input/marginal representations. Learned VC
    # tree leaves are not extra baseline terms; their PC inputs are included.
    design = tuple(np.column_stack([
        role.design[p][:, :role.width], pcs["oral"][p], pcs["gut"][p],
        *(role.markers[m][p] for m in mm.MARKERS),
        *(role.markers[m][p] ** 2 for m in mm.MARKERS), panel[p], panel[p] ** 2,
        *(value[p] for site in ("oral", "gut") for value in views[site].values()),
        *(value[p] for value in tensors.values()),
        *(value[p] for value in modifiers.values()),
    ]) for p in (0, 1))
    if any(np.isinf(x).any() for x in design):
        raise ValueError("INFINITE_BASELINE_DESIGN")
    role = replace(role, design=design)
    return Prepared(role, pcs, modifiers, fixed, views, tensors, dict(
        actual_pc_ranks=ranks, baseline_columns=design[0].shape[1],
        left_columns=role.width, rff_blocks=len(modifiers), rff_columns=288,
        shared_host_columns=list(role.shared_columns),
        compositional_views=composition_audit,
        tensor_blocks=len(tensors), tensor_columns=sum(v[0].shape[1] for v in tensors.values()),
        baseline_constraint="host/diet/context OR microbial block; age/sex shared only",
    ))


def predictions(prepared, residual, chosen_settings):
    role = prepared.role
    out, contrasts, details = [], [], {}
    for name, c in chosen_settings:
        if name in FIXED:
            z = prepared.fixed[name]
            modifier = prepare_modifier(*z, c)
        else:
            z = (microbial(prepared.pcs, c.site, c.n_components) if c.view == "pc"
                 else prepared.views[c.site][c.view])
            modifier = (prepared.modifiers.get((c.site, c.n_components)) if c.kind == "rbf"
                        else prepared.tensors.get((c.site, c.n_components, "tensor"))
                        if c.kind == "tensor" else None)
        pred, contrast, detail = fit_correction(
            *z, *role.dose, role.delta, residual, c, prepared_modifier=modifier)
        if not detail["converged"]:
            raise RuntimeError("INTERACTION_DID_NOT_CONVERGE " + setting_key(name, c))
        out.append(pred)
        contrasts.append(contrast)
        details[setting_key(name, c)] = detail
    return np.column_stack(out), np.column_stack(contrasts), details


def policies(y, base, corrections):
    specs = settings()
    keys = [setting_key(n, c) for n, c in specs]
    choices, weights = {}, {}

    def choose(name, indices):
        result = select_correction(y, base, corrections[:, indices], names=[keys[i] for i in indices])
        w = np.zeros(len(specs))
        if result["index"] is not None:
            w[indices[result["index"]]] = 1
        weights[name] = w
        choices[name] = {k: v for k, v in result.items() if k != "scores"}

    for name in grid():
        choose(name, [i for i, (n, _) in enumerate(specs) if n == name])
    choose("nested_selector", list(range(len(specs))))
    for site in ("gut", "oral", "both"):
        choose("selector_" + site, [i for i, (n, c) in enumerate(specs) if
                                    c.site == site and (n not in FIXED or site == "gut")])
    choose("selector_original", [i for i, (n, c) in enumerate(specs) if
        n in FIXED or (c.view == "pc" and c.kind in ("ridge", "pls", "elasticnet", "rbf"))])

    # First choose penalties within a route using the same inner predictions;
    # then average routes within each algorithm family, and families equally.
    # Scale selection reuses inner validation; its full procedure is assessed
    # only by the untouched outer fold. This is not an inner-unbiased stack CI.
    for stack, site, kinds in (("stack_balanced", None, ("ridge", "pls", "elasticnet", "rbf")),
                               ("stack_gut", "gut", ("ridge", "pls", "elasticnet", "rbf")),
                               ("stack_linear", None, ("ridge", "pls", "elasticnet"))):
        family_weights = []
        for kind in kinds:
            names = [n for n, cs in grid().items() if n not in FIXED and
                     cs[0].view == "pc" and cs[0].kind == kind and (site is None or cs[0].site == site)]
            family_weights.append(np.mean([weights[n] for n in names], axis=0))
        w = np.mean(family_weights, axis=0)
        scales = (.25, .5, 1.)
        candidates = np.column_stack([corrections @ (w * s) for s in scales])
        chosen = select_correction(y, base, candidates, names=[f"scale_{s:g}" for s in scales])
        scale = 0. if chosen["index"] is None else scales[chosen["index"]]
        weights[stack] = w * scale
        choices[stack] = {k: v for k, v in chosen.items() if k != "scores"}
        choices[stack]["nonzero_setting_weights"] = {keys[i]: float(v) for i, v in enumerate(w * scale) if v}
    # These added stacks balance REPRESENTATION classes so the 120-column
    # balance view and a family with more settings cannot dominate by count.
    groups = {
        "pc": [n for n, cs in grid().items() if n not in FIXED and cs[0].view == "pc"
               and cs[0].kind not in ("tensor", "vc_boost")],
        "taxa": [n for n, cs in grid().items() if cs[0].view == "taxa"],
        "balance": [n for n, cs in grid().items() if cs[0].view == "balance"],
        "two_part": [n for n, cs in grid().items() if cs[0].view == "two_part"],
        "tensor": [n for n, cs in grid().items() if cs[0].kind == "tensor"],
        "vc_boost": [n for n, cs in grid().items() if cs[0].kind == "vc_boost"],
    }
    for stack, classes in (("stack_compositional", ("taxa", "balance", "two_part")),
                           ("stack_expanded", tuple(groups))):
        w = np.mean([np.mean([weights[n] for n in groups[g]], axis=0) for g in classes], axis=0)
        scales = (.25, .5, 1.)
        chosen = select_correction(y, base, np.column_stack([corrections @ (w * s) for s in scales]),
                                   names=[f"scale_{s:g}" for s in scales])
        scale = 0. if chosen["index"] is None else scales[chosen["index"]]
        weights[stack] = w * scale
        choices[stack] = {k: v for k, v in chosen.items() if k != "scores"}
        choices[stack]["nonzero_setting_weights"] = {keys[i]: float(v) for i, v in enumerate(w * scale) if v}
    return weights, choices


def run_fold(source, repeat, fold, *, bases=mm.BASES, emit=lambda _: None):
    started = time.monotonic()
    inner, outer = source.roles(repeat, fold)
    held_groups = source.groups[outer.rows]
    if any(np.intersect1d(source.groups[np.r_[s.train_rows, s.rows]], held_groups).size for s in inner):
        raise ValueError("INNER_OUTER_HOUSEHOLD_OVERLAP")
    prepared = [prepare(source, split) for split in inner]
    emit(f"EXP2_V6_INNER_BASE r{repeat} f{fold}")
    fits = []
    for config in bases:
        predicted = [mm.fit_base(p.role.design[0], p.role.y[0], p.role.width, config,
                                shared_columns=p.role.shared_columns).predict(p.role.design[1]) for p in prepared]
        loss = sum(np.sum((p.role.y[1] - v) ** 2) for p, v in zip(prepared, predicted, strict=True))
        fits.append((float(loss), config, predicted))
    _, config, predicted = min(fits, key=lambda v: v[0])
    emit(f"EXP2_V6_INNER_INTERACTIONS r{repeat} f{fold} settings={len(settings())}")
    inner_corrections = []
    for p in prepared:
        residual = mm.residuals(p.role, config)
        correction, _, _ = predictions(p, residual, settings())
        inner_corrections.append(correction)
    y = np.concatenate([p.role.y[1] for p in prepared])
    base = np.concatenate(predicted)
    corrections = np.vstack(inner_corrections)
    weights, choices = policies(y, base, corrections)
    p = prepare(source, outer)
    role = p.role
    prediction = mm.fit_base(role.design[0], role.y[0], role.width, config,
                              shared_columns=role.shared_columns).predict(role.design[1])
    used = np.flatnonzero(np.any(np.vstack(list(weights.values())) != 0, axis=0))
    pred = np.zeros((len(role.rows), len(settings())))
    contrast = pred.copy()
    detail = {}
    if len(used):
        residual = mm.residuals(role, config)
        selected = [settings()[i] for i in used]
        pred[:, used], contrast[:, used], detail = predictions(p, residual, selected)
    names = route_names()
    w = np.column_stack([weights[n] for n in names])
    full, contrasts = prediction[:, None] + pred @ w, contrast @ w
    if not np.isfinite(full).all() or not np.isfinite(contrasts).all():
        raise ValueError("NONFINITE_OUTER_PREDICTIONS")
    arrays = dict(rows=role.rows, y=role.y[1], baseline=prediction, full=full,
                  contrasts=contrasts, groups=source.groups[role.rows].astype(str))
    metadata = dict(repeat=repeat, fold=fold, names=names, baseline=config[0],
                    choices=choices, outer_fit_details=detail, feature_audit=p.feature_audit,
                    people=len(role.rows), seconds=round(time.monotonic() - started, 2))
    return arrays, metadata


def summarize(source, folds, *, draws=1999):
    output = dict(repeats=[], scope="discovery only", reserved_outcomes_read=False,
                  ratio_definition="100 * (R2_full - R2_baseline) / R2_full",
                  primary_procedure="nested_selector", route_count=len(route_names()),
                  settings=len(settings()), exposure="fiber_density", outcome="D_Glu",
                  historical_v11_r2_reference_only=list(source.historical_scores))
    for repeat in (0, 1):
        parts = [folds[(repeat, f)] for f in range(5)]
        rows = np.concatenate([a[0]["rows"] for a in parts])
        expected = np.flatnonzero(np.isfinite(source.reference["y"][repeat]))
        if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), expected):
            raise ValueError("OOF_COHORT_INCOMPLETE_OR_DUPLICATED")
        arrays = {key: np.concatenate([a[0][key] for a in parts]) for key in
                  ("y", "baseline", "full", "contrasts", "groups")}
        order = np.argsort(rows)
        if not np.array_equal(arrays["y"][order], source.reference["y"][repeat, expected]):
            raise ValueError("OOF_OUTCOME_DIFFERS_FROM_V11_ANCHOR")
        result = summarize_paired(arrays["y"], arrays["baseline"], arrays["full"], arrays["groups"],
                                   names=route_names(), draws=draws, seed=SEED + repeat)
        result["repeat"] = repeat
        comparison_names = ["nested_selector", "stack_compositional", "stack_expanded"]
        result["added_methods_vs_original_selector"] = summarize_paired(
            arrays["y"], arrays["full"][:, route_names().index("selector_original")],
            arrays["full"][:, [route_names().index(name) for name in comparison_names]],
            arrays["groups"], names=comparison_names, draws=draws, seed=SEED + repeat)
        result["added_methods_vs_original_selector"]["interpretation"] = (
            "Additional predictive gain versus the original-family selector on this same expanded baseline; "
            "the comparator already contains interactions, so this is not the primary no-interaction contrast.")
        for j, name in enumerate(route_names()):
            selected = [m["choices"][name] for _, m in parts]
            result["routes"][name]["active_folds"] = sum(v["name"] != "zero" for v in selected)
            result["routes"][name]["choices"] = dict(Counter(v["name"] for v in selected))
            result["routes"][name]["contrast_sd_10_to_15_mg_dl"] = float(np.std(arrays["contrasts"][:, j]))
        valid = [(n, v) for n, v in result["routes"].items() if v["rankable_positive_gain"]]
        valid.sort(key=lambda item: item[1]["ratio_over_full_pct"], reverse=True)
        result["exploratory_observed_max"] = None if not valid else dict(route=valid[0][0], **valid[0][1])
        stable = [(n, v) for n, v in valid if v["ratio_over_full_status"] ==
                  "paired_percentile_denominator_positive_ci"]
        result["exploratory_denominator_stable_max"] = None if not stable else dict(route=stable[0][0], **stable[0][1])
        output["repeats"].append(result)
    recurring = []
    for name in route_names():
        r = [v["routes"][name] for v in output["repeats"]]
        if all(v["rankable_positive_gain"] for v in r):
            recurring.append(dict(route=name, minimum_ratio_pct=min(v["ratio_over_full_pct"] for v in r),
                                  ratios_pct=[v["ratio_over_full_pct"] for v in r],
                                  deltas_r2=[v["delta_r2"] for v in r],
                                  denominator_stable_both=all(v["ratio_over_full_status"] ==
                                      "paired_percentile_denominator_positive_ci" for v in r)))
    output["exploratory_positive_both_repeats"] = sorted(recurring, key=lambda r: r["minimum_ratio_pct"], reverse=True)
    output["limits"] = [
        "Both repeats use the same people; they are not independent replications.",
        "Delta/full is a relative prediction score, not a causal mediated proportion or fraction of a diet total effect.",
        "Bootstrap intervals condition on fitted predictions and do not cover historical search or selection of the largest displayed result.",
        "Residual nuisance learners are cross-fitted; feature transforms are fitted on the current enclosing training role, not each nuisance subfold.",
        "V5 reserved outcomes have already been inspected and are not reused to select V6 models.",
        "The V11 total R2 is a historical reference, not the denominator of these matched contrasts.",
    ]
    return output
