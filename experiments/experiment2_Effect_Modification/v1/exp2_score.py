"""Pooled out-of-fold score and fixed-prediction household uncertainty."""

from collections import Counter

import numpy as np

from exp2_engine import ANCHOR_WORKFLOWS, BASE, WORKFLOWS


def r2(y, predictions):
    denom = np.sum((y - y.mean()) ** 2)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("R2 needs nonconstant finite targets")
    return 1 - np.sum((y[:, None] - predictions) ** 2, axis=0) / denom


def house_intervals(y, pred, households, comparators, reps, seed):
    """Paired bootstrap resamples whole households; training and OOF stay fixed."""
    names, group = np.unique(households, return_inverse=True)
    h = len(names)
    yc = y - y.mean()
    errors = (y[:, None] - pred) ** 2
    improve = np.column_stack([errors[:, j:j + 1] - errors for j in comparators])
    counts = np.bincount(group)
    ysum = np.bincount(group, weights=yc)
    ysq = np.bincount(group, weights=yc ** 2)
    stats = np.column_stack([counts, ysum, ysq] + [
        np.bincount(group, weights=c) for c in improve.T
    ])
    rng, parts = np.random.default_rng(seed), []
    for j in range(0, reps, 48):
        rows = rng.integers(0, h, size=(min(48, reps - j), h))
        # Chunking bounds memory. Counts preserve household cluster size.
        total = np.vstack([np.bincount(a, minlength=h) for a in rows]) @ stats
        denom = total[:, 2] - total[:, 1] ** 2 / total[:, 0]
        if not (denom > 0).all():
            raise ValueError("Degenerate household resample")
        parts.append(total[:, 3:] / denom[:, None])
    intervals = np.quantile(np.concatenate(parts), (0.025, 0.975), axis=0).T
    return intervals.reshape(len(comparators), pred.shape[1], 2)


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1e-14 else None


def validate_fold(dataset, engine, repeat, fold, arrays):
    ref = dataset.reference
    expected = np.flatnonzero((ref["outer"][repeat] == fold) & np.isfinite(ref["y"][repeat]))
    n, k = len(expected), len(engine.names)
    shapes = {"rows": (n,), "y": (n,), "raw_y": (n,), "support": (n,),
              "predictions": (n, k), "contrast": (n, k), "anchor": (n, len(ANCHOR_WORKFLOWS))}
    if set(arrays) != set(shapes) or any(arrays[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Unexpected checkpoint shape for r%d f%d" % (repeat, fold))
    if not np.issubdtype(arrays["rows"].dtype, np.integer) or not np.array_equal(arrays["rows"], expected):
        raise ValueError("Checkpoint rows differ from r%d f%d assignment" % (repeat, fold))
    if not np.array_equal(arrays["y"], ref["y"][repeat, expected]):
        raise ValueError("Checkpoint processed Y differs from reference")
    if not np.array_equal(arrays["raw_y"], dataset.y[expected]):
        raise ValueError("Checkpoint raw Y differs from reference")
    if arrays["support"].dtype != np.dtype(bool) or not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Invalid checkpoint values")


def summarize(dataset, engine, units):
    cfg, names = engine.cfg, engine.names
    groups = dataset.index.split_unit_id.astype(str).to_numpy()
    expected = np.flatnonzero(np.isfinite(dataset.reference["y"][0]))
    all_repeats, scores, quality, joint_comparison = [], [], [], []
    if len(units) != cfg["outer_repeats"] * cfg["outer_folds"]:
        raise ValueError("Incomplete OOF coverage")
    for repeat in range(cfg["outer_repeats"]):
        entries = [units[(repeat, k)] for k in range(cfg["outer_folds"])]
        parts, metadata = zip(*entries)
        for fold, arrays in enumerate(parts):
            validate_fold(dataset, engine, repeat, fold, arrays)
        merged = {key: np.concatenate([a[key] for a in parts]) for key in parts[0]}
        order = np.argsort(merged["rows"])
        merged = {key: a[order] for key, a in merged.items()}
        rows = merged["rows"]
        if not np.array_equal(rows, expected) or not np.array_equal(merged["y"], dataset.reference["y"][repeat, rows]):
            raise ValueError("Final OOF score input differs from the frozen rows/outcomes")
        if not np.array_equal(merged["raw_y"], dataset.y[rows]):
            raise ValueError("Raw-y diagnostic alignment mismatch")
        y, pred, cont = (merged[k] for k in ("y", "predictions", "contrast"))
        rsq, raw = r2(y, pred), r2(merged["raw_y"], pred)
        ridge_idx = names.index("select_ridge")
        core_idx = names.index("select_core_other")
        v2_idx = names.index("select_v2_other")
        v1_release_idx = names.index("select_v1_release_other")
        additive_idx = names.index("ridge_pc10_both")
        joint_idx = names.index("oral_gut_plus_both")
        legacy_idx = names.index("select_other")
        previous_idx = names.index("select_expanded")
        refs = (0, ridge_idx, core_idx, v2_idx, v1_release_idx, additive_idx, legacy_idx, previous_idx)
        ci, ridge_ci, core_ci, v2_ci, release_ci, joint_ci, legacy_ci, previous_ci = house_intervals(
            y, pred, groups[rows], refs, cfg["prediction_bootstraps"], cfg["model_seed"] + repeat,
        )
        joint_comparison.append(dict(delta=float(rsq[joint_idx] - rsq[additive_idx]),
                                     ci=joint_ci[joint_idx].tolist()))
        correction_size = np.abs(pred - pred[:, :1])
        fold_gains = []
        for a in parts:
            if np.ptp(a["y"]) > 0:
                fold_r2 = r2(a["y"], a["predictions"])
                fold_gains.append(fold_r2 - fold_r2[0])
        worst_fold = np.min(fold_gains, axis=0) if fold_gains else [None] * len(names)
        supported = merged["support"].astype(bool)
        sds = cont.std(0)
        sds_supported = cont[supported].std(0) if supported.any() else [None] * len(names)
        support_strata = {}
        for label, mask in (("supported", supported), ("outside", ~supported)):
            values = r2(y[mask], pred[mask]) if mask.sum() > 1 and np.ptp(y[mask]) > 0 else None
            support_strata[label] = dict(
                people=int(mask.sum()), households=len(set(groups[rows][mask])),
                delta=None if values is None else values - values[0],
                contrast_quantiles=(np.quantile(cont[mask], (0.10, 0.50, 0.90), axis=0).T
                                    if mask.any() else None))
        record = {}
        for j, key in enumerate(names):
            if key == BASE:
                active = 0
            elif key == "blend":
                active = sum(sum(m["weights"]) > 1e-10 for m in metadata)
            elif key in WORKFLOWS:
                active = sum(m["selected"][key] is not None for m in metadata)
            else:
                active = sum(m["choices"][key] > 0 for m in metadata)
            record[key] = dict(r2=float(rsq[j]), delta=float(rsq[j] - rsq[0]), ci=ci[j].tolist(),
                               vs_ridge=float(rsq[j] - rsq[ridge_idx]), vs_ridge_ci=ridge_ci[j].tolist(),
                               vs_core=float(rsq[j] - rsq[core_idx]), vs_core_ci=core_ci[j].tolist(),
                               vs_v2=float(rsq[j] - rsq[v2_idx]), vs_v2_ci=v2_ci[j].tolist(),
                               vs_v1_release=float(rsq[j] - rsq[v1_release_idx]),
                               vs_v1_release_ci=release_ci[j].tolist(),
                               vs_legacy=float(rsq[j] - rsq[legacy_idx]), vs_legacy_ci=legacy_ci[j].tolist(),
                               vs_previous=float(rsq[j] - rsq[previous_idx]), vs_previous_ci=previous_ci[j].tolist(),
                               applied_folds=sum(key in m["applied"] for m in metadata),
                               correction_p99=float(np.quantile(correction_size[:, j], 0.99)),
                               correction_max=float(correction_size[:, j].max()),
                               worst_fold_delta=None if worst_fold[j] is None else float(worst_fold[j]),
                               raw_y_delta=float(raw[j] - raw[0]), active_folds=active,
                               contrast_sd=float(sds[j]),
                               supported_contrast_sd=None if sds_supported[j] is None else float(sds_supported[j]))
            for label, subset in support_strata.items():
                record[key][label] = dict(
                    delta=None if subset["delta"] is None else float(subset["delta"][j]),
                    contrast_quantiles=(None if subset["contrast_quantiles"] is None
                                        else subset["contrast_quantiles"][j].tolist()))
        anchor_idx = [names.index(key) for key in ANCHOR_WORKFLOWS]
        fixed = r2(y, pred[:, :1] + merged["anchor"])
        varying = r2(y, pred[:, anchor_idx] - merged["anchor"])
        for j, key in enumerate(ANCHOR_WORKFLOWS):
            record[key]["dose_replay"] = dict(
                fixed_only=float(fixed[j] - rsq[0]),
                varying_only=float(varying[j] - rsq[0]),
                full_vs_fixed=float(rsq[anchor_idx[j]] - fixed[j]))
        status = Counter()
        for m in metadata:
            status.update(m["status"])
        quality.append(dict(people=len(rows), households=len(set(groups[rows])), support=float(supported.mean()),
                            main_penalties=[m["main_penalty"] for m in metadata],
                            selected=[m["selected"] for m in metadata],
                            warnings=sorted({w for m in metadata for w in m["warnings"]}),
                            status=dict(status),
                            failed_routes=sorted({key for m in metadata for key in m["failures"]}),
                            seconds=sum(m["seconds"] for m in metadata),
                            guard=[m["guard"] for m in metadata],
                            full_guard=[m["full_guard"] for m in metadata],
                            support_strata={key: {k: value[k] for k in ("people", "households")}
                                            for key, value in support_strata.items()},
                            projected_ranks=[m["projected_rank"] for m in metadata],
                            max_projected_condition=max(m["projected_condition"] for m in metadata)))
        all_repeats.append(cont)
        scores.append(record)
    targets = [cfg["ridge_target"]] + cfg["stretch_targets"]
    consistency = {name: correlation(all_repeats[0][:, j], all_repeats[1][:, j])
                   for j, name in enumerate(names)}
    return dict(version=cfg["version"], exposure=cfg["exposure"], outcome=cfg["outcome"], unit=cfg["unit"],
                primary_repeat=0, primary_workflow="select_full", names=names,
                workflows=list(WORKFLOWS), core_routes=[s["name"] for s in engine.specs if s["core"]],
                added_routes=[s["name"] for s in engine.specs if s["edition"] == 2],
                v3_routes=[s["name"] for s in engine.specs if s["edition"] == 3],
                release_added_routes=[s["name"] for s in engine.specs if s["edition"] == 4],
                expansion_routes=[s["name"] for s in engine.specs if s["edition"] == 5],
                followup_routes=[s["name"] for s in engine.specs if s["edition"] == 6],
                anchor_workflows=list(ANCHOR_WORKFLOWS),
                joint_comparison=joint_comparison,
                scores=scores, quality=quality, contrast=cfg["contrast"], contrast_correlation=consistency,
                targets={key: [all(s[key]["delta"] >= t for s in scores) for t in targets] for key in names},
                thresholds=targets, bootstrap_draws=cfg["prediction_bootstraps"],
                confirmation_outcomes_read=False)
