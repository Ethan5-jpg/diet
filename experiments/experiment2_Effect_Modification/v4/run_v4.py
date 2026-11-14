"""Nested family-level evaluation of single and combined fiber interactions."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
FILES = {"association.py", "interactions.py", "models.py", "run_v4.py", "server_run.sh"}
DEPENDENCIES = {
    "v2": {
        "marker_effects.py": "a75851ff89f328c613cc59e67da069de9eead23b5c36be7bdac2be66f14ef96b",
        "moderation_models.py": "aaa64447d8cbd7cb271f307be7b9752b13025b91e69d13ddab73c1f79dfb8b81",
        "person_inputs.py": "e13bc27e4b0e4487d406171b5f2172ba558f957b4af314a905df9ef945aaeb1a",
    },
    "v3": {"fermentation_score.py": "87950719455e3b7678ad942b4abef211ab47b79f1bca55b051e7174ccddbe6ef"},
}
VERSIONS = {
    "numpy": "1.26.4", "scipy": "1.11.4", "pandas": "2.1.0", "pyarrow": "15.0.2",
    "scikit-learn": "1.4.1.post1", "threadpoolctl": "3.3.0",
}


def emit(name, payload):
    print(f"EXP2_V4_{name} " + json.dumps(payload, allow_nan=False), flush=True)


def check_release():
    manifest = json.loads((ROOT / "upload_manifest.json").read_text(encoding="utf-8"))["files"]
    if set(manifest) != FILES:
        raise ValueError("UPLOAD_FILE_SET_CHANGED")
    for folder, checks in ((ROOT, manifest), *((ROOT.parent / v, m) for v, m in DEPENDENCIES.items())):
        for name, expected in checks.items():
            if sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"UPLOAD_HASH_MISMATCH {folder.name}/{name}")
    try:
        actual = {name: version(name) for name in VERSIONS}
    except PackageNotFoundError as exc:
        raise ValueError(f"HPP_ENV_MISSING {exc.name}") from None
    differences = [f"{name}={actual[name]} (need {pin})" for name, pin in VERSIONS.items() if actual[name] != pin]
    if differences:
        raise ValueError("HPP_ENV_MISMATCH " + ", ".join(differences))
    emit("RELEASE_OK", {"v4_files": len(FILES), "pinned_sibling_files": sum(map(len, DEPENDENCIES.values()))})


def panel_for(source, eligible):
    import numpy as np
    from fermentation_score import build_score, describe

    try:
        score, matches, abundances = build_score(source.raw["gut"], source.taxa["gut"])
        summary = describe(score, matches, abundances, eligible)
    except ValueError as exc:
        reason = str(exc)
        emit("PANEL", {"status": "unavailable", "reason": reason})
        return None
    q25, _, q75 = summary["score_q25_q50_q75"]
    if summary["scored"] != summary["rows"] or q75 <= q25:
        emit("PANEL", {"status": "unavailable", "reason": "INCOMPLETE_OR_CONSTANT_FROZEN_PANEL", **summary})
        return None
    if not np.isfinite(score).all():
        emit("PANEL", {"status": "unavailable", "reason": "INCOMPLETE_PANEL_IN_ARCHIVED_DISCOVERY", **summary})
        return None
    emit("PANEL", {"status": "ok", **summary})
    return score


def analyze(source, panel, repeats):
    import numpy as np
    from association import joint_fold, summarize_joint
    from marker_effects import MARKERS, partial_fold, summarize_effects
    from models import run_fold, summarize_prediction

    fiber = source.index.fiber_density.to_numpy(dtype=float, na_value=np.nan)
    shannon = MARKERS.index("gut_shannon")
    history = {}
    for repeat in repeats:
        fold_rows, fold_y, effects, joint = [], [], [], []
        predictions = {"core": []}
        if panel is not None:
            predictions["panel"] = []
        for fold in range(5):
            role, fitted = run_fold(source, repeat, fold, panel)
            rows, y, markers = role.rows, role.y[1], role.raw_eco[1]
            fold_rows.append(rows)
            fold_y.append(y)
            effects.append(partial_fold(rows, y, role.association_main, fiber[rows], markers))
            if panel is not None:
                joint.append(joint_fold(rows, y, role.association_main, fiber[rows],
                                        np.column_stack([markers[:, shannon], panel[rows]])))
            for name, parts in predictions.items():
                parts.append(fitted[name])
            emit("OUTER_COMPLETE", {"repeat": repeat, "fold": fold, "people": len(rows)})

        rows, y = np.concatenate(fold_rows), np.concatenate(fold_y)
        order = np.argsort(rows)
        reference = source.reference["y"][repeat]
        expected = np.flatnonzero(np.isfinite(reference))
        if not np.array_equal(rows[order], expected) or not np.array_equal(y[order], reference[expected]):
            raise ValueError("POOLED_ROWS_OR_D_GLU_DIFFERS_FROM_V11")
        groups = source.groups[expected]
        label = {"repeat": repeat, "people": len(y), "households": len(np.unique(groups))}
        source.verify_sources()
        emit("EFFECT", {**label, **summarize_effects(effects, source.groups)})
        emit("JOINT", {**label, **(summarize_joint(joint, source.groups) if panel is not None else
                                 {"status": "unavailable", "reason": "PANEL_UNAVAILABLE"})})
        for name, parts in predictions.items():
            result, contrasts = summarize_prediction(parts, y[order], groups, order, name=name, repeat=repeat)
            history.setdefault(name, []).append((result["routes"], contrasts))
            emit("PREDICT", {**label, "background": name, **result})
            stack = f"stack_{name}"
            pairs = result["comparisons"]
            emit("HIGHLIGHTS", {
                "repeat": repeat, "background": name,
                "delta_r2": {key: result["routes"][key]["delta_r2"] for key in
                             ("gut_shannon", "selector", "selector_stable", stack, "stack_stable")},
                "stack_ci95": result["routes"][stack]["ci95"],
                "stack_minus_shannon": pairs[f"{stack}_minus_gut_shannon"],
                "stack_minus_selector": pairs[f"{stack}_minus_selector"],
                "stable_stack_minus_stack": pairs[f"stack_stable_minus_{stack}"],
                **({"stack_panel_minus_stack_core": pairs["stack_panel_minus_stack_core"]} if name == "panel" else {}),
            })
    for name, entries in history.items():
        if len(entries) == 2:
            emit("STABILITY", {"background": name, **stability(*entries)})
    emit("COMPLETE", {"repeats": list(repeats), "scope": "discovery", "panel": panel is not None})


def stability(first, second):
    """Compare fitted dose contrasts for the same people across the two splits."""
    import numpy as np

    score0, contrast0 = first
    score1, contrast1 = second
    if not (score0.keys() == score1.keys() == contrast0.keys() == contrast1.keys()):
        raise ValueError("INCONSISTENT_REPEATED_ROUTE_NAMES")
    routes = {}
    for name in score0:
        x, y = contrast0[name], contrast1[name]
        if x.shape != y.shape:
            raise ValueError("INCONSISTENT_REPEATED_CONTRAST_SHAPES")
        sd = [float(x.std()), float(y.std())]
        correlation = float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1)) if min(sd) > 1e-10 else None
        deltas = [score0[name]["delta_r2"], score1[name]["delta_r2"]]
        routes[name] = {
            "delta_r2_r0_r1": deltas,
            "both_positive": min(deltas) > 0,
            "contrast_sd_mg_dl_r0_r1": sd,
            "contrast_correlation": correlation,
        }
    ranked = sorted(routes, key=lambda key: -min(routes[key]["delta_r2_r0_r1"]))
    return {"scope": "same discovery people", "top_six_by_worse_repeat": ranked[:6], "routes": routes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    check_release()
    sys.path[1:1] = [str(ROOT.parent / v) for v in DEPENDENCIES]

    import numpy as np
    from threadpoolctl import threadpool_limits
    from person_inputs import load_discovery

    with threadpool_limits(limits=args.threads):
        if args.preflight:
            from association import joint_selftest
            from fermentation_score import score_selftest
            from interactions import interaction_selftest
            from marker_effects import effect_selftest
            from models import model_selftest

            joint_selftest()
            score_selftest()
            interaction_selftest()
            effect_selftest()
            model_selftest()
            emit("PREFLIGHT_DONE", {"scope": "generated data"})
            return
        source = load_discovery(ROOT)
        eligible = np.isfinite(source.reference["y"][0])
        panel = panel_for(source, eligible)
        emit("SOURCE_READY", {"people": int(eligible.sum()), "households": len(np.unique(source.groups[eligible]))})
        analyze(source, panel, (0, 1) if args.repeat is None else (args.repeat,))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        detail = str(exc)
        allowed = ("UPLOAD_HASH_MISMATCH ", "HPP_ENV_")
        if not detail.startswith(allowed) and not (detail.isascii() and detail.replace("_", "").isalnum() and detail.isupper()):
            detail = type(exc).__name__
        print(f"EXP2_V4_STOP {detail}", file=sys.stderr, flush=True)
        sys.exit(1)
