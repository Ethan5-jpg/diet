"""Compare fiber modification by gut Shannon and a fixed three-species score."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
V2_ROOT = ROOT.parent / "v2"
FILES = {"association.py", "fermentation_score.py", "prediction.py", "run_v3.py", "server_run.sh"}
V2_DEPS = {
    "person_inputs.py": "e13bc27e4b0e4487d406171b5f2172ba558f957b4af314a905df9ef945aaeb1a",
    "moderation_models.py": "aaa64447d8cbd7cb271f307be7b9752b13025b91e69d13ddab73c1f79dfb8b81",
}
VERSIONS = {
    "numpy": "1.26.4", "scipy": "1.11.4", "pandas": "2.1.0", "pyarrow": "15.0.2",
    "scikit-learn": "1.4.1.post1", "threadpoolctl": "3.3.0",
}


def emit(tag, payload):
    print(tag + " " + json.dumps(payload, ensure_ascii=True, allow_nan=False), flush=True)


def check_release():
    manifest = json.loads((ROOT / "upload_manifest.json").read_text(encoding="utf-8"))["files"]
    if set(manifest) != FILES:
        raise ValueError("UPLOAD_FILE_SET_CHANGED")
    for folder, expected in ((ROOT, manifest), (V2_ROOT, V2_DEPS)):
        for file, checksum in expected.items():
            if sha256((folder / file).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"UPLOAD_HASH_MISMATCH {folder.name}/{file}")
    try:
        found = {name: version(name) for name in VERSIONS}
    except PackageNotFoundError as exc:
        raise ValueError(f"HPP_ENV_MISSING {exc.name}") from None
    different = [f"{name}={found[name]} (need {pin})" for name, pin in VERSIONS.items() if found[name] != pin]
    if different:
        raise ValueError("HPP_ENV_MISMATCH " + ", ".join(different))
    emit("EXP2_V3_RELEASE_OK", {"v3_files": len(FILES), "v2_dependencies": len(V2_DEPS), "environment": "pinned"})


def analyze(source, score, repeats):
    import numpy as np
    from association import fold_effect, summarize_effects
    from moderation_models import MARKERS, SEED, ZERO
    from prediction import ROUTES, run_fold, summarize_prediction

    fiber = source.index.fiber_density.to_numpy(dtype=float, na_value=np.nan)
    shannon_column = MARKERS.index("gut_shannon")
    for repeat in repeats:
        folds, effects = [], []
        for fold in range(5):
            role, base, full, _, choices, base_setting = run_fold(source, repeat, fold, score)
            rows, y = role.rows, role.y[1]
            markers = np.column_stack((role.raw_eco[1][:, shannon_column], score[rows]))
            effects.append(fold_effect(rows, y, role.association_main, fiber[rows], markers))
            folds.append((rows, y, base, full, choices))
            emit("EXP2_V3_OUTER_COMPLETE",
                 {"repeat": repeat, "fold": fold, "people": len(y), "baseline": base_setting})
        rows, y, base, full = (np.concatenate([part[k] for part in folds]) for k in range(4))
        order = np.argsort(rows)
        reference = source.reference["y"][repeat]
        expected = np.flatnonzero(np.isfinite(reference))
        if not np.array_equal(rows[order], expected) or not np.array_equal(y[order], reference[expected]):
            raise ValueError("POOLED_ROWS_OR_D_GLU_DIFFERS_FROM_V11")
        association = summarize_effects(effects, source.groups)
        prediction = summarize_prediction(y[order], base[order], full[order],
                                          source.groups[expected], seed=SEED + repeat)
        for name in ROUTES:
            prediction["routes"][name]["active_folds"] = sum(part[4][name] != ZERO for part in folds)
        source.verify_sources()
        emit("EXP2_V3_EFFECT", {"repeat": repeat, **association})
        emit("EXP2_V3_PREDICT", {"repeat": repeat, **prediction})
    emit("EXP2_V3_COMPLETE", {"repeats": list(repeats), "scope": "discovery"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Generated-data tests only")
    modes.add_argument("--availability", action="store_true", help="Aggregate score coverage without outcomes")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    check_release()
    sys.path.insert(1, str(V2_ROOT))

    import numpy as np
    from threadpoolctl import threadpool_limits
    from fermentation_score import build_score, check_archive_score, describe, score_selftest
    from person_inputs import N_DISCOVERY, V11, V11_RUN, digest, load_discovery, read_json

    with threadpool_limits(limits=args.threads):
        if args.preflight:
            from association import association_selftest
            from prediction import prediction_selftest

            score_selftest()
            association_selftest()
            prediction_selftest()
            emit("EXP2_V3_PREFLIGHT_DONE", {"scope": "generated data"})
            return
        if args.availability:
            score, matched, panel = check_archive_score(ROOT.parent.parent, digest=digest, read_json=read_json,
                                                         v11=V11, run_name=V11_RUN)
            if len(score) != N_DISCOVERY:
                raise ValueError("GUT_ARCHIVE_DISCOVERY_ROW_COUNT_CHANGED")
            emit("EXP2_V3_AVAILABILITY",
                 {"scope": "all archived discovery gut profiles", **describe(score, matched, panel, slice(None))})
            return
        source = load_discovery(ROOT)
        eligible = np.isfinite(source.reference["y"][0])
        score, matched, panel = build_score(source.raw["gut"], source.taxa["gut"])
        qc = describe(score, matched, panel, eligible)
        emit("EXP2_V3_PANEL", {"scope": "V11 evaluated discovery people", **qc})
        if qc["scored"] != qc["rows"]:
            raise ValueError("MISSING_FERMENTATION_SCORE_IN_FROZEN_COHORT")
        q25, _, q75 = qc["score_q25_q50_q75"]
        if q75 <= q25:
            raise ValueError("ZERO_FERMENTATION_SCORE_IQR")
        del panel
        repeats = (0, 1) if args.repeat is None else (args.repeat,)
        analyze(source, score, repeats)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        # Do not emit native paths or arbitrary exception text from the data loaders.
        detail = str(exc)
        tag = detail.split(maxsplit=1)[0] if detail else ""
        allowed = ("PANEL_TAXON_", "V11_REGISTERED_GUT_SOURCE_MISMATCH", "UPLOAD_HASH_MISMATCH", "HPP_ENV_")
        message = detail if tag.startswith(allowed) else tag if tag.isupper() else type(exc).__name__
        print(f"EXP2_V3_STOP {message}", file=sys.stderr, flush=True)
        sys.exit(1)
