"""Run Experiment 2 v5 discovery selection or reserved-household evaluation."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
FILES = {
    "run_discovery.py", "select_discovery.py", "server_run.sh", "validate_reserved.py",
    "frozen_model.py", "reserved_inputs.py", "reserved_outcome.py",
}
DEPENDENCIES = {
    "v2": {
        "person_inputs.py": "e13bc27e4b0e4487d406171b5f2172ba558f957b4af314a905df9ef945aaeb1a",
        "moderation_models.py": "aaa64447d8cbd7cb271f307be7b9752b13025b91e69d13ddab73c1f79dfb8b81",
    },
    "v3": {"fermentation_score.py": "87950719455e3b7678ad942b4abef211ab47b79f1bca55b051e7174ccddbe6ef"},
    "v4": {
        "interactions.py": "7ade9afcbec1062e21ee1b215b0d62ca459c7859916575a112ff6c0d198107b5",
        "models.py": "6b8ad09a4612cce1f8b5dea650ac49d57408f54c9c823723f65a25b5ee4a0560",
    },
}
VERSIONS = {
    "numpy": "1.26.4", "scipy": "1.11.4", "pandas": "2.1.0", "pyarrow": "15.0.2",
    "scikit-learn": "1.4.1.post1", "threadpoolctl": "3.3.0",
}


def emit(tag, **payload):
    print("EXP2_V5_" + tag, json.dumps(payload, allow_nan=False), flush=True)


def verify_release():
    manifest = json.loads((ROOT / "upload_manifest.json").read_text(encoding="utf-8"))["files"]
    if set(manifest) != FILES:
        raise ValueError("UPLOAD_FILE_SET_CHANGED")
    files = [(ROOT, manifest)] + [(ROOT.parent / folder, checks) for folder, checks in DEPENDENCIES.items()]
    for folder, checks in files:
        for name, expected in checks.items():
            path = folder / name
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"RELEASE_FILE_MISMATCH {folder.name}/{name}")
    try:
        actual = {name: version(name) for name in VERSIONS}
    except PackageNotFoundError as exc:
        raise ValueError(f"RELEASE_MISSING_PACKAGE {exc.name}") from None
    differences = [f"{name}={actual[name]} (need {pin})" for name, pin in VERSIONS.items() if actual[name] != pin]
    if differences:
        raise ValueError("RELEASE_ENV_MISMATCH " + ", ".join(differences))
    emit("RELEASE_OK", files=len(manifest), sibling_files=sum(map(len, DEPENDENCIES.values())))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--preflight", action="store_true", help="run checks on generated data")
    phase.add_argument("--freeze", action="store_true", help="fit discovery and lock reserved predictions")
    phase.add_argument("--score", action="store_true", help="score the locked predictions on reserved D_Glu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    release = verify_release()
    sys.path[1:1] = [str(ROOT.parent / version_name) for version_name in DEPENDENCIES]

    import numpy as np
    from threadpoolctl import threadpool_limits
    from fermentation_score import build_score, score_selftest
    from interactions import interaction_selftest
    from person_inputs import load_discovery
    from select_discovery import select, selector_selftest

    with threadpool_limits(limits=args.threads):
        if args.freeze or args.score:
            from validate_reserved import run_phase
            run_phase(ROOT, release, emit, freeze_only=args.freeze)
            return
        if args.preflight:
            score_selftest()
            interaction_selftest()
            selector_selftest()
            from validate_reserved import validation_selftest
            validation_selftest()
            emit("PREFLIGHT_DONE", scope="generated data")
            return
        source = load_discovery(ROOT)
        score, _, _ = build_score(source.raw["gut"], source.taxa["gut"])
        if not np.isfinite(score).all():
            raise ValueError("DISCOVERY_PANEL_INCOMPLETE")
        chosen = select(source, score)
        emit("DISCOVERY_SELECTION", **chosen)
        emit("DISCOVERY_COMPLETE", scope="discovery only", reserved_outcome_values_read=False)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        project = ROOT.parents[2]
        try:
            missing = Path(exc.filename).relative_to(project)
        except (TypeError, ValueError):
            missing = None
        print("EXP2_V5_STOP FILE_MISSING" + (f" {missing}" if missing is not None else ""),
              file=sys.stderr, flush=True)
        sys.exit(1)
    except (AssertionError, ArithmeticError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        detail = str(exc)
        if not detail.startswith(("RELEASE_", "PANEL_TAXON_")) and not (
                detail.isascii() and detail.replace("_", "").isalnum() and detail.isupper()):
            detail = type(exc).__name__
        print("EXP2_V5_STOP " + detail, file=sys.stderr, flush=True)
        sys.exit(1)
