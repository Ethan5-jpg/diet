#!/usr/bin/env python3
"""Portable whole-cohort Diet x Gut interaction analysis, Python 3.8+."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import traceback
import uuid

PACKAGE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", help="Path to the shared Data directory; default: this package's parent")
    p.add_argument("--config", default=str(PACKAGE / "config" / "analysis.json"))
    p.add_argument("--output-dir", help="New, non-existing run directory; never overwrite a prior run")
    p.add_argument("--check-only", action="store_true", help="Validate real input content and complete-case counts; no fits")
    p.add_argument("--list-inputs", action="store_true", help="Check configured paths without importing scientific libraries")
    p.add_argument("--demo", action="store_true", help="Run clearly labelled synthetic data, no real-data access")
    p.add_argument("--skip-plots", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    root = Path(args.data_root).expanduser().resolve() if args.data_root else PACKAGE.parent
    if args.list_inputs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        missing = 0
        for label, value in config["inputs"].items():
            path = Path(value).expanduser()
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            enabled = label != "protocol" or config["protocol_mode"] != "disabled"
            required = label != "protocol" or config["protocol_mode"] == "required"
            found = path.is_file()
            missing += int(enabled and required and not found)
            print("{} | {} | {}".format(label, "DISABLED" if not enabled else ("FOUND" if found else ("MISSING_REQUIRED" if required else "OPTIONAL_MISSING")), path))
        return 2 if missing else 0
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(name, "1")
    try:
        modules = {name: importlib.import_module(name) for name in ["numpy", "pandas", "scipy", "matplotlib"]}
    except ImportError as error:
        print("Dependency missing: {}\nInstall with this Python: {} -m pip install -r {}".format(error, sys.executable, PACKAGE / "requirements.txt"), file=sys.stderr)
        return 2
    from interaction import VERSION
    from interaction.common import require, sha256, write_json
    from interaction.inputs import load_config, load_inputs, resolve_paths
    from interaction.pipeline import run_models
    from interaction.reporting import bundle_results, make_figures, write_screens, write_tables
    kind = "SYNTHETIC" if args.demo else "SERVER_INPUTS"
    run_id = ("demo_" if args.demo else "run_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else PACKAGE / "outputs" / run_id
    if out.exists():
        print("Output directory already exists; choose a new path: " + str(out), file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    for directory in ["tables", "figures"]:
        (out / directory).mkdir()
    initial = {str(p.relative_to(PACKAGE)): sha256(p) for pattern in ["interaction/*.py", "*.py", "*.sh", "config/*.json", "requirements.txt"] for p in PACKAGE.glob(pattern)}
    manifest = dict(version=VERSION, run_id=run_id, data_kind=kind, status="RUNNING", data_root=str(root),
                    command=sys.argv if argv is None else argv, config_path=str(config_path),
                    config_sha256=sha256(config_path), code_sha256=initial,
                    software=dict(python=platform.python_version(), **{k: m.__version__ for k, m in modules.items()}))
    temp = None
    write_json(out / "manifest.json", manifest)
    try:
        config = load_config(config_path)
        manifest["config"] = config
        write_json(out / "analysis_config.json", config)
        if args.demo:
            from interaction.synthetic import create_fixture
            # Refuse absolute paths: synthetic generation must never write into real inputs.
            require(all(not Path(x).expanduser().is_absolute() and ".." not in Path(x).parts for x in config["inputs"].values()), "Demo requires safe relative paths")
            temp = tempfile.TemporaryDirectory(prefix="hpp_interaction_synthetic_")
            root = Path(temp.name)
            manifest["synthetic_fixture"] = create_fixture(root, config)
            manifest["data_root"] = str(root)
        print("INPUT_CHECK | " + kind, flush=True)
        master, metadata = load_inputs(config, resolve_paths(config, root))
        manifest["inputs"] = metadata["files"]
        manifest["input_summary"] = metadata["summary"]
        write_json(out / "manifest.json", manifest)
        results, slopes, flows, families = run_models(master, metadata, config, args.check_only)
        write_tables(out, metadata, results, slopes, flows, families)
        plotting = "SKIPPED" if args.skip_plots or args.check_only else "PENDING"
        if plotting == "PENDING":
            make_figures(out, results, slopes, config, kind)
            plotting = "COMPLETE"
        for item in metadata["files"]:
            require(sha256(item["path"]) == item["sha256"], "Input changed during run: " + item["label"])
        require(sha256(config_path) == manifest["config_sha256"], "Config changed during run")
        require(all(sha256(PACKAGE / name) == digest for name, digest in initial.items()), "Analysis code changed during run")
        if args.check_only:
            status = "INPUT_CHECK_PASSED"
        else:
            focus = results.loc[results.model.eq("primary") & results.family.eq("focus")]
            if not focus.status.eq("OK").any():
                status = "NO_ESTIMABLE_FOCUS_MODELS"
            else:
                limited_support = results.get("limited_joint_support", modules["pandas"].Series(dtype=bool)).fillna(False).any()
                status = "COMPLETE_WITH_LIMITATIONS" if (not results.status.eq("OK").all() or limited_support) else "COMPLETE"
        manifest.update(status=status, plots=plotting, model_counts=results.status.value_counts().to_dict(),
                        limitations=[config["interpretation"], "Historical rEDIH unit provenance is inherited, not resolved here.",
                                     "CGM/taxon collection timing is not re-audited; no prospective microbiome prediction claim.",
                                     "Observed support, model form and effect sizes must be reviewed; q<0.05 is not a mechanism."],
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_screens(out, metadata, results, config, status, kind)
        for name in ["README.md", "METHODS.md"]:
            if (PACKAGE / name).is_file():
                shutil.copyfile(PACKAGE / name, out / name)
        manifest["output_sha256"] = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
        bundle_results(out, PACKAGE)
        if not args.check_only and status.startswith("COMPLETE"):
            pointer = PACKAGE / "outputs" / ("LATEST_SYNTHETIC.txt" if args.demo else "LATEST_REAL.txt")
            pointer.parent.mkdir(exist_ok=True)
            pointer.write_text(str(out) + "\n", encoding="utf-8")
        print("\n" + (out / "SCREENSHOT_SUMMARY.txt").read_text(encoding="utf-8"))
        print("RESULT_DIR=" + str(out))
        return 2 if status == "NO_ESTIMABLE_FOCUS_MODELS" else 0
    except Exception as error:
        manifest.update(status="FAILED", error=str(error), completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
        (out / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("FAILED: {}\nDetails: {}".format(error, out / "manifest.json"), file=sys.stderr)
        return 2
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
