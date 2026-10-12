#!/usr/bin/env python3
"""Add six sensitivities for every diet, preserving the original run and its identity."""
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
    p.add_argument("--reference-run", help="Existing completed original .2 run; required for real analysis")
    p.add_argument("--data-root", help="Shared Data root; defaults to two levels above this package")
    p.add_argument("--output-dir", help="New output directory; existing directories are never overwritten")
    p.add_argument("--demo", action="store_true", help="Generate a synthetic reference and follow-up; no real input access")
    p.add_argument("--skip-plots", action="store_true")
    return p


def make_demo_reference(out, data_root):
    import numpy as np
    import pandas as pd
    from interaction.common import require, sha256, write_json
    from interaction.extension_sensitivity import CORE_FILES
    from interaction.inputs import load_config, load_inputs, resolve_paths
    from interaction.pipeline import run_models
    from interaction.synthetic import create_fixture
    config = load_config(PACKAGE / "config/analysis.json")
    require(all(not Path(value).expanduser().is_absolute() and ".." not in Path(value).parts
                for value in config["inputs"].values()), "Demo requires safe relative input paths")
    fixture = create_fixture(data_root, config)
    paths = resolve_paths(config, data_root)
    # Add known extension signals to the existing synthetic generator, not to real data.
    diet = pd.read_csv(paths["current4"])
    micro = pd.read_csv(paths["microbiome"])
    taxon = next(c for c in micro if "s__Eubacterium_rectale" in c)
    product = diet.AMED_z.to_numpy() * micro[taxon].to_numpy()
    cgm = pd.read_csv(paths["cgm"])
    for outcome in ["cgm_cv", "cgm_sdw"]:
        values = cgm[outcome + "_clean"].to_numpy() - 1.3 * product
        if values.min() <= 0:
            values = values - values.min() + 1
        cgm[outcome + "_clean"] = values
        cgm[outcome + "_z"] = (values-values.mean()) / np.std(values, ddof=1)
    cgm.to_csv(paths["cgm"], index=False)
    fixture["additional_extension_interactions"] = {"AMED:T068:CV": -1.3, "AMED:T068:SDw": -1.3}
    master, metadata = load_inputs(config, paths)
    results, unused_slopes, unused_flows, unused_fdr = run_models(master, metadata, config)
    root = out / "reference_demo"
    (root / "tables").mkdir(parents=True)
    write_json(root / "analysis_config.json", config)
    results.to_csv(root / "tables/all_interactions.csv", index=False, encoding="utf-8-sig")
    has_focus = results.loc[results.family.eq("focus") & results.model.eq("primary"), "status"].eq("OK").any()
    limited = results.get("limited_joint_support", pd.Series(dtype=bool)).fillna(False).any()
    reference_status = ("NO_ESTIMABLE_FOCUS_MODELS" if not has_focus else
                        "COMPLETE_WITH_LIMITATIONS" if limited or not results.status.eq("OK").all() else "COMPLETE")
    manifest = dict(version=config["version"], data_kind="SYNTHETIC", status=reference_status,
                    config=config, inputs=metadata["files"], synthetic_fixture=fixture,
                    code_sha256={name: sha256(PACKAGE / name) for name in CORE_FILES},
                    output_sha256={name: sha256(root / name) for name in ["analysis_config.json", "tables/all_interactions.csv"]})
    write_json(root / "manifest.json", manifest)
    return root, master, metadata


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.demo and (args.reference_run or args.data_root):
        p.error("--demo cannot use --reference-run or --data-root")
    if not args.demo and not args.reference_run:
        p.error("Real follow-up requires --reference-run pointing to the original completed run")
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(name, "1")
    try:
        modules = {name: importlib.import_module(name) for name in ["numpy", "pandas", "scipy", "matplotlib"]}
    except ImportError as error:
        print("Dependency missing: {}\nInstall with {} -m pip install -r {}".format(error, sys.executable, PACKAGE / "requirements.txt"), file=sys.stderr)
        return 2
    from interaction.common import require, sha256, write_json
    from interaction.extension_sensitivity import (FOLLOWUP_VERSION, MODELS, compare_reference,
                                                   load_reference, run_extended, stability_tables, verify_input_identity)
    from interaction.extension_reporting import make_followup_figures, write_followup_screens, write_followup_tables
    from interaction.inputs import load_inputs, resolve_paths
    from interaction.reporting import bundle_results
    kind = "SYNTHETIC" if args.demo else "SERVER_INPUTS"
    run_id = ("demo_" if args.demo else "extension_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else PACKAGE / "outputs_extension_sensitivity" / run_id
    reference_root = Path(args.reference_run).expanduser().resolve() if args.reference_run else None
    if out.exists():
        print("Output already exists; choose a new directory: " + str(out), file=sys.stderr)
        return 2
    if reference_root is not None and (out == reference_root or reference_root in out.parents):
        print("Follow-up output must not be inside the original run", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    for name in ["tables", "figures"]:
        (out / name).mkdir()
    initial = {str(p.relative_to(PACKAGE)): sha256(p) for pattern in ["interaction/*.py", "*.py", "*.sh", "config/*.json", "requirements.txt"] for p in PACKAGE.glob(pattern)}
    root = Path(args.data_root).expanduser().resolve() if args.data_root else PACKAGE.parent.parent
    manifest = dict(version=FOLLOWUP_VERSION, data_kind=kind, status="RUNNING", run_id=run_id,
                    data_root=str(root), reference_run=str(reference_root) if reference_root else None,
                    code_sha256=initial, command=sys.argv if argv is None else argv,
                    software=dict(python=platform.python_version(), **{k: m.__version__ for k, m in modules.items()}))
    write_json(out / "manifest.json", manifest)
    temp = None
    try:
        if args.demo:
            temp = tempfile.TemporaryDirectory(prefix="hpp_extension_synthetic_")
            root = Path(temp.name)
            reference_root, master, metadata = make_demo_reference(out, root)
        reference, config, old, reference_hashes = load_reference(reference_root, PACKAGE, kind)
        manifest.update(reference_run=str(reference_root), reference_file_sha256=reference_hashes,
                        reference_version=reference["version"], data_root=str(root), config=config)
        targets = old[old.model.eq("primary") & old.family.eq("extension") & old.status.eq("OK") & old.q_all.lt(config["alpha"])]
        manifest["reference_extension_targets"] = targets[["exposure", "taxon", "outcome", "q_all"]].to_dict("records")
        write_json(out / "reference_manifest.json", reference)
        write_json(out / "analysis_config.json", config)
        write_json(out / "followup_design.json", dict(version=FOLLOWUP_VERSION, specifications=["primary"]+MODELS,
                   planned_tests_per_specification=84, extension_tests_per_specification=60, planned_fits=588,
                   q_all84_denominator=84, q_focus24_denominator=24,
                   common_all7_exposures=config["focus_exposures"]+config["extension_exposures"],
                   reference_selection="Original primary extension q_all < alpha; frozen before sensitivity fitting",
                   timing="Follow-up requested after inspecting primary results; not preregistered or independent validation"))
        print("EXTENSION INPUT_CHECK | " + kind, flush=True)
        if not args.demo:
            master, metadata = load_inputs(config, resolve_paths(config, root))
        verify_input_identity(reference, metadata)
        manifest.update(inputs=metadata["files"], input_summary=metadata["summary"],
                        outcome_scales=metadata["outcome_scales"], score_mapping=metadata["score_mapping"],
                        taxa_mapping=metadata["taxa_mapping"])
        write_json(out / "manifest.json", manifest)
        results, slopes, flows, families = run_extended(master, metadata, config)
        comparison = compare_reference(old, results, config)
        comparison.to_csv(out / "tables/reference_reproduction.csv", index=False, encoding="utf-8-sig")
        require(len(comparison) == 204 and comparison.matches.all(), "Reference reproduction failed; see tables/reference_reproduction.csv")
        stability, summary = stability_tables(old, results, config["alpha"])
        write_followup_tables(out, results, slopes, flows, families, comparison, stability, summary)
        plots = "SKIPPED" if args.skip_plots else "COMPLETE"
        if not args.skip_plots:
            make_followup_figures(out, results, config, kind)
        for item in metadata["files"]:
            require(sha256(item["path"]) == item["sha256"], "Input changed during follow-up: " + item["label"])
        for name, digest in reference_hashes.items():
            require(sha256(reference_root / name) == digest, "Reference changed during follow-up: " + name)
        require(all(sha256(PACKAGE / name) == digest for name, digest in initial.items()), "Code/config changed during follow-up")
        any_extension = results.loc[results.family.eq("extension"), "status"].eq("OK").any()
        limited = results.get("limited_joint_support", modules["pandas"].Series(dtype=bool)).fillna(False).any()
        status = ("NO_ESTIMABLE_EXTENSION_MODELS" if not any_extension else
                  "COMPLETE_WITH_LIMITATIONS" if not results.status.eq("OK").all() or limited else "COMPLETE")
        manifest.update(status=status, plots=plots, reference_reproduced_tests=int(comparison.matches.sum()),
                        model_counts=results.status.value_counts().to_dict(), reference_extension_hits=len(stability),
                        original_hits_retained_all_six=int(stability.retained_all_six.sum()),
                        limitations=["Follow-up after viewing primary results in the same cohort; not independent validation.",
                                     "All7 common sample differs from the original two-diet common-focus sample.",
                                     "Same direction/q retention is descriptive; CIs are pointwise and samples overlap.",
                                     "High-leverage influence has not been separately refitted; HC3 does not remove influential observations.",
                                     "Historical rEDIH unit provenance and timing limitations remain."],
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_followup_screens(out, results, config, comparison, stability, summary, status, kind, plots=plots)
        for name in ["EXTENSION_METHODS.md", "METHODS.md", "README.md"]:
            if (PACKAGE / name).is_file():
                shutil.copyfile(PACKAGE / name, out / name)
        manifest["output_sha256"] = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file() and p != out / "manifest.json"}
        write_json(out / "manifest.json", manifest)
        bundle_results(out, PACKAGE)
        if status.startswith("COMPLETE"):
            pointer = PACKAGE / "outputs_extension_sensitivity" / ("LATEST_SYNTHETIC.txt" if args.demo else "LATEST_REAL.txt")
            pointer.parent.mkdir(exist_ok=True)
            pointer.write_text(str(out)+"\n", encoding="utf-8")
        print("\n"+(out / "SCREENSHOT_SUMMARY.txt").read_text(encoding="utf-8"))
        print("RESULT_DIR="+str(out))
        return 0 if status.startswith("COMPLETE") else 2
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
