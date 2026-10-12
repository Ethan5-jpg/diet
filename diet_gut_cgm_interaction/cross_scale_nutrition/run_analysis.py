#!/usr/bin/env python3
"""Common-definition two-taxon localization and same-sample nutrition analysis."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

HERE = Path(__file__).resolve().parent
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", default=str(HERE.parent / "daily_within_person/alcohol_revision/outputs/revised_20261011T103556_258035Z_9583d5"))
    parser.add_argument("--output-dir", help="New directory only; defaults to this package's outputs/")
    parser.add_argument("--synthetic-reference", action="store_true", help="Local validation only; refuses SERVER_INPUTS")
    parser.add_argument("--check-only", action="store_true", help="Verify input, historical models and new sample counts; no new hypothesis fits or files")
    args = parser.parse_args(argv)
    try:
        import numpy as np
        import pandas as pd
        import scipy
    except ImportError as exc:
        print("DEPENDENCY_MISSING: {}. Use {} -m pip install -r {}".format(exc, sys.executable, HERE / "requirements.txt"), file=sys.stderr)
        return 2
    from crossscale import VERSION
    from crossscale.common import code_hashes, require, sha256, write_json
    from crossscale.inputs import open_reference, reproduce_reference, REFERENCE_FILES
    from crossscale.analysis import VARIANTS, select_samples, describe_samples, run_models
    from crossscale.reporting import render_screens
    manifest, out, owns_run = None, None, False
    try:
        require(not (args.check_only and args.output_dir), "--check-only writes no output directory")
        protocol = json.loads((HERE / "config/protocol.json").read_text())
        require(protocol["version"] == VERSION and protocol["variants"] == [v["name"] for v in VARIANTS], "Version/model registry mismatch")
        code = code_hashes(HERE)
        kind = "SYNTHETIC" if args.synthetic_reference else "SERVER_INPUTS"
        print("COMMON INPUT CHECK | " + kind, flush=True)
        reference = Path(args.reference_run).expanduser().resolve()
        panel, config, old, ref, ref_hash, verify = open_reference(reference, protocol, args.synthetic_reference)
        historical = reproduce_reference(panel, config, old, protocol, args.synthetic_reference)
        samples, flow = select_samples(panel, config, protocol)
        description = describe_samples(samples)
        print("Historical numerical/sample checkpoints {}/8 verified.".format(len(historical)), flush=True)
        if args.check_only:
            verify()
            require(code_hashes(HERE) == code, "Code changed during check")
            print(description.drop(columns=["sample_sha256", "people_sha256"]).to_string(index=False))
            print("CHECK_ONLY_COMPLETE: no new hypothesis models, files, or raw CGM reads.")
            return 0
        run_id = ("synthetic_" if args.synthetic_reference else "common_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
        out = Path(args.output_dir).expanduser().resolve() if args.output_dir else HERE / "outputs" / run_id
        require(not out.exists(), "Output already exists; no overwrite")
        require(out != reference and reference not in out.parents, "Output must be outside reference run")
        os.umask(0o077)
        out.mkdir(parents=True)
        owns_run = True
        for name in ["tables", "private", "logs"]:
            (out / name).mkdir()
        manifest = dict(version=VERSION, run_id=run_id, data_kind=kind, status="RUNNING", protocol=protocol,
                        code_sha256=code, reference_run=str(reference), reference_manifest_sha256=ref_hash,
                        reference_output_sha256={n: ref["output_sha256"][n] for n in REFERENCE_FILES},
                        reference_software=ref["software"], software=dict(python=platform.python_version(), numpy=np.__version__,
                        pandas=pd.__version__, scipy=scipy.__version__), hypothesis_scope=protocol["study_scope"])
        write_json(out / "manifest.json", manifest)
        write_json(out / "config.json", protocol)
        historical.to_csv(out / "tables/00_reference_reproduction.csv", index=False)
        flow.to_csv(out / "tables/01_sample_flow.csv", index=False)
        description.to_csv(out / "tables/02_sample_summary.csv", index=False)
        membership = samples["F"][["participant_id", "connection_id", "date", "day_id"]].copy()
        membership["nutrition_sample"] = membership.day_id.isin(samples["N"].day_id)
        membership.to_csv(out / "private/sample_membership.csv", index=False)
        tables = run_models(samples, protocol, args.synthetic_reference)
        filenames = dict(interactions="10_interactions.csv", comparisons="11_comparisons.csv", diagnostics="12_model_diagnostics.csv",
                         coefficients="13_all_coefficients.csv", slopes="14_conditional_slopes.csv", anchors="15_microbe_anchors.csv",
                         sample_audit="03_model_sample_audit.csv")
        for key, filename in filenames.items():
            tables[key].to_csv(out / "tables" / filename, index=False)
        verify()
        require(code_hashes(HERE) == code, "Code changed during analysis")
        r, c = tables["interactions"], tables["comparisons"]
        main = r.loc[r.variant.eq("F_joint") & r.outcome.eq("SD")]
        estimated = r.status.eq("ESTIMATED")
        manifest["status"] = ("NO_ESTIMABLE_PRIMARY_MODELS" if not main.status.eq("ESTIMATED").any() else
                              "COMPLETE_WITH_LIMITATIONS" if not estimated.all() or not c.status.eq("ESTIMATED").all() or r.limited_support.any() else "COMPLETE")
        manifest["summary"] = dict(fits=int(tables["diagnostics"].status.eq("ESTIMATED").sum()), planned_fits=26,
                                   interactions=int(estimated.sum()), planned_interactions=44,
                                   contrasts=int(c.status.eq("ESTIMATED").sum()), planned_contrasts=40,
                                   historical_checkpoints=8, limited_support_rows=int(r.limited_support.sum()))
        manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        screens = render_screens(out, manifest, description, tables, historical)
        files = [p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json" and "logs" not in p.relative_to(out).parts]
        manifest["output_sha256"] = {str(p.relative_to(out)): sha256(p) for p in sorted(files)}
        write_json(out / "manifest.json", manifest)
        for screen in screens:
            print("\n" + screen, flush=True)
        print("\nRESULT_DIR=" + str(out), flush=True)
        return 2 if manifest["status"] == "NO_ESTIMABLE_PRIMARY_MODELS" else 0
    except (Exception, KeyboardInterrupt) as exc:
        if owns_run and manifest is not None:
            manifest["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            manifest["error"] = str(exc)
            write_json(out / "manifest.json", manifest)
            (out / "logs/error.txt").write_text(traceback.format_exc())
        print("COMMON STOPPED: " + str(exc), file=sys.stderr, flush=True)
        if owns_run:
            print("RESULT_DIR=" + str(out), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
