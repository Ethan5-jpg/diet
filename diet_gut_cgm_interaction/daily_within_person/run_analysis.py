#!/usr/bin/env python3
"""Standalone third-step daily within-person analysis; Python 3.8+."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

PACKAGE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", help="Data folder; defaults to two levels above this folder")
    p.add_argument("--config", default=str(PACKAGE / "config" / "analysis.json"))
    p.add_argument("--stage", choices=["check", "build", "analyze", "all"], default="all")
    p.add_argument("--output-dir", help="New output directory; existing directories are rejected")
    p.add_argument("--run-dir", help="Exact BUILD_COMPLETE directory, only with --stage analyze")
    p.add_argument("--demo", action="store_true", help="Use newly generated fictional inputs only")
    return p


def code_hashes(sha):
    return {str(p.relative_to(PACKAGE)): sha(p) for pattern in ["*.py", "*.sh", "daily/*.py", "config/*.json", "requirements.txt"] for p in sorted(PACKAGE.glob(pattern))}


def source_files(paths):
    sources = [(k, p) for k, p in paths.items() if k not in ["cgm_glob", "cgm_files"] and p is not None and p.is_file()]
    return sources + [("cgm", p) for p in paths["cgm_files"]]


def fingerprint(paths, sha):
    rows = []
    sources = source_files(paths)
    for i, (label, p) in enumerate(sources, 1):
        rows.append(dict(label=label, path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
        if i % 50 == 0:
            print("INPUT fingerprint {}/{}".format(i, len(sources)), flush=True)
    return rows


def verify_sources(manifest, paths, sha, require):
    actual = {(label, str(p)) for label, p in source_files(paths)}
    expected = {(r["label"], r["path"]) for r in manifest["inputs"]}
    require(actual == expected, "Input file collection changed, including optional metadata")
    for i, r in enumerate(manifest["inputs"], 1):
        require(sha(r["path"]) == r["sha256"], "Input content changed: " + r["label"])
        if i % 50 == 0:
            print("INPUT verification {}/{}".format(i, len(manifest["inputs"])), flush=True)


def load_saved_panel(path):
    import pandas as pd
    from daily.common import CATEGORICAL, header, require, unique
    # A numeric-looking category must remain a category, and literal IDs such as
    # "NA" must survive exactly as they did in the original strict CSV reader.
    text_columns = {"participant_id", "connection_id", "day_id", "cohort", "research_stage",
                    "date", "sample_date", "timing_status", "diet_reasons"} | CATEGORICAL
    columns = header(path)
    missing = ["", "nan", "NaN", "None", "null", "<NA>"]
    d = pd.read_csv(path, dtype={c: str for c in columns if c in text_columns},
                    keep_default_na=False, na_values={c: missing for c in columns if c not in text_columns},
                    float_precision="round_trip")
    require(isinstance(d.index, pd.RangeIndex), "Malformed saved panel row width")
    require(d.participant_id.ne("").all() and d.connection_id.ne("").all(), "Missing saved panel identity")
    unique(d, ["participant_id", "date"], "saved panel")
    for col in ["diet_pass", "cgm_pass", "cgm_strict", "primary_eligible", "strict_eligible", "exclusion_unknown", "has_covariate_record"]:
        if d[col].dtype != bool:
            values = d[col].astype(str).str.lower()
            if not values.isin(["true", "false"]).all():
                raise ValueError("Malformed saved boolean: " + col)
            d[col] = values.eq("true")
    return d


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(key, "1")
    try:
        import numpy as np
        import pandas as pd
        import scipy
    except ImportError as exc:
        print("DEPENDENCY_MISSING: {}\nUse this interpreter: {} -m pip install -r {}".format(exc, sys.executable, PACKAGE / "requirements.txt"))
        return 2
    from daily import VERSION
    from daily.common import require, sha256, validate_config, write_json
    from daily.inputs import load_people, paths_for, preflight
    from daily.build import construct_panel
    from daily.models import feasibility, fit_all
    from daily.reporting import output_hashes, qc_tables, table, write_screens

    out, manifest, lock, owns_run = None, None, None, False
    try:
        require(not (args.run_dir and (args.output_dir or args.demo)), "--run-dir cannot be combined with --output-dir or --demo")
        require((args.stage == "analyze") == bool(args.run_dir), "Use --run-dir only with --stage analyze; new runs use check/build/all")
        config_path = Path(args.config).expanduser().resolve()
        config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
        config_digest = sha256(config_path)
        code = code_hashes(sha256)
        versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)
        root = Path(args.data_root).expanduser().resolve() if args.data_root else PACKAGE.parent.parent
        if args.run_dir:
            out = Path(args.run_dir).expanduser().resolve()
            require(out.is_dir(), "Run directory not found")
            import fcntl
            lock = (out / ".run.lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            require(manifest["status"] == "BUILD_COMPLETE", "Only an intact BUILD_COMPLETE run can enter analyze")
            require(manifest["code_sha256"] == code, "Code/config bundle changed since build; make a new run")
            require(manifest["software"] == versions, "Interpreter/library versions changed since build")
            require(manifest["config_sha256"] == config_digest, "Configuration file changed since build")
            if args.data_root:
                require(str(root) == manifest["data_root"], "Resume requires the recorded Data root")
            root, config = Path(manifest["data_root"]), manifest["config"]
            paths = paths_for(root, config)
            for name, digest in manifest["output_sha256"].items():
                require((out / name).is_file() and sha256(out / name) == digest, "Build output changed: " + name)
            verify_sources(manifest, paths, sha256, require)
            panel = load_saved_panel(out / "private" / "daily_panel.csv")
            feasible = feasibility(panel, config)
            owns_run = True
            manifest["status"] = "RUNNING_ANALYSIS"
            write_json(out / "manifest.json", manifest)
        else:
            run_id = ("demo_" if args.demo else "daily_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
            out = Path(args.output_dir).expanduser().resolve() if args.output_dir else PACKAGE / "outputs" / run_id
            require(not out.exists(), "Output directory exists; never overwrite an earlier run")
            os.umask(0o077)
            out.mkdir(parents=True)
            for name in ["tables", "private", "logs"]: (out / name).mkdir()
            import fcntl
            lock = (out / ".run.lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owns_run = True
            manifest = dict(version=VERSION, run_id=run_id, data_kind="SYNTHETIC" if args.demo else "SERVER_INPUTS",
                            status="RUNNING", config_path=str(config_path), config_sha256=config_digest,
                            code_sha256=code, software=versions, requested_stage=args.stage, inputs=[])
            write_json(out / "manifest.json", manifest)
            if args.demo:
                require(all(not Path(v).is_absolute() and ".." not in Path(v).parts for v in config["inputs"].values() if v), "Demo requires contained relative input paths")
                config = json.loads(json.dumps(config))
                config["analysis"]["minimum_people"] = 30
                config["analysis"]["minimum_tail_people"] = 8
                root = out / "private" / "synthetic_inputs"
                from daily.synthetic import create_fixture
                manifest["synthetic_fixture"] = create_fixture(root, config)
            manifest.update(config=config, data_root=str(root))
            write_json(out / "analysis_config.json", config)
            paths = paths_for(root, config)
            print("DAILY INPUT_CHECK | " + manifest["data_kind"], flush=True)
            check = preflight(paths, config)
            table(out, "00_preflight.csv", check)
            bad = check.loc[check.status.eq("MISSING_OR_INVALID")]
            if not bad.empty:
                for row in bad.itertuples(): print("{}: {} | {}".format(row.source, row.detail, row.path), flush=True)
                raise ValueError("Required inputs missing or structurally invalid; see tables/00_preflight.csv")
            if args.stage == "check":
                manifest["status"] = "PREFLIGHT_PASSED"
                manifest["check_scope"] = "headers and exact species mapping only; no cohort counts or file-content certification"
                manifest["inputs_inspected"] = [{"label": label, "path": str(p)} for label, p in source_files(paths)]
                write_json(out / "manifest.json", manifest)
                print("PREFLIGHT_PASSED: headers only. Run --stage all for actual daily QC and models.")
                print("RESULT_DIR=" + str(out))
                return 0
            manifest["inputs"] = fingerprint(paths, sha256)
            write_json(out / "manifest.json", manifest)
            people, scope, mapping = load_people(paths, config)
            table(out, "00_scope.csv", scope)
            table(out, "00_taxa_mapping.csv", mapping)
            manifest["person_summary"] = dict(selected=len(people), missing_cov=int((~people.has_covariate_record).sum()),
                                              unknown=int(people.exclusion_unknown.sum()), both_taxa=int(people[["T068", "T096"]].notna().all(axis=1).sum()))
            panel, audit = construct_panel(out, paths, people, config)
            manifest["build_audit"] = audit
            feasible = feasibility(panel, config)
            qc_tables(out, panel, people, feasible, audit)
            manifest["status"] = "BUILD_COMPLETE"
        results, slopes = None, None
        if args.stage in ["all", "analyze"]:
            manifest["status"] = "RUNNING_ANALYSIS"
            write_json(out / "manifest.json", manifest)
            results, coefficients, flows, slopes, anchors = fit_all(panel, config)
            for name, frame in [("10_interactions.csv", results), ("11_all_coefficients.csv", coefficients),
                                ("12_model_flow.csv", flows), ("13_primary_slopes.csv", slopes), ("14_primary_anchors.csv", anchors)]:
                table(out, name, frame)
            estimated = results.status.eq("ESTIMATED")
            primary = results.spec.eq("primary") & results.outcome.eq("SD")
            manifest["model_summary"] = dict(estimated=int(estimated.sum()), planned=len(results), primary_sd_estimated=int((estimated & primary).sum()))
            manifest["status"] = ("NO_ESTIMABLE_PRIMARY_MODELS" if not (estimated & primary).any() else
                                   "COMPLETE_WITH_LIMITATIONS" if not estimated.all() or results.limited_support.any() else "COMPLETE")
        verify_sources(manifest, paths_for(root, config), sha256, require)
        require(code_hashes(sha256) == manifest["code_sha256"], "Code/config bundle changed during execution")
        require(sha256(config_path) == manifest["config_sha256"], "Configuration changed during execution")
        manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["interpretation"] = "Same-visit observational, selected candidate taxa, no causal/clinical-benefit or independent-replication claim"
        screens = write_screens(out, manifest, feasible, results, slopes)
        manifest["output_sha256"] = output_hashes(out)
        write_json(out / "manifest.json", manifest)
        for screen in screens:
            print("\n" + screen, flush=True)
        print("\nRESULT_DIR=" + str(out), flush=True)
        return 2 if manifest["status"] == "NO_ESTIMABLE_PRIMARY_MODELS" else 0
    except (Exception, KeyboardInterrupt) as exc:
        if owns_run and out is not None and manifest is not None:
            manifest["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            manifest["error"] = str(exc)
            write_json(out / "manifest.json", manifest)
            (out / "logs" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("DAILY STOPPED: " + str(exc), file=sys.stderr, flush=True)
        if out is not None: print("RESULT_DIR=" + str(out), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        if lock is not None: lock.close()


if __name__ == "__main__":
    sys.exit(main())
