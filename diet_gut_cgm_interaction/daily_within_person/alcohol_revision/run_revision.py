#!/usr/bin/env python3
"""Reanalyze an intact daily panel with audited AMED alcohol; no raw CGM reads."""
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
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")


def own_hashes(sha):
    return {p.name: sha(p) for pattern in ["*.py", "*.sh", "*.json"] for p in sorted(HERE.glob(pattern))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--data-root", help="Optional Data-root relocation; saved panel and AMED hash must still match")
    parser.add_argument("--output-dir", help="New directory only")
    parser.add_argument("--synthetic", action="store_true", help="Local synthetic validation only; rejects SERVER_INPUTS")
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    import scipy
    from daily.common import require, sha256, validate_config, write_json
    from daily.models import choose_sample as old_sample
    from run_analysis import code_hashes, load_saved_panel
    from source_audit import attach_alcohol, checkpoints, verify_paired_samples
    from revision_models import SPECS, fit_all, feasibility
    from revision_reporting import write_screens

    out, manifest, owns_run = None, None, False
    try:
        protocol = json.loads((HERE / "protocol.json").read_text())
        own_code = own_hashes(sha256)
        reference = Path(args.reference_run).expanduser().resolve()
        ref_manifest_hash = sha256(reference / "manifest.json")
        ref = json.loads((reference / "manifest.json").read_text())
        kind = "SYNTHETIC" if args.synthetic else "SERVER_INPUTS"
        require(ref["data_kind"] == kind, "Reference data kind conflicts with explicit synthetic/real mode")
        require(ref["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS", "NO_ESTIMABLE_PRIMARY_MODELS"], "Reference must be a completed original analysis")
        require(code_hashes(sha256) == ref["code_sha256"] == protocol["parent_code_sha256"], "Parent .3 runtime changed; preserve original parent files")
        required_outputs = ["analysis_config.json", "private/daily_panel.csv", "tables/10_interactions.csv"]
        def verify_reference():
            require(sha256(reference / "manifest.json") == ref_manifest_hash, "Reference manifest changed")
            for name in required_outputs:
                require(sha256(reference / name) == ref["output_sha256"][name], "Reference output changed: " + name)
        verify_reference()
        config = validate_config(json.loads((reference / "analysis_config.json").read_text()))
        require(config == ref["config"], "Reference configuration mismatch")
        require(SPECS == protocol["specifications"], "Frozen specification registry changed")
        if not args.synthetic:
            require(ref["run_id"] == protocol["reference_run_id"], "Use the exact audited real reference run")
            require(config == json.loads((PARENT / "config/analysis.json").read_text()), "Real reference thresholds differ from frozen configuration")
        root = Path(args.data_root).expanduser().resolve() if args.data_root else Path(ref["data_root"])
        source = root / protocol["amed_relative_path"]
        source_hash = sha256(source)
        if not args.synthetic:
            require(source_hash == protocol["amed_sha256"], "AMED source differs from 18:09 screenshot; audit the change before fitting")
        panel = load_saved_panel(reference / "private/daily_panel.csv").replace([np.inf, -np.inf], np.nan)
        old = pd.read_csv(reference / "tables/10_interactions.csv")
        old = old.loc[old.spec.eq("primary") & old.outcome.eq("SD")].set_index(["exposure", "taxon"])
        require(len(old) == 4 and old.index.is_unique, "Original SD registry invalid")
        for (exposure, taxon), row in old.iterrows():
            d, _ = old_sample(panel, config, exposure, taxon, "SD", "primary")
            require((d.participant_id.nunique(), len(d)) == (int(row.n_people), int(row.n_days)), "Original sample counts not reproduced")
        revised, audit = attach_alcohol(panel, source, config)
        counts = checkpoints(revised, config)
        if not args.synthetic:
            require(counts == protocol["expected_counts"], "18:09 screenshot sample counts not reproduced")
            require(audit["qc_days"] == protocol["expected_qc_days"], "QC day count changed")
            for name in ["observed", "recovered"]:
                require(audit["candidate_" + name + "_days"] == protocol["expected_candidates"][name], "Alcohol candidate counts changed")
        paired = verify_paired_samples(revised, config)
        require(len(paired) == 24 and paired.identical.all(), "Incomplete paired-sample audit")
        require(sha256(source) == source_hash, "AMED source changed during reading")
        print("REVISION CHECK | {} | original 4/4; source counts verified; paired samples 24/24".format(kind), flush=True)
        run_id = ("synthetic_" if args.synthetic else "revised_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
        out = Path(args.output_dir).expanduser().resolve() if args.output_dir else HERE / "outputs" / run_id
        require(not out.exists(), "Output directory exists; no overwrite")
        require(out != reference and reference not in out.parents, "New output must stay outside the reference run")
        os.umask(0o077)
        out.mkdir(parents=True)
        owns_run = True
        for name in ["private", "tables", "logs"]:
            (out / name).mkdir()
        manifest = dict(version=protocol["version"], run_id=run_id, data_kind=kind, status="RUNNING",
                        config=config, protocol=protocol, code_sha256=own_code, alcohol_audit=audit,
                        reference_run=str(reference), reference_manifest_sha256=ref_manifest_hash,
                        reference_output_sha256={n: ref["output_sha256"][n] for n in required_outputs},
                        parent_code_sha256=ref["code_sha256"], amed_source=str(source), amed_sha256=source_hash,
                        data_root=str(root), software=dict(python=platform.python_version(), numpy=np.__version__,
                        pandas=pd.__version__, scipy=scipy.__version__), reference_software=ref["software"],
                        checkpoint_scope="synthetic self-consistency only" if args.synthetic else "all 12 rows of 18:09 screenshot",
                        interpretation="Same-visit observational follow-up; source audit is not measurement validation; no independent replication or causal mediation claim")
        write_json(out / "manifest.json", manifest)
        write_json(out / "analysis_config.json", config)
        write_json(out / "protocol.json", protocol)
        write_json(out / "source_audit.json", audit)
        revised.to_csv(out / "private/daily_panel.csv", index=False)
        pd.DataFrame(counts).to_csv(out / "tables/01_source_checkpoint.csv", index=False)
        paired.to_csv(out / "tables/02_paired_sample_identity.csv", index=False)
        feasible = feasibility(revised, config)
        feasible.to_csv(out / "tables/03_primary_feasibility.csv", index=False)
        results, coefficients, flows, slopes, anchors = fit_all(revised, config)
        require(len(results) == 132, "Frozen 132-model registry incomplete")
        for pair in paired.itertuples():
            d = results.loc[results.exposure.eq(pair.exposure) & results.taxon.eq(pair.taxon) & results.outcome.eq(pair.outcome)]
            for spec in [pair.adjusted, pair.unadjusted]:
                row = d.loc[d.spec.eq(spec)].iloc[0]
                if row.status == "ESTIMATED":
                    require(row.sample_sha256 == pair.sample_sha256, "Fitted model used a different person-day sample")
        for name, frame in [("10_interactions.csv", results), ("11_all_coefficients.csv", coefficients),
                            ("12_model_flow.csv", flows), ("13_primary_slopes.csv", slopes), ("14_primary_anchors.csv", anchors)]:
            frame.to_csv(out / "tables" / name, index=False)
        verify_reference()
        require(sha256(source) == source_hash, "AMED source changed during analysis")
        require(own_hashes(sha256) == own_code and code_hashes(sha256) == ref["code_sha256"], "Code changed during analysis")
        good = results.status.eq("ESTIMATED")
        main_sd = results.spec.eq("primary") & results.outcome.eq("SD")
        manifest["model_summary"] = dict(estimated=int(good.sum()), planned=132, primary_sd_estimated=int((good & main_sd).sum()))
        manifest["status"] = ("NO_ESTIMABLE_PRIMARY_MODELS" if not (good & main_sd).any() else
                              "COMPLETE_WITH_LIMITATIONS" if not good.all() or results.limited_support.any() else "COMPLETE")
        manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        screens = write_screens(out, manifest, feasible, results, slopes)
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
        print("REVISION STOPPED: " + str(exc), file=sys.stderr, flush=True)
        if owns_run:
            print("RESULT_DIR=" + str(out), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
