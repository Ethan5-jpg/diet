#!/usr/bin/env python3
"""Read a completed meal run; write follow-up results to a fresh sibling directory."""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Completed original meal_... run")
    parser.add_argument("--data-root", help="Data root; default: original manifest root")
    parser.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--show", help="Print SUMMARY.md from a completed repeated_... run")
    args = parser.parse_args(argv)
    if args.show:
        try:
            print((Path(args.show).expanduser() / "SUMMARY.md").read_text(encoding="utf-8"))
            return 0
        except OSError as e:
            print("Cannot read SUMMARY.md; use the exact RUN_DIR printed in the log. " + str(e), file=sys.stderr)
            return 2
    if not args.run_dir:
        parser.error("--run-dir is required unless --show is used")
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required", file=sys.stderr)
        return 2
    run, info = None, dict(status="RUNNING")
    differences, statuses, gains = None, [], None
    try:
        import numpy as np
        import pandas as pd
        from hppmeal.common import Run, require, write_json
        from followup import VERSION
        from followup.io_utils import (verify_bundle, inspect_source, read_inputs, verify_unchanged, cohort_meals)
        from followup.repeats import COHORTS, OUTCOMES, checked_pairs, difference_summary, template_structure, structure_summary
        from followup.prediction import predict_cohort
        from followup.reporting import summary, figures

        require(200 <= args.bootstrap <= 10000, "--bootstrap must be between 200 and 10000; default 1000")
        require(0 <= args.seed < 2 ** 32, "--seed must be in [0, 2**32)")
        bundle = verify_bundle()
        source = Path(args.run_dir).expanduser().resolve()
        old = inspect_source(source)
        root = Path(args.data_root or old["root"]).expanduser().resolve()
        require(root.is_dir(), "Data root missing")
        require(not (root / "SYNTHETIC_TEST_ONLY.json").exists() or old["data_kind"] == "SYNTHETIC",
                "Synthetic input cannot be labelled REAL")
        out_root = Path(args.output_root).expanduser().resolve()
        require(out_root != source and source not in out_root.parents, "Outputs must be outside the original run directory")
        out = out_root / datetime.now(timezone.utc).strftime("repeated_%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
        run = Run(root, out, old["config"])
        run.status["followup"] = info
        run.save()
        summary(run, info)
        package_files = dict(bundle["files"])
        package_files["followup/MANIFEST.json"] = bundle["_manifest_sha256"]
        for name, expected in package_files.items():
            recorded = run.record(PACKAGE / name, "followup_code")
            observed = next(x["sha256"] for x in run.inputs if x["path"] == str(recorded))
            require(observed == expected, "Followup package changed after inspection: " + name)
        write_json(out / "analysis_spec.json", dict(version=VERSION, bootstrap_draws=args.bootstrap, seed=args.seed,
                   minimum_people=50, ridge_penalty=10.0, source_run=str(source), original_config=old["config"],
                   estimand="observational repeat differences and later-date prediction; no ICC or causal effect"))
        print("RUN_DIR=" + str(out), flush=True)
        print("VERIFY source meals, pairing rule and covariate cohort", flush=True)
        meals, stored_pairs = read_inputs(run, source, old)
        pairs = checked_pairs(meals, stored_pairs)
        run.table(pairs, "01_verified_pairs.csv", private=True)
        flow = []
        for cohort in COHORTS:
            selected = cohort_meals(meals, cohort)
            before = pairs if cohort == "all_descriptive" else pairs.loc[pairs[cohort + "_eligible"]]
            good = before.loc[before.complete]
            flow.append(dict(cohort=cohort, eligible_people=selected.participant_id.nunique(), eligible_meals=len(selected),
                             pairs_before_common_complete=len(before), pairs=len(good), people=good.participant_id.nunique(),
                             excluded_incomplete_pairs=len(before) - len(good),
                             different_or_unknown_connections=int((~good.same_connection).sum())))
        run.table(pd.DataFrame(flow), "01_cohort_flow.csv")
        templates = template_structure(meals)
        run.table(templates, "02_templates.csv", private=True)
        run.table(structure_summary(templates), "02_template_structure.csv")
        all_differences, quantiles, contributions = [], [], []
        for cohort in COHORTS:
            print("REPEATS {}: participant bootstrap {} draws".format(cohort, args.bootstrap), flush=True)
            all_differences.append(difference_summary(pairs, cohort, args.bootstrap, args.seed))
            p = pairs.loc[pairs.complete].copy()
            if cohort != "all_descriptive":
                p = p.loc[p[cohort + "_eligible"]]
            for outcome in OUTCOMES:
                for absolute in [False, True]:
                    values = p[outcome].abs() if absolute else p[outcome]
                    for q in [0.1, 0.5, 0.9]:
                        quantiles.append(dict(cohort=cohort, outcome=outcome, absolute=absolute, quantile=q,
                                              value=values.quantile(q) if len(values) else np.nan, unit="mg/dL"))
            counts = p.groupby("participant_id").size()
            contributions.append(dict(cohort=cohort, people=len(counts), pairs=len(p),
                people_one_pair=int(counts.eq(1).sum()), people_two_or_more=int(counts.ge(2).sum()),
                median_pairs=counts.median(), max_pairs=counts.max(), mean_pairs=counts.mean()))
        differences = pd.concat(all_differences, ignore_index=True)
        run.table(differences, "03_pair_differences.csv")
        run.table(pd.DataFrame(quantiles), "03_pair_quantiles.csv")
        run.table(pd.DataFrame(contributions), "03_pair_counts.csv")
        metrics_list, gains_list, coefficient_list, prediction_flows = [], [], [], []
        for cohort in ["primary", "strict"]:
            print("PREDICTION {}: split by local date, train-only template support".format(cohort), flush=True)
            values = predict_cohort(cohort_meals(meals, cohort), cohort, args.bootstrap, args.seed)
            status, pflow, assignment, predictions, metrics, pgains, coefficients = values
            statuses.append(status)
            prediction_flows.append(pflow)
            run.table(assignment, "04_{}_date_split.csv".format(cohort), private=True)
            if not predictions.empty:
                run.table(predictions, "04_{}_predictions.csv".format(cohort), private=True)
                metrics_list.append(metrics)
                gains_list.append(pgains)
                coefficient_list.append(coefficients)
            print("PREDICTION {}={}: {} test people, {} test meals".format(
                cohort, status["status"], status["test_people"], status["test_meals"]), flush=True)

        def joined(frames, columns):
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

        metrics = joined(metrics_list, ["cohort", "model", "people", "test_meals", "unit", "person_equal_mae", "person_equal_rmse", "meal_weighted_mae", "meal_weighted_rmse"])
        gains = joined(gains_list, ["cohort", "previous", "newer", "people", "test_meals", "mae_improvement", "ci_low", "ci_high", "unit", "direction", "ci_method", "ci_scope", "bootstrap_draws"])
        coefficients = joined(coefficient_list, ["cohort", "model", "feature", "coefficient", "train_within_template_scale", "status"])
        run.table(pd.concat(prediction_flows, ignore_index=True), "04_prediction_flow.csv")
        run.table(pd.DataFrame(statuses), "04_prediction_status.csv")
        run.table(metrics, "04_prediction_metrics.csv")
        run.table(gains, "04_prediction_gains.csv")
        run.table(coefficients, "04_prediction_coefficients.csv")
        figures(run, differences, gains)
        print("VERIFY source files unchanged", flush=True)
        verify_unchanged(run)
        limited = (not differences.status.eq("ESTIMATED").all()
                   or any(s["status"] != "ESTIMATED" for s in statuses))
        info.update(status="COMPLETE_WITH_LIMITATIONS" if limited else "COMPLETE", version=VERSION,
                    verified_original_pairs=len(pairs), predictions=statuses,
                    reason="Some confidence intervals/models were not estimable; see statuses" if limited else "")
        run.save()
        summary(run, info, differences, statuses, gains)
        print("FOLLOWUP={}\nSUMMARY={}".format(info["status"], out / "SUMMARY.md"), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        interrupted = isinstance(e, KeyboardInterrupt)
        reason = "Interrupted; this output is incomplete. Re-run into a fresh directory." if interrupted else str(e)
        if run is not None:
            info.update(status="INTERRUPTED" if interrupted else "FAILED", reason=reason)
            (run.out / "logs" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            run.save()
            summary(run, info, differences, statuses, gains)
        print("FOLLOWUP ERROR: " + reason, file=sys.stderr)
        return 130 if interrupted else 2


if __name__ == "__main__":
    sys.exit(main())
