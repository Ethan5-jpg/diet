#!/usr/bin/env python3
"""Seven personal dietary variables and observed meal-CGM responses."""
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
    parser.add_argument("--run-dir", help="Completed original meal_... directory")
    parser.add_argument("--data-root", help="Data directory; default original run root")
    parser.add_argument("--sources-config", default=str(PACKAGE / "seven_diet_meals/sources.json"))
    parser.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    parser.add_argument("--stage", choices=["check", "all"], default="all")
    parser.add_argument("--show", help="Print SUMMARY.md in an existing seven_diets_... directory")
    args = parser.parse_args(argv)
    if args.show:
        try:
            print((Path(args.show).expanduser() / "SUMMARY.md").read_text(encoding="utf-8"))
            return 0
        except OSError as e:
            print("Cannot read summary; use exact RUN_DIR: " + str(e), file=sys.stderr)
            return 2
    if not args.run_dir:
        parser.error("--run-dir is required")
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required", file=sys.stderr)
        return 2
    run, state, results = None, {"status": "RUNNING"}, None
    try:
        from hppmeal.common import Run, require, write_json
        from seven_diet_meals import VERSION
        from seven_diet_meals.inputs import (bundle, inspect, record_expected, unchanged, load)
        from seven_diet_meals.models import run_models
        from seven_diet_meals.reporting import summary, score_reports, figures
        print("CHECK seven-diet package, original meals and all score sources", flush=True)
        verified = bundle()
        source = Path(args.run_dir).expanduser().resolve()
        root, c, paths, expected, audit = inspect(source, args.data_root, args.sources_config)
        expected.update(verified)
        out_root = Path(args.output_root).expanduser().resolve()
        require(out_root != source and source not in out_root.parents, "New outputs must be outside original meal run")
        out = out_root / datetime.now(timezone.utc).strftime("seven_diets_%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
        run = Run(root, out, c)
        state.update(version=VERSION, stage=args.stage)
        run.status["seven_diets"] = state
        run.save()
        summary(run, state)
        print("RUN_DIR=" + str(out), flush=True)
        record_expected(run, expected)
        run.table(audit, "00_preflight.csv")
        write_json(out / "analysis_spec.json", dict(version=VERSION, original_meal_run=str(source), sources={k: str(v) for k, v in paths.items()},
            stage=args.stage, min_people=50, min_residual_df=30,
            A=dict(min_meals=5, min_dates=3, weighting="one person one row", se="HC3"),
            B=dict(min_meals=10, min_dates=3, min_carb_range_g=30, weighting="person total weight 1", se="participant CR1 with absorbed FE"),
            outcomes=["mean_increment", "peak_rise_0_120"], planned_models=126,
            fdr="per setting: primary14 (A-M2+B increment), secondary14 (A-M2+B peak), background14 (A-M1 both outcomes)"))
        meals, people = load(run, paths)
        if args.stage == "check":
            unchanged(run)
            state.update(status="CHECK_PASS", reason="Inputs and row-level contracts checked; models have not run")
        else:
            run.table(people.reset_index(), "01_person_inputs.csv", private=True)
            results, flow, coefficients, membership, descriptive = run_models(meals, people, lambda x: print(x, flush=True))
            run.table(results, "all_models.csv")
            run.table(results.loc[results.branch.eq("A")], "02_diet_person_models.csv")
            run.table(results.loc[results.branch.eq("B")], "03_diet_carb_interactions.csv")
            run.table(flow, "01_sample_flow.csv")
            run.table(membership, "02_model_membership.csv", private=True)
            run.table(descriptive, "05_person_response_descriptives.csv")
            run.table(coefficients, "06_coefficients.csv")
            comparison = results.pivot(index=["diet", "branch", "outcome", "model"], columns="setting",
                                       values=["estimate", "ci_low", "ci_high", "q_family14", "people", "meals", "status"])
            comparison.columns = ["__".join(x) for x in comparison.columns]
            run.table(comparison.reset_index(), "04_sensitivity_comparison.csv")
            score_reports(run, people, membership)
            figures(run, results)
            print("VERIFY all source files unchanged", flush=True)
            unchanged(run)
            count = int(results.status.eq("ESTIMATED").sum())
            state.update(status="COMPLETE" if count == len(results) else "COMPLETE_WITH_LIMITATIONS",
                         estimated_models=count, planned_models=len(results), reason="" if count == len(results) else "Some models not estimable; see all_models.csv reasons")
        run.save()
        summary(run, state, results)
        print("SEVEN_DIETS={}\nSUMMARY={}".format(state["status"], out / "SUMMARY.md"), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        reason = "Interrupted; rerun into a new output directory" if isinstance(e, KeyboardInterrupt) else str(e)
        if run is not None:
            state.update(status="INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "FAILED", reason=reason)
            (run.out / "logs/error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            run.save()
            summary(run, state, results)
        print("SEVEN_DIETS ERROR: " + reason, file=sys.stderr, flush=True)
        return 130 if isinstance(e, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
