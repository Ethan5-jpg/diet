#!/usr/bin/env python3
"""Two-week CGM/diet QC and same-visit diet/gut associations."""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def protect_output(output, data_root, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or source in output.parents:
            raise ValueError("Output must not be inside an existing source run")
    for source in PACKAGE.iterdir():
        if source.is_dir() and source.name != "outputs" and (output == source or source in output.parents):
            raise ValueError("Output must not be inside code/archives")
    root = Path(data_root).resolve() if data_root else PACKAGE.parent
    allowed = PACKAGE / "outputs"
    if (output == root or root in output.parents) and output != allowed and allowed not in output.parents:
        raise ValueError("Inside Data, use meal_cgm_exploration/outputs only")
    for parent in [output] + list(output.parents):
        if (parent / "manifest.json").exists():
            raise ValueError("Output root must not be inside an existing run")


def early(out, state):
    (out / "manifest.json").write_text(json.dumps(dict(data_kind="UNVERIFIED",
        status=dict(window_qc_followup=state)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root")
    p.add_argument("--meal-run", default=str(PACKAGE / "outputs/meal_20260924T143630_752200Z"))
    p.add_argument("--seven-run", default=str(PACKAGE / "outputs/seven_diets_20260925T093822_426339Z"))
    p.add_argument("--nutrition-run", default=str(PACKAGE / "outputs/nutrition_bridge_20260926T071736_080048Z"))
    p.add_argument("--gut-run", default=str(PACKAGE / "outputs/gut_meal_bridge_v2_20260929T044951_469958Z"))
    p.add_argument("--sources-config", default=str(PACKAGE / "gut_meal_bridge_v2/sources.json"))
    p.add_argument("--score-sources", default=str(PACKAGE / "seven_diet_meals/sources.json"))
    p.add_argument("--diary")
    p.add_argument("--cgm-metadata")
    p.add_argument("--window-starts")
    p.add_argument("--diet-main-threshold", type=float, choices=[.05, .10], default=.05)
    p.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    p.add_argument("--stage", choices=["check", "audit", "all"], default="all")
    p.add_argument("--show")
    p.add_argument("--page", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--brief", action="store_true")
    args = p.parse_args(argv)
    if sys.version_info < (3, 8):
        print("Python>=3.8 required", file=sys.stderr)
        return 2
    if args.show:
        try:
            from window_qc_followup.reporting import show
            show(args.show, args.output_root, args.page)
            return 0
        except Exception as error:
            print("SHOW ERROR: " + str(error), file=sys.stderr)
            return 2
    if args.page is not None or args.brief:
        p.error("--page/--brief require --show")
    out, run = None, None
    state = dict(status="RUNNING", stage=args.stage, diet_main_threshold=args.diet_main_threshold)
    try:
        output = Path(args.output_root).expanduser().resolve()
        protect_output(output, args.data_root, [args.meal_run, args.seven_run, args.nutrition_run, args.gut_run])
        out = output / datetime.now(timezone.utc).strftime("window_qc_followup_%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
        early(out, state)
        print("RUN_DIR=" + str(out), flush=True)
        from hppmeal.common import Run, write_json, sha256
        from seven_diet_meals.inputs import record_expected, unchanged
        from window_qc_followup import VERSION
        from window_qc_followup.inputs import bundle, inspect, load, numeric_models
        from window_qc_followup.coverage import build, attach
        from window_qc_followup.analysis import run_models, threshold_summary
        from window_qc_followup.reporting import build_pages
        print("WINDOW_QC_VERSION=" + VERSION, flush=True)
        print("CHECK upload, dependencies and frozen sources (large cache hash may take time)", flush=True)
        expected = bundle()
        root, config, paths, pinned, audit, mapping = inspect(args)
        expected.update(pinned)
        run = Run(root, out, config)
        run.status["window_qc_followup"] = state
        state.update(version=VERSION)
        run.save()
        record_expected(run, expected)
        run.table(audit, "00_input_audit.csv")
        write_json(out / "analysis_spec.json", dict(version=VERSION, data_kind=config["data_kind"],
            cgm_duration_hours=336, cadence_minutes=15, expected_slots=1344, slot_tolerance_seconds=180,
            cadence_phase_policy="first timestamp fixes sensor cadence phase; independent of glucose/coverage",
            comparison="strictly_less_than", diet_main_threshold=args.diet_main_threshold,
            sensitivity_thresholds=[.10, .20], same_visit_window_confirmed_by="user reports supervisor confirmation, 2026-10-04",
            microbiome_date_policy="audit; do not exclude after-CGM records; not a causal ordering claim",
            default_cgm_anchor="first recorded timestamp proxy, not verified sensor activation",
            unanchored_missing_rate="NA; do not misclassify an unmatched window as absent CGM records",
            meal_window="baseline start >= window start; post-120-minute endpoint < exclusive window end",
            diet_coverage="14 calendar date presence; cannot measure within-day meal omissions",
            diet_score_window="unchanged upstream score record range; not recalculated from the QC 14 calendar dates",
            whole_cgm_reference="frozen original same-connection Z values; may extend beyond fixed 336h",
            planned_new_models=dict(diet=168, gut=96), sources=paths, plan_sha256=sha256(Path(__file__).parent / "PLAN.md")))
        meals, people, diary, plan, metadata = load(run, paths, mapping)
        dtable, gtable = None, None
        if args.stage == "check":
            state.update(status="CHECK_PASS", reason="Inputs and identities checked; no coverage or effects computed")
        else:
            quality = build(run, paths, diary, plan, metadata, lambda x: print(x, flush=True))
            tagged = attach(meals, quality)
            run.table(threshold_summary(quality, tagged, args.diet_main_threshold), "02_threshold_flow.csv")
            run.table(tagged[["participant_id", "meal_id", "connection_id", "quality_audited", "inside_window",
                             "cgm_missing_rate", "diet_missing_day_rate"]], "meal_quality.csv", private=True)
            if args.stage == "audit":
                state.update(status="AUDIT_COMPLETE", reason="Coverage only; no new association models estimated")
            else:
                dtable, gtable = run_models(run, meals, people, quality,
                    numeric_models(paths["previous_diet"]), numeric_models(paths["previous_gut"]),
                    args.diet_main_threshold, lambda x: print(x, flush=True))
                n_d = int(dtable.loc[dtable.setting.str.startswith("qc"), "status"].eq("ESTIMATED").sum())
                n_g = int(gtable.loc[gtable.setting.str.startswith("qc"), "status"].eq("ESTIMATED").sum())
                state.update(status="COMPLETE_WITH_LIMITATIONS" if n_d == 168 and n_g == 96 else "PARTIAL",
                    reproduction="PASS:42 diet+48 gut", diet_new_estimated=n_d, gut_new_estimated=n_g,
                    reason="Same-visit selected-candidate associations; window may use first-record proxy; diet is day-presence only. NE rows retained; see full tables.")
        print("VERIFY all frozen sources unchanged", flush=True)
        unchanged(run)
        build_pages(run, state, dtable, gtable)
        state["result_sha256"] = {str(f.relative_to(out)): sha256(f) for folder in ["tables", "private", "pages"]
            for f in sorted((out / folder).glob("*")) if f.is_file()}
        state["result_sha256"]["analysis_spec.json"] = sha256(out / "analysis_spec.json")
        state["result_sha256"]["SUMMARY.md"] = sha256(out / "SUMMARY.md")
        run.save()
        print("WINDOW_QC={}\nDATA={}".format(state["status"], config["data_kind"]), flush=True)
        print((out / "pages/01.txt").read_text(encoding="utf-8"), flush=True)
        print('VIEW: python3 window_qc_followup/run_analysis.py --show "{}" --brief'.format(out), flush=True)
        return 2 if state["status"] == "PARTIAL" else 0
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", reason=str(error) or "Interrupted")
        if out is not None:
            (out / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if run is None:
                early(out, state)
            else:
                run.save()
        print("WINDOW_QC ERROR: " + state["reason"], file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
