#!/usr/bin/env python3
"""Four old gut candidates and existing meal phenotypes; no new mediation."""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def protect_output(output, sources):
    frozen = list(sources) + [p for p in PACKAGE.iterdir() if p.is_dir() and p.name != "outputs"]
    for folder in frozen:
        folder = folder.resolve()
        if output == folder or folder in output.parents:
            raise ValueError("Output root must be outside source runs/code/archive: " + str(folder))
    for folder in [output] + list(output.parents):
        if (folder / "manifest.json").exists():
            raise ValueError("Output root must not be inside an existing run")


def early_state(out, state):
    (out / "manifest.json").write_text(json.dumps(dict(data_kind="UNVERIFIED",
        status=dict(gut_meal_bridge_v2=state)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meal-run", default=str(PACKAGE / "outputs/meal_20260924T143630_752200Z"))
    p.add_argument("--nutrition-run", default=str(PACKAGE / "outputs/nutrition_bridge_20260926T071736_080048Z"))
    p.add_argument("--data-root")
    p.add_argument("--sources-config", default=str(PACKAGE / "gut_meal_bridge_v2/sources.json"))
    p.add_argument("--score-sources", default=str(PACKAGE / "seven_diet_meals/sources.json"))
    p.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    p.add_argument("--stage", choices=["check", "all"], default="all")
    p.add_argument("--show")
    p.add_argument("--brief", action="store_true")
    p.add_argument("--page", type=int, choices=[1, 2, 3])
    args = p.parse_args(argv)
    if sys.version_info < (3, 8):
        print("Python >=3.8 required", file=sys.stderr)
        return 2
    if args.show:
        try:
            from gut_meal_bridge_v2.reporting import show
            show(args.show, args.output_root, args.page)
            return 0
        except Exception as error:
            print("SHOW ERROR: " + str(error), file=sys.stderr)
            return 2
    if args.brief or args.page is not None:
        p.error("--brief/--page require --show")
    state = dict(status="RUNNING", stage=args.stage)
    out, run = None, None
    try:
        source = Path(args.meal_run).expanduser().resolve()
        nutrition = Path(args.nutrition_run).expanduser().resolve()
        output = Path(args.output_root).expanduser().resolve()
        protect_output(output, [source, nutrition])
        out = output / datetime.now(timezone.utc).strftime("gut_meal_bridge_v2_%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
        early_state(out, state)
        print("RUN_DIR=" + str(out), flush=True)
        print("CHECK upload, frozen inputs and exact candidate taxa", flush=True)
        from hppmeal.common import Run, sha256, write_json
        from seven_diet_meals.inputs import record_expected, unchanged
        from gut_meal_bridge_v2 import VERSION
        from gut_meal_bridge_v2.inputs import bundle, inspect, load, contract
        from gut_meal_bridge_v2.analysis import prepare, analyze
        from gut_meal_bridge_v2.reporting import pages, summary
        expected = bundle()
        print("GUT_MEAL_BRIDGE_VERSION=" + VERSION, flush=True)
        root, config, paths, pinned, audit, mapping = inspect(source, nutrition, args.data_root,
            args.sources_config, args.score_sources)
        expected.update(pinned)
        run = Run(root, out, config)
        state.update(version=VERSION)
        run.status["gut_meal_bridge_v2"] = state
        run.save()
        record_expected(run, expected)
        run.table(audit, "00_input_audit.csv")
        write_json(out / "analysis_spec.json", dict(version=VERSION, data_kind=config["data_kind"],
            meal_run=str(source), nutrition_run=str(nutrition), sources=paths, contract=contract(),
            contract_sha256=sha256(PACKAGE / "gut_meal_bridge_v2/CONTRACT.json")))
        meals, people = load(run, paths, mapping)
        persons, members, flow, mismatch = prepare(meals, people)
        run.table(flow, "02_sample_flow.csv")
        run.table(persons, "person_inputs.csv", private=True)
        run.table(members, "meal_membership.csv", private=True)
        if args.stage == "check":
            unchanged(run)
            state.update(status="CHECK_PASS", common_people=len(persons),
                reason="Input identities, dates and common sample checked; no models estimated")
        else:
            result = analyze(persons, members, lambda message: print(message, flush=True))
            result.update(flow=flow, mismatch=mismatch)
            for name in ["models", "coefficients", "sample", "timing", "descriptive"]:
                run.table(result[name], name + ".csv")
            rows = result["models"]
            state.update(status="COMPLETE_WITH_LIMITATIONS", estimated=int(rows.status.eq("ESTIMATED").sum()),
                primary_estimated=int((rows.outcome.eq("mean_increment") & rows.status.eq("ESTIMATED")).sum()),
                reason="Same-cohort selected-candidate associations; timing may be unknown; whole-CGM reference shares readings. No causal mediation or independent validation.")
            pages(run, result, state)
            print("VERIFY source files unchanged", flush=True)
            unchanged(run)
            state["result_sha256"] = {str(f.relative_to(out)): sha256(f) for folder in ["tables", "private", "pages"]
                for f in sorted((out / folder).glob("*")) if f.is_file()}
        run.save()
        summary(run, state)
        print("GUT_MEAL_BRIDGE={}\nDATA={}".format(state["status"], config["data_kind"]), flush=True)
        if args.stage == "all":
            print('VIEW: python3 gut_meal_bridge_v2/run_analysis.py --show "{}" --brief'.format(out), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", reason=str(error) or "Interrupted")
        if out is not None:
            (out / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if run is None:
                early_state(out, state)
            else:
                run.save()
        print("GUT_MEAL_BRIDGE ERROR: " + state["reason"], file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
