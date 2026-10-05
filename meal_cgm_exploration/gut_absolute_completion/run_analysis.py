#!/usr/bin/env python3
"""Complete absolute glucose support analyses and export full tables/figures."""
import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in [None, ""]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy
from hppmeal.common import require, sha256, write_json
from gut_absolute_robustness.inputs import unchanged
from gut_absolute_completion import VERSION
from gut_absolute_completion.inputs import MODULE, PACKAGE, bundle, contract, load_inputs
from gut_absolute_completion.analysis import complete
from gut_absolute_completion.reporting import save_pages, show, write_results
from gut_absolute_completion.figures import make_figures


def protect_output(source, previous, root):
    root = Path(root).expanduser().resolve()
    for origin in [source, previous]:
        origin = Path(origin).expanduser().resolve()
        require(root != origin and origin not in root.parents, "Output must be outside both input runs")
    for p in [root] + list(root.parents):
        require(not (p / "manifest.json").exists(), "Output cannot be nested inside an existing run")
    return root


def save_table(out, name, frame, private=False):
    folder = Path(out) / ("private" if private else "tables")
    folder.mkdir(exist_ok=True)
    if not private:
        frame = frame.drop(columns=[x for x in ["participant_id", "meal_id", "person_sha256", "meal_sha256"] if x in frame])
    frame.to_csv(folder / (name + ".csv"), index=False, float_format="%.17g")


def run(args):
    require(sys.version_info >= (3, 8), "Python >=3.8 required")
    source, previous = Path(args.source).expanduser().resolve(), Path(args.previous).expanduser().resolve()
    root = protect_output(source, previous, args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / datetime.now(timezone.utc).strftime("gut_absolute_completion_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(exist_ok=False)
    state = dict(version=VERSION, status="RUNNING", data_kind="UNVERIFIED", source=str(source), previous=str(previous),
                 hypothesis_timing="exploratory_after_prior_results", new_planned=32, reused_common=40,
                 date_order_filter=False, additional_14day_completeness_filter=False, figures_status="NOT_RUN")
    write_json(out / "manifest.json", state)
    print("RUN_DIR=" + str(out), flush=True)
    try:
        print("CHECK uploaded module, both frozen runs, common members and previous FDR", flush=True)
        pins = bundle()
        data = load_inputs(source, previous)
        pins.update(data["pins"])
        state.update(data_kind=data["data_kind"], input_sha256=pins, source_checks="PASS",
                     software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
        write_json(out / "manifest.json", state)
        write_json(out / "analysis_spec.json", dict(version=VERSION, data_kind=state["data_kind"], contract=contract(),
                   contract_sha256=sha256(MODULE / "CONTRACT.json"), source=str(source), previous=str(previous)))
        print("SUPPORT_COUNTS_FIXED", flush=True)
        print(data["flow"][["setting", "threshold", "people", "meals"]].to_string(index=False), flush=True)
        common, algebra, reuse = complete(data, lambda x: print(x, flush=True))
        original = data["previous_b"].copy()
        tables = dict(common_models=common, original_models=original, source_models_48=data["original"],
                      support_flow=data["flow"], selection_balance=data["balance"], algebra_check=algebra, reuse_audit=reuse)
        all_models = pd.concat([common.assign(block="common_adjustment"), original.assign(block="original_adjustment")], ignore_index=True)
        all_models["shared_pre_estimate"] = all_models.outcome.eq("baseline_glucose")
        tables["all_meal_models"] = all_models
        for name, frame in tables.items():
            save_table(out, name, frame)
        save_table(out, "participant_membership", data["membership"], private=True)
        save_table(out, "group_membership_hashes", data["flow"][["setting", "threshold", "people", "meals", "person_sha256", "meal_sha256"]], private=True)
        columns = ["block", "setting", "threshold", "taxon", "outcome", "people", "reason"]
        save_table(out, "not_estimated", all_models.loc[all_models.status.ne("ESTIMATED"), columns])
        state.update(common_estimated=int(common.status.eq("ESTIMATED").sum()),
                     new_estimated=int(common.loc[common.estimate_origin.eq("NEW_FIT"), "status"].eq("ESTIMATED").sum()),
                     original_reference_estimated=int(original.status.eq("ESTIMATED").sum()),
                     algebra_pass=int(algebra.status.eq("PASS_ALGEBRA_ONLY").sum()))
        write_results(out, state["data_kind"], common, original, data["flow"])
        (out / "METHODS.md").write_text("DATA=" + state["data_kind"] + "\n\n" + (MODULE / "METHODS.md").read_text(encoding="utf-8"), encoding="utf-8")
        try:
            version = make_figures(out, state["data_kind"], common, original, data["flow"])
            state["software"]["matplotlib"] = version
            state["figures_status"] = "COMPLETE_2_PNG_PDF_SVG"
        except ImportError as error:
            state["figures_status"] = "UNAVAILABLE_IMPORT_ERROR"
            (out / "FIGURES_ERROR.txt").write_text(str(error) + "\nAnalysis tables are complete; plotting dependencies are unavailable.\n", encoding="utf-8")
        state["status"] = "COMPLETE_WITH_LIMITATIONS" if state["common_estimated"] == 72 else "PARTIAL"
        if state["common_estimated"] == 72 and state["figures_status"] != "COMPLETE_2_PNG_PDF_SVG":
            state["status"] = "COMPLETE_ANALYSIS_FIGURES_UNAVAILABLE"
        unchanged(pins)
        state["sources_unchanged"] = "PASS"
        save_pages(out, state, common, original, data["flow"], data["balance"])
        state["result_sha256"] = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", state)
        show(out, root)
        print("VIEW: python3 gut_absolute_completion/run_analysis.py --show '{}' --brief".format(str(out).replace("'", "'\"'\"'")), flush=True)
        return 0 if state["status"] == "COMPLETE_WITH_LIMITATIONS" else 2
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error) or "Interrupted")
        write_json(out / "manifest.json", state)
        (out / "ERROR.txt").write_text(state["error"] + "\n", encoding="utf-8")
        print("GUT_ABSOLUTE_COMPLETION={}: {}".format(state["status"], state["error"]), file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(PACKAGE / "outputs" / contract()["original_run"]))
    parser.add_argument("--previous", default=str(PACKAGE / "outputs" / contract()["previous_run"]))
    parser.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    parser.add_argument("--show", metavar="RUN_OR_LATEST")
    parser.add_argument("--page", type=int, choices=[1, 2, 3])
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.show:
            show(args.show, args.output_root, args.page)
            return 0
        require(args.page is None and not args.brief, "Use --show with --page or --brief")
        return run(args)
    except (Exception, KeyboardInterrupt) as error:
        print("GUT_ABSOLUTE_COMPLETION_FAILED: " + (str(error) or "Interrupted"), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
