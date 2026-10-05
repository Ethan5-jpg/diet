#!/usr/bin/env python3
"""Same-visit four-taxon absolute glucose and meal-count sensitivity extension."""
import argparse
import json
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
from gut_absolute_robustness import VERSION
from gut_absolute_robustness.inputs import MODULE, PACKAGE, bundle, contract, load_source, unchanged
from gut_absolute_robustness.analysis import reproduce, support_tables, analyze
from gut_absolute_robustness.reporting import save_pages, show


def protect_output(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    require(root != source and source not in root.parents, "Output must be outside the original source run")
    for p in [root] + list(root.parents):
        require(not (p / "manifest.json").exists(), "Output cannot be inside an existing run")
    return root


def save_table(out, name, frame, private=False):
    folder = Path(out) / ("private" if private else "tables")
    folder.mkdir(exist_ok=True)
    if not private:
        frame = frame.drop(columns=[x for x in ["participant_id", "meal_id", "person_sha256", "meal_sha256"] if x in frame])
    frame.to_csv(folder / (name + ".csv"), index=False, float_format="%.17g")


def run(args):
    require(sys.version_info >= (3, 8), "Python >=3.8 required")
    source = Path(args.source).expanduser().resolve()
    root = protect_output(source, Path(args.output_root).expanduser())
    root.mkdir(parents=True, exist_ok=True)
    out = root / datetime.now(timezone.utc).strftime("gut_absolute_robustness_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(exist_ok=False)
    state = dict(version=VERSION, status="RUNNING", data_kind="UNVERIFIED", source=str(source),
                 source_reproduction="NOT_RUN", source_files_unchanged="NOT_CHECKED",
                 date_order_filter=False, additional_14day_completeness_filter=False,
                 hypothesis_timing="exploratory_after_prior_results", summary_pages=3)
    write_json(out / "manifest.json", state)
    print("RUN_DIR=" + str(out), flush=True)
    pins = {}
    try:
        print("CHECK complete upload, original private summaries and member hashes", flush=True)
        pins.update(bundle())
        persons, members, old, samples, kind, source_pins = load_source(source)
        pins.update(source_pins)
        state["data_kind"] = kind
        state["input_sha256"] = pins
        state["software"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)
        write_json(out / "manifest.json", state)
        write_json(out / "analysis_spec.json", dict(version=VERSION, data_kind=kind, contract=contract(),
                   contract_sha256=sha256(MODULE / "CONTRACT.json"), source=str(source),
                   person_summary_scope="unchanged original common participants/meals; high-support groups filter people only"))
        flow, balance, membership = support_tables(persons, members)
        save_table(out, "support_flow", flow)
        save_table(out, "selection_balance", balance)
        save_table(out, "participant_membership", membership, private=True)
        save_table(out, "group_membership_hashes", flow[["setting", "threshold", "people", "meals", "person_sha256", "meal_sha256"]], private=True)
        print("SUPPORT_COUNTS (fixed before new models)", flush=True)
        print(flow[["setting", "threshold", "people", "meals"]].to_string(index=False), flush=True)
        reproduction = reproduce(persons, members, old, lambda x: print(x, flush=True))
        state["source_reproduction"] = "PASS_48"
        save_table(out, "source_reproduction", reproduction)
        save_table(out, "source_models_48", old)
        unchanged(pins)
        result = analyze(persons, members, old, lambda x: print(x, flush=True))
        tables = dict(absolute_models=result["absolute"], support_models=result["sensitivity"],
                      algebra_check=result["identity"], support_flow=flow, selection_balance=balance)
        for name, table in tables.items():
            save_table(out, name, table)
        reasons = []
        for block, name in [("A", "absolute_models"), ("B", "support_models")]:
            for row in tables[name].loc[tables[name].status.ne("ESTIMATED")].itertuples():
                reasons.append(dict(block=block, setting=row.setting, threshold=row.threshold, taxon=row.taxon, outcome=row.outcome, people=row.people, reason=row.reason))
        save_table(out, "not_estimated", pd.DataFrame(reasons, columns=["block", "setting", "threshold", "taxon", "outcome", "people", "reason"]))
        new_b = result["sensitivity"].loc[result["sensitivity"].threshold.ne("m5d3")]
        state.update(absolute_estimated=int(result["absolute"].status.eq("ESTIMATED").sum()),
                     support_new_estimated=int(new_b.status.eq("ESTIMATED").sum()),
                     source_meal_references=24, absolute_original_pre_references=8,
                     algebra_pass=int(result["identity"].status.eq("PASS_ALGEBRA_ONLY").sum()))
        state["status"] = "COMPLETE_WITH_LIMITATIONS" if state["absolute_estimated"] == 24 and state["support_new_estimated"] == 48 else "PARTIAL"
        unchanged(pins)
        state["source_files_unchanged"] = "PASS"
        save_pages(out, state, tables)
        (out / "METHODS.md").write_text("DATA=" + kind + "\n\n" + (MODULE / "METHODS.md").read_text(encoding="utf-8"), encoding="utf-8")
        state["result_sha256"] = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", state)
        print("SOURCE_REPRODUCTION=PASS_48; SOURCES_UNCHANGED=PASS", flush=True)
        show(out, root)
        print("VIEW: python3 gut_absolute_robustness/run_analysis.py --show latest --brief", flush=True)
        return 0 if state["status"] == "COMPLETE_WITH_LIMITATIONS" else 2
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error) or "Interrupted")
        write_json(out / "manifest.json", state)
        (out / "ERROR.txt").write_text(state["error"] + "\n", encoding="utf-8")
        print("GUT_ABSOLUTE_ROBUSTNESS={}: {}".format(state["status"], state["error"]), file=sys.stderr, flush=True)
        return 130 if state["status"] == "INTERRUPTED" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(PACKAGE / "outputs" / contract()["source_run"]))
    parser.add_argument("--output-root", default=str(PACKAGE / "outputs"))
    parser.add_argument("--show", metavar="RUN_OR_LATEST")
    parser.add_argument("--page", type=int, choices=[1, 2, 3])
    parser.add_argument("--brief", action="store_true", help="Show the three compact pages (default summary format)")
    args = parser.parse_args(argv)
    try:
        if args.show:
            show(args.show, args.output_root, args.page)
            return 0
        require(args.page is None and not args.brief, "Use --show latest with --page or --brief")
        return run(args)
    except (Exception, KeyboardInterrupt) as error:
        print("GUT_ABSOLUTE_ROBUSTNESS_FAILED: " + (str(error) or "Interrupted"), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    sys.exit(main())
