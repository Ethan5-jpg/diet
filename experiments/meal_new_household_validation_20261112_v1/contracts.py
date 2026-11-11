"""Read frozen experiment receipts and keep the validation stages separate."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
MEAL = ROOT.parent / "meal_carbohydrate_modification_20261112_v1"
ARCHIVE = MEAL / "runs_TRE_ONLY/models-1"
FEASIBILITY = (ROOT.parent / "meal_coffee_modification_20261112_v1" /
               "runs_TRE_ONLY/ADDITIONAL_MEAL_COHORT_FEASIBILITY_TRE_ONLY.json")
sys.path.insert(0, str(MEAL))
try:
    from analyze import bases
    from paired import archived_inputs, verified_result
    from prepare import (
        NUTRIENTS, align, base_modules, digest, episodes, food_groups, helper, load,
        require, scoped_read, verified_roster,
    )
finally:
    sys.path.pop(0)

PAIRED_SHA = "6495e7bf2fe831a971be31a1dbc1ddaf582773c13a04d89dbf7641a478239b0a"
KEYS = ["input_row", "start"]
ENDPOINTS = ("net_p0", "peak_p0", "net_m15", "net_p15", "pre_p0")


def run_path(name):
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", name), "Unexpected run name")
    return ROOT / "runs_TRE_ONLY" / name


def validate_args(args):
    package = args.package.resolve()
    require(package / "experiments" / ROOT.name == ROOT and re.fullmatch(r"[0-9a-f]{40}", args.commit),
            "Unexpected study package or commit")
    return package


def check_sources(sources):
    require(all(digest(Path(p)) == sha for p, sha in sources.items()), "A registered source changed")


def completed(path, filename):
    receipt = load(path / "COMPLETE.json")
    registration, summary, payload = [path / name for name in ("registration.json", "summary.json", filename)]
    require(all(digest(file) == receipt[key] for key, file in
                (("registration", registration), ("summary", summary), ("payload", payload))),
            "A completed run receipt differs")
    return load(registration), load(summary), payload, receipt


def finish(path, summary, payload, io):
    io.write(path / "summary.json", summary)
    io.write(path / "COMPLETE.json", dict(
        registration=digest(path / "registration.json"), summary=digest(path / "summary.json"),
        payload=digest(payload),
    ))


def archived_contract():
    require(digest(MEAL / "paired.py") == PAIRED_SHA, "Archived replay helper differs")
    fitted, _ = verified_result(ARCHIVE)
    prep = MEAL / "runs_TRE_ONLY" / fitted["prepared"]
    registration = prep / "registration.json"
    require(digest(registration) == load(prep / "COMPLETE.json")["registration"],
            "Archived meal preparation registration differs")
    return fitted, registration, load(registration)
