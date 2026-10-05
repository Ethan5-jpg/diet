"""Verify both frozen runs before reusing estimates; never read raw CGM."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hppmeal.common import require, sha256
from gut_meal_bridge_v2.analysis import model_columns
from gut_absolute_robustness import VERSION as PREVIOUS_VERSION
from gut_absolute_robustness.inputs import (DEP_FILES as BASE_DEPS, SOURCE_FILES,
    load_source, unchanged, contract as prior_contract)
from gut_absolute_robustness.analysis import (support_tables, common_columns,
    absolute_fdr, sensitivity_fdr, original_fdr)
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent
DEP_FILES = BASE_DEPS + ["gut_absolute_robustness/" + n for n in
    ["__init__.py", "inputs.py", "analysis.py", "reporting.py", "CONTRACT.json"]]
PREVIOUS_FILES = ["tables/absolute_models.csv", "tables/support_models.csv",
    "tables/source_models_48.csv", "tables/support_flow.csv", "tables/selection_balance.csv",
    "private/group_membership_hashes.csv", "private/participant_membership.csv"]
KEYS = ["setting", "threshold", "taxon", "outcome"]
NUMERIC = ["estimate", "se", "ci_low", "ci_high", "p"]


def contract():
    return json.loads((MODULE / "CONTRACT.json").read_text(encoding="utf-8"))


def bundle():
    path = MODULE / "MANIFEST.json"
    require(path.is_file(), "Upload the entire gut_absolute_completion folder")
    body = json.loads(path.read_text(encoding="utf-8"))
    required = {"__init__.py", "inputs.py", "analysis.py", "reporting.py", "figures.py", "run_analysis.py", "CONTRACT.json", "METHODS.md"}
    require(body.get("version") == VERSION and required.issubset(body.get("files", {})), "Incomplete new module manifest")
    require(set(body.get("dependencies", {})) == set(DEP_FILES), "Incomplete dependency manifest")
    pins = {str(path): sha256(path)}
    for section, root in [("files", MODULE), ("dependencies", PACKAGE)]:
        for name, digest in body[section].items():
            p = (root / name).resolve()
            require(root in p.parents, "Invalid manifest path")
            pins[str(p)] = digest
    unchanged(pins)
    c, old = contract(), prior_contract()
    require(c["previous_version"] == PREVIOUS_VERSION, "Previous module version differs")
    for key in ["settings", "taxa", "thresholds"]:
        require(c[key] == old[key], "Fixed design changed: " + key)
    require(c["outcomes"] == old["absolute_outcomes"], "Fixed outcomes changed")
    return pins


def compare(left, right, keys, numeric, strings=()):
    require(not left.duplicated(keys).any() and not right.duplicated(keys).any(), "Duplicate comparison keys")
    d = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    require(len(d) == len(left) == len(right), "Frozen comparison keys differ")
    for name in numeric:
        require(np.allclose(d[name + "_left"], d[name + "_right"], rtol=1e-10, atol=1e-12, equal_nan=True),
                "Frozen values differ: " + name)
    for name in strings:
        require(d[name + "_left"].fillna("").eq(d[name + "_right"].fillna("")).all(), "Frozen labels differ: " + name)


def validate_models(d, thresholds, outcomes, common):
    c = contract()
    expected = {(s, h, t, o) for s in c["settings"] for h in thresholds for t in c["taxa"] for o in outcomes}
    require(not d.duplicated(KEYS).any() and len(d) == len(expected) and
            set(d[KEYS].itertuples(index=False, name=None)) == expected, "Previous model specifications incomplete")
    require(d.status.isin(["ESTIMATED", "NOT_ESTIMATED"]).all(), "Invalid previous status")
    estimated = d.status.eq("ESTIMATED")
    require(np.isfinite(d.loc[estimated, NUMERIC].to_numpy(float)).all(), "Nonfinite estimated result")
    require(d.loc[~estimated, NUMERIC].isna().all().all(), "NE rows contain numeric results")
    for r in d.itertuples():
        expected_cols = common_columns(r.taxon) if common else model_columns(r.taxon, r.outcome)
        require(r.adjustment == ";".join(expected_cols[1:]), "Previous adjustment differs")
        require(r.unit == "mg/dL per original CLR-Z", "Previous units differ")
        if r.status == "ESTIMATED":
            require(r.se > 0 and r.ci_low <= r.estimate <= r.ci_high and 0 <= r.p <= 1, "Invalid estimate/CI/p")
        else:
            require(pd.notna(r.reason) and str(r.reason).strip(), "NE reason missing")


def load_inputs(source, previous):
    source, previous = Path(source).expanduser().resolve(), Path(previous).expanduser().resolve()
    persons, members, old, samples, kind, pins = load_source(source)
    path = previous / "manifest.json"
    require(path.is_file(), "Previous absolute run missing: " + str(path))
    state = json.loads(path.read_text(encoding="utf-8"))
    require(state.get("version") == contract()["previous_version"], "Previous result version differs")
    require(state.get("status") in ["COMPLETE_WITH_LIMITATIONS", "PARTIAL"], "Previous run did not finish")
    require(state.get("data_kind") == kind, "REAL/SYNTHETIC source mismatch")
    require(state.get("source_reproduction") == "PASS_48" and state.get("source_files_unchanged") == "PASS", "Previous source verification did not pass")
    # Match file CONTENT after a run is moved; never silently switch source cohorts.
    saved_source = Path(state.get("source", ""))
    for rel in ["manifest.json"] + SOURCE_FILES:
        recorded = state.get("input_sha256", {}).get(str(saved_source / rel))
        require(recorded and recorded == pins[str(source / rel)], "Previous run belongs to different source content: " + rel)
    pins[str(path)] = sha256(path)
    frames = {}
    for name in PREVIOUS_FILES:
        p = previous / name
        digest = state.get("result_sha256", {}).get(name)
        require(digest and p.is_file() and sha256(p) == digest, "Missing/changed previous result: " + name)
        pins[str(p)] = digest
        frames[name] = pd.read_csv(p, float_precision="round_trip", dtype={"participant_id": str, "person_sha256": str, "meal_sha256": str})
    a, b = frames["tables/absolute_models.csv"], frames["tables/support_models.csv"]
    validate_models(a, ["m5d3"], contract()["outcomes"], True)
    validate_models(b, [x["name"] for x in contract()["thresholds"]], prior_contract()["original_meal_outcomes"], False)
    require(int(a.status.eq("ESTIMATED").sum()) == state.get("absolute_estimated"), "Previous A estimated count differs")
    require(int(b.loc[b.threshold.ne("m5d3"), "status"].eq("ESTIMATED").sum()) == state.get("support_new_estimated"), "Previous B estimated count differs")
    compare(a, absolute_fdr(a), KEYS, ["q_panel12", "planned_family_size"])
    compare(b, sensitivity_fdr(b), KEYS, ["q_family", "q_meal12", "planned_family_size"], ["family"])
    k = ["setting", "taxon", "outcome"]
    compare(old, original_fdr(old), k, ["q_family", "q_meal12", "planned_family_size"])
    compare(frames["tables/source_models_48.csv"], old, k, NUMERIC + ["people", "meals", "q_family", "q_meal12"], ["status", "adjustment", "unit", "family"])
    old_meals = old.loc[old.outcome.isin(prior_contract()["original_meal_outcomes"])].copy()
    compare(b.loc[b.threshold.eq("m5d3")], old_meals, k, NUMERIC + ["people", "meals", "q_family", "q_meal12"], ["status", "adjustment", "unit"])
    compare(a.loc[a.outcome.eq("baseline_glucose")], old_meals.loc[old_meals.outcome.eq("baseline_glucose")], k,
            NUMERIC + ["people", "meals"], ["status", "adjustment", "unit"])
    flow, balance, membership = support_tables(persons, members)
    group_keys = ["setting", "threshold"]
    compare(frames["tables/support_flow.csv"], flow, group_keys,
            ["people", "meals", "min_meals", "min_dates", "retained_fraction", "excluded_people", "meals_p50", "dates_p50"])
    compare(frames["private/group_membership_hashes.csv"], flow, group_keys, ["people", "meals"], ["person_sha256", "meal_sha256"])
    compare(frames["private/participant_membership.csv"], membership, group_keys + ["participant_id"], [])
    compare(frames["tables/selection_balance.csv"], balance, group_keys + ["field"],
            ["retained_people", "excluded_people", "retained_mean", "excluded_mean", "retained_sd", "excluded_sd", "standardized_difference"], ["smd_status"])
    for frame in [a, b]:
        for (setting, threshold), g in frame.groupby(group_keys):
            f = flow.loc[flow.setting.eq(setting) & flow.threshold.eq(threshold)].iloc[0]
            require(g.people.eq(f.people).all() and g.meals.eq(f.meals).all(), "Previous model/group counts differ")
    unchanged(pins)
    return dict(persons=persons, members=members, original=old, previous_a=a, previous_b=b,
                flow=flow, balance=balance, membership=membership, data_kind=kind, pins=pins,
                previous=previous, source=source)
