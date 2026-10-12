"""Post-primary sensitivity follow-up with frozen reference identity and 84-test FDR."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import need, require, sha256
from .inputs import load_config
from .models import bh_adjust
from .pipeline import COEFFICIENT_COLUMNS, SENSITIVITIES, run_models

FOLLOWUP_VERSION = "2026-10-10.ext1"
CORE_FILES = ["interaction/__init__.py", "interaction/common.py", "interaction/inputs.py",
              "interaction/models.py", "interaction/pipeline.py"]
KEY = ["exposure", "taxon", "outcome"]
MODELS = ["strict", "protocol_same_sample", "protocol_adjusted", "quadratic",
          "covariate_interactions", "common_all7_sample"]
MODEL_CODES = dict(zip(MODELS, ["S", "P", "A", "Q", "C", "U"]))


def load_reference(root, package, expected_kind="SERVER_INPUTS"):
    root, package = Path(root), Path(package)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("data_kind") == expected_kind, "Reference data kind mismatch")
    require(manifest.get("status") in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"], "Reference is not a completed fitted run")
    require(manifest.get("version") == "2026-10-10.2", "Unsupported reference version")
    fingerprints = {"manifest.json": sha256(manifest_path)}
    for name in ["analysis_config.json", "tables/all_interactions.csv"]:
        expected = manifest.get("output_sha256", {}).get(name)
        require(expected and sha256(root / name) == expected, "Reference output changed: " + name)
        fingerprints[name] = expected
    config = load_config(root / "analysis_config.json")
    require(config == manifest.get("config"), "Reference config/manifest mismatch")
    require(config["focus_exposures"] == ["hPDI", "NOVA4"], "Unexpected focus registry")
    require(set(config["extension_exposures"]) == {"AHEI", "AMED", "rEDIH", "EAT13", "Carbohydrate_pct"}, "Unexpected extension registry")
    require([t["id"] for t in config["taxa"]] == ["T023", "T046", "T068", "T096"], "Unexpected taxon registry")
    require([o["id"] for o in config["outcomes"]] == ["Mean", "CV", "SDw"], "Unexpected outcome registry")
    for name in CORE_FILES:
        require(manifest.get("code_sha256", {}).get(name) == sha256(package / name), "Core analysis code differs from reference: " + name)
    results = pd.read_csv(root / "tables/all_interactions.csv")
    need(results, KEY + ["model", "family", "status", "n", "sample_sha256", "q_all", "q_focus"] + COEFFICIENT_COLUMNS, "reference results")
    expected_keys = set()
    for e, t, o in itertools.product(config["focus_exposures"] + config["extension_exposures"],
                                     [t["id"] for t in config["taxa"]], [o["id"] for o in config["outcomes"]]):
        for model in ["primary"] + (SENSITIVITIES if e in config["focus_exposures"] else []):
            expected_keys.add((e, t, o, model))
    actual = list(results[KEY + ["model"]].itertuples(index=False, name=None))
    require(len(actual) == len(set(actual)) and set(actual) == expected_keys, "Reference model grid is incomplete or duplicated")
    expected_family = np.where(results.exposure.isin(config["focus_exposures"]), "focus", "extension")
    require((results.family == expected_family).all(), "Reference family labels disagree with registry")
    # Freeze file identities and recheck them at completion as well.
    return manifest, config, results, fingerprints


def verify_input_identity(reference, metadata):
    def identities(rows):
        require(len(rows) == len(set(x["label"] for x in rows)), "Duplicate input labels")
        return {x["label"]: x["sha256"] for x in rows}
    require(identities(reference["inputs"]) == identities(metadata["files"]),
            "Input identity differs from reference; do not combine changed data with old results")


def run_extended(master, metadata, config):
    expanded = copy.deepcopy(config)
    expanded["focus_exposures"] = config["focus_exposures"] + config["extension_exposures"]
    expanded["extension_exposures"] = []
    # The existing engine's common-focus branch now explicitly means ALL seven diets.
    results, slopes, flows, unused_fdr = run_models(master, metadata, expanded)
    for frame in [results, slopes, flows]:
        frame["model"] = frame.model.replace({"common_focus_sample": "common_all7_sample"})
        frame["family"] = np.where(frame.exposure.isin(config["focus_exposures"]), "focus", "extension")
    results = results.rename(columns={"q_all": "q_all84"}).drop(columns=["q_focus"])
    results["q_focus24"] = np.nan
    audit = []
    for model in ["primary"] + MODELS:
        for label, family in [("all84", None), ("focus24", "focus")]:
            mask = results.model.eq(model)
            if family:
                mask &= results.family.eq(family)
            subset = results.loc[mask]
            planned = 84 if family is None else 24
            require(len(subset) == planned, "Unexpected fixed FDR denominator")
            good = subset.status.eq("OK") & subset.interaction_p.notna()
            q = bh_adjust(subset.interaction_p.where(good, 1.0).to_numpy(float))
            field = "q_" + label
            results.loc[subset.index[good], field] = q[good.to_numpy()]
            audit.append(dict(model=model, family=label, planned_tests=planned,
                              estimable_tests=int(good.sum()), nonestimable_in_denominator=int((~good).sum()),
                              correction="BH; failed/skipped p=1; own q missing"))
    common = results[results.model.eq("common_all7_sample")]
    require(common.sample_sha256.nunique() == 1, "Seven-diet common sample is not identical across all 84 tests")
    return results, slopes, flows, pd.DataFrame(audit)


def compare_reference(reference, current, config):
    """Check identical estimands, excluding the deliberately changed common sample."""
    old = reference[~reference.model.eq("common_focus_sample")].copy()
    new = current.set_index(KEY + ["model"])
    numeric = list(COEFFICIENT_COLUMNS) + ["n", "parameters", "residual_df", "r2", "interaction_delta_r2",
                                         "d_center", "m_center", "d25", "d50", "d75", "m25", "m50", "m75"]
    rows = []
    for _, a in old.iterrows():
        key = tuple(a[k] for k in KEY + ["model"])
        b = new.loc[key]
        mismatches = []
        for name in ["status", "sample_sha256"]:
            if str(a[name]) != str(b[name]):
                mismatches.append(name)
        for name in numeric:
            av, bv = a.get(name, np.nan), b.get(name, np.nan)
            if not np.isclose(av, bv, rtol=1e-7, atol=1e-10, equal_nan=True):
                mismatches.append(name)
        if a.model == "primary" and not np.isclose(a.q_all, b.q_all84, rtol=1e-7, atol=1e-10, equal_nan=True):
            mismatches.append("primary_q_all84")
        if a.family == "focus" and not np.isclose(a.q_focus, b.q_focus24, rtol=1e-7, atol=1e-10, equal_nan=True):
            mismatches.append("q_focus24")
        rows.append(dict(zip(KEY + ["model"], key), matches=not mismatches,
                         differing_fields=";".join(mismatches), old_n=int(a.n), current_n=int(b.n)))
    return pd.DataFrame(rows)


def stability_tables(reference, results, alpha):
    primary = reference[reference.model.eq("primary") & reference.family.eq("extension")]
    selected = primary[primary.status.eq("OK") & primary.q_all.lt(alpha)].sort_values(KEY)
    selected_keys = set(selected[KEY].itertuples(index=False, name=None))
    all_new = results[results.family.eq("extension")].set_index(KEY + ["model"])
    rows = []
    for _, a in selected.iterrows():
        key = tuple(a[k] for k in KEY)
        row = dict(zip(KEY, key), primary_n=int(a.n), reference_beta_native=a.interaction_beta_native,
                   reference_q_all84=a.q_all)
        estimable, same, retained, qs, ns = [], [], [], [], []
        for model in MODELS:
            b = all_new.loc[key + (model,)]
            good = b.status == "OK" and np.isfinite(b.q_all84) and np.isfinite(b.interaction_beta_native)
            direction = good and a.interaction_beta_native * b.interaction_beta_native > 0
            keep = direction and b.q_all84 < alpha
            estimable.append(good); same.append(direction); retained.append(keep)
            qs.append(b.q_all84 if good else np.nan); ns.append(int(b.n))
            for name in ["status", "n", "interaction_beta_native", "interaction_ci_low_native", "interaction_ci_high_native", "q_all84"]:
                row[model + "__" + name] = b[name]
            row[model + "__same_direction"] = bool(direction)
            row[model + "__retained"] = bool(keep)
        row.update(all_six_estimable=all(estimable), same_direction_all_six=all(same),
                   retained_all_six=all(retained), retained_sensitivity_count=sum(retained),
                   max_sensitivity_q_all84=max(qs) if all(estimable) else np.nan,
                   min_sensitivity_n=min(ns), max_sensitivity_n=max(ns))
        rows.append(row)
    columns = KEY + ["primary_n", "reference_beta_native", "reference_q_all84"]
    for model in MODELS:
        columns += [model + "__" + field for field in ["status", "n", "interaction_beta_native", "interaction_ci_low_native", "interaction_ci_high_native", "q_all84", "same_direction", "retained"]]
    columns += ["all_six_estimable", "same_direction_all_six", "retained_all_six", "retained_sensitivity_count",
                "max_sensitivity_q_all84", "min_sensitivity_n", "max_sensitivity_n"]
    stability = pd.DataFrame(rows, columns=columns)
    summaries = []
    for model in MODELS:
        subset = results[results.model.eq(model) & results.family.eq("extension")]
        good = subset.status.eq("OK")
        sig = subset[good & subset.q_all84.lt(alpha)]
        sig_keys = set(sig[KEY].itertuples(index=False, name=None))
        summaries.append(dict(model=model, extension_planned=60, extension_fitted=int(good.sum()),
                              extension_q_all84_below_alpha=len(sig), reference_hits=len(selected),
                              reference_hits_retained=int(stability[model + "__retained"].sum()),
                              new_significant_outside_reference=len(sig_keys-selected_keys),
                              n_min=int(subset.n.min()), n_max=int(subset.n.max()),
                              limited_support_tests=int(subset.get("limited_joint_support", pd.Series(False, index=subset.index)).fillna(False).sum())))
    return stability, pd.DataFrame(summaries)
