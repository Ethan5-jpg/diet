"""Prespecified grid. No outcome-based model selection or silent redefinition."""
import numpy as np
import pandas as pd

from .common import sample_hash
from .inputs import covariates
from .models import ModelUnavailable, apply_fdr, build_reference, fit_interaction

SENSITIVITIES = ["strict", "protocol_same_sample", "protocol_adjusted", "quadratic",
                 "covariate_interactions", "common_focus_sample"]
COEFFICIENT_COLUMNS = (["interaction_" + key for key in ["beta", "se", "ci_low", "ci_high", "p"]]
                       + [prefix + key + "_native" for prefix in ["interaction_", "double_iqr_"]
                          for key in ["beta", "se", "ci_low", "ci_high"]])
SLOPE_COLUMNS = ["exposure", "taxon", "species", "outcome", "outcome_field", "unit", "family", "model",
                 "microbial_quantile", "microbial_original_z", "beta_native", "ci_low_native", "ci_high_native",
                 "p_pointwise", "slope_note"]


def complete_sample(master, exposure, taxon, outcome, config, model):
    cols = [exposure, taxon, outcome] + covariates(exposure)
    protocol = model in ["protocol_same_sample", "protocol_adjusted"]
    if protocol:
        cols += ["antibiotic_use", "ppi_use"]
    if model == "common_focus_sample":
        # Same people for both focus diets, all four taxa and all three endpoints.
        cols += config["focus_exposures"] + [x["id"] for x in config["taxa"]] + [x["id"] for x in config["outcomes"]]
        for e in config["focus_exposures"]:
            cols += covariates(e)
    cols = list(dict.fromkeys(cols))
    eligible = master["strict_eligible" if model == "strict" else "primary_eligible"]
    present = eligible & master.has_microbiome_record
    mask = present & master[cols].notna().all(axis=1)
    frame = master.loc[mask].copy()
    flow = dict(eligible_n=int(eligible.sum()), plus_microbiome_record_n=int(present.sum()),
                complete_case_n=len(frame), missing_excluded_n=int(present.sum()) - len(frame),
                unknown_exclusion_in_model_n=int(frame.exclude_diabetes_or_a10.isna().sum()),
                sample_sha256=sample_hash(frame.index), required_fields=";".join(cols))
    return frame, flow


def run_models(master, metadata, config, check_only=False):
    rows, slope_rows, flows = [], [], []
    exposures = config["focus_exposures"] + config["extension_exposures"]
    n_hypotheses = len(exposures) * len(config["taxa"]) * len(config["outcomes"])
    number = 0
    for exposure in exposures:
        for taxon in config["taxa"]:
            for outcome in config["outcomes"]:
                number += 1
                identity = dict(exposure=exposure, taxon=taxon["id"], species=taxon["species"],
                                outcome=outcome["id"], outcome_field=outcome["field"], unit=outcome["unit"],
                                family="focus" if exposure in config["focus_exposures"] else "extension")
                models = ["primary"] + (SENSITIVITIES if identity["family"] == "focus" else [])
                primary, unused_flow = complete_sample(master, exposure, taxon["id"], outcome["id"], config, "primary")
                ref = build_reference(primary, exposure, taxon["id"]) if len(primary) >= 2 else None
                for model in models:
                    frame, flow = complete_sample(master, exposure, taxon["id"], outcome["id"], config, model)
                    flow.update(identity, model=model)
                    flows.append(flow)
                    record = dict(identity, model=model, status="NOT_ESTIMABLE", reason="", n=len(frame), interaction_p=np.nan,
                                  sample_sha256=flow["sample_sha256"])
                    record.update({col: np.nan for col in COEFFICIENT_COLUMNS})
                    if model.startswith("protocol") and not metadata["summary"]["protocol_available"]:
                        record.update(status="SKIPPED", reason="Protocol source unavailable/disabled")
                    elif check_only:
                        record.update(status="CHECK_ONLY", reason="Inputs checked; no model fitted")
                    else:
                        try:
                            if ref is None:
                                raise ModelUnavailable("No primary reference sample")
                            cols = covariates(exposure, protocol=model == "protocol_adjusted")
                            variant = model if model in ["quadratic", "covariate_interactions"] else "linear"
                            values, slopes = fit_interaction(frame, exposure, taxon["id"], outcome["id"], cols,
                                                             ref, variant, config, metadata["outcome_scales"][outcome["id"]]["sd"])
                            record.update(values, status="OK")
                            for row in slopes:
                                slope_rows.append(dict(identity, model=model, **row))
                        except ModelUnavailable as error:
                            record["reason"] = str(error)
                    rows.append(record)
                if not check_only and (number % 12 == 0 or number == n_hypotheses):
                    print("ANALYSIS {}/{} hypotheses".format(number, n_hypotheses), flush=True)
    results = pd.DataFrame(rows)
    results, families = apply_fdr(results)
    # Named models must use identical membership where that is their purpose.
    key = ["exposure", "taxon", "outcome"]
    hashes = results.pivot(index=key, columns="model", values="sample_sha256")
    for first, second in [("primary", "quadratic"), ("primary", "covariate_interactions"),
                          ("protocol_same_sample", "protocol_adjusted")]:
        if first in hashes and second in hashes:
            common = hashes[[first, second]].dropna()
            if not common[first].eq(common[second]).all():
                raise RuntimeError("Internal sample inconsistency: {} vs {}".format(first, second))
    return results, pd.DataFrame(slope_rows, columns=SLOPE_COLUMNS), pd.DataFrame(flows), families
