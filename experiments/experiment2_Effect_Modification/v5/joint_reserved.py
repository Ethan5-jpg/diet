"""Secondary v5 joint linear association on the original scored reserved cohort."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pickle
import sys

from run_discovery import DEPENDENCIES, ROOT, emit, verify_release


ASSOCIATION_SHA256 = "bfa5488e22922ad23c6c0183c4f666ae02081ab2e1eeb215bebe5dafb2411f62"
DISCOVERY_QUARTILES = {
    "gut_shannon": (3.3375784804500706, 3.986831892141875),
    "fermentation": (3.939667155027373, 5.5595733238663385),
}
CODE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
ASSOCIATION_PATH = ROOT.parent / "v4/association.py"


def association_module():
    """Load the verified v4 module explicitly because v3 has the same basename."""
    if sha256(ASSOCIATION_PATH.read_bytes()).hexdigest() != ASSOCIATION_SHA256:
        raise ValueError("JOINT_ASSOCIATION_DEPENDENCY_CHANGED")
    spec = spec_from_file_location("hpp_v5_joint_v4_association", ASSOCIATION_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def support(fiber, mask):
    import numpy as np

    values = fiber[mask]
    if not len(values):
        return {"people": 0, "q05_q95": None, "near_10": 0, "near_15": 0, "covers_10_to_15": False}
    lo, hi = np.quantile(values, (0.05, 0.95))
    return {"people": len(values), "q05_q95": [float(lo), float(hi)],
            "near_10": int(np.count_nonzero(np.abs(values - 10) <= 1)),
            "near_15": int(np.count_nonzero(np.abs(values - 15) <= 1)),
            "covers_10_to_15": bool(lo <= 10 and hi >= 15)}


def joint_association(y, background, fiber, markers, groups, quartiles):
    import numpy as np

    engine = association_module()
    # FWL and CR1 reuse the archived v4 fit. Its reported contrasts use the
    # held-sample IQR, so rescale the linear contrast to the fixed discovery IQR.
    part = engine.joint_fold(np.arange(len(y)), y, background, fiber, markers)
    result = engine.summarize_joint([part], groups)
    for j, name in enumerate(engine.MARKERS):
        entry = result["markers"][name]
        q25, q75 = quartiles[name]
        values = markers[:, j]
        lo, hi = support(fiber, values <= q25), support(fiber, values >= q75)
        reserved_iqr = entry.pop("q75") - entry.pop("q25")
        entry["discovery_q25_q75"] = [q25, q75]
        entry["support"] = {
            "low": lo, "high": hi,
            "both_tails_cover_10_to_15": lo["covers_10_to_15"] and hi["covers_10_to_15"],
            "both_reference_markers_in_reserved_range": bool(values.min() <= q25 < q75 <= values.max()),
        }
        if entry["status"] != "ok":
            continue
        ratio = (q75 - q25) / reserved_iqr
        for key in ("estimate_mg_dl", "se_mg_dl"):
            entry[key] *= ratio
        for key in ("ci95_mg_dl", "family95_mg_dl"):
            entry[key] = [bound * ratio for bound in entry[key]]
    result.update(scope="secondary association on the previously scored HPP reserved households",
                  outcome="D_Glu with frozen discovery preprocessing", interval="household CR1; Bonferroni two contrasts",
                  feature_source="discovery-fitted V5 transforms and V4 association background")
    return result


def predictor_background(state, discovery, reserved, rows, fit_rows):
    import numpy as np

    from fermentation_score import build_score
    from frozen_model import _context, _diet_basis, _replay_clr, _replay_host
    from moderation_models import MARKERS, SITES, ecology, finite_inference_background
    from validate_reserved import check

    table = reserved.index.iloc[rows]
    fiber = table.fiber_density.to_numpy(float)
    dose = _diet_basis(fiber, state.dose)
    host = _replay_host(table, dose, state.host)
    ctx = (_context(discovery.extra, fit_rows, state.alcohol_levels),
           _context(reserved.extra, rows, state.alcohol_levels))
    pc, raw_eco = [], []
    for site in SITES:
        raw = reserved.raw[site][rows]
        z, supported = _replay_clr(raw, state.clr[site])
        check(supported.all(), "SCORED_CLR_SUPPORT_CHANGED")
        pc.append((None, state.pc_scale[site].apply(state.pca[site].transform(z))))
        raw_eco.append(ecology(raw))
    raw_eco = np.column_stack(raw_eco)
    eco = state.eco_scale.apply(raw_eco)
    background = finite_inference_background(host, ctx, pc, (None, eco))
    panel, matches, _ = build_score(reserved.raw["gut"][rows], reserved.taxa["gut"])
    check(tuple(matches.items()) == state.panel_taxa, "SCORED_PANEL_MAPPING_CHANGED")
    return background, fiber, np.column_stack((raw_eco[:, MARKERS.index("gut_shannon")], panel))


def close_numeric(actual, expected):
    import numpy as np

    return np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=False)


def association_inputs(run, release):
    import numpy as np

    from frozen_model import predict_model, training_rows, transform_target
    from person_inputs import digest, load_discovery, read_json
    from reserved_inputs import load_reserved
    from reserved_outcome import load_outcome, summarize
    from validate_reserved import (
        COMPLETE, FROZEN, KEY, MODEL, PREDICTIONS, SUMMARY, check, cohort_arrays, discover_training,
        load_frozen, verify_hashes,
    )

    receipt, index, arrays = load_frozen(run, release)
    complete_path = run / COMPLETE
    done_hash = digest(complete_path)
    done = read_json(complete_path)
    proof = {complete_path: done_hash, run / FROZEN: done["freeze_sha256"], run / SUMMARY: done["summary_sha256"],
             run / MODEL: receipt["model_sha256"], run / PREDICTIONS: receipt["predictions_sha256"]}
    verify_hashes(proof | done["outcome_sources"], "PRIOR_SCORE_RECEIPT_OR_SOURCE_CHANGED")
    archived = read_json(run / SUMMARY)
    model = (run / MODEL).read_bytes()
    check(sha256(model).hexdigest() == receipt["model_sha256"], "FROZEN_MODEL_CHANGED")
    state = pickle.loads(model)
    discovery = load_discovery(ROOT)
    check(discover_training(ROOT, discovery) == receipt["training"], "DISCOVERY_TRAINING_CHANGED")
    fit_rows = training_rows(discovery)
    check(len(fit_rows) == state.training_rows == receipt["training"]["training_eligible"]
          and np.unique(discovery.groups[fit_rows]).size == state.training_households == receipt["training_households"]
          and state.target == receipt["target"], "FROZEN_TRAINING_STATE_CHANGED")

    reserved = load_reserved(ROOT, discovery)
    inputs = {str(path): value for path, value in (discovery.source_hashes | reserved.hashes).items()}
    check(inputs == receipt["inputs"], "FROZEN_ASSOCIATION_INPUTS_CHANGED")
    rows, baseline, correction, contrast = predict_model(state, reserved)
    keys = cohort_arrays(reserved.index.iloc[rows])
    check(all(np.array_equal(keys[name], arrays[name]) for name in KEY)
          and close_numeric(baseline, arrays["baseline"])
          and close_numeric(baseline + correction, arrays["full"])
          and close_numeric(contrast, arrays["contrast"]), "RESERVED_PREDICTION_REPLAY_CHANGED")

    raw, outcome_paths = load_outcome(ROOT, index)
    check({str(path): value for path, value in outcome_paths.items()} == done["outcome_sources"],
          "RESERVED_OUTCOME_SOURCE_SET_CHANGED")
    target = transform_target(receipt["target"], raw)
    keep = np.isfinite(target)
    y, groups = target[keep], arrays["split_unit_id"][keep]
    check(int(np.isfinite(raw).sum()) == archived["with_qualified_raw_outcome"]
          and int(np.count_nonzero(np.isfinite(raw) & ~keep)) == archived["outcome_removed_by_discovery_limits"],
          "RESERVED_OUTCOME_ELIGIBILITY_CHANGED")
    replay = summarize(y, arrays["baseline"][keep], arrays["full"][keep], groups, arrays["contrast"][keep])
    check(all(replay[key] == archived[key] for key in ("people", "households", "bootstrap_draws"))
          and all(close_numeric(replay[key], archived[key])
                  for key in ("base_r2", "full_r2", "delta_r2", "delta_r2_ci95", "contrast_sd_10_to_15_mg_dl")),
          "RESERVED_SCORE_REPLAY_CHANGED")
    background, fiber, markers = predictor_background(state, discovery, reserved, rows[keep], fit_rows)
    return (y, background, fiber, markers, groups), proof | inputs | done["outcome_sources"]


def selftest():
    import numpy as np
    from scipy.stats import t

    from validate_reserved import check

    association_module().joint_selftest()
    rng = np.random.default_rng(52024)
    groups = np.repeat(np.arange(80), 3)
    fiber = rng.uniform(7, 19, len(groups))
    markers = rng.normal(size=(len(groups), 2))
    markers[:, 1] += 0.5 * markers[:, 0]
    background = np.column_stack((np.ones(len(groups)), rng.normal(size=len(groups))))
    dose = (fiber - 10) / 5
    beta = np.array([0.5, -0.8])
    design = np.column_stack((background, dose, markers, markers ** 2, dose[:, None] * markers))
    y = design[:, :-2] @ np.arange(design.shape[1] - 2) + design[:, -2:] @ beta + rng.normal(0, .1, len(groups))
    quartiles = {"gut_shannon": (-1.5, 1.2), "fermentation": (-1.1, 1.6)}
    got = joint_association(y, background, fiber, markers, groups, quartiles)
    full = np.linalg.lstsq(design, y, rcond=None)[0]
    scores = np.zeros((80, design.shape[1]))
    np.add.at(scores, groups, design * (y - design @ full)[:, None])
    covariance_factor = scores @ np.linalg.inv(design.T @ design)
    cr1 = 80 / 79 * (len(y) - 1) / (len(y) - design.shape[1])
    for j, name in enumerate(quartiles):
        lo, hi = quartiles[name]
        point = full[-2 + j] * (hi - lo)
        margin = t.ppf(.9875, 79) * np.sqrt(cr1) * np.linalg.norm(covariance_factor[:, -2 + j]) * (hi - lo)
        row = got["markers"][name]
        np.testing.assert_allclose(row["estimate_mg_dl"], point, atol=1e-10)
        np.testing.assert_allclose(row["family95_mg_dl"], [point - margin, point + margin], atol=1e-10)
        check(row["support"]["low"]["people"] == (markers[:, j] <= lo).sum(), "TEST_FIXED_TAIL")
    check(support(fiber, fiber < 0)["q05_q95"] is None, "TEST_EMPTY_TAIL")
    duplicate = np.column_stack((markers[:, 0], 2 * markers[:, 0]))
    check(joint_association(y, background, fiber, duplicate, groups, quartiles)["status"] == "unavailable",
          "TEST_RANK_DEFICIENCY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    release = verify_release()
    sys.path[1:1] = [str(ROOT.parent / name) for name in DEPENDENCIES]
    from threadpoolctl import threadpool_limits
    from person_inputs import digest
    from validate_reserved import RUN, check, verify_hashes

    with threadpool_limits(limits=2):
        if args.preflight:
            selftest()
            emit("JOINT_PREFLIGHT_DONE", scope="generated data")
            return
        run = ROOT / RUN
        check(not (run / "RUNNING.lock").exists(), "ORIGINAL_V5_RUN_ACTIVE")
        inputs, proofs = association_inputs(run, release)
        result = joint_association(*inputs, DISCOVERY_QUARTILES)
        verify_hashes(proofs, "JOINT_SOURCE_CHANGED")
        check(digest(ASSOCIATION_PATH) == ASSOCIATION_SHA256, "JOINT_ASSOCIATION_DEPENDENCY_CHANGED")
        check(digest(ROOT / "joint_reserved.py") == CODE_SHA256, "JOINT_CODE_CHANGED")
        emit("JOINT_RESERVED", **result)
        emit("JOINT_COMPLETE", people=result["n"], households=result["households"], status=result["status"])


if __name__ == "__main__":
    try:
        main()
    except (ArithmeticError, AssertionError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        detail = str(exc)
        if not (detail.isascii() and detail.replace("_", "").isalnum() and detail.isupper()):
            detail = type(exc).__name__
        raise SystemExit("EXP2_V5_JOINT_STOP " + detail) from None
