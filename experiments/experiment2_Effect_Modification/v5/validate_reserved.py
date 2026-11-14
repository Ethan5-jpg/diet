"""Fit on discovery, freeze reserved predictions, and score the paired comparison."""

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
from uuid import uuid4

import numpy as np
import pandas as pd

from frozen_model import ROUTE, fit_model, predict_model, training_rows, transform_target
from person_inputs import digest, imported, load_discovery, read_json, registered_sources
from reserved_inputs import load_reserved
from reserved_outcome import summarize

CANDIDATE = ROUTE
KEY = ("participant_id", "cohort", "research_stage", "split_unit_id")
FORMAT = "exp2_v5_reserved_1"
RUN = Path("runs/TRE_ONLY/v5_reserved")
FROZEN = "FREEZE.json"
COMPLETE = "COMPLETE.json"
MODEL = "discovery_model.pkl"
PREDICTIONS = "reserved_predictions.npz"
SUMMARY = "reserved_summary.json"
PREDICTORS = ("baseline", "full", "contrast")


def check(condition, code):
    if not condition:
        raise ValueError(code)


def verify_hashes(paths, error):
    for path, expected in paths.items():
        path = Path(path)
        check(path.is_file() and digest(path) == expected, error)


def json_bytes(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()


def write_new(path, payload):
    """Publish only closed, verified sequential writes on the TRE filesystem."""
    check(not path.exists(), "OUTPUT_ALREADY_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
        check(digest(temporary) == sha256(payload).hexdigest(), "OUTPUT_READBACK_MISMATCH")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = run / "RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": os.uname().nodename}, stream)
    except FileExistsError:
        raise ValueError("RUN_ALREADY_LOCKED") from None
    try:
        yield
    finally:
        lock.unlink()


def cohort_arrays(index):
    participant = index["participant_id"].to_numpy()
    check(np.issubdtype(participant.dtype, np.integer) and (participant >= 0).all(),
          "INVALID_RESERVED_PARTICIPANT_IDS")
    ids = participant.astype(np.int64)
    check(np.array_equal(ids, participant), "INVALID_RESERVED_PARTICIPANT_IDS")
    result = {"participant_id": ids}
    for name in KEY[1:]:
        series = index[name]
        check(series.notna().all(), "MISSING_RESERVED_IDENTITY")
        value = series.astype(str).to_numpy(dtype=str)
        check(np.all(np.char.str_len(value) > 0), "MISSING_RESERVED_IDENTITY")
        result[name] = value
    check(len(ids) == len(np.unique(ids)) and np.all(result["cohort"] == "10k")
          and np.all(result["research_stage"] == "00_00_visit"), "INVALID_RESERVED_IDENTITY")
    return result


def validate_predictions(arrays):
    check(set(arrays) == set((*KEY, *PREDICTORS)),
          "INVALID_FROZEN_PREDICTIONS")
    n = len(arrays["participant_id"])
    check(n > 0 and all(a.shape == (n,) for a in arrays.values()), "INVALID_FROZEN_PREDICTIONS")
    check(all(np.isfinite(arrays[name]).all() for name in PREDICTORS),
          "NONFINITE_FROZEN_PREDICTIONS")
    index = pd.DataFrame({name: arrays[name] for name in KEY})
    cohort_arrays(index)
    return index


def load_frozen(run, release):
    receipt = read_json(run / FROZEN)
    check(receipt["format"] == FORMAT and receipt["candidate"] == CANDIDATE
          and receipt["release"] == release and receipt["reserved_outcomes_read"] is False,
          "FREEZE_RECEIPT_MISMATCH")
    verify_hashes({run / MODEL: receipt["model_sha256"], run / PREDICTIONS: receipt["predictions_sha256"]},
                  "FROZEN_ARTIFACT_MISMATCH")
    verify_hashes(receipt["inputs"], "FROZEN_INPUT_CHANGED")
    raw = (run / PREDICTIONS).read_bytes()
    check(sha256(raw).hexdigest() == receipt["predictions_sha256"], "FROZEN_ARTIFACT_MISMATCH")
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    index = validate_predictions(arrays)
    check(len(index) == receipt["prediction_people"]
          and index.split_unit_id.nunique() == receipt["prediction_households"], "FREEZE_COHORT_MISMATCH")
    return receipt, index, arrays


def discover_training(root, source):
    old = np.isfinite(source.reference["y"][0])
    rows = training_rows(source)
    current = np.zeros(len(old), dtype=bool)
    current[rows] = True
    if not np.array_equal(current, old):
        package = root.parent.parent
        _, expected_paths, helpers = registered_sources(package)
        context = imported("hpp_v5_context_for_full_training", helpers[2])
        extra, paths = context.read_context(package, source.index, current)
        check(set(paths) == set(expected_paths) and extra["audit"]["daily_outcome_columns_read"] == [],
              "DISCOVERY_CONTEXT_CHANGED")
        source.extra = extra
    return {"training_eligible": int(current.sum()),
            "new_vs_r0": int(np.count_nonzero(current & ~old)),
            "lost_vs_r0": int(np.count_nonzero(old & ~current))}


def freeze(root, run, release, emit):
    if (run / FROZEN).exists():
        receipt, _, _ = load_frozen(run, release)
        emit("FREEZE_COMPLETE", reused=True, **freeze_summary(receipt))
        return

    source = load_discovery(root)
    training = discover_training(root, source)
    reserved = load_reserved(root, source)
    all_ids = cohort_arrays(reserved.index)
    check(not np.intersect1d(all_ids["participant_id"], source.index.participant_id).size
          and not np.intersect1d(all_ids["split_unit_id"], source.groups).size,
          "DISCOVERY_RESERVED_OVERLAP")

    state = fit_model(source)
    check(state.training_rows == training["training_eligible"], "TRAINING_ELIGIBILITY_CHANGED")
    rows, baseline, correction, contrast = predict_model(state, reserved)
    check(np.issubdtype(rows.dtype, np.integer) and rows.ndim == 1 and np.all(np.diff(rows) > 0)
          and np.all((rows >= 0) & (rows < len(reserved.index))), "INVALID_PREDICTION_ROW_ORDER")
    arrays = {name: value[rows] for name, value in all_ids.items()}
    arrays.update(baseline=baseline, full=baseline + correction, contrast=contrast)
    validate_predictions(arrays)
    model = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    replay = predict_model(pickle.loads(model), reserved)
    check(all(np.array_equal(a, b) for a, b in zip(replay, (rows, baseline, correction, contrast), strict=True)),
          "SERIALIZED_MODEL_REPLAY_MISMATCH")
    packed = BytesIO()
    np.savez_compressed(packed, **arrays)
    check(all(source.source_hashes.get(path, value) == value for path, value in reserved.hashes.items()),
          "FREEZE_SOURCE_HASH_DISAGREEMENT")
    inputs = {str(path): value for path, value in (source.source_hashes | reserved.hashes).items()}
    verify_hashes(inputs, "FREEZE_INPUT_CHANGED")
    receipt = {
        "format": FORMAT, "candidate": CANDIDATE, "release": release, "inputs": inputs,
        "model_sha256": sha256(model).hexdigest(), "predictions_sha256": sha256(packed.getvalue()).hexdigest(),
        "reserved_outcomes_read": False, "target": state.target, "training": training,
        "training_households": state.training_households, "reserved_roster_people": len(reserved.index),
        "prediction_people": len(rows), "prediction_households": int(np.unique(arrays["split_unit_id"]).size),
        "predictor_audit": reserved.audit,
    }
    receipt_payload = json_bytes(receipt)
    # A failed publish can leave uncommitted artifacts. They have no scoring receipt.
    for name in (MODEL, PREDICTIONS):
        (run / name).unlink(missing_ok=True)
    write_new(run / MODEL, model)
    write_new(run / PREDICTIONS, packed.getvalue())
    write_new(run / FROZEN, receipt_payload)
    load_frozen(run, release)
    emit("FREEZE_COMPLETE", reused=False, **freeze_summary(receipt))


def freeze_summary(receipt):
    keys = ("candidate", "training", "training_households", "reserved_roster_people",
            "prediction_people", "prediction_households", "reserved_outcomes_read")
    return {name: receipt[name] for name in keys}


def score(root, run, release, emit):
    check((run / FROZEN).is_file(), "FREEZE_REQUIRED_BEFORE_OUTCOME")
    receipt, index, arrays = load_frozen(run, release)
    frozen_hash = digest(run / FROZEN)
    if (run / COMPLETE).exists():
        done = read_json(run / COMPLETE)
        check(done["freeze_sha256"] == frozen_hash, "SCORE_RECEIPT_MISMATCH")
        verify_hashes({run / SUMMARY: done["summary_sha256"]}, "SCORE_SUMMARY_MISMATCH")
        verify_hashes(done["outcome_sources"], "SCORED_OUTCOME_SOURCE_CHANGED")
        emit("RESERVED_RESULT", **read_json(run / SUMMARY))
        emit("RESERVED_COMPLETE", reused=True)
        return

    from reserved_outcome import load_outcome

    raw, outcome_sources = load_outcome(root, index)
    y = transform_target(receipt["target"], raw)
    check(y.shape == raw.shape == (len(index),), "INVALID_RESERVED_OUTCOME_ALIGNMENT")
    keep = np.isfinite(y)
    result = summarize(y[keep], arrays["baseline"][keep], arrays["full"][keep],
                       arrays["split_unit_id"][keep], arrays["contrast"][keep])
    result.update(candidate=CANDIDATE, prediction_people=len(index),
                  delta_r2_percentage_points=100 * result["delta_r2"],
                  with_qualified_raw_outcome=int(np.isfinite(raw).sum()),
                  outcome_removed_by_discovery_limits=int(np.count_nonzero(np.isfinite(raw) & ~keep)),
                  evaluated_outcome="D_Glu with frozen discovery preprocessing", outcome_unit="mg/dL",
                  contrast_fiber_g_per_1000_kcal=[10.0, 15.0],
                  scope="HPP reserved households", reserve_outcomes_used_for_training=False)
    hashes = {str(path): value for path, value in outcome_sources.items()}
    verify_hashes(hashes, "OUTCOME_SOURCE_CHANGED")
    check(digest(run / FROZEN) == frozen_hash, "FREEZE_RECEIPT_CHANGED")
    load_frozen(run, release)
    payload = json_bytes(result)
    (run / SUMMARY).unlink(missing_ok=True)
    write_new(run / SUMMARY, payload)
    write_new(run / COMPLETE, json_bytes({
        "freeze_sha256": frozen_hash, "summary_sha256": sha256(payload).hexdigest(), "outcome_sources": hashes,
    }))
    emit("RESERVED_RESULT", **result)
    emit("RESERVED_COMPLETE", reused=False)


def run_phase(root, release, emit, *, freeze_only):
    run = root / RUN
    if not freeze_only:
        check((run / FROZEN).is_file(), "FREEZE_REQUIRED_BEFORE_OUTCOME")
    with run_lock(run):
        phase = freeze if freeze_only else score
        phase(root, run, release, emit)


def validation_selftest():
    """Check paired metrics, strict identity and the before-outcome gate."""
    from tempfile import TemporaryDirectory
    from frozen_model import model_selftest

    model_selftest()

    y = np.arange(20, dtype=float)
    base = y + 1
    full = y + .5
    groups = np.repeat(np.arange(10), 2)
    result = summarize(y, base, full, groups, np.zeros(len(y)))
    sst = np.sum((y - y.mean()) ** 2)
    np.testing.assert_allclose(
        [result["base_r2"], result["full_r2"], result["delta_r2"]],
        [1 - 20 / sst, 1 - 5 / sst, 15 / sst],
    )
    check(result["people"] == 20 and result["households"] == 10, "TEST_PAIRED_COUNTS")
    with TemporaryDirectory(prefix="exp2_v5_preflight_") as path:
        try:
            run_phase(Path(path), {}, None, freeze_only=False)
        except ValueError as exc:
            check(str(exc) == "FREEZE_REQUIRED_BEFORE_OUTCOME", "TEST_SCORING_GATE")
        else:
            raise AssertionError("Scoring ran without a freeze receipt")
        check(not (Path(path) / RUN).exists(), "TEST_PRE_FREEZE_WRITE")
