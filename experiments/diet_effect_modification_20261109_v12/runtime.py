"""Native readiness gate, source identity and closed-file output management."""

import importlib.metadata
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from bridge import V11, adapter_sources, sha, write_json


def valid_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Invalid run ID")
    return value


def check_config(config):
    fixed = dict(exposure="fiber_density", outcome="D_Glu", unit="mg/dL", confirmation_outcomes_read=False,
                 expected_people=3343, outer_repeats=2, outer_folds=5, inner_folds=3,
                 outer_seed=20261104, primary_repeat=0, contrast=[10, 15],
                 minimum_effective_households_for_inference=30)
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Unexpected population, target or validation configuration")
    if config["modifier_components_per_site"] > config["main_components_per_site"]:
        raise ValueError("Modifier PCs must also occur in the matched main background")
    for key in ("main_penalties", "interaction_penalties"):
        if not config[key] or any(value <= 0 for value in config[key]) or len(set(config[key])) != len(config[key]):
            raise ValueError("Distinct positive penalties required")
    for key in ("score_multipliers", "prediction_bootstraps"):
        if config[key] < 199:
            raise ValueError("Insufficient native resampling draws")
    valid_id(config["v11_run_id"])


def environment(config):
    versions = {name: importlib.metadata.version(name) for name in config["dependencies"]}
    if versions != config["dependencies"]:
        raise ValueError("Dependency versions differ from the verified existing environment")
    versions["threadpoolctl"] = importlib.metadata.version("threadpoolctl")
    return versions


def readiness(config, prior_summary_reviewed=False, previous_root=None):
    """No participant inputs or native model checkpoints are opened by this gate."""
    if not prior_summary_reviewed:
        raise RuntimeError("WAIT: finish V11 and review the V6-V11 summary before using --prior-summary-reviewed")
    result = v11_completion(config, previous_root)
    result.update(v6_v11_summary_reviewed=True,
                  acknowledgement="User-supplied --prior-summary-reviewed; no claim of automatic review")
    return result


def v11_completion(config, previous_root=None):
    """Metadata-only prerequisite for the user-authorized background inspection."""
    folder = Path(previous_root) if previous_root is not None else V11
    run = folder / "runs" / config["v11_run_id"]
    if (run / "RUNNING.lock").exists():
        raise RuntimeError("WAIT: V11 still has a run lock")
    path = run / "summary.json"
    if not path.is_file():
        raise RuntimeError("WAIT: completed native V11 summary is missing")
    result = json.loads(path.read_text())
    if result.get("status") != "COMPLETE" or result.get("confirmation_outcomes_read") is not False:
        raise RuntimeError("WAIT: V11 completion/scope check failed")
    repeats = {row.get("repeat") for row in result.get("scores", [])}
    if repeats != {0, 1} or any(row.get("N") != 3343 for row in result.get("scores", [])):
        raise RuntimeError("WAIT: V11 summary does not show the expected repeated evaluation population")
    return dict(v11_summary=str(path), v11_summary_sha256=sha(path))


def identity(root, config, dataset, versions, gate, threads):
    sources = sorted(Path(root).glob("*.py")) + [Path(root) / "config.json", Path(root) / "requirements.txt", Path(root) / "review_context.json"]
    sources += adapter_sources() + dataset.paths + [Path(gate["v11_summary"])]
    sources = list(dict.fromkeys(sources))
    return dict(
        config=config, hashes={str(p): sha(p) for p in sources}, versions=versions,
        python=sys.version, threads=threads, prerequisite_review=gate,
        context_audit=dataset.extra["audit"], primary_repeat=0,
        historical_discovery_search=True, independent_confirmation=False,
        confirmation_outcomes_read=False, participant_data_exported=False,
    )


def audit(record):
    for path, digest in record["hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Registered source/input changed: " + path)


def register(run, record, resume=False):
    path = Path(run) / "registration.json"
    if resume:
        if not path.is_file() or json.loads(path.read_text()) != record:
            raise ValueError("Resume requires identical sources, inputs, settings and environment")
    else:
        Path(run).mkdir(parents=True, exist_ok=False)
        write_json(path, record)


@contextmanager
def run_lock(run):
    path = Path(run) / "RUNNING.lock"
    token = dict(pid=os.getpid(), token=uuid.uuid4().hex)
    try:
        write_json(path, token)
    except FileExistsError as error:
        raise RuntimeError("Run lock exists; inspect its PID before any recovery") from error
    try:
        yield
    except BaseException:
        # Leave a visible lock after failure/interruption, as well as completed units.
        raise
    else:
        if json.loads(path.read_text()) == token:
            path.unlink()


def publish_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Output must not be a symlink")
    if path.exists():
        if json.loads(path.read_text()) == value:
            return
        raise ValueError("Completed aggregate differs; keep the existing file")
    partial = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    write_json(partial, value)
    if json.loads(partial.read_text()) != value:
        raise ValueError("Aggregate readback failed")
    partial.rename(path)
