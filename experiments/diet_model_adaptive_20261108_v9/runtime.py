"""Registration, strict resume identity and exclusive, retained-on-crash locks."""

import importlib
import importlib.metadata
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from bridge import V6, V8, sha, write_json


def _publish(path, payload, json_value=None):
    """Publish closed result files; retain partial predecessors for inspection."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Result file must not be a symlink")
    if path.exists():
        previous = path.read_bytes()
        if previous == payload:
            return
        if json_value is not None:
            try:
                stored = json.loads(previous)
            except (ValueError, UnicodeError):
                pass
            else:
                if stored == json_value:
                    return
                raise ValueError("Existing completed result differs; retained without overwrite")
        elif not payload.startswith(previous):
            raise ValueError("Existing summary differs; retained without overwrite")
    token = uuid.uuid4().hex
    temporary = path.with_name(path.name + ".partial_" + token)
    with temporary.open("xb") as stream:
        stream.write(payload)
    if path.exists():
        path.rename(path.with_name(path.name + ".incomplete_" + token))
    temporary.replace(path)


def publish_json(path, value):
    payload = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    _publish(path, payload, json_value=value)


def publish_text(path, value):
    _publish(path, value.encode("ascii"))


def valid_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Invalid run-id; use 1-64 letters, digits, hyphens or underscores")
    return value


def check_config(config):
    expected = dict(
        outcome="D_Glu",
        exposure="fiber_density",
        confirmation_outcomes_read=False,
        outer_seed=20261104,
        outer_repeats=2,
        outer_folds=5,
        inner_folds=3,
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected discovery target or household CV configuration")
    valid_id(config["reference_run"])
    valid_id(config["v7_reference_run"])
    valid_id(config["v8_reference_run"])
    from registry import names, pools

    if (
        len(names(config)) != len(set(names(config)))
        or any(not set(v) <= set(names(config)) or len(v) != len(set(v)) for v in pools(config).values())
        or any(config[k] < 1 for k in ("completion_iterations", "neural_seeds", "meta_tree_iterations"))
    ):
        raise ValueError("Invalid model names or blend members")


def environment(config):
    versions = {name: importlib.metadata.version(name) for name in config["dependencies"]}
    for name, version in config["dependencies"].items():
        if versions[name] != version:
            raise ValueError(
                "Version mismatch for %s: expected %s, found %s" % (name, version, versions[name])
            )
    importlib.import_module("lightgbm")
    importlib.import_module("catboost")
    for name in ("joblib", "threadpoolctl"):
        versions[name] = importlib.metadata.version(name)
    return versions


def identity(root, config, dataset, reference, versions, threads):
    sources = sorted(root.glob("*.py")) + [root / "config.json", root / "requirements.txt"]
    sources += [V6 / name for name in ("data.py", "numerics.py", "storage.py")]
    sources.append(V8 / "features.py")
    paths = list(dict.fromkeys(sources + dataset.paths + reference.paths))
    return dict(
        config=config,
        hashes={str(p): sha(p) for p in paths},
        versions=versions,
        python=sys.version,
        threads=threads,
        daily_extension=dataset.extra["audit"],
        prior_discovery_selection=True,
        server_data_exported=False,
    )


def register(run, current, resume):
    path = run / "registration.json"
    if resume:
        if not path.is_file() or json.loads(path.read_text()) != current:
            raise ValueError("Resume identity differs; keep old run and choose a new run-id")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(path, current)


@contextmanager
def run_lock(run):
    path = run / "RUNNING.lock"
    token = dict(pid=os.getpid(), token=uuid.uuid4().hex)
    try:
        write_json(path, token)
    except FileExistsError as error:
        raise RuntimeError("Run lock exists; inspect its PID before any manual recovery") from error
    try:
        yield
    finally:
        if path.is_file() and json.loads(path.read_text()) == token:
            path.unlink()


def audit(current):
    for path, digest in current["hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Input, source or reference changed during execution: " + path)
