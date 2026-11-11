"""Sequential-write checkpoints and an exclusive worker lock."""

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import uuid


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path, data):
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
    with temp.open("xb") as stream:
        stream.write(data)
    # Check the closed object before exposing it; works on sequential-write mounts.
    if sha(temp) != hashlib.sha256(data).hexdigest():
        raise IOError("Sequential write checksum mismatch")
    os.replace(temp, path)


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, data)


def verify_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError("Missing required file: " + str(path))
    if sha(path) != expected:
        raise ValueError("SHA256 mismatch: " + str(path))


@contextmanager
def locked(run):
    lock = run / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), stream)
    try:
        yield
    finally:
        lock.unlink()


def recover_stale(run):
    lock = run / "RUNNING.lock"
    guard = run / "RECOVER.lock"
    with guard.open("x"):
        pass
    try:
        if not lock.exists():
            raise ValueError("No lock to recover")
        owner = read_json(lock)
        if owner["host"] != socket.gethostname() or not isinstance(owner["pid"], int) or owner["pid"] < 2:
            raise ValueError("Cannot verify an owner on a different host")
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            # Preserve evidence. An active worker with that PID is never reclaimed.
            lock.rename(run / ("STALE_LOCK_" + uuid.uuid4().hex + ".json"))
        else:
            raise ValueError("Worker PID is alive")
    finally:
        guard.unlink()


def register(run, payload, resume):
    path = run / "registration.json"
    if path.exists():
        if not resume or read_json(path) != payload:
            raise ValueError("Run registration differs; use a new run ID for changed code, data or settings")
    else:
        if any(run.glob("r?_f?.json")) or (run / "COMPLETE.json").exists():
            raise ValueError("Run has outputs but no registration")
        write_json(path, payload)
    return sha(path)


def unit_path(run, repeat, fold):
    return run / ("r%d_f%d.json" % (repeat, fold))


def save_unit(run, repeat, fold, reg, arrays, meta):
    import numpy as np

    path = unit_path(run, repeat, fold)
    if path.exists():
        raise FileExistsError("Refusing to overwrite a committed fold")
    npz = path.with_suffix(".npz")
    packed = io.BytesIO()
    np.savez_compressed(packed, **arrays)
    atomic_bytes(npz, packed.getvalue())
    write_json(path, dict(registration=reg, repeat=repeat, fold=fold,
                          arrays_sha=sha(npz), meta=meta))


def load_unit(run, repeat, fold, reg):
    import numpy as np

    path = unit_path(run, repeat, fold)
    receipt = read_json(path)
    if (receipt["registration"], receipt["repeat"], receipt["fold"]) != (reg, repeat, fold):
        raise ValueError("Fold registration mismatch")
    npz = path.with_suffix(".npz")
    payload = npz.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["arrays_sha"]:
        raise ValueError("Fold prediction file checksum mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as src:
        arrays = {key: src[key] for key in src.files}
    return arrays, receipt["meta"]


def publish(run, summary, schedule, reg):
    summary_file = run / "summary.json"
    write_json(summary_file, summary)
    write_json(run / "COMPLETE.json", dict(
        registration=reg, summary_sha=sha(summary_file),
        folds={unit_path(run, r, f).name: sha(unit_path(run, r, f)) for r, f in schedule},
    ))


def read_summary(run):
    marker = read_json(run / "COMPLETE.json")
    verify_file(run / "registration.json", marker["registration"])
    verify_file(run / "summary.json", marker["summary_sha"])
    reg = read_json(run / "registration.json")
    expected = {unit_path(run, r, f).name: (r, f) for r, f in reg["schedule"]}
    if len(expected) != len(reg["schedule"]) or set(marker["folds"]) != set(expected):
        raise ValueError("Completion receipt does not cover the registered folds")
    for name, digest in marker["folds"].items():
        verify_file(run / name, digest)
        receipt = read_json(run / name)
        if (receipt["registration"], receipt["repeat"], receipt["fold"]) != (
                marker["registration"], *expected[name]):
            raise ValueError("Fold registration mismatch")
        verify_file((run / name).with_suffix(".npz"), receipt["arrays_sha"])
    return read_json(run / "summary.json")
