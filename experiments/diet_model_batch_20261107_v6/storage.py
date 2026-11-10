"""Small, sequential, hash-checked checkpoints; all payloads stay inside TRE."""

import hashlib
import json
import pickle
import re
import uuid
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )
    with Path(path).open("xb") as stream:
        stream.write(payload + b"\n")


class Memo:
    """Resume completed units. Never reuse an incomplete or corrupted payload."""

    def __init__(self, root, log=lambda message: None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log, self.created = log, 0

    def get(self, key, compute=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in (".", ".."):
            raise ValueError("Invalid checkpoint key")
        folder = self.root / key
        if folder.is_symlink():
            raise ValueError("Checkpoint directory must not be a symlink")
        folder.mkdir(exist_ok=True)
        for receipt in sorted(folder.glob("done_*.json")):
            try:
                record = json.loads(receipt.read_text())
            except json.JSONDecodeError:
                self.log("INCOMPLETE_RECEIPT " + key)
                continue
            path = folder / record["file"]
            if (
                Path(record["file"]).name != record["file"]
                or path.is_symlink()
                or sha(path) != record["sha256"]
            ):
                raise ValueError("Corrupted checkpoint: " + key)
            return pickle.loads(path.read_bytes())
        if compute is None:
            raise FileNotFoundError(key)
        value = compute()
        token = uuid.uuid4().hex
        path = folder / (token + ".pkl")
        payload = pickle.dumps(value, protocol=4)
        with path.open("xb") as stream:
            stream.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if sha(path) != digest:
            raise ValueError("Checkpoint readback failed: " + key)
        write_json(
            folder / ("done_" + token + ".json"), dict(file=path.name, sha256=digest)
        )
        self.created += 1
        self.log("DONE " + key)
        return value
