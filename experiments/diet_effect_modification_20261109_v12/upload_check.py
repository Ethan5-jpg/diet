"""Verify prepared source bytes, without opening native participant inputs."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "upload_manifest.json").read_text())
    expected_runtime = {str(p.relative_to(root)) for p in root.iterdir()
                        if p.is_file() and p.suffix in (".py", ".sh")}
    if not expected_runtime.issubset(set(manifest["files"])):
        raise ValueError("Unexpected runtime source outside the upload manifest")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid manifest path")
        path = root / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("Upload checksum mismatch: " + name)
    for name, expected in manifest["native_source_dependencies"].items():
        path = root.parent / name
        if digest(path) != expected:
            raise ValueError("Existing adapter source changed: " + name)
    return len(manifest["files"])


if __name__ == "__main__":
    print("UPLOAD_CHECK_OK files=%d" % verify(Path(__file__).resolve().parent))
