"""Standard-library-only upload checksum verification; no data access."""

import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "upload_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Invalid upload manifest path")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Upload checksum mismatch: " + name)
    return len(manifest["sha256"])


if __name__ == "__main__":
    print("UPLOAD_CHECK_OK files=%d" % verify(Path(__file__).resolve().parent))
