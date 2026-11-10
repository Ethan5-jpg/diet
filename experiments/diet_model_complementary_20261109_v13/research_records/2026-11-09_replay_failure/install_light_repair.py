"""Update only the unregistered repair helper; retain strict normal resume checks."""

import ast
import hashlib
from pathlib import Path

OLD = "89af414429ade9b6aa38ec082a31e6051f875f3a20c7649848270c0da7cd17f1"
NEW = "9f0a7388f62ccca98e8b9cf21bcf000a5accae4ca6ad3b5acd73dff0b5373be5"
CANONICAL = "ea789975ffb0822f7ebd63ab10acbcf62f9099f37c7dc1267afbc31565f6d243"
SOURCE_HELPERS = """

def verify_upload(root):
    count = verify(root)
    print("UPLOAD_CHECK_OK files=%d" % count, flush=True)
    return count


def source_audit(current):
    paths = [(path, expected) for path, expected in current["hashes"].items()
             if Path(path).suffix == ".py" or Path(path).name in ("config.json", "requirements.txt")]
    print("CODE_CHECK files=%d; full data/cache hashes remain enforced by --resume" % len(paths), flush=True)
    for i, (path, expected) in enumerate(paths, 1):
        print("CODE %d/%d %s" % (i, len(paths), path), flush=True)
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError("Registered code/config changed: " + path)
"""


def build(old):
    new = old.replace("from runtime import audit, run_lock", "from runtime import run_lock")
    new = new.replace("verify(root)", "verify_upload(root)")
    new = new.replace("audit(json.loads(current[1]))", "source_audit(json.loads(current[1]))")
    new = new.replace("audit(new_reg)", "source_audit(new_reg)")
    new = new.replace("\n\ndef apply(root):", SOURCE_HELPERS + "\n\ndef apply(root):")
    new = new.replace(
        "verify upload and registered sources/inputs; no model fitting",
        "code-only repair; data/cache integrity is checked during strict --resume",
    )
    new = new.replace(
        '        keys = [key for key in old_reg["hashes"] if Path(key).resolve() == engine]',
        """        keys = [str(engine)] if str(engine) in old_reg["hashes"] else [
            key for key in old_reg["hashes"]
            if Path(key).name == "engine.py" and Path(key).resolve() == engine
        ]""",
    )
    new = new.replace(
        "            checkpoint_files_changed=False,",
        """            checkpoint_files_changed=False,
            data_hashes_retained=True,
            full_input_check="unchanged strict resume identity before model/cache use",
            repair_check_scope="registered Python/config sources; no bulk data/cache scan",""",
    )
    return new


def install(root):
    root = Path(root).resolve()
    if (root / "runs/complementary_v13_20261109/RUNNING.lock").exists():
        raise RuntimeError("Run lock remains; inspect the old repair process before retrying")
    path = root / "research_records/apply_replay_fix.py"
    print("UPDATING_REPAIR_HELPER " + str(path), flush=True)
    old = path.read_bytes()
    sha = hashlib.sha256(old).hexdigest()
    if sha in (NEW, CANONICAL):
        print("LIGHT_REPAIR_ALREADY_INSTALLED", flush=True)
        return
    if sha != OLD:
        raise ValueError("Unexpected repair helper version; original file retained")
    new = build(old.decode("ascii")).encode("ascii")
    if hashlib.sha256(new).hexdigest() != NEW:
        raise ValueError("Generated helper checksum differs; original file retained")
    ast.parse(new)
    backup = path.with_name("apply_replay_fix.full_audit_backup.py")
    if backup.exists():
        if backup.read_bytes() != old:
            raise ValueError("Existing helper backup differs")
    else:
        with backup.open("xb") as stream:
            stream.write(old)
    temporary = path.with_name("apply_replay_fix.light_partial.py")
    temporary.write_bytes(new)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != NEW:
        raise ValueError("Helper write readback failed")
    temporary.replace(path)
    print("LIGHT_REPAIR_INSTALLED; source checks have per-file progress", flush=True)
    print("FULL_DATA_HASH_CHECK remains in the unchanged normal --resume entry", flush=True)


if __name__ == "__main__":
    install(Path.cwd())
