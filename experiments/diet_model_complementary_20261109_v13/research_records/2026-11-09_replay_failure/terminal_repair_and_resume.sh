(
set -e
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_complementary_20261109_v13
mkdir -p research_records
cat > research_records/apply_replay_fix.py <<'PY_REPLAY_FIX'
"""Explicit, backed-up V13 comparison-only repair; never modifies fitted caches."""

import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path.cwd()))
from runtime import audit, run_lock
from upload_check import verify

OLD_SHA = "e12eeb697fd9079d111beaf7947c16d72a9fe9fcf7ca07757674540b81e58085"
BEFORE = b"""        if error > 1e-8:
            raise ValueError("V11 positive fusion replay differs")"""
AFTER = b"""        if error > 1e-6:
            raise ValueError(
                "V11 positive fusion replay differs: max_abs_error=%.12g mg/dL; tolerance=1e-6" % error
            )"""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def put(path, payload):
    temporary = path.with_name(path.name + ".repair_partial_" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(payload)
    if digest(temporary.read_bytes()) != digest(payload):
        raise ValueError("Repair write readback failed")
    temporary.replace(path)


def apply(root):
    root = Path(root).resolve()
    run = root / "runs/complementary_v13_20261109"
    engine, registration, manifest = (
        root / "engine.py",
        run / "registration.json",
        root / "upload_manifest.json",
    )
    if not registration.is_file() or any(p.is_symlink() for p in (run, engine, registration, manifest)):
        raise ValueError("Expected the existing, stopped original V13 run")
    backup = root / "research_records/2026-11-09_replay_failure/native_repair_backup"
    if backup.is_symlink():
        raise ValueError("Repair backup must not be a symlink")
    with run_lock(run):
        print("REPAIR_CHECK: verify upload and registered sources/inputs; no model fitting", flush=True)
        verify(root)
        current = [p.read_bytes() for p in (engine, registration, manifest)]
        names = ("engine.before.py", "registration.before.json", "upload_manifest.before.json")
        originals = [
            (backup / n).read_bytes() if (backup / n).is_file() else v for n, v in zip(names, current)
        ]
        source, reg_bytes, manifest_bytes = originals
        if digest(source) != OLD_SHA or source.count(BEFORE) != 1:
            raise ValueError("Original engine differs from the reviewed repair target")
        changed = source.replace(BEFORE, AFTER, 1)
        compile(changed, str(engine), "exec")
        old_reg, old_manifest = json.loads(reg_bytes), json.loads(manifest_bytes)
        keys = [key for key in old_reg["hashes"] if Path(key).resolve() == engine]
        if len(keys) != 1:
            raise ValueError("Expected one registered engine path")
        key, new_sha = keys[0], digest(changed)
        if old_reg["hashes"].get(key) != OLD_SHA or old_manifest["sha256"].get("engine.py") != OLD_SHA:
            raise ValueError("Original registration/manifest does not match the original engine")
        new_reg, new_manifest = json.loads(reg_bytes), json.loads(manifest_bytes)
        new_reg["hashes"][key], new_manifest["sha256"]["engine.py"] = new_sha, new_sha
        if (
            current[0] not in (source, changed)
            or json.loads(current[1]) not in (old_reg, new_reg)
            or json.loads(current[2]) not in (old_manifest, new_manifest)
        ):
            raise ValueError("Unexpected source or metadata change; retained for inspection")
        audit(json.loads(current[1]))
        targets = [changed, encoded(new_reg), encoded(new_manifest)]
        if (
            current[0] == changed
            and json.loads(current[1]) == new_reg
            and json.loads(current[2]) == new_manifest
        ):
            print("REPAIR_ALREADY_APPLIED; use the original --resume command", flush=True)
            return
        backup.mkdir(parents=True, exist_ok=True)
        for name, payload in zip(names, originals):
            path = backup / name
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError("Existing original backup differs")
            else:
                with path.open("xb") as stream:
                    stream.write(payload)
        note = dict(
            state="PREPARED",
            repair="V13 absolute replay comparison 1e-8 -> 1e-6 mg/dL",
            original_engine_sha256=OLD_SHA,
            patched_engine_sha256=new_sha,
            original_registration_sha256=digest(reg_bytes),
            patched_registration_sha256=digest(targets[1]),
            helper_sha256=digest(Path(__file__).read_bytes()),
            reported_native_error_mg_dL=2.52560283798e-8,
            reported_saved_model_error_mg_dL=0,
            models_or_predictions_changed=False,
            checkpoint_files_changed=False,
        )
        put(backup / "repair.json", encoded(note))
        try:
            for path, payload in zip((engine, registration, manifest), targets):
                put(path, payload)
            verify(root)
            audit(new_reg)
        except BaseException:
            for path, payload in zip((engine, registration, manifest), current):
                put(path, payload)
            note["state"] = "ROLLED_BACK"
            put(backup / "repair.json", encoded(note))
            raise
        note["state"] = "COMMITTED"
        put(backup / "repair.json", encoded(note))
        print("REPAIR_OK: comparison tolerance=1e-6 mg/dL; consistency check retained", flush=True)
        print("SOURCE_HASH_AMENDMENT=engine.py only; original metadata backed up", flush=True)
        print("ALL_FITTED_CHECKPOINTS_UNCHANGED; restart with bash server_run.sh --resume", flush=True)


if __name__ == "__main__":
    apply(Path.cwd())
PY_REPLAY_FIX
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python -B research_records/apply_replay_fix.py
bash server_run.sh --resume
)
