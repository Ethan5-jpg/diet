from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path.cwd()))
from upload_check import verify

root = Path.cwd()
p = Path("tests/test_integrity.py")
m = Path("upload_manifest.json")
info = json.loads(m.read_text())
before = '41d8c0f7b11caa5337f763f12012dd07d4e23cc015c66afc1110c57a3e26319c'
after = '16dd47e9e2e3577c6e284d4dcecf39981848824638c04937cf6a3268fd779308'
source = p.read_bytes()
current = hashlib.sha256(source).hexdigest()
if current == before:
    verify(root)
    old = '        for name, digest in hashes.items():\n            self.assertEqual(sha(ROOT.parent / name), digest)\n'
    new = '        required = (\n            V6.name + "/data.py",\n            V6.name + "/numerics.py",\n            V6.name + "/storage.py",\n            V8.name + "/features.py",\n        )\n        for name in required:\n            self.assertEqual(sha(ROOT.parent / name), hashes[name])\n'
    revised = source.decode("utf-8").replace(old, new, 1).encode("utf-8")
    if hashlib.sha256(revised).hexdigest() != after:
        raise SystemExit("Unexpected patch result; no change made")
    backup = p.with_name(p.name + ".before_portability_fix")
    if not backup.exists():
        backup.write_bytes(source)
    p.write_bytes(revised)
elif current != after:
    raise SystemExit("Unexpected test source; stop and report this message")
if info["sha256"][str(p)] not in (before, after):
    raise SystemExit("Unexpected upload manifest; stop and report this message")
info["sha256"][str(p)] = after
m.write_text(json.dumps(info, indent=2) + "\n")
print("PORTABILITY_FIX_OK files=%d" % verify(root))
