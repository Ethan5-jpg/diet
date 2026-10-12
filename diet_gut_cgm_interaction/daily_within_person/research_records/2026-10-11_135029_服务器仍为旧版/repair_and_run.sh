conda activate pheno &&
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person &&
python - <<'PYREPAIR' &&
from pathlib import Path
import hashlib
import importlib
import json
import os
import shutil
import sys

root = Path.cwd()
patterns = ["*.py", "*.sh", "daily/*.py", "config/*.json", "requirements.txt"]
paths = [p for pattern in patterns for p in sorted(root.glob(pattern))]
data = {str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in paths}
required = ["daily/common.py", "daily/build.py", "daily/__init__.py", "config/analysis.json"]
if not all(name in data for name in required):
    raise SystemExit("REPAIR_STOPPED: run this inside daily_within_person")

old = "    return s.map(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype(float)"
new = '''    # Series.map preserves datetime dtype for an empty datetime Series, whose
    # astype(float) raises. Build float values explicitly, including empty/NaT.
    values = np.fromiter((x.timestamp() if pd.notna(x) else np.nan for x in s),
                         dtype=np.float64, count=len(s))
    return pd.Series(values, index=s.index, name=s.name, dtype="float64")'''
data["daily/common.py"] = data["daily/common.py"].replace(old, new)
old = "            d = d.loc[selected].copy()\n            times = parse_times"
new = '''            d = d.loc[selected].copy()
            if d.empty:
                # The participant is selected but this file/chunk can contain
                # only their other connections. They are already counted above.
                continue
            times = parse_times'''
data["daily/build.py"] = data["daily/build.py"].replace(old, new)
for name in ["daily/common.py", "daily/__init__.py", "config/analysis.json"]:
    data[name] = data[name].replace("2026-10-11.2", "2026-10-11.3")

hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in data.items()}
digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if digest != "71aed0de1b38c258c62033fc3c06d1f07c463eaa75911136f5fc41a6ba8bb2b5":
    raise SystemExit("REPAIR_STOPPED: unexpected file contents; send this message, no files changed")
for name, text in data.items():
    path = root / name
    if path.read_bytes() != text.encode("utf-8"):
        temp = path.with_name(path.name + ".repair_tmp")
        temp.write_bytes(text.encode("utf-8"))
        os.replace(str(temp), str(path))
cache = root / "daily" / "__pycache__"
if cache.is_dir():
    shutil.rmtree(cache)
sys.dont_write_bytecode = True
importlib.invalidate_caches()
from daily import VERSION
from daily.common import epoch, parse_times, validate_config
import pandas as pd
assert VERSION == "2026-10-11.3"
validate_config(json.loads(data["config/analysis.json"]))
assert epoch(parse_times(pd.Series([], dtype=str))).dtype == "float64"
print("REPAIR_OK | CODE_VERSION=" + VERSION + " | verified runtime files; empty-time test passed")
PYREPAIR
PYTHON=python bash run_analysis.sh \
  --data-root /home/ec2-user/Desktop/HPP/Data \
  --stage all
