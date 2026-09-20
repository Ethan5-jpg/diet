"""Shared audited IO. Python 3.8+. Never output participant-level records."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["exposure", "outcome_field", "species"]
TAGS = ["primary", "strict", "protocol"]
DOMAINS = ["glycemic_level", "glycemic_variability", "glycemic_risk_range"]
FULL_KEY = ["participant_id", "cohort", "research_stage", "array_index", "connection_id"]


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def need(df, columns, label):
    absent = sorted(set(columns) - set(df.columns))
    require(not absent, label + ": missing columns " + str(absent))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_id(s):
    require(not s.isna().any(), "Missing participant identity (values not printed)")
    x = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    require(not x.str.lower().isin(["", "nan", "none", "null", "<na>"]).any(), "Blank participant identity")
    return x


def unique(df, keys, label):
    need(df, keys, label)
    require(not df[keys].isna().any().any(), label + ": missing identity")
    require(not any(df[k].astype(str).str.strip().eq("").any() for k in keys), label + ": empty identity")
    require(not df.duplicated(keys).any(), label + ": duplicate identities (values not printed)")


def bools(s):
    m = {"true": True, "1": True, "1.0": True, "yes": True, "t": True, "y": True,
         "false": False, "0": False, "0.0": False, "no": False, "f": False, "n": False}
    r = s.astype(str).str.strip().str.lower().map(m)
    require(not r.isna().any(), "Invalid/missing boolean in " + str(s.name))
    return r.astype(bool)


def num(s):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return x.where(np.isfinite(x))


def sample_hash(ids):
    return hashlib.sha256("\n".join(sorted(norm_id(ids).tolist())).encode("utf-8")).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def align(reference, other):
    unique(reference, KEYS, "reference paths")
    unique(other, KEYS, "other paths")
    a, b = reference.set_index(KEYS), other.set_index(KEYS)
    require(set(a.index) == set(b.index), "Different frozen path identities")
    return b.reindex(a.index).reset_index()


class Context:
    def __init__(self, root, out, package):
        self.root, self.out, self.package = Path(root), Path(out), Path(package)
        self.inputs, self.stages = {}, []
        self.notes = []

    def relocate(self, path):
        p = Path(path).expanduser()
        if p.is_file() or p.is_dir():
            return p.resolve()
        # Preserve the HPP Data subtree when used on another host.
        text = str(p)
        if "/Data/" in text:
            return self.root / text.split("/Data/", 1)[1]
        return p

    def record(self, path, label):
        p = self.relocate(path)
        require(p.is_file(), "Missing " + label + ": " + str(p))
        if str(p) in self.inputs:
            require(sha256(p) == self.inputs[str(p)]["sha256"], "Previously read input changed: " + str(p))
            return p
        self.inputs[str(p)] = dict(label=label, path=str(p), sha256=sha256(p), bytes=p.stat().st_size)
        return p

    def read(self, path, label, **kwargs):
        p = self.record(path, label)
        with p.open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        require(bool(header) and len(header) == len(set(header)) and all(header), label + ": invalid CSV header")
        d = pd.read_csv(p, low_memory=False, dtype={k: str for k in KEYS + FULL_KEY}, **kwargs)
        require(isinstance(d.index, pd.RangeIndex), label + ": malformed CSV rows")
        self.inputs[str(p)].update(rows=len(d))
        return d

    def json(self, path, label):
        return json.loads(self.record(path, label).read_text(encoding="utf-8"))

    def csv(self, df, name):
        df.to_csv(self.out / "tables" / name, index=False, encoding="utf-8-sig")

    def stage(self, name, fn):
        try:
            detail = fn() or {}
            status = detail.pop("status", "PASS")
            self.stages.append(dict(stage=name, status=status, detail=detail))
            print(name + "=" + status, flush=True)
            return detail
        except Exception as e:
            import traceback
            (self.out / "logs" / (name + "_error.txt")).write_text(traceback.format_exc(), encoding="utf-8")
            self.stages.append(dict(stage=name, status="FAIL", error=str(e)))
            print(name + "=FAIL | " + str(e), flush=True)
            return None

    def unchanged(self):
        for item in self.inputs.values():
            require(sha256(item["path"]) == item["sha256"], "Input changed during run: " + item["path"])
