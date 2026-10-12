import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class InputError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InputError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path, body):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(safe(body), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


MISSING = {"", "na", "n/a", "nan", "none", "null", "<na>", "#n/a"}


def text_series(series):
    text = series.astype(str).str.strip()
    return text.mask(series.isna() | text.str.lower().isin(MISSING))


def numeric(series, label):
    text = text_series(series)
    result = pd.to_numeric(text, errors="coerce").astype(float)
    require(not (text.notna() & (~np.isfinite(result))).any(), label + ": invalid numeric/nonfinite values")
    return result


def norm_ids(series):
    ids = text_series(series).str.replace(r"\.0$", "", regex=True)
    require(ids.notna().all() and ids.ne("").all(), "Missing participant identity (IDs not printed)")
    return ids


def sample_hash(index):
    return hashlib.sha256("\n".join(sorted(str(x) for x in index)).encode("utf-8")).hexdigest()


def need(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, label + ": missing columns " + str(missing))


def header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle, strict=True), [])
    require(columns and all(columns) and len(columns) == len(set(columns)), "Empty/duplicate CSV header: " + str(path))
    return columns


def read_csv(path, usecols=None):
    columns = header(path)
    # Detect malformed wide files before pandas can infer an accidental row index.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        next(reader)
        for row in reader:
            require(not row or len(row) == len(columns), "Malformed CSV row at line {}: {}".format(reader.line_num, path))
    frame = pd.read_csv(str(path), dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=usecols)
    require(not frame.empty, "Empty input: " + str(path))
    return frame
