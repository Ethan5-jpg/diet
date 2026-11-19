"""Read a registered discovery snapshot. No confirmation outcomes are addressable."""

from dataclasses import dataclass
from hashlib import file_digest, sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np
import pyarrow.parquet as pq

import protocol as p


class AuditError(ValueError):
    """Safe error text with paths/field names only, never participant values."""


def check(ok, code):
    if not ok:
        raise AuditError(code)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def required_files(tasks):
    tasks = tuple(tasks)
    files = [p.INDEX, p.ROLES, p.FLOW]
    for site in ("oral", "gut"):
        files += [f"{p.INPUTS}/{site}_taxa.json", f"{p.INPUTS}/{site}_raw_TRE_ONLY.npz"]
    files += [p.fold_file(fold) for fold in (*p.FOLDS, None)]
    if any(x == "UPF" for x, _ in tasks):
        files += [p.upf_metadata(fold) for fold in (*p.FOLDS, None)]
    files += [p.outcome_file(e) for e in dict.fromkeys(e for _, e in tasks)]
    return files


class Snapshot:
    def __init__(self, root, files):
        self.root = Path(root).resolve(strict=True)
        self.paths = {}
        self.manifest = self._path(p.MANIFEST)
        contents = self.manifest.read_bytes()
        self.manifest_sha = sha256(contents).hexdigest()
        entries = json.loads(contents)
        check(isinstance(entries, list), "INVALID_SNAPSHOT_MANIFEST")
        catalog = {}
        for entry in entries:
            rel = entry["relative_path"]
            check(rel not in catalog, "DUPLICATE_SNAPSHOT_MANIFEST_PATH")
            catalog[rel] = entry
        for rel in files:
            check(rel in catalog, "UNREGISTERED_INPUT " + rel)
            path = self._path(rel)
            entry = catalog[rel]
            check(path.is_file() and path.stat().st_size == entry["bytes"],
                  "MISSING_OR_SIZE_MISMATCH " + rel)
            check(hex_sha(entry["sha256"]), "INVALID_INPUT_SHA " + rel)
            self.paths[rel] = (path, entry["sha256"])
        self.verified = set()

    def _path(self, rel):
        path = (self.root / rel).resolve()
        check(path.is_relative_to(self.root), "INPUT_PATH_OUTSIDE_SNAPSHOT")
        return path

    def report(self):
        return {rel: sha for rel, (_, sha) in sorted(self.paths.items())}

    def bytes(self, rel):
        path, expected = self.paths[rel]
        value = path.read_bytes()
        check(sha256(value).hexdigest() == expected, "INPUT_HASH_MISMATCH " + rel)
        self.verified.add(rel)
        return value

    def json(self, rel):
        return json.loads(self.bytes(rel))

    def parquet(self, rel):
        return pq.ParquetFile(BytesIO(self.bytes(rel)), pre_buffer=False)

    def table(self, rel, columns):
        with self.parquet(rel) as parquet:
            check(set(columns) <= set(parquet.schema_arrow.names), "INPUT_SCHEMA_MISMATCH " + rel)
            return parquet.read(columns=list(columns), use_threads=False)

    def footer(self, rel, columns):
        path, _ = self.paths[rel]
        with pq.ParquetFile(path, pre_buffer=False) as parquet:
            check(set(columns) <= set(parquet.schema_arrow.names), "INPUT_SCHEMA_MISMATCH " + rel)
            return parquet.metadata.num_rows

    def npz(self, rel, fields):
        value = self.bytes(rel)
        try:
            archive = np.load(BytesIO(value), allow_pickle=False)
            check(isinstance(archive, np.lib.npyio.NpzFile), "EXPECTED_NPZ_ARCHIVE " + rel)
            with archive:
                check(set(fields) <= set(archive.files), "INPUT_ARRAY_FIELDS_MISMATCH " + rel)
                return {name: archive[name] for name in fields}
        except AuditError:
            raise
        except (BadZipFile, ValueError, OSError, EOFError) as exc:
            raise AuditError("INVALID_INPUT_NPZ " + rel) from exc

    def verify_manifest(self):
        check(digest(self.manifest) == self.manifest_sha, "SNAPSHOT_MANIFEST_CHANGED")


def inventory(snapshot, tasks):
    # Footers and file metadata only. No outcome column is materialized.
    cov = snapshot.footer(p.INDEX, p.INDEX_COLS)
    roster = snapshot.footer(p.ROLES, (*p.KEY, "split_unit_id", "final_role"))
    return dict(discovery_rows=cov, roster_rows=roster, registered_files=len(snapshot.paths),
                raw_outcome_rows={e: snapshot.footer(p.outcome_file(e),
                                                    (*p.KEY, "input_row", "Y_value", "Y_unit"))
                                  for e in dict.fromkeys(e for _, e in tasks)},
                registrations="metadata only", confirmation_outcomes_read=False)


def numbers(table, column):
    return np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=float)


def integers(table, column):
    values = table[column].to_pylist()
    for value in values:
        check(type(value) in (int, float) and math.isfinite(value)
              and abs(value) < 2**53 and int(value) == value, "NONINTEGER_COLUMN " + column)
    return np.asarray(values, dtype=np.int64)


def labels(table, column):
    values = table[column].to_pylist()
    check(all(v is not None and str(v).strip() and
              (not isinstance(v, float) or math.isfinite(v)) for v in values),
          "MISSING_LABEL " + column)
    return np.asarray([str(v) for v in values], dtype=str)


def keys(table):
    people = integers(table, "participant_id")
    cohort, stage = (labels(table, c) for c in p.KEY[1:])
    check(np.all(cohort == "10k") and np.all(stage == "00_00_visit"),
          "UNEXPECTED_PARTICIPANT_COHORT_OR_VISIT")
    check(len(np.unique(people)) == len(people), "DUPLICATED_PARTICIPANT_KEY")
    return people, cohort, stage


def in_role_family(family, role):
    check(len(set(zip(family.tolist(), role.tolist()))) == len(np.unique(family)),
          "HOUSEHOLD_CROSSES_ROLE_OR_FOLD")


def species_indices(taxa):
    check(isinstance(taxa, list) and all(isinstance(x, str) and x for x in taxa)
          and len(set(taxa)) == len(taxa), "INVALID_TAXON_DICTIONARY")
    hits = {}
    for i, label in enumerate(taxa):
        term = label.rsplit("|", 1)[-1].strip().lower()
        if term.startswith("s__"):
            term = term[3:]
        elif "|" in label or "__" in term:
            continue
        hits.setdefault(re.sub(r"\s+", "_", term), []).append(i)
    mapping = {}
    for name, aliases in p.SPECIES.items():
        ix = [c for alias in aliases for c in hits.get(alias, ())]
        check(len(ix) == 1, f"THREE_SPECIES_MATCH_COUNT {name} {len(ix)}")
        mapping[name] = ix[0]
    return mapping


def raw_ecology(abundance):
    total = abundance.sum(axis=1)
    check(np.isfinite(abundance).all() and not np.any(abundance < 0)
          and np.allclose(total, 1, rtol=0, atol=0.001), "INVALID_RAW_RELATIVE_ABUNDANCE_FRACTIONS")
    rel = abundance / total[:, None]
    logs = np.log(rel, out=np.zeros_like(rel), where=rel > 0)
    return np.column_stack((-(rel * logs).sum(axis=1), rel.max(axis=1)))


@dataclass(frozen=True)
class Background:
    role: np.ndarray
    supported: np.ndarray
    pcs: np.ndarray
    archived: dict
    upf: dict | None


@dataclass(frozen=True)
class Outcome:
    y: np.ndarray
    unit: str


class Study:
    def __init__(self, snapshot, *, with_upf=False, expected=p.EXPECTED_ROLE_COUNTS):
        self.snapshot = snapshot
        self.native = expected is not None
        cov = snapshot.table(p.INDEX, p.INDEX_COLS)
        self.keys = keys(cov)
        self.n = len(cov)
        check(np.array_equal(integers(cov, "input_row"), np.arange(self.n)), "DISCOVERY_ROW_INDEX")
        check(np.all(labels(cov, "final_role") == "discovery"), "NONDISCOVERY_INDEX_ROW")
        self.family, self.fold = labels(cov, "split_unit_id"), integers(cov, "cv_fold")
        check(set(self.fold) == set(p.FOLDS), "INVALID_DISCOVERY_FOLDS")
        in_role_family(self.family, self.fold)

        roster = snapshot.table(p.ROLES, (*p.KEY, "split_unit_id", "final_role"))
        rk, rfamily, rrole = keys(roster), labels(roster, "split_unit_id"), labels(roster, "final_role")
        check(set(rrole) == set(p.ROLE_NAMES), "ROSTER_ROLE_LABELS")
        in_role_family(rfamily, rrole)
        rc = tuple((int((rrole == nm).sum()), len(np.unique(rfamily[rrole == nm]))) for nm in p.ROLE_NAMES)
        if self.native:
            check(rc == expected, "DISCOVERY_OR_RESERVED_ROSTER_SIZE_MISMATCH")
        discovery_rows = {int(k): fam for k, fam in zip(rk[0][rrole == "discovery"],
                                                        rfamily[rrole == "discovery"], strict=True)}
        check(discovery_rows == dict(zip(self.keys[0], self.family, strict=True)), "DISCOVERY_ROSTER_MISMATCH")
        self.role_counts = rc

        numeric = ("age_calendar_years", "sex_code", "fiber_density",
                   *(c for cols in p.EXPOSURES.values() for c in cols[:2]))
        self.numeric = {c: numbers(cov, c) for c in numeric}
        self.smoking = np.array([str(v) if v is not None and str(v).strip() else "__UNKNOWN__"
                                 for v in cov["smoking_status"].to_pylist()], dtype=str)
        self.backgrounds = {f: self._background(f, with_upf) for f in (*p.FOLDS, None)}
        self.flow = {}
        for entry in snapshot.json(p.FLOW):
            key = (entry["preprocessing"], entry["analysis_id"], entry["exposure_id"])
            check(key not in self.flow, "DUPLICATE_ARCHIVED_FLOW")
            self.flow[key] = entry

        ecology = []
        for site in ("oral", "gut"):
            taxa = snapshot.json(f"{p.INPUTS}/{site}_taxa.json")
            check(isinstance(taxa, list) and all(isinstance(t, str) and t for t in taxa)
                  and len(taxa) == len(set(taxa)), "INVALID_TAXON_DICTIONARY " + site)
            raw = snapshot.npz(f"{p.INPUTS}/{site}_raw_TRE_ONLY.npz", ("abundance",))["abundance"]
            raw = np.asarray(raw, dtype=float)
            check(raw.shape == (self.n, len(taxa)) and len(taxa) >= 3, "RAW_PROFILE_ALIGNMENT " + site)
            ecology.append(raw_ecology(raw))
            if site == "gut":
                matches = species_indices(taxa)
                panel = raw[:, list(matches.values())]
                score = np.log1p(panel.sum(axis=1) / p.PANEL_UNIT)
                self.panel_qc = {name: {"terminal_label": taxa[i], "detected": int((raw[:, i] > 0).sum())}
                                 for name, i in matches.items()}
        self.ecology = np.column_stack(ecology)
        self.markers = np.column_stack((score, self.ecology[:, 2]))

    def _background(self, fold, with_upf):
        fields = ("training", "supported", "oral_PC", "gut_PC", "fiber_u", "fiber_h")
        if with_upf:
            fields += ("UPF_u", "UPF_h")
        arrays = self.snapshot.npz(p.fold_file(fold), fields)
        expected = self.fold != fold if fold is not None else np.ones(self.n, bool)
        train, supported = arrays["training"], arrays["supported"]
        check(train.dtype == supported.dtype == bool and supported.shape == (self.n,)
              and np.array_equal(train, expected), "FOLD_CACHE_ROLE_OR_SUPPORT_MISMATCH")
        for site in ("oral", "gut"):
            a = arrays[f"{site}_PC"]
            check(a.ndim == 2 and a.shape[0] == self.n and a.shape[1] >= p.N_PC
                  and np.isfinite(a[supported, :p.N_PC]).all(), "FOLD_PC_ALIGNMENT " + site)
        pcs = np.column_stack([arrays[f"{s}_PC"][:, :p.N_PC] for s in ("oral", "gut")])
        archived = {}
        for x, (_, _, prefix) in p.EXPOSURES.items():
            if x == "UPF" and not with_upf:
                continue
            cols = arrays[f"{prefix}_u"], arrays[f"{prefix}_h"]
            check(all(a.shape == (self.n,) for a in cols), "ARCHIVED_EXPOSURE_DIMENSION " + x)
            archived[x] = np.column_stack(cols).astype(float)
        upf = self.snapshot.json(p.upf_metadata(fold))["UPF"] if with_upf else None
        if upf is not None:
            check(np.isfinite(upf["center"]) and 0 <= upf["center"] <= 1
                  and upf["scale"] == 0.1, "UPF_CALIBRATION_SCALE_MISMATCH")
        return Background(train, supported, pcs, archived, upf)

    def outcome(self, endpoint):
        with self.snapshot.parquet(p.outcome_file(endpoint)) as parquet:
            indexes = parquet.read(columns=[*p.KEY, "input_row"], use_threads=False)
            rows = integers(indexes, "input_row")
            check(np.unique(rows).size == len(rows) and np.all((0 <= rows) & (rows < self.n)),
                  "OUTCOME_ROW_INDEX " + endpoint)
            check(all(np.array_equal(x[rows], y) for x, y in zip(self.keys, keys(indexes), strict=True)),
                  "OUTCOME_COMPOSITE_KEY_MISMATCH " + endpoint)
            # Verify the discovery key in the audited bytes before decoding Y.
            t = parquet.read(columns=["Y_value", "Y_unit"], use_threads=False)
        recorded_units = t["Y_unit"].to_pylist()
        units = {u for u in recorded_units if isinstance(u, str) and u.strip()}
        y_raw = numbers(t, "Y_value")
        check(len(units) == 1 and all(u in units for u, v in zip(recorded_units, y_raw, strict=True)
                                      if np.isfinite(v)), "OUTCOME_UNIT " + endpoint)
        unit = next(iter(units))
        check(endpoint != "D_Glu" or unit == "mg/dL", "GLUCOSE_UNIT_MISMATCH")
        y = np.full(self.n, np.nan)
        y[rows] = y_raw
        return Outcome(y, unit)

    def eligible(self, exposure, endpoint, fold, cleaned):
        bg = self.backgrounds[fold]
        days, energy, _ = p.EXPOSURES[exposure]
        day, kcal = self.numeric[days], self.numeric[energy]
        ok = ((day >= 3) & bg.supported & np.isfinite(self.numeric["age_calendar_years"])
              & np.isin(self.numeric["sex_code"], (0, 1)) & np.isfinite(kcal)
              & np.isfinite(bg.archived[exposure]).all(axis=1) & np.isfinite(cleaned))
        split = "full" if fold is None else f"fold{fold}"
        entry = self.flow.get((split, endpoint, exposure))
        check(entry is not None and
              (int(ok.sum()), int((ok & bg.role).sum()), int((ok & ~bg.role).sum())) ==
              tuple(entry[term] for term in ("eligible", "training", "validation")),
              f"ARCHIVED_ELIGIBILITY_MISMATCH {exposure} {endpoint} {split}")
        check(np.isfinite(self.markers[ok]).all() and np.all(kcal[ok] > 0), "ELIGIBLE_INPUT_NONFINITE_OR_ENERGY")
        if exposure == "fiber_density":
            fiber = self.numeric[exposure][ok]
            check(np.isfinite(fiber).all() and np.all(fiber >= 0), "INVALID_FIBER_DENSITY")
        else:
            amount = bg.archived[exposure][ok, 0] * bg.upf["scale"] + bg.upf["center"]
            check(np.all((-1e-8 <= amount) & (amount <= 1 + 1e-8)), "INVALID_UPF_FRACTION")
        return ok


def clean_outcome(raw, training):
    """W20 endpoint transform, recomputed only from each outer training role."""
    finite = np.isfinite(raw)
    ordered = np.sort(raw[training & finite])
    check(len(ordered) >= 3, "INSUFFICIENT_TRAINING_OUTCOME")
    k = math.ceil(0.95 * len(ordered))
    begin = int(np.argmin(ordered[k - 1:] - ordered[:len(ordered) - k + 1]))
    core = ordered[begin:begin + k]
    mu, sd = float(core.mean()), float(core.std(ddof=1))
    check(np.isfinite(sd) and sd > 0, "DEGENERATE_TRAINING_OUTCOME")
    retained = finite & (raw >= mu - 8 * sd) & (raw <= mu + 8 * sd)
    result = np.full(raw.shape, np.nan)
    result[retained] = np.clip(raw[retained], mu - 5 * sd, mu + 5 * sd)
    return result
