"""Score the registered V10 fixed setting only on exactly matched glucose folds."""

from hashlib import sha256
from io import BytesIO
import json
from zipfile import BadZipFile

import numpy as np

from metrics import prediction
import protocol as p
from source import AuditError, check, digest, read_json


def receipts(root):
    """Fingerprint the optional reference and the exact fold files it used."""
    run = root / "runs/TRE_ONLY/v10"

    def fingerprint(path):
        try:
            return digest(path)
        except FileNotFoundError:
            return None
        except OSError:
            return "unreadable"

    result = {"COMPLETE.json": fingerprint(run / "COMPLETE.json")}
    if result["COMPLETE.json"] is None:
        return result
    result.update({rel: fingerprint(run / rel) for rel in
                   ("registration.json", *(f"r0_f{f}/{name}" for f in p.FOLDS
                                          for name in ("receipt.json", "metadata.json", "arrays.npz")))})
    result["design.json"] = fingerprint(root / "design.json")
    return result


def load_v10(root, snapshot, held, families, draws):
    run = root / "runs/TRE_ONLY/v10"
    if not (run / "COMPLETE.json").is_file():
        return dict(status="not_available", reason="V10_COMPLETION_RECEIPT_NOT_FOUND")
    try:
        receipt = read_json(run / "COMPLETE.json")
        reg_path = run / "registration.json"
        artifacts = receipt["artifacts"]
        check(receipt["version"] == "v10" and digest(reg_path) == artifacts["registration.json"],
              "V10_REGISTRATION_HASH")
        registration = read_json(reg_path)
        design_file = root / "design.json"
        check(digest(design_file) == p.V10_DESIGN_SHA and
              registration["release"]["files"]["design.json"] == p.V10_DESIGN_SHA, "V10_RECIPE_HASH")
        design = read_json(design_file)
        check(registration["protocol"]["paired_column_sha256"] == design["paired_column_sha256"]
              and registration["protocol"]["outcome_scope"] == "discovery only", "V10_PROTOCOL")
        paths = [p.INDEX, p.outcome_file("D_Glu"), *(f"{p.INPUTS}/{s}_raw_TRE_ONLY.npz" for s in ("oral", "gut"))]
        check(all(registration["source_hashes"].get(str(snapshot.paths[k][0])) == snapshot.paths[k][1]
                  for k in paths), "V10_INPUT_HASH_MISMATCH")

        arms, settings, adaptive = (design[name] for name in ("family_order", "settings_order", "adaptive"))
        prefix = p.V10_SETTING.split("@")[0]
        check(len(set(arms)) == len(arms) and len(set(settings)) == len(settings)
              and settings.count(p.V10_SETTING) == 1, "V10_PAIRED_COLUMN_NAMES")
        off_index = [arms.index(name.rsplit("@", 1)[0]) for name in settings]
        column_hash = sha256(json.dumps([arms, settings, off_index, adaptive], separators=(",", ":")).encode()).hexdigest()
        check(column_hash == design["paired_column_sha256"], "V10_PAIRED_COLUMNS")
        j = settings.index(p.V10_SETTING)

        y, g, both = [], [], []
        check(tuple(x.fold for x in held) == p.FOLDS, "V10_INCOMPLETE_MATCH")
        for matched in held:
            fold = matched.fold
            folder = run / f"r0_f{fold}"
            verified = read_json(folder / "receipt.json")
            meta = (folder / "metadata.json").read_bytes()
            data = (folder / "arrays.npz").read_bytes()
            check(sha256(meta).hexdigest() == verified["metadata_sha256"] and
                  sha256(data).hexdigest() == verified["arrays_sha256"], f"V10_FOLD_HASH f{fold}")
            info = json.loads(meta)
            check(info["fold"] == fold and info["repeat"] == 0 and info["paired_column_sha256"] == column_hash
                  and info["additive_models_checked"] == len(arms), f"V10_FOLD_METADATA f{fold}")
            with np.load(BytesIO(data), allow_pickle=False) as a:
                rows = a["rows"]
                check(rows.dtype.kind in "iu" and rows.ndim == 1 and np.unique(rows).size == len(rows),
                      f"V10_FOLD_ROW_INDEX f{fold}")
                order = np.argsort(rows)
                check(np.array_equal(rows[order], matched.rows), f"V10_SCORED_MEMBERSHIP_DIFFERS f{fold}")
                n = len(rows)
                check(a["y"].shape == a["groups"].shape == (n,) and info["people"] == n and
                      np.array_equal(a["groups"][order].astype(str), families[matched.rows]), f"V10_HOUSEHOLD_ALIGNMENT f{fold}")
                check(np.array_equal(a["y"][order], matched.y), f"V10_SCORED_OUTCOME_DIFFERS f{fold}")
                off, on = a["setting_off"], a["setting_on"]
                check(off.shape == (n, len(arms)) and on.shape == (n, len(settings)) and
                      off_index[j] == arms.index(prefix), f"V10_FOLD_PREDICTION_LAYOUT f{fold}")
                both.append(np.column_stack((off[order, off_index[j]], on[order, j])))
            y.append(matched.y)
            g.append(families[matched.rows])
        stats = prediction(np.concatenate(y), np.vstack(both), np.concatenate(g), draws, p.BOOTSTRAP_SEED,
                           names=("v10_off", "v10_on"),
                           pairs={"all_interactions": ("v10_off", "v10_on", False)})
        return dict(status="matched", setting=p.V10_SETTING, repeat=0, folds=len(held), scoring=stats,
                    registration_sha256=artifacts["registration.json"])
    except AuditError as exc:
        return dict(status="not_matched", reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        return dict(status="not_verified", reason="V10_ARCHIVE_" + type(exc).__name__)
