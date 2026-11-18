"""After scoring, compare repeat zero to the historical held-out predictions, if archived."""
from pathlib import Path
import json
import numpy as np


HISTORICAL_OFF = 0.07827223012893003
HISTORICAL_ON = 0.08367503706727109
HISTORICAL_RATIO = 100 * (HISTORICAL_ON - HISTORICAL_OFF) / HISTORICAL_ON


def historical_check(results_dir, folds, digest):
    import pyarrow as pa
    import pyarrow.parquet as pq
    results_dir = Path(results_dir)
    name = 'held_predictions_and_contrasts_TRE_ONLY.parquet'
    manifest = results_dir / 'artifact_manifest_TRE_ONLY.json'
    path = results_dir / name
    if not (path.is_file() and manifest.is_file()):
        return {'status': 'historical_predictions_or_manifest_unavailable'}
    try:
        hashes = json.loads(manifest.read_text())
        if not isinstance(hashes, dict) or digest(path) != hashes.get(name):
            return {'status': 'historical_prediction_hash_mismatch'}
        expected = hashes[name]
    except (ValueError, OSError):
        return {'status': 'historical_manifest_unreadable'}
    cols = ['input_row', 'split_unit_id', 'cv_fold', 'Y_processed',
            'matched_null_prediction', 'ridge_PC10_gut_prediction']
    try:
        file = pq.ParquetFile(path, pre_buffer=False)
        schema = file.schema_arrow
        if not all(schema.names.count(col) == 1 for col in cols):
            return {'status': 'historical_schema_differs'}
        # Legacy identifiers may also be integer-valued float columns with
        # missing entries. Reject nonnumeric physical fields before arithmetic.
        for col in (cols[0], *cols[2:]):
            dtype = schema.field(col).type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
                return {'status': 'historical_schema_differs'}
        table = file.read(columns=cols).to_pandas(ignore_metadata=True)
    except (pa.ArrowException, OSError):
        return {'status': 'historical_parquet_unreadable'}
    if (table.input_row.isna().any() or not table.input_row.is_unique
            or not np.isfinite(table.input_row).all()
            or not np.equal(table.input_row, np.floor(table.input_row)).all()):
        return {'status': 'historical_row_index_invalid'}
    table = table.set_index('input_row').sort_index()
    eligible = table[np.isfinite(table.Y_processed)]
    distances = []
    current_rows = []
    for fold in range(5):
        arrays, meta = folds[0, fold]
        which = meta['names'].index('reference_pc10_p01')
        rows = arrays['rows']
        if not np.isin(rows, eligible.index).all():
            return {'status': 'evaluated_membership_differs'}
        old = eligible.loc[rows]
        if not np.all(old.cv_fold == fold) or not np.array_equal(
                old.split_unit_id.astype(str).to_numpy(), arrays['groups'].astype(str)):
            return {'status': 'historical_household_or_fold_differs'}
        current_rows.extend(rows.tolist())
        distances.append([np.max(np.abs(old[column].to_numpy() - new)) for column, new in (
            ('Y_processed', arrays['y']), ('matched_null_prediction', arrays['off'][:, which]),
            ('ridge_PC10_gut_prediction', arrays['on'][:, which]))])
    if not np.array_equal(np.sort(current_rows), eligible.index.to_numpy()):
        return {'status': 'evaluated_membership_differs'}
    maximum = np.max(distances, axis=0)
    if not np.isfinite(maximum).all():
        return {'status': 'nonfinite_historical_predictions'}
    y = eligible.Y_processed.to_numpy()
    sst = np.sum((y - y.mean()) ** 2)
    scores = [None, None]
    if np.isfinite(sst) and sst > 0:
        for i, col in enumerate(cols[-2:]):
            score = 1 - np.sum((y - eligible[col].to_numpy())**2)/sst
            scores[i] = float(score) if np.isfinite(score) else None
    ratio = None
    if all(s is not None and s > 1e-8 for s in scores):
        ratio = 100 * (scores[1] - scores[0]) / scores[1]
    identical = bool(maximum[0] <= 1e-12 and max(maximum[1:]) <= 2e-8)
    return dict(status='predictions_match' if identical else 'predictions_differ',
                people=len(eligible), tolerance_predictions=2e-8,
                historical_off_r2=scores[0], historical_on_r2=scores[1],
                historical_ratio_pct=ratio,
                max_abs_y_off_on=maximum.tolist(), historical_file_sha256=expected)
