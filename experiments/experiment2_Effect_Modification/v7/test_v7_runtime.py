"""Exercise the full nested estimator and immutable checkpoints with generated data."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_v7
if str(run_v7.V6) not in sys.path:
    sys.path.insert(1, str(run_v7.V6))
from integration_fixture import Generated
from threadpoolctl import threadpool_limits
import v7_eval
import v7_reference


class RuntimeTest(unittest.TestCase):
    def test_generated_nested_run_replay_and_checkpoint_integrity(self):
        source = Generated(n=180, seed=406, effect=5.)
        release = {'fixture': 'generated'}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), threadpool_limits(2):
            run = Path(tmp)/'run'
            fit = v7_eval.run_fold

            def interrupt(source, repeat, fold):
                if (repeat, fold) == (0, 1):
                    raise InterruptedError('generated interruption')
                return fit(source, repeat, fold)

            with patch('v7_eval.run_fold', side_effect=interrupt), self.assertRaises(InterruptedError):
                run_v7.execute(source, run, release, 2, draws=99)
            self.assertTrue((run/'r0_f0'/'receipt.json').is_file())
            self.assertFalse((run/'COMPLETE.json').exists())
            with patch('v7_eval.run_fold', wraps=fit) as resumed:
                s = run_v7.execute(source, run, release, 2, draws=99)
                self.assertEqual(resumed.call_count, 9)
            self.assertEqual(len(s['repeats']), 2)
            for repeat in s['repeats']:
                self.assertGreater(repeat['routes']['ratio_selector']['delta_r2'], 0.02)
                self.assertEqual(repeat['people'], 180)
                self.assertEqual(repeat['households'], 90)
            run_v7.compact(s)
            with patch('v7_eval.run_fold', side_effect=AssertionError('refitted completed run')):
                self.assertEqual(run_v7.execute(source, run, release, 2, draws=99), s)
            with self.assertRaisesRegex(ValueError, 'registration differs'):
                run_v7.execute(source, run, {'changed':True}, 2, draws=99)
            (run/'r0_f2'/'arrays.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run_v7.read_complete(run, release)

    def test_legacy_parquet_audit_aligns_by_row_and_requires_hash(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            folds, records = {}, []
            for fold in range(5):
                rows = np.array([2*fold, 2*fold+1])
                y = rows.astype(float)
                off, on = y+.5, y+.1
                groups = np.array([f'g{fold}']*2)
                folds[0,fold] = (dict(rows=rows,groups=groups,y=y,off=off[:,None],on=on[:,None]),
                                 dict(names=['reference_pc10_p01']))
                records.extend(zip(rows,groups,[fold]*2,y,off,on))
            frame = pd.DataFrame(records, columns=['input_row','split_unit_id','cv_fold','Y_processed',
                                                   'matched_null_prediction','ridge_PC10_gut_prediction'])
            archive = path/'held_predictions_and_contrasts_TRE_ONLY.parquet'
            frame.iloc[::-1].to_parquet(archive,index=False)
            m = path/'artifact_manifest_TRE_ONLY.json'
            m.write_text(json.dumps({archive.name:run_v7.digest(archive)}))
            result = v7_reference.historical_check(path, folds, run_v7.digest)
            self.assertEqual(result['status'], 'predictions_match')
            folds[0,0][0]['on'][0,0] += .1
            self.assertEqual(v7_reference.historical_check(path,folds,run_v7.digest)['status'],'predictions_differ')
            m.write_text('{}')
            self.assertEqual(v7_reference.historical_check(path,folds,run_v7.digest)['status'],'historical_prediction_hash_mismatch')
            # A valid archive can have poor predictions. The identity of the
            # two prediction arrays can still be verified without a ratio.
            for arr, _ in folds.values():
                arr['on'][:, 0] = arr['y'] + 100
            frame['ridge_PC10_gut_prediction'] = frame['Y_processed'] + 100
            frame.to_parquet(archive, index=False)
            m.write_text(json.dumps({archive.name:run_v7.digest(archive)}))
            result = v7_reference.historical_check(path, folds, run_v7.digest)
            self.assertEqual(result['status'], 'predictions_match')
            self.assertLess(result['historical_on_r2'], 0)
            self.assertIsNone(result['historical_ratio_pct'])
            frame['Y_processed'] = 'unknown'
            frame.to_parquet(archive, index=False)
            m.write_text(json.dumps({archive.name:run_v7.digest(archive)}))
            self.assertEqual(v7_reference.historical_check(path, folds, run_v7.digest)['status'],
                             'historical_schema_differs')
