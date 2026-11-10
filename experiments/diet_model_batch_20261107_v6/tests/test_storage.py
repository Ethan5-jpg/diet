"""Checkpoint integrity and safe recovery from incomplete writes."""

import json
import tempfile
import unittest
from pathlib import Path

from storage import Memo


class StorageChecks(unittest.TestCase):
    def test_finished_checkpoint_resumes_without_computing(self):
        with tempfile.TemporaryDirectory() as name:
            memo = Memo(name)
            self.assertEqual(memo.get("step", lambda: dict(value=3)), dict(value=3))
            self.assertEqual(Memo(name).get("step"), dict(value=3))
            self.assertEqual(memo.created, 1)
            with self.assertRaises(FileNotFoundError):
                memo.get("unfinished")

    def test_incomplete_receipt_is_retained_and_recomputed(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name) / "step"
            folder.mkdir()
            incomplete = folder / "done_partial.json"
            incomplete.write_text("{")
            self.assertEqual(Memo(name).get("step", lambda: 7), 7)
            self.assertEqual(incomplete.read_text(), "{")
            self.assertEqual(Memo(name).get("step"), 7)

    def test_corruption_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            memo = Memo(name)
            memo.get("step", lambda: dict(prediction=[1.0, 2.0]))
            receipt = next((Path(name) / "step").glob("done_*.json"))
            record = json.loads(receipt.read_text())
            (receipt.parent / record["file"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Corrupted checkpoint"):
                memo.get("step")
            with self.assertRaises(ValueError):
                memo.get("../outside", lambda: 1)
