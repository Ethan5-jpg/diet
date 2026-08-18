import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_DIR
    / "scripts"
    / "hpdi"
    / "04b_audit_hpdi_missing_weight_direction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hpdi_missing_weight_direction_audit", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载测试模块：{}".format(SCRIPT_PATH))
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules["hpdi_missing_weight_direction_audit"] = AUDIT
SPEC.loader.exec_module(AUDIT)


def make_participant(
    participant_id,
    missing_counts=None,
    event_counts=None,
):
    missing_counts = missing_counts or {}
    event_counts = event_counts or {}
    row = {
        "participant_id": participant_id,
        "cohort": "10k",
        "research_stage": "00_00_visit",
    }
    for component, _ in AUDIT.COMPONENT_DEFINITIONS:
        row[AUDIT.missing_weight_column(component)] = missing_counts.get(
            component, 0
        )
        row[AUDIT.component_event_column(component)] = event_counts.get(
            component, 10
        )
    return row


class HpdiMissingWeightDirectionAuditTests(unittest.TestCase):
    def test_direction_counts_and_participant_overlap_are_separate(self):
        participants = pd.DataFrame(
            [
                make_participant("p1", {"fruits": 2}),
                make_participant("p2", {"dairy": 3}),
                make_participant("p3", {"vegetables": 1, "meat": 1}),
                make_participant("p4"),
            ]
        )

        component, direction, patterns = AUDIT.audit_missing_weight_direction(
            participants
        )

        by_direction = direction.set_index("scoring_direction")
        self.assertEqual(
            int(by_direction.loc["positive", "missing_weight_event_count"]), 3
        )
        self.assertEqual(
            int(by_direction.loc["reverse", "missing_weight_event_count"]), 4
        )
        self.assertEqual(
            int(by_direction.loc["positive", "affected_participant_count"]), 2
        )
        self.assertEqual(
            int(by_direction.loc["reverse", "affected_participant_count"]), 2
        )

        by_pattern = patterns.set_index("participant_pattern")
        self.assertEqual(
            int(by_pattern.loc["positive_only", "participant_count"]), 1
        )
        self.assertEqual(
            int(by_pattern.loc["reverse_only", "participant_count"]), 1
        )
        self.assertEqual(
            int(by_pattern.loc["both_directions", "participant_count"]), 1
        )
        self.assertEqual(
            int(by_pattern.loc["any_missing_weight", "participant_count"]), 3
        )
        self.assertEqual(
            int(by_pattern.loc["no_missing_weight", "participant_count"]), 1
        )

        fruits = component.set_index("hpdi_component").loc["fruits"]
        self.assertEqual(int(fruits["missing_weight_event_count"]), 2)
        self.assertEqual(int(fruits["affected_participant_count"]), 1)

    def test_duplicate_participant_rows_are_rejected(self):
        participants = pd.DataFrame(
            [make_participant("p1"), make_participant("p1")]
        )
        with self.assertRaisesRegex(ValueError, "重复参与者"):
            AUDIT.audit_missing_weight_direction(participants)


if __name__ == "__main__":
    unittest.main()
