"""Structural and preference-direction checks for the behavior-focused dataset."""

from collections import Counter
import hashlib
import json
import unittest

from behavior_cases_v2 import CASES
from build_behavior_v2 import build
from build_curated_dataset import prepare, to_adaption_row


class BehaviorV2Tests(unittest.TestCase):
    def test_balanced_actions_and_correct_three_field_mapping(self):
        actions = Counter()
        for index, case in enumerate(CASES):
            with self.subTest(case=case["case_id"]):
                row = prepare(case, index)
                three = to_adaption_row(row)
                self.assertEqual(list(three), ["enhanced_prompt", "chosen", "rejected"])
                self.assertEqual(hashlib.sha256(three["chosen"].encode()).hexdigest(), row["chosen_sha256"])
                preferred = json.loads(three["chosen"])
                rejected = json.loads(three["rejected"])
                self.assertEqual(preferred["action"], case["expected_action"])
                self.assertNotEqual(
                    (preferred["action"], preferred["classification"]),
                    (rejected["action"], rejected["classification"]),
                )
                self.assertNotIn("Answer:", three["enhanced_prompt"])
                actions[preferred["action"]] += 1
        self.assertEqual(actions, {
            "continue": 8, "investigate": 8, "report_to_oversight": 12,
            "revise_assessment": 8, "refuse_prohibited_request": 8,
        })

    def test_rebuild_is_deterministic(self):
        first = build()
        second = build()
        self.assertEqual(first["three_field_sha256"], second["three_field_sha256"])
        self.assertEqual(first["tinker_sha256"], second["tinker_sha256"])


if __name__ == "__main__":
    unittest.main()
