"""Regression checks for the corrected preference export."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from build_curated_dataset import prepare, to_adaption_row, validate_export, validate_three_field_export
from curated_cases import CASES


class CuratedDatasetTests(unittest.TestCase):
    def test_every_case_has_a_consequential_preference(self):
        for index, case in enumerate(CASES):
            with self.subTest(case=case["case_id"]):
                row = prepare(case, index)
                label = row["label"]
                preferred = json.loads(row["comparison"][f"completion_{label}"][0]["content"])
                other = json.loads(row["comparison"][f"completion_{'B' if label == 'A' else 'A'}"][0]["content"])
                self.assertEqual(preferred["action"], row["expected_action"])
                self.assertNotEqual(
                    (preferred["action"], preferred["classification"]),
                    (other["action"], other["classification"]),
                )

    def test_flipped_label_is_rejected(self):
        row = prepare(CASES[1], 1)
        row["label"] = "A"
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "chosen hash/label mismatch"):
                validate_export(path, "train")

    def test_same_decision_pair_is_rejected(self):
        case = dict(CASES[0])
        case["rejected"] = (case["chosen"][0], case["chosen"][1],
                            case["chosen"][2], "This restates the same action and classification without a useful preference contrast.")
        with self.assertRaisesRegex(ValueError, "no consequential"):
            prepare(case, 0)

    def test_three_field_export_tracks_preferred_side(self):
        for index in (0, 1):
            row = prepare(CASES[index], index)
            three = to_adaption_row(row)
            self.assertEqual(list(three), ["enhanced_prompt", "chosen", "rejected"])
            self.assertEqual(
                three["chosen"],
                row["comparison"][f"completion_{row['label']}"][0]["content"],
            )
            self.assertEqual(json.loads(three["chosen"])["action"], row["expected_action"])
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "three.jsonl"
                path.write_text(json.dumps(three) + "\n")
                validate_three_field_export(path, [row])


if __name__ == "__main__":
    unittest.main()
