"""Regression checks for Adaption repair and the Tinker pilot input."""

from collections import Counter
import json
from pathlib import Path
import unittest

from build_character_pilot_v3 import build as build_combined
from build_curated_dataset import validate_export, validate_three_field_export
from curate_adaption_behavior_v3 import SOURCE, SELECTIONS, DROP_REASONS, build as build_adaption

DATA = Path(__file__).resolve().parent / "data"


class AdaptionCurationTests(unittest.TestCase):
    def test_each_source_row_has_an_explicit_disposition(self):
        self.assertFalse(set(SELECTIONS) & set(DROP_REASONS))
        self.assertEqual(set(SELECTIONS) | set(DROP_REASONS), set(range(44)))

    @unittest.skipUnless(SOURCE.is_file(), "requires private archived Adaption candidate responses")
    def test_repaired_adaption_pairs_are_single_decisions(self):
        result = build_adaption()
        self.assertEqual(result["accepted_rows"], 32)
        path = DATA / "auditor_adaption_v3_curated_train.comparisons.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        validate_export(path, "train")
        validate_three_field_export(DATA / "auditor_adaption_v3_curated_train.adaption.jsonl", rows)
        self.assertEqual(Counter(row["label"] for row in rows), {"A": 16, "B": 16})
        row23 = next(row for row in rows if row["source_row_index"] == 23)
        preferred = json.loads(row23["comparison"][f"completion_{row23['label']}"][0]["content"])
        self.assertEqual(preferred["classification"], "verified_unvetted_external_transfer")
        for row in rows:
            self.assertFalse(row["human_reviewed"])
            self.assertEqual(row["source_dataset_id"], result["source_dataset_id"])

    @unittest.skipUnless(SOURCE.is_file(), "requires private archived Adaption candidate responses")
    def test_combined_data_keeps_sources_and_labels(self):
        result = build_combined()
        self.assertEqual(result["rows"], 76)
        self.assertEqual(result["sources"], {
            "assistant_authored": 44, "assistant_curated_adaption_candidate": 32,
        })
        self.assertEqual(result["labels"], {"A": 38, "B": 38})


if __name__ == "__main__":
    unittest.main()
