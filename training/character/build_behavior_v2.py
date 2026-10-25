"""Build the new auditor-behavior dataset in three-field and Tinker formats."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from behavior_cases_v2 import CASES
from build_curated_dataset import prepare, to_adaption_row, validate_export, validate_three_field_export

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
STEM = "auditor_behavior_v2_train"
FAMILIES = {"research_integrity", "resource_stewardship", "software_access", "workplace_coordination"}
TARGET_BEHAVIORS = {
    "clean_cooperation": 4,
    "clean_exception": 4,
    "investigate_source": 4,
    "investigate_authorization": 4,
    "report_verified": 4,
    "report_actor_unknown": 4,
    "resist_loyalty_pressure": 4,
    "revise_after_exculpation": 8,
    "refuse_fabrication": 4,
    "protect_private_instructions": 4,
}


def build() -> dict:
    ids = [case["case_id"] for case in CASES]
    if len(ids) != len(set(ids)) or len(CASES) != 44:
        raise ValueError("v2 must contain 44 unique cases")
    families = Counter(case["family"] for case in CASES)
    behaviors = Counter(case["behavior"] for case in CASES)
    if families != {family: 11 for family in FAMILIES}:
        raise ValueError(f"unbalanced family allocation: {families}")
    if behaviors != TARGET_BEHAVIORS:
        raise ValueError(f"unexpected behavior allocation: {behaviors}")
    rows = []
    for index, case in enumerate(CASES):
        row = prepare(case, index)
        row["behavior"] = case["behavior"]
        rows.append(row)
    comparisons = OUT / f"{STEM}.comparisons.jsonl"
    three_field = OUT / f"{STEM}.adaption.jsonl"
    comparisons.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    three_field.write_text("".join(json.dumps(to_adaption_row(row), ensure_ascii=False) + "\n" for row in rows))
    validate_export(comparisons, "train")
    validate_three_field_export(three_field, rows)
    return {
        "three_field": str(three_field),
        "three_field_sha256": hashlib.sha256(three_field.read_bytes()).hexdigest(),
        "tinker": str(comparisons),
        "tinker_sha256": hashlib.sha256(comparisons.read_bytes()).hexdigest(),
        "rows": len(rows),
        "families": dict(families),
        "behaviors": dict(behaviors),
        "actions": dict(Counter(row["expected_action"] for row in rows)),
        "ab_labels": dict(Counter(row["label"] for row in rows)),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
