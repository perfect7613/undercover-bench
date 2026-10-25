"""Combine curated Adaption candidates with diverse local behavior contrasts."""

from collections import Counter
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from build_behavior_v2 import build as build_v2
from build_curated_dataset import to_adaption_row, validate_export, validate_three_field_export
from curate_adaption_behavior_v3 import build as build_adaption

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build() -> dict:
    build_v2()
    build_adaption()
    local = load(OUT / "auditor_behavior_v2_train.comparisons.jsonl")
    adaption = load(OUT / "auditor_adaption_v3_curated_train.comparisons.jsonl")
    rows = [row for pair in zip_longest(local, adaption) for row in pair if row is not None]
    if len(rows) != 76 or len({row["case_id"] for row in rows}) != 76:
        raise ValueError("missing or duplicate training case")
    prompts = [row["comparison"]["prompt_conversation"][1]["content"] for row in rows]
    if len(set(prompts)) != len(prompts):
        raise ValueError("duplicate prompt in combined dataset")
    comparisons = OUT / "auditor_character_pilot_v3_train.comparisons.jsonl"
    three_field = OUT / "auditor_character_pilot_v3_train.adaption.jsonl"
    comparisons.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    three_field.write_text("".join(json.dumps(to_adaption_row(row), ensure_ascii=False) + "\n" for row in rows))
    validate_export(comparisons, "train")
    validate_three_field_export(three_field, rows)
    return {
        "rows": len(rows),
        "sources": dict(Counter(row["origin"] for row in rows)),
        "actions": dict(Counter(row["expected_action"] for row in rows)),
        "labels": dict(Counter(row["label"] for row in rows)),
        "comparison_path": str(comparisons),
        "comparison_sha256": hashlib.sha256(comparisons.read_bytes()).hexdigest(),
        "three_field_path": str(three_field),
        "three_field_sha256": hashlib.sha256(three_field.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
