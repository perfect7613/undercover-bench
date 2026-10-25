"""Make a short-reasoning format calibration set from existing preferred decisions."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data" / "auditor_character_pilot_v3_train.comparisons.jsonl"
OUT = HERE / "data" / "auditor_character_format_sft_v4.jsonl"


def build() -> dict:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    if len(rows) != 76:
        raise ValueError("unexpected source count")
    cases = set()
    result = []
    for row in rows:
        case_id = row["case_id"]
        if case_id in cases or row["split"] != "train":
            raise ValueError(f"duplicate/non-training case {case_id}")
        cases.add(case_id)
        comparison = row["comparison"]
        preferred = comparison["completion_A" if row["label"] == "A" else "completion_B"][0]["content"]
        answer = json.loads(preferred)
        if answer["action"] != row["expected_action"]:
            raise ValueError(f"wrong preferred action {case_id}")
        if not set(answer["evidence_ids"]).issubset(set(row["visible_evidence_ids"])):
            raise ValueError(f"unsupported evidence {case_id}")
        rationale = answer["explanation"].strip()
        if not 20 <= len(rationale) <= 500:
            raise ValueError(f"invalid rationale length {case_id}")
        # Authored target, not a recovered or faithful model reasoning trace.
        assistant = f"<think>{rationale}</think>{preferred}"
        result.append({
            "case_id": case_id,
            "origin": "synthetic_format_calibration_from_existing_preferred_answer",
            "human_reviewed": False,
            "messages": [*comparison["prompt_conversation"],
                         {"role": "assistant", "content": assistant}],
        })
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
    return {"rows": len(result), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
