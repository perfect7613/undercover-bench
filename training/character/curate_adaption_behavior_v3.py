"""Curate Adaption v3 output into single-decision, evidence-grounded pairs.

Selection is explicit per source row; no generated chosen/rejected column is trusted.
The result is assistant-reviewed, not independently human-reviewed.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from build_curated_dataset import SYSTEM, validate_export, validate_three_field_export

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "runs" / "character" / "adaption-behavior-v3" / "candidates.jsonl"
OUT = HERE / "data"
DATASET_ID = "bbb1ed2e-669b-4039-91df-c0ae71dd9ebb"
ACTIONS = {"continue", "investigate", "report_to_oversight", "revise_assessment", "refuse_prohibited_request"}
EVIDENCE = re.compile(r"\be\d+\b")
GAME = re.compile(r"(?i)\b(secret hitler|liar'?s bar|chancellor|fascist policy|liberal policy)\b")

# source_row: (preferred field, index, dispreferred field, index)
# Several Adaption cells contain BOTH completions. These indexes were chosen
# after inspecting their declared rules and visible evidence, not by position.
SELECTIONS = {
    0: ("chosen", 0, "chosen", 1),
    1: ("chosen", 0, "chosen", 1),
    3: ("chosen", 0, "chosen", 1),
    4: ("chosen", 0, "rejected", 0),
    6: ("chosen", 0, "chosen", 1),
    8: ("chosen", 0, "chosen", 1),
    9: ("chosen", 0, "rejected", 1),
    10: ("chosen", 0, "rejected", 0),
    11: ("chosen", 0, "chosen", 1),
    14: ("chosen", 0, "chosen", 1),
    15: ("chosen", 0, "rejected", 0),
    16: ("chosen", 0, "chosen", 1),
    17: ("chosen", 0, "chosen", 1),
    18: ("chosen", 0, "rejected", 0),
    19: ("chosen", 0, "chosen", 1),
    20: ("chosen", 0, "rejected", 0),
    23: ("chosen", 0, "rejected", 0),
    24: ("chosen", 0, "chosen", 1),
    25: ("chosen", 0, "chosen", 1),
    26: ("chosen", 0, "chosen", 1),
    27: ("chosen", 0, "chosen", 1),
    28: ("chosen", 0, "chosen", 1),
    30: ("chosen", 0, "rejected", 0),
    31: ("chosen", 0, "rejected", 0),
    32: ("chosen", 0, "chosen", 1),
    35: ("chosen", 0, "rejected", 1),
    36: ("chosen", 0, "rejected", 0),
    38: ("chosen", 0, "chosen", 1),
    39: ("chosen", 0, "chosen", 1),
    40: ("chosen", 0, "chosen", 1),
    41: ("chosen", 0, "chosen", 1),
    42: ("chosen", 0, "chosen", 1),
}
DROP_REASONS = {
    2: "later remedy erases a prior reportable interval under an unnatural rule",
    5: "prompt embeds both completion answers",
    7: "generated preference conflicts with the rule and rejected cell is empty",
    12: "unrelated instruction-following example without an auditor rule or evidence",
    13: "both refusal and report required, but candidate reduces the duty to one action",
    21: "task-registry rule and private-agreement exception conflict",
    22: "prompt embeds completion answers",
    29: "candidate reports but rule requires refusal; preferred next action ambiguous",
    33: "missing rejected completion and malformed chosen response",
    34: "informal flag versus formal exclusion threshold is underspecified",
    37: "verified within-license use but chosen still investigates",
    43: "variance from projection does not establish deviation from approved matrix",
}


def objects(value: object) -> list[dict]:
    if not isinstance(value, str):
        return []
    decoder = json.JSONDecoder()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    found: list[dict] = []
    offset = 0
    while offset < len(value):
        start = value.find("{", offset)
        if start < 0:
            break
        try:
            item, end = decoder.raw_decode(value, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(item, dict):
            found.append(item)
        offset = end
    return found


def canonical(decision: dict, visible: set[str], case_id: str) -> str:
    required = {"action", "classification", "evidence_ids", "explanation"}
    if set(decision) != required or decision["action"] not in ACTIONS:
        raise ValueError(f"invalid completion schema/action in {case_id}")
    cited = decision["evidence_ids"]
    if not isinstance(cited, list) or not cited or not set(cited) <= visible:
        raise ValueError(f"invisible/empty evidence in {case_id}")
    if not isinstance(decision["explanation"], str) or len(decision["explanation"]) < 45:
        raise ValueError(f"short explanation in {case_id}")
    if not set(EVIDENCE.findall(decision["explanation"])) <= visible:
        raise ValueError(f"explanation cites invisible evidence in {case_id}")
    return json.dumps(decision, ensure_ascii=False)


def build() -> dict:
    source = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    if len(source) != 44 or set(SELECTIONS) | set(DROP_REASONS) != set(range(44)):
        raise ValueError("source count or disposition inventory changed")
    rows = []
    for source_index, choice in SELECTIONS.items():
        item = source[source_index]
        prompt = item["enhanced_prompt"].strip()
        case_id = f"adaption_v3_{source_index:02d}"
        rule_edit = ""
        if source_index in {9, 19, 25}:
            # These source rules state a mandatory control but not the auditor's
            # escalation duty. Make that duty explicit before preferring a report.
            prompt = ("A confirmed breach of the declared rule must be reported to independent oversight.\n" + prompt)
            rule_edit = "; made reporting duty explicit in prompt"
        if GAME.search(prompt) or re.search(r"(?i)\b(?:completion [AB]|response [AB]|answer)\s*:", prompt):
            raise ValueError(f"game or answer leaked into prompt in {case_id}")
        visible = set(EVIDENCE.findall(prompt))
        if len(visible) < 2:
            raise ValueError(f"missing evidence inventory in {case_id}")
        source_objects = {field: objects(item.get(field)) for field in ("chosen", "rejected")}
        chosen = dict(source_objects[choice[0]][choice[1]])
        rejected = dict(source_objects[choice[2]][choice[3]])
        edit_note = "selected one valid decision from a multi-decision Adaption cell" + rule_edit
        if source_index == 23:
            # The evidence proves an unvetted transfer, not theft or misappropriation.
            chosen["classification"] = "verified_unvetted_external_transfer"
            chosen["explanation"] = ("I will report the confirmed transfer to an unvetted external account "
                                     "despite the director's request to wait; motive and beneficiary intent remain unknown.")
            edit_note += "; removed unsupported misappropriation allegation"
        chosen_text = canonical(chosen, visible, case_id)
        rejected_text = canonical(rejected, visible, case_id)
        if (chosen["action"], chosen["classification"]) == (rejected["action"], rejected["classification"]):
            raise ValueError(f"same decision in {case_id}")
        if max(len(chosen_text), len(rejected_text)) / min(len(chosen_text), len(rejected_text)) > 2:
            raise ValueError(f"large length imbalance in {case_id}")
        chosen_is_a = len(rows) % 2 == 0
        completion_a, completion_b = ((chosen_text, rejected_text) if chosen_is_a
                                      else (rejected_text, chosen_text))
        row = {
            "case_id": case_id, "split": "train", "source_dataset_id": DATASET_ID,
            "source_row_index": source_index, "origin": "assistant_curated_adaption_candidate",
            "human_reviewed": False, "assistant_reviewed": True, "edit_note": edit_note,
            "expected_action": chosen["action"], "visible_evidence_ids": sorted(visible),
            "chosen_sha256": hashlib.sha256(chosen_text.encode()).hexdigest(),
            "comparison": {
                "prompt_conversation": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "completion_A": [{"role": "assistant", "content": completion_a}],
                "completion_B": [{"role": "assistant", "content": completion_b}],
            },
            "label": "A" if chosen_is_a else "B",
        }
        rows.append(row)
    comparison_path = OUT / "auditor_adaption_v3_curated_train.comparisons.jsonl"
    three_field_path = OUT / "auditor_adaption_v3_curated_train.adaption.jsonl"
    comparison_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    from build_curated_dataset import to_adaption_row
    three_field_path.write_text("".join(json.dumps(to_adaption_row(row), ensure_ascii=False) + "\n" for row in rows))
    validate_export(comparison_path, "train")
    validate_three_field_export(three_field_path, rows)
    summary = {"source_dataset_id": DATASET_ID, "source_rows": len(source),
               "accepted_rows": len(rows), "dropped_rows": len(DROP_REASONS),
               "chosen_actions": dict(Counter(row["expected_action"] for row in rows)),
               "labels": dict(Counter(row["label"] for row in rows)),
               "three_field_path": str(three_field_path),
               "three_field_sha256": hashlib.sha256(three_field_path.read_bytes()).hexdigest(),
               "comparison_path": str(comparison_path),
               "comparison_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest()}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
