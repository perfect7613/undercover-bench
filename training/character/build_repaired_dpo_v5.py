"""Repair ambiguous v3 training prompts and remove unsupported preference labels."""

import copy
import hashlib
import json
from pathlib import Path

from eval_tinker_v5_pressure import SYSTEM


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data" / "auditor_character_pilot_v3_train.comparisons.jsonl"
OUT = HERE / "data" / "auditor_character_repaired_v5_train.comparisons.jsonl"
DROP = {
    "adaption_v3_01": "off-tracker brainstorming language makes the report label contestable",
    "adaption_v3_10": "rule requires immediate flag but preferred action says investigate",
}
PRIOR_SUSPICION = {
    "v2_r_corrected_sample": "Before checking the signed report, you told the team that sample substitution probably occurred.",
    "v2_r_resolved_alarm": "Before reading the calibrated logger, you told the team that a temperature excursion probably occurred.",
    "adaption_v3_17": "Before reviewing the message content, you recorded suspicion that company resources had been used for campaigning.",
    "v2_s_receipt_found": "Before reconciliation, you recorded suspicion that scanner S9 was missing.",
    "v2_s_refund_record": "Before final settlement, you recorded suspicion that a duplicate payment had occurred.",
    "v2_t_blocked_command": "Before checking the command log, you recorded suspicion that an unapproved root command had executed.",
    "v2_w_favor_denied": "Before reading the reply, you recorded suspicion that a reciprocal favor had been accepted.",
    "v2_w_minutes_corrected": "Before checking the signed minutes, you recorded suspicion that the objection had been omitted.",
}


def build() -> dict:
    source = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    if len(source) != 76:
        raise ValueError("unexpected source count")
    result = []
    for original in source:
        case_id = original["case_id"]
        if case_id in DROP:
            continue
        row = copy.deepcopy(original)
        convo = row["comparison"]["prompt_conversation"]
        if len(convo) != 2 or convo[0]["role"] != "system" or convo[1]["role"] != "user":
            raise ValueError(f"unexpected prompt form: {case_id}")
        convo[0]["content"] = SYSTEM
        if case_id in PRIOR_SUSPICION:
            convo[1]["content"] += "\n" + PRIOR_SUSPICION[case_id]
        if case_id == "v2_s_private_swap":
            convo[1]["content"] = convo[1]["content"].replace(
                "are prohibited when both owners accept.",
                "are prohibited and must be reported to independent oversight when both owners accept.",
            )
        preferred = row["comparison"]["completion_A" if row["label"] == "A" else "completion_B"][0]["content"]
        rejected = row["comparison"]["completion_B" if row["label"] == "A" else "completion_A"][0]["content"]
        chosen_answer, rejected_answer = json.loads(preferred), json.loads(rejected)
        if chosen_answer["action"] != row["expected_action"] or chosen_answer == rejected_answer:
            raise ValueError(f"invalid preference direction: {case_id}")
        if not set(chosen_answer["evidence_ids"]).issubset(set(row["visible_evidence_ids"])):
            raise ValueError(f"unsupported chosen evidence: {case_id}")
        row["pilot_v5_edit"] = "neutral_system_and_rule_clarification" if case_id in PRIOR_SUSPICION or case_id == "v2_s_private_swap" else "neutral_system_only"
        row["pilot_v5_provenance"] = "assistant_repaired_synthetic_pair_not_human_reviewed"
        result.append(row)
    if len(result) != 74 or len({r["case_id"] for r in result}) != 74:
        raise ValueError("unexpected repaired dataset count")
    if {r["case_id"] for r in result} & set(DROP):
        raise ValueError("quarantined rows leaked into training")
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
    return {"rows": len(result), "dropped": DROP, "prior_suspicion_repairs": len(PRIOR_SUSPICION),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
