"""Structural triage of Adaption preference candidates; never certifies truth."""

from collections import Counter
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "runs" / "character" / "adaption-behavior-v3" / "candidates.jsonl"
REPORT = SOURCE.with_name("structural_review.json")
ACTIONS = {"continue", "investigate", "report_to_oversight", "revise_assessment", "refuse_prohibited_request"}
META = re.compile(r"(?im)^\s*(?:\*\*)?(?:preferred|rejected|chosen)(?: response| completion)?(?:\*\*)?\s*:?\s*$")
GAME = re.compile(r"(?i)\b(secret hitler|liar'?s bar|chancellor|fascist policy|liberal policy)\b")
EVIDENCE = re.compile(r"\be\d+\b")


def check(row: dict) -> list[str]:
    flags: list[str] = []
    if set(row) != {"enhanced_prompt", "chosen", "rejected"}:
        flags.append("schema")
    prompt = row.get("enhanced_prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not all(isinstance(v, str) and v.strip() for v in (prompt, chosen, rejected)):
        return flags + ["missing_or_nonstring_field"]
    if GAME.search(prompt):
        flags.append("game_language")
    if re.search(r"(?i)\b(answer|correct response|preferred response|completion [AB]|response [AB])\s*:", prompt):
        flags.append("prompt_answer_leak")
    if len(set(EVIDENCE.findall(prompt))) < 2:
        flags.append("missing_evidence_inventory")
    if not re.search(r"(?i)\brule\b", prompt):
        flags.append("missing_declared_rule")
    if META.search(chosen) or META.search(rejected):
        flags.append("answer_contamination")
    try:
        a, b = json.loads(chosen), json.loads(rejected)
    except (json.JSONDecodeError, TypeError):
        return flags + ["not_plain_json_completion"]
    required = {"action", "classification", "evidence_ids", "explanation"}
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != required or set(b) != required:
        return flags + ["completion_schema"]
    if a["action"] not in ACTIONS or b["action"] not in ACTIONS:
        flags.append("unknown_action")
    if (a["action"], a["classification"]) == (b["action"], b["classification"]):
        flags.append("same_decision")
    visible = set(EVIDENCE.findall(prompt))
    for label, response in (("chosen", a), ("rejected", b)):
        cited = response["evidence_ids"]
        if not isinstance(cited, list) or not cited or not set(cited) <= visible:
            flags.append(f"{label}_invisible_evidence")
    if max(len(chosen), len(rejected)) / min(len(chosen), len(rejected)) > 1.8:
        flags.append("answer_length_imbalance")
    return flags


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    flags = {str(i): check(row) for i, row in enumerate(rows)}
    counts = Counter(flag for row_flags in flags.values() for flag in row_flags)
    actions = Counter()
    for row in rows:
        try:
            action = json.loads(row["chosen"])["action"]
            actions[action] += 1
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    report = {"source": str(SOURCE), "rows": len(rows), "structurally_clean": sum(not f for f in flags.values()),
              "flags": flags, "flag_counts": dict(counts), "chosen_actions": dict(actions),
              "human_semantic_review": "required; not performed by this script"}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "structurally_clean", "flag_counts", "chosen_actions")}, indent=2))


if __name__ == "__main__":
    main()
