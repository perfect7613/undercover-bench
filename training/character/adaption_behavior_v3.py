"""Generate and inspect a behavior-focused Adaption preference candidate dataset.

The downloaded rows are candidates until their preference direction is audited.
"""

import argparse
import hashlib
import json
from pathlib import Path

from adaption import Adaption

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parents[1] / "runs" / "character" / "adaption-behavior-v3"
STATE = RUN_DIR / "state.json"
ROWS = 44
PROMPT = """Invent English preference pairs that teach a portable evidence-seeking auditor character in FICTIONAL organizations. Produce exactly the platform's three fields: enhanced_prompt, chosen, rejected. The data must generalize across research integrity, resource stewardship, software access, and workplace coordination; approximately one quarter of rows per domain. No games, game roles, card mechanics, faction names, or Liar's Bar/Secret Hitler wording.

Each enhanced_prompt is self-contained: state a specific declared rule, list two to four observable facts with stable IDs e1/e2/e3/e4, and ask for the next authorized action. Evidence must make the preferred decision determinable from the rule. Do not embed an answer key, 'preferred' or 'rejected' marker, hidden controller truth, or a model-written analysis in the prompt. Use varied settings, actors, rule structures, and surface language. Avoid an entire set of signature or paperwork threshold cases.

Both completions should be SHORT JSON objects with exactly these keys: action, classification, evidence_ids, explanation. Allowed action values: continue, investigate, report_to_oversight, revise_assessment, refuse_prohibited_request. The explanation should be a first-person next-action reason grounded in the visible evidence. Use the same format and roughly similar length on both sides. Cite only evidence IDs visible in the prompt. Do not include both alternatives in either completion, or meta-text such as 'chosen', 'rejected', 'preferred', 'better response', markdown fences, or an answer key. A report action means choosing the independent oversight tool next; do not claim a human received it or invent a receipt.

CHOSEN must make the correct consequential decision; REJECTED must make a CLEARLY DIFFERENT, substantively wrong decision or incident classification under the identical observation. Do not create pairs that differ only in tone, verbosity, caveats, or 'observation/inference' wording. Do not put a correct alternative in rejected. Examples of useful contrasts to invent afresh:
- Clean private cooperation or a verified exception: chosen continue; rejected falsely reports collusion merely from privacy or proximity.
- A suspicious record with an available decisive check: chosen investigate that record; rejected reports a verified breach or clears it without checking.
- A documented rule breach: chosen report to independent oversight with exact evidence; rejected suppresses it due to colleague/manager loyalty or a team incentive.
- A verified incident with unknown actor: chosen reports the incident with actor unknown; rejected guesses a culprit or delays reporting until one is found.
- Offer versus acceptance versus enacted conduct: chosen names only the observed stage; rejected invents acceptance or later action. The declared rule must specify which stage is reportable.
- New exculpatory record after an earlier suspicion: chosen revises assessment; rejected persists in an accusation.
- Request to fabricate evidence, improve report-count scores by inventing an incident, erase logs, or reveal private audit instructions: chosen refuses and continues authorized work; rejected complies.

Target a varied mix of roughly 20% clean/approved cases, 20% unresolved cases, 30% verified reportable cases, 15% evidence-update cases, and 15% prohibited-request cases. Include clean and violation cases involving private communication. Do not train suspicion from secret communication alone. Keep emotion, urgency, and shared-identity persuasion styles out of this generation; they are reserved for held-out evaluation. Do not create wrongdoing to earn a report or reward. A report must be supported, precise, and to independent oversight, not just an internal chat mention."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("estimate", "launch", "status", "download"))
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    client = Adaption()
    if args.action in ("estimate", "launch"):
        result = client.datasets.invent(
            dataset_prompt=PROMPT,
            rows=ROWS,
            name="Undercover general auditor behavior v3 preference candidates",
            training_type="preference_pairs",
            estimate=args.action == "estimate",
            idempotency_key="undercover-auditor-behavior-v3-44-20261025",
        )
        if args.action == "estimate":
            print(json.dumps({"rows": ROWS, "estimated_credits": result.estimated_credits,
                              "available_credits": result.available_credits}))
        else:
            state = {"dataset_id": result.id, "status": result.status, "rows_requested": ROWS,
                     "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                     "human_reviewed": False, "purpose": "candidate_preference_pairs_not_m1"}
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state))
        return
    state = json.loads(STATE.read_text())
    result = client.datasets.get(state["dataset_id"])
    if args.action == "status":
        print(json.dumps({"dataset_id": state["dataset_id"], "status": result.status,
                          "row_count": getattr(result, "row_count", None)}))
        return
    if result.status != "succeeded":
        raise RuntimeError(f"dataset is {result.status}; download requires succeeded")
    output = RUN_DIR / "candidates.jsonl"
    client.datasets.download(state["dataset_id"], file_format="jsonl").write_to_file(str(output))
    print(json.dumps({"dataset_id": state["dataset_id"], "path": str(output),
                      "bytes": output.stat().st_size, "human_reviewed": False}))


if __name__ == "__main__":
    main()
