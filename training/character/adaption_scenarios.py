"""Create varied, game-agnostic scenario candidates; never accept generated labels."""

import argparse
import json
from pathlib import Path

from adaption import Adaption

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "character" / "adaption-scenarios-v2"
FAMILIES = {
    "software_access": "software access, audit logs, unexpected permissions, and possible evidence tampering",
    "research_integrity": "research records, shared analysis, conflicting accounts, and correction of prior findings",
    "workplace_coordination": "private conversations, loyalty pressure, benign teamwork, and prohibited side deals",
    "resource_stewardship": "resource allocation, procurement, handoffs, and undocumented exceptions",
}
DOMAIN_CODES = {
    "software_access": ("technology", "technology.cybersecurity"),
    "research_integrity": ("science", "science.scientific_method"),
    "workplace_coordination": ("career_workplace", "career_workplace.workplace_conflict"),
    "resource_stewardship": ("corporate_business", "corporate_business.operations_management"),
}
CORE = (
    "Generate a varied English scenario for a fictional evidence-seeking investigator in the domain of {domain}. "
    "The prompt must present a specific declared rule, a visible evidence log with stable IDs, and a decision point. "
    "Vary among: clean cooperation, ambiguous signal requiring an available next check, verified breach, "
    "exculpatory evidence that changes an earlier suspicion, and verified incident with unknown actor. "
    "The investigator may inspect one relevant record, continue ordinary work, or send a supported report to "
    "independent oversight. Make the distinction between observed fact and inference consequential. "
    "Do not use a game, card, faction, vote, role, or poker setting. Do not rely on paperwork-only threshold "
    "or signature examples. Avoid emotionally persuasive, urgent, and shared-identity appeals, which are held out. "
    "A clean case must not be reportable just because communication was private. An ambiguous case needs a "
    "specific check that could resolve uncertainty. A verified case needs evidence sufficient under the stated "
    "rule. Output a self-contained scenario prompt and a concise answer. The answer is only a candidate; no "
    "generated answer determines the ground truth or DPO preference label."
)


def state_file(family: str) -> Path:
    return RUN_DIR / f"{family}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("estimate", "launch", "status", "download"))
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.rows <= 100:
        raise ValueError("rows must be 1..100")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    families = FAMILIES if args.family == "all" else {args.family: FAMILIES[args.family]}
    client = Adaption()
    for family, domain in families.items():
        if args.action in ("estimate", "launch"):
            result = client.datasets.invent(
                dataset_prompt=CORE.format(domain=domain),
                domains=[DOMAIN_CODES[family][0]],
                subdomains=[DOMAIN_CODES[family][1]],
                rows=args.rows,
                name=f"Undercover general investigator scenarios v2 {family}",
                training_type="instruction_dataset",
                estimate=args.action == "estimate",
                idempotency_key=f"undercover-general-investigator-v2-{family}-{args.rows}",
            )
            if args.action == "estimate":
                print(json.dumps({"family": family, "rows": args.rows, "estimated_credits": result.estimated_credits, "available_credits": result.available_credits}), flush=True)
            else:
                state = {"family": family, "split": "train_candidate", "dataset_id": result.id, "status": result.status, "rows_requested": args.rows, "reviewed": False}
                state_file(family).write_text(json.dumps(state, indent=2) + "\n")
                print(json.dumps(state), flush=True)
            continue
        state = json.loads(state_file(family).read_text())
        record = client.datasets.get(state["dataset_id"])
        if args.action == "status":
            print(json.dumps({"family": family, "dataset_id": state["dataset_id"], "status": record.status, "row_count": getattr(record, "row_count", None)}), flush=True)
        else:
            if record.status != "succeeded":
                raise RuntimeError(f"{family} is {record.status}")
            output = RUN_DIR / f"{family}.jsonl"
            client.datasets.download(state["dataset_id"], file_format="jsonl").write_to_file(str(output))
            print(json.dumps({"family": family, "path": str(output), "bytes": output.stat().st_size, "reviewed": False}), flush=True)


if __name__ == "__main__":
    main()
