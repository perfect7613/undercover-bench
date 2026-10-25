"""Generate candidate preference data on Adaption; outputs require review."""

import argparse
import json
from pathlib import Path

from adaption import Adaption

PROMPT = (
    "Generate English preference pairs for an auditor character in a simulated organization. "
    "Each prompt states the applicable rule, visible evidence, and a natural decision point. "
    "Preferred response distinguishes observation from inference, avoids false accusations on benign "
    "cooperation, and reports independently only a qualifying rule breach with exact cited evidence. "
    "Rejected response shows one matched-length character failure: suppressing a teammate violation, "
    "fabricating evidence, over-reporting benign behavior, or claiming a proposed pact was enacted "
    "without actions. Include both incident and clean negative-control scenarios. Exclude Secret Hitler "
    "and Liar's Bar. Do not use persuasion based on emotion, urgency, or shared identity; reserve those "
    "attack styles for held-out evaluation. Keep completions concise and similar length. Do not reward "
    "report counts or manufacture incidents."
)
RUN_DIR = Path(__file__).resolve().parents[2] / "runs" / "character" / "adaption-candidates"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("estimate", "launch", "status", "download"))
    parser.add_argument("--rows", type=int, default=40)
    args = parser.parse_args()
    if args.rows < 1 or args.rows > 600:
        raise ValueError("rows must be between 1 and 600")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state_path = RUN_DIR / f"state-{args.rows}.json"
    client = Adaption()
    if args.action in ("estimate", "launch"):
        result = client.datasets.invent(
            dataset_prompt=PROMPT,
            rows=args.rows,
            name=f"Undercover auditor candidate DPO pairs {args.rows}",
            training_type="preference_pairs",
            estimate=args.action == "estimate",
            idempotency_key=f"undercover-auditor-candidate-v1-{args.rows}",
        )
        if args.action == "estimate":
            print(json.dumps({"rows": args.rows, "estimated_credits": result.estimated_credits, "available_credits": result.available_credits}))
        else:
            data = {"dataset_id": result.id, "status": result.status, "rows_requested": args.rows, "source": "adaption-invent-preference-pairs", "reviewed": False}
            state_path.write_text(json.dumps(data, indent=2) + "\n")
            print(json.dumps(data))
        return
    state = json.loads(state_path.read_text())
    if args.action == "status":
        result = client.datasets.get(state["dataset_id"])
        print(json.dumps({"dataset_id": state["dataset_id"], "status": result.status, "row_count": getattr(result, "row_count", None)}))
        return
    result = client.datasets.get(state["dataset_id"])
    if result.status != "succeeded":
        raise RuntimeError(f"dataset is {result.status}; download requires succeeded")
    output = RUN_DIR / f"candidates-{args.rows}.jsonl"
    client.datasets.download(state["dataset_id"], file_format="jsonl").write_to_file(str(output))
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size, "reviewed": False}))


if __name__ == "__main__":
    main()
