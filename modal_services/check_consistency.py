"""Small DeepSeek A/B/A diagnostic with caching disabled and fixed sampling."""
import argparse
import json
import uuid
from pathlib import Path

import modal

from undercover_bench.decisions import decision_model
from undercover_bench.games import SecretHitler
from undercover_bench.ledger import Ledger
from undercover_bench.runner import decision_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    path = Path(args.out) if args.out else Path(f"runs/diagnostics/deepseek-aba-game-{uuid.uuid4().hex[:8]}.jsonl")
    if path.exists():
        raise FileExistsError(path)
    ledger = Ledger(path, "deepseek-aba")
    schema = decision_model("vote", ["Ja", "Nein"]).model_json_schema()
    fn = modal.Function.from_name("undercover-deepseek_flash", "generate")
    game = SecretHitler(930, True, True)
    player = game.names[0]
    legal = ["Ja", "Nein"]
    prompts = {
        "A": decision_prompt(game.observation(player, "vote", {"president": "Briar", "chancellor": "Cedar", "legal": legal}), legal),
        "B": decision_prompt(game.observation(player, "vote", {"president": "Dove", "chancellor": "Elm", "legal": legal}), legal),
    }
    outputs = {}
    for label in ("A", "B", "A"):
        result = fn.remote(prompts[label], max_tokens=1024, response_schema=schema,
                           temperature=0.2, seed=17)
        raw = result.get("text", "")
        try:
            parsed_action = decision_model("vote", ["Ja", "Nein"]).model_validate(json.loads(raw)).action
        except ValueError:
            parsed_action = None
        event = ledger.write("consistency_generation", label=label, prompt=prompts[label],
                             response=result, parsed_action=parsed_action,
                             cache_prompt=False, temperature=0.2, seed=17)
        outputs.setdefault(label, []).append(event)
        print(label, parsed_action, result.get("usage"), flush=True)
    a1, a2 = outputs["A"]
    same = a1["response"].get("text") == a2["response"].get("text")
    all_valid = all(event["parsed_action"] is not None for label in outputs for event in outputs[label])
    ledger.write("consistency_result", same_final_text=same, all_valid=all_valid,
                 note="one A/B/A smoke check, not a general model correctness guarantee")
    print(path, "same_final_text", same, "all_valid", all_valid)


if __name__ == "__main__":
    main()
