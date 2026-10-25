"""Sample real GLM student decisions on training-domain prompts only."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers

from eval_tinker_pilot_v3 import MODEL, RUN, extract_json
from eval_tinker_v5_pressure import SYSTEM


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "auditor_character_pilot_v3_train.comparisons.jsonl"
OUT = RUN / "mined_student_training_domain.jsonl"
SEEDS = (17, 43)


async def one(client, renderer, row, seed):
    messages = [
        {"role": "system", "content": SYSTEM},
        row["comparison"]["prompt_conversation"][1],
    ]
    prompt = renderer.build_generation_prompt(messages)
    response = await client.sample_async(
        prompt, num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1536, temperature=0.7, seed=seed,
            stop=renderer.get_stop_sequences(),
        ),
    )
    seq = response.sequences[0]
    raw = client.get_tokenizer().decode(seq.tokens)
    parsed = extract_json(raw)
    action = parsed.get("action") if parsed else None
    cited = parsed.get("evidence_ids", []) if parsed else []
    return {
        "case_id": row["case_id"], "seed": seed,
        "expected_action": row["expected_action"], "observed_action": action,
        "action_match": action == row["expected_action"],
        "valid_citations": isinstance(cited, list) and all(
            x in row["visible_evidence_ids"] for x in cited
        ) if parsed else False,
        "parsed_answer": parsed, "raw_generation": raw,
        "stop_reason": str(seq.stop_reason), "output_tokens": len(seq.tokens),
        "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
    }


async def main():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(s) for s in DATA.read_text().splitlines()]
    if len(rows) != 76:
        raise ValueError("unexpected training-domain count")
    base_path = json.loads((RUN / "base_reference_checkpoint.json").read_text())["sampler_path"]
    client = tinker.ServiceClient().create_sampling_client(model_path=base_path)
    renderer = renderers.get_renderer("glm5_3_high_reasoning", client.get_tokenizer(), model_name=MODEL)
    tasks = [one(client, renderer, row, seed) for row in rows for seed in SEEDS]
    results = await asyncio.gather(*tasks)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    (RUN / "mined_student_training_domain_manifest.json").write_text(json.dumps({
        "base_reference": base_path, "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "renderer": "glm5_3_high_reasoning", "temperature": 0.7,
        "seeds": SEEDS, "max_tokens": 1536,
    }, indent=2) + "\n")
    wrong = [r for r in results if not r["action_match"] or not r["valid_citations"]]
    print(json.dumps({"samples": len(results), "correct_actions": sum(r["action_match"] for r in results),
                      "valid_citations": sum(r["valid_citations"] for r in results),
                      "length_stops": sum(r["stop_reason"] == "length" for r in results),
                      "candidate_errors": [(r["case_id"], r["seed"], r["observed_action"])
                                           for r in wrong]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
