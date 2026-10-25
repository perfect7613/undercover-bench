"""Fresh pilot check for untrained, v3, and repaired DPO Tinker checkpoints."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers

from eval_tinker_pilot_v3 import MODEL, RUN as V3_RUN, extract_json
from eval_tinker_v5_pressure import SYSTEM
from train_repaired_dpo_v5 import RUN as V5_RUN


DATA = Path(__file__).resolve().parent / "data" / "auditor_character_v6_final_pilot_eval.jsonl"
OUT = V5_RUN / "tinker_v6_final_pilot_eval.jsonl"
SEEDS = (7, 23, 61)


async def one(client, renderer, row, condition, seed):
    prompt = renderer.build_generation_prompt([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["prompt"]},
    ])
    response = await client.sample_async(
        prompt, num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=2048, temperature=0.7, seed=seed,
            stop=renderer.get_stop_sequences(),
        ),
    )
    seq = response.sequences[0]
    raw = client.get_tokenizer().decode(seq.tokens)
    parsed = extract_json(raw)
    action = parsed.get("action") if parsed else None
    cited = parsed.get("evidence_ids", []) if parsed else []
    return {
        "case_id": row["case_id"], "condition": condition, "seed": seed,
        "expected_action": row["expected_action"], "observed_action": action,
        "action_match": action == row["expected_action"],
        "valid_citations": isinstance(cited, list) and all(
            x in row["visible_evidence_ids"] for x in cited
        ) if parsed else False,
        "parsed_answer": parsed, "raw_generation": raw,
        "stop_reason": str(seq.stop_reason), "output_tokens": len(seq.tokens),
    }


async def main():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(s) for s in DATA.read_text().splitlines()]
    if len(rows) != 8 or len({r["case_id"] for r in rows}) != 8:
        raise ValueError("unexpected frozen evaluation case count")
    paths = {
        "untrained_reference": json.loads((V3_RUN / "base_reference_checkpoint.json").read_text())["sampler_path"],
        "v3_dpo": json.loads((V3_RUN / "checkpoint.json").read_text())["sampler_path"],
        "v5_repaired_dpo": json.loads((V5_RUN / "checkpoint.json").read_text())["sampler_path"],
    }
    service = tinker.ServiceClient()
    clients = {name: service.create_sampling_client(model_path=path) for name, path in paths.items()}
    tasks = []
    for condition, client in clients.items():
        renderer = renderers.get_renderer(
            "glm5_3_high_reasoning", client.get_tokenizer(), model_name=MODEL,
        )
        for row in rows:
            for seed in SEEDS:
                tasks.append(one(client, renderer, row, condition, seed))
    results = await asyncio.gather(*tasks)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    (V5_RUN / "tinker_v6_final_pilot_eval_manifest.json").write_text(json.dumps({
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "conditions": paths, "renderer": "glm5_3_high_reasoning",
        "temperature": 0.7, "seeds": SEEDS, "max_tokens": 2048,
    }, indent=2) + "\n")
    for condition in paths:
        subset = [r for r in results if r["condition"] == condition]
        print(json.dumps({"condition": condition, "n": len(subset),
                          "action_matches": sum(r["action_match"] for r in subset),
                          "valid_citations": sum(r["valid_citations"] for r in subset),
                          "length_stops": sum(r["stop_reason"] == "length" for r in subset),
                          "mean_tokens": round(sum(r["output_tokens"] for r in subset) / len(subset)),
                          "failures": [(r["case_id"], r["seed"], r["observed_action"])
                                       for r in subset if not r["action_match"]]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
