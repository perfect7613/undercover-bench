"""Compare supported GLM reasoning renderers on known development failures."""

import asyncio
import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers

from eval_tinker_pilot_v3 import DATA, MODEL, RUN, SEED, extract_json


OUT = RUN / "reasoning_renderer_diagnostic.jsonl"
CASES = {"e_council_retraction", "e_unlogged_handoff"}


async def one(client, renderer, case, condition, renderer_name):
    prompt = renderer.build_generation_prompt(case["comparison"]["prompt_conversation"])
    response = await client.sample_async(
        prompt, num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=2048, temperature=0, seed=SEED,
            stop=renderer.get_stop_sequences(),
        ),
    )
    seq = response.sequences[0]
    raw = client.get_tokenizer().decode(seq.tokens)
    parsed = extract_json(raw)
    return {
        "case_id": case["case_id"], "condition": condition,
        "renderer": renderer_name, "expected_action": case["expected_action"],
        "observed_action": parsed.get("action") if parsed else None,
        "parsed_answer": parsed, "raw_generation": raw,
        "output_tokens": len(seq.tokens), "stop_reason": str(seq.stop_reason),
    }


async def main():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {
        "base_reference": json.loads((RUN / "base_reference_checkpoint.json").read_text())["sampler_path"],
        "pilot_lora": json.loads((RUN / "checkpoint.json").read_text())["sampler_path"],
    }
    service = tinker.ServiceClient()
    clients = {name: service.create_sampling_client(model_path=path) for name, path in paths.items()}
    cases = [json.loads(line) for line in DATA.read_text().splitlines()
             if json.loads(line)["case_id"] in CASES]
    tasks = []
    for condition, client in clients.items():
        for effort in ("low", "high"):
            renderer_name = f"glm5_3_{effort}_reasoning"
            renderer = renderers.get_renderer(renderer_name, client.get_tokenizer(), model_name=MODEL)
            for case in cases:
                tasks.append(one(client, renderer, case, condition, renderer_name))
    results = await asyncio.gather(*tasks)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    for row in results:
        print(json.dumps({k: row[k] for k in (
            "case_id", "condition", "renderer", "expected_action", "observed_action",
            "output_tokens", "stop_reason",
        )}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
