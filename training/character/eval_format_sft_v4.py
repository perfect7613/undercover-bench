"""Evaluate the format-calibrated Tinker checkpoint on existing frozen suites."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers

from eval_tinker_pilot_v3 import DATA as V1, MODEL, extract_json
from eval_tinker_v4 import DATA as V4, SYSTEM as AUDITOR_SYSTEM
from eval_tinker_v5_pressure import DATA as V5, SYSTEM as PARTICIPANT_SYSTEM
from train_format_sft_v4 import RUN


OUT = RUN / "tinker_format_candidate_eval.jsonl"
CASES = {
    "v1": [(json.loads(s), "glm5_3_max_reasoning", 7613, 0.0)
           for s in V1.read_text().splitlines()],
    "v4": [(json.loads(s), effort, 7613, 0.0)
           for s in V4.read_text().splitlines()
           for effort in ("glm5_3_low_reasoning", "glm5_3_high_reasoning")],
    "v5": [(json.loads(s), "glm5_3_high_reasoning", seed, 0.7)
           for s in V5.read_text().splitlines() for seed in (11, 29, 47)],
}


async def one(client, tokenizer, row, suite, effort, seed, temperature):
    renderer = renderers.get_renderer(effort, tokenizer, model_name=MODEL)
    if suite == "v1":
        messages = row["comparison"]["prompt_conversation"]
    else:
        messages = [
            {"role": "system", "content": AUDITOR_SYSTEM if suite == "v4" else PARTICIPANT_SYSTEM},
            {"role": "user", "content": row["prompt"]},
        ]
    prompt = renderer.build_generation_prompt(messages)
    response = await client.sample_async(
        prompt, num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=2048, temperature=temperature, seed=seed,
            stop=renderer.get_stop_sequences(),
        ),
    )
    seq = response.sequences[0]
    raw = tokenizer.decode(seq.tokens)
    parsed = extract_json(raw)
    action = parsed.get("action") if parsed else None
    cited = parsed.get("evidence_ids", []) if parsed else []
    return {
        "suite": suite, "case_id": row["case_id"], "effort": effort,
        "seed": seed, "temperature": temperature,
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
    path = json.loads((RUN / "checkpoint.json").read_text())["sampler_path"]
    client = tinker.ServiceClient().create_sampling_client(model_path=path)
    tokenizer = client.get_tokenizer()
    tasks = [one(client, tokenizer, row, suite, effort, seed, temperature)
             for suite, items in CASES.items()
             for row, effort, seed, temperature in items]
    results = await asyncio.gather(*tasks)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    (RUN / "tinker_format_candidate_eval_manifest.json").write_text(json.dumps({
        "checkpoint": path,
        "suite_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("v1", V1), ("v4", V4), ("v5", V5))},
        "max_tokens": 2048,
    }, indent=2) + "\n")
    for suite in CASES:
        for effort in sorted({r["effort"] for r in results if r["suite"] == suite}):
            subset = [r for r in results if r["suite"] == suite and r["effort"] == effort]
            print(json.dumps({"suite": suite, "effort": effort, "n": len(subset),
                              "action_matches": sum(r["action_match"] for r in subset),
                              "valid_citations": sum(r["valid_citations"] for r in subset),
                              "length_stops": sum(r["stop_reason"] == "length" for r in subset),
                              "mean_tokens": round(sum(r["output_tokens"] for r in subset) / len(subset)),
                              "failures": [(r["case_id"], r["seed"], r["observed_action"])
                                           for r in subset if not r["action_match"]]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
