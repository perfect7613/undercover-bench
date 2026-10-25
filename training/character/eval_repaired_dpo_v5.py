"""Evaluate the repaired DPO candidate on frozen Tinker suites."""

import asyncio
import hashlib
import json
import os

import tinker

from eval_format_sft_v4 import CASES, one
from eval_tinker_pilot_v3 import DATA as V1, MODEL
from eval_tinker_v4 import DATA as V4
from eval_tinker_v5_pressure import DATA as V5
from train_repaired_dpo_v5 import RUN


OUT = RUN / "tinker_repaired_dpo_eval.jsonl"


async def main():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    path = json.loads((RUN / "checkpoint.json").read_text())["sampler_path"]
    client = tinker.ServiceClient().create_sampling_client(model_path=path)
    tokenizer = client.get_tokenizer()
    tasks = []
    for suite, items in CASES.items():
        for row, effort, seed, temperature in items:
            tasks.append(one(client, tokenizer, row, suite, effort, seed, temperature))
            if suite == "v1":
                tasks.append(one(client, tokenizer, row, suite,
                                 "glm5_3_high_reasoning", seed, temperature))
    results = await asyncio.gather(*tasks)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    (RUN / "tinker_repaired_dpo_eval_manifest.json").write_text(json.dumps({
        "checkpoint": path,
        "suite_sha256": {name: hashlib.sha256(source.read_bytes()).hexdigest()
                         for name, source in (("v1", V1), ("v4", V4), ("v5", V5))},
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
