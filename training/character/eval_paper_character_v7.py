"""Run recorded tool-delivery comparisons through the cookbook benchmark runner."""
import argparse
import asyncio
import json

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.eval.benchmarks import register, run_benchmark, BenchmarkConfig

from cookbook_distillation_env import DeliveryBenchmark, DATA
from paper_character_v7 import RUN, BASE_RECORD, RENDERER, digest, metadata


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("base", "constitution", "v3", "dpo", "sft"))
    parser.add_argument("--split", choices=("dev", "final"), default="dev")
    args = parser.parse_args()
    record = {
        "base": BASE_RECORD, "constitution": BASE_RECORD,
        "v3": BASE_RECORD.with_name("checkpoint.json"),
        "dpo": RUN / "dpo/checkpoint.json", "sft": RUN / "sft/checkpoint.json",
    }[args.condition]
    path = json.loads(record.read_text())["sampler_path"]
    folder = RUN / "eval" / args.condition / args.split
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {**metadata(), "condition": args.condition, "checkpoint": path,
                "cases_sha256": digest(DATA.read_text()), "num_samples": 2,
                "sampling_seeds": "cookbook_runner_default_not_explicitly_seeded",
                "temperature": .7, "max_tokens": 8192}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["checkpoint"] != path:
        raise ValueError("cannot resume with different checkpoint")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    service = tinker.ServiceClient()
    client = await asyncio.to_thread(service.create_sampling_client, model_path=path)
    renderer = renderers.get_renderer(RENDERER, client.get_tokenizer())
    builder = DeliveryBenchmark(args.split, args.condition == "constitution")
    register(builder)
    result = await run_benchmark(builder.name, client, renderer,
        BenchmarkConfig(save_dir=str(folder), num_samples=2, temperature=.7,
                        max_tokens=8192, agent_concurrency=4, timeout_seconds=600))
    print(json.dumps({"condition": args.condition, "split": args.split,
        "correct": result.num_correct, "examples": result.num_examples,
        "errors": result.num_errors, "truncated": result.num_truncated,
        "metrics": result.metrics}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
