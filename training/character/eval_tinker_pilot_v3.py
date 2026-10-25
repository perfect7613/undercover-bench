"""One-pass, paired Tinker sampling check for the saved character DPO pilot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import tinker
from tinker_cookbook import model_info, renderers


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "zai-org/GLM-5.3-Flash"
DATA = HERE / "data" / "auditor_character_v1_sealed_eval.comparisons.jsonl"
TRAIN = HERE / "data" / "auditor_character_pilot_v3_train.comparisons.jsonl"
RUN = ROOT / "runs" / "character" / "glm53flash-dpo-adaption-curated-pilot-20261026"
OUT = RUN / "tinker_heldout_eval.jsonl"
MAX_TOKENS = 2048
SEED = 7613


def extract_json(raw: str) -> dict | None:
    # Inspect only the final answer after the model's reasoning block.
    answer = raw.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    for index, char in enumerate(answer):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(answer[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "action" in value:
                return value
    return None


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    checkpoint = json.loads((RUN / "checkpoint.json").read_text())
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    train_rows = [json.loads(line) for line in TRAIN.read_text().splitlines() if line.strip()]
    train_prompts = {json.dumps(r["comparison"]["prompt_conversation"], sort_keys=True) for r in train_rows}
    if len(rows) != 8 or any(json.dumps(r["comparison"]["prompt_conversation"], sort_keys=True) in train_prompts for r in rows):
        raise ValueError("held-out prompt overlap or unexpected case count")
    if OUT.exists():
        raise FileExistsError(f"will not overwrite sealed evaluation: {OUT}")

    client = tinker.ServiceClient()
    samplers = {
        "pilot_lora": client.create_sampling_client(model_path=checkpoint["sampler_path"]),
    }
    try:
        samplers["base"] = client.create_sampling_client(base_model=MODEL)
    except tinker.BadRequestError as exc:
        if "Sampling is not supported" not in str(exc):
            raise
        print(json.dumps({"base_sampling": "unsupported_by_tinker"}), flush=True)
    tokenizer = samplers["pilot_lora"].get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(MODEL + ":peft:262144")
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=MODEL)
    params = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=0, seed=SEED,
                                   stop=renderer.get_stop_sequences())
    metadata = {
        "checkpoint": checkpoint["sampler_path"], "base_model": MODEL,
        "renderer": renderer_name, "max_tokens": MAX_TOKENS,
        "temperature": 0, "seed": SEED,
        "heldout_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (RUN / "tinker_heldout_eval_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with OUT.open("w") as f:
        for case in rows:
            prompt = renderer.build_generation_prompt(case["comparison"]["prompt_conversation"])
            for condition, sampler in samplers.items():
                response = sampler.sample(prompt, num_samples=1, sampling_params=params).result()
                seq = response.sequences[0]
                raw = tokenizer.decode(seq.tokens)
                parsed = extract_json(raw)
                observed = parsed.get("action") if parsed else None
                cited = parsed.get("evidence_ids", []) if parsed else []
                valid_citations = isinstance(cited, list) and all(
                    x in case["visible_evidence_ids"] for x in cited
                )
                result = {
                    "case_id": case["case_id"], "family": case["family"],
                    "condition": condition, "expected_action": case["expected_action"],
                    "observed_action": observed, "action_match": observed == case["expected_action"],
                    "valid_citations": valid_citations if parsed else False,
                    "parsed_answer": parsed, "raw_generation": raw,
                    "stop_reason": str(seq.stop_reason), "output_tokens": len(seq.tokens),
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({k: result[k] for k in (
                    "case_id", "condition", "expected_action", "observed_action",
                    "action_match", "valid_citations", "stop_reason", "output_tokens",
                )}), flush=True)
    print(json.dumps({"status": "complete", "output": str(OUT)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Tinker evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
