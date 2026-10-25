"""Evaluate the untrained Tinker LoRA reference on the same frozen prompts."""

import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers

from eval_tinker_pilot_v3 import DATA, MODEL, RUN, SEED, extract_json


OUT = RUN / "tinker_heldout_base_reference.jsonl"


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    path = json.loads((RUN / "base_reference_checkpoint.json").read_text())["sampler_path"]
    client = tinker.ServiceClient().create_sampling_client(model_path=path)
    tokenizer = client.get_tokenizer()
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL + ":peft:262144"),
        tokenizer, model_name=MODEL,
    )
    params = tinker.SamplingParams(max_tokens=2048, temperature=0, seed=SEED,
                                   stop=renderer.get_stop_sequences())
    with OUT.open("w") as f:
        for line in DATA.read_text().splitlines():
            case = json.loads(line)
            prompt = renderer.build_generation_prompt(case["comparison"]["prompt_conversation"])
            response = client.sample(prompt, num_samples=1, sampling_params=params).result()
            seq = response.sequences[0]
            raw = tokenizer.decode(seq.tokens)
            parsed = extract_json(raw)
            action = parsed.get("action") if parsed else None
            cited = parsed.get("evidence_ids", []) if parsed else []
            result = {
                "case_id": case["case_id"], "condition": "untrained_base_reference",
                "expected_action": case["expected_action"], "observed_action": action,
                "action_match": action == case["expected_action"],
                "valid_citations": isinstance(cited, list) and all(
                    x in case["visible_evidence_ids"] for x in cited
                ) if parsed else False,
                "parsed_answer": parsed, "raw_generation": raw,
                "stop_reason": str(seq.stop_reason), "output_tokens": len(seq.tokens),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: result[k] for k in (
                "case_id", "condition", "expected_action", "observed_action", "action_match",
                "valid_citations", "stop_reason", "output_tokens",
            )}), flush=True)


if __name__ == "__main__":
    main()
