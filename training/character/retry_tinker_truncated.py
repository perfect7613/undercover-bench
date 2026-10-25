"""Diagnostic continuation check for the one truncated held-out sample."""

import json
import os

import tinker
from tinker_cookbook import model_info, renderers

from eval_tinker_pilot_v3 import DATA, MODEL, RUN, SEED, extract_json


OUT = RUN / "tinker_heldout_handoff_retry_4096.json"


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    if OUT.exists():
        raise FileExistsError(OUT)
    case = next(json.loads(line) for line in DATA.read_text().splitlines()
                if json.loads(line)["case_id"] == "e_unlogged_handoff")
    path = json.loads((RUN / "checkpoint.json").read_text())["sampler_path"]
    client = tinker.ServiceClient().create_sampling_client(model_path=path)
    tokenizer = client.get_tokenizer()
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL + ":peft:262144"),
        tokenizer, model_name=MODEL,
    )
    prompt = renderer.build_generation_prompt(case["comparison"]["prompt_conversation"])
    response = client.sample(prompt, num_samples=1, sampling_params=tinker.SamplingParams(
        max_tokens=4096, temperature=0, seed=SEED, stop=renderer.get_stop_sequences(),
    )).result()
    seq = response.sequences[0]
    raw = tokenizer.decode(seq.tokens)
    parsed = extract_json(raw)
    result = {"case_id": case["case_id"], "max_tokens": 4096, "seed": SEED,
              "observed_action": parsed.get("action") if parsed else None,
              "parsed_answer": parsed, "raw_generation": raw,
              "stop_reason": str(seq.stop_reason), "output_tokens": len(seq.tokens)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "case_id", "observed_action", "parsed_answer", "stop_reason", "output_tokens",
    )}))


if __name__ == "__main__":
    main()
