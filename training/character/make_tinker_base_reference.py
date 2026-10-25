"""Save an untrained rank-16 GLM LoRA as a Tinker-only base reference."""

import json
import os
from pathlib import Path

import tinker


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "character" / "glm53flash-dpo-adaption-curated-pilot-20261026"
OUT = RUN / "base_reference_checkpoint.json"


def main() -> None:
    if OUT.exists():
        print(OUT.read_text())
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    client = tinker.ServiceClient()
    training = client.create_lora_training_client(
        base_model="zai-org/GLM-5.3-Flash:peft:262144", rank=16,
        user_metadata={"purpose": "untrained_base_reference_for_auditor_pilot"},
    )
    checkpoint = training.save_weights_for_sampler("untrained_base_reference").result()
    OUT.write_text(json.dumps({"sampler_path": checkpoint.path,
                               "training_steps": 0, "rank": 16}, indent=2) + "\n")
    print(json.dumps({"saved": True, "sampler_path": checkpoint.path}))


if __name__ == "__main__":
    main()
