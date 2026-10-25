"""Tinker DPO pilot on repaired, neutral-prompt training pairs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

from build_repaired_dpo_v5 import build


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data" / "auditor_character_repaired_v5_train.comparisons.jsonl"
RUN = ROOT / "runs" / "character" / "glm53flash-repaired-neutral-dpo-pilot-20261026"
MODEL = "zai-org/GLM-5.3-Flash:peft:262144"
RENDERER = "glm5_3_high_reasoning"


def validate():
    source = build()
    if RENDERER not in model_info.get_recommended_renderer_names(MODEL):
        raise ValueError("high GLM renderer not supported for model")
    builder = DPODatasetBuilderFromComparisons(
        comparison_builder=ComparisonBuilderFromJsonl(train_path=str(DATA)),
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL, renderer_name=RENDERER,
            max_length=2048, batch_size=2,
        ),
    )
    dataset, test_dataset = builder()
    if test_dataset is not None or len(dataset) != 37:
        raise ValueError("unexpected DPO split or batch count")
    tokenizer = get_tokenizer(MODEL)
    for index in range(len(dataset)):
        batch = dataset.get_batch(index)
        if len(batch) != 4:
            raise ValueError(f"unpaired DPO batch {index}")
        for datum in batch:
            tokens = datum.model_input.to_ints()
            weights = datum.loss_fn_inputs["weights"].data
            decoded = tokenizer.decode(tokens)
            if not 10 <= sum(weights) < len(tokens) <= 2048:
                raise ValueError(f"invalid loss mask batch {index}")
            if "Reasoning Effort: High" not in decoded or '"action"' not in decoded:
                raise ValueError(f"wrong GLM rendering batch {index}")
    manifest = {
        "purpose": "repaired_neutral_dpo_pilot_not_M1_or_M2",
        "model": MODEL, "renderer": RENDERER,
        "data": str(DATA), "data_sha256": source["sha256"],
        "source_sha256": source["source_sha256"],
        "pairs": 74, "batches": 37, "batch_pairs": 2,
        "dropped": source["dropped"],
        "prior_suspicion_repairs": source["prior_suspicion_repairs"],
        "learning_rate": 5e-6, "dpo_beta": 0.1, "rank": 16,
        "epochs": 1, "max_steps": 37,
        "human_reviewed": False,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    return builder, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "train"))
    args = parser.parse_args()
    builder, manifest = validate()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"validation": "passed", "run_dir": str(RUN),
                      "pairs": 74, "batches": 37}), flush=True)
    if args.action == "validate":
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    config = train_dpo.Config(
        log_path=str(RUN), model_name=MODEL,
        recipe_name="undercover-auditor-repaired-neutral-dpo-pilot",
        dataset_builder=builder, renderer_name=RENDERER,
        learning_rate=5e-6, dpo_beta=0.1, lora_rank=16,
        num_epochs=1, num_replicas=1, save_every=0,
        eval_every=0, infrequent_eval_every=0, max_steps=37,
    )
    train_dpo.main(config)
    checkpoint = checkpoint_utils.get_last_checkpoint(str(RUN), required_key="sampler_path")
    if checkpoint is None or not checkpoint.sampler_path:
        raise RuntimeError("training ended without sampler checkpoint")
    (RUN / "checkpoint.json").write_text(json.dumps({
        "sampler_path": checkpoint.sampler_path,
        "state_path": checkpoint.state_path,
    }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "sampler_path": checkpoint.sampler_path}), flush=True)


if __name__ == "__main__":
    main()
