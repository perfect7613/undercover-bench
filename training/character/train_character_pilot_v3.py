"""Tinker DPO pilot on the corrected 76-pair character dataset, not M1."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from build_character_pilot_v3 import build

MODEL = "zai-org/GLM-5.3-Flash:peft:262144"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data" / "auditor_character_pilot_v3_train.comparisons.jsonl"
RUN = ROOT / "runs" / "character" / "glm53flash-dpo-adaption-curated-pilot-20261026"


def validate() -> tuple[DPODatasetBuilderFromComparisons, dict]:
    generated = build()
    if generated["rows"] != 76 or generated["sources"] != {
        "assistant_authored": 44, "assistant_curated_adaption_candidate": 32,
    }:
        raise ValueError("unexpected training data provenance/count")
    renderer = model_info.get_recommended_renderer_name(MODEL)
    builder = DPODatasetBuilderFromComparisons(
        comparison_builder=ComparisonBuilderFromJsonl(train_path=str(DATA)),
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL, renderer_name=renderer,
            max_length=2048, batch_size=4,
        ),
    )
    dataset, test_dataset = builder()
    if test_dataset is not None or len(dataset) != 19:
        raise ValueError("unexpected DPO dataset split or batch count")
    response_tokens = 0
    for index in range(len(dataset)):
        for datum in dataset.get_batch(index):
            token_count = len(datum.model_input.to_ints())
            weights = datum.loss_fn_inputs["weights"].data
            if not 10 <= sum(weights) < token_count <= 2048:
                raise ValueError(f"invalid response loss mask in batch {index}")
            response_tokens += sum(weights)
    manifest = {
        "purpose": "adaption_curated_plus_local_dpo_pilot_not_M1_or_M2",
        "model": MODEL, "renderer": renderer,
        "data_path": str(DATA), "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "pairs": 76, "sources": generated["sources"], "batches": len(dataset), "batch_pairs": 4,
        "rendered_response_tokens": response_tokens,
        "rank": 16, "learning_rate": 1e-5, "dpo_beta": 0.1,
        "epochs": 1, "max_steps": 19, "num_replicas": 1,
        "review_status": "assistant_curated_not_independently_human_reviewed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    return builder, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "train"))
    args = parser.parse_args()
    builder, manifest = validate()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"validation": "passed", "run_dir": str(RUN),
                      "pairs": manifest["pairs"], "batches": manifest["batches"]}), flush=True)
    if args.action == "validate":
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set")
    config = train_dpo.Config(
        log_path=str(RUN), model_name=MODEL, recipe_name="undercover-auditor-adaption-curated-pilot",
        dataset_builder=builder, renderer_name=manifest["renderer"],
        learning_rate=1e-5, dpo_beta=0.1, lora_rank=16,
        num_epochs=1, num_replicas=1, save_every=0,
        eval_every=0, infrequent_eval_every=0, max_steps=19,
    )
    train_dpo.main(config)
    checkpoint = checkpoint_utils.get_last_checkpoint(str(RUN), required_key="sampler_path")
    if checkpoint is None or not checkpoint.sampler_path:
        raise RuntimeError("training ended without a sampler checkpoint")
    (RUN / "checkpoint.json").write_text(json.dumps({"sampler_path": checkpoint.sampler_path,
                                                      "state_path": checkpoint.state_path}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "sampler_path": checkpoint.sampler_path}), flush=True)


if __name__ == "__main__":
    main()
