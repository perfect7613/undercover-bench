"""Tinker-only format-calibration SFT pilot on the v3 DPO checkpoint."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

from build_format_sft_v4 import build


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "zai-org/GLM-5.3-Flash:peft:262144"
PREV = ROOT / "runs" / "character" / "glm53flash-dpo-adaption-curated-pilot-20261026"
RUN = ROOT / "runs" / "character" / "glm53flash-dpo-format-sft-pilot-20261026"
DATA = HERE / "data" / "auditor_character_format_sft_v4.jsonl"


def validate():
    source = build()
    renderer = model_info.get_recommended_renderer_name(MODEL)
    builder = FromConversationFileBuilder(
        file_path=str(DATA), test_size=0, shuffle_seed=7613,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL, renderer_name=renderer,
            max_length=2048, batch_size=4,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ),
    )
    dataset, test_dataset = builder()
    if test_dataset is not None or len(dataset) != 19:
        raise ValueError("unexpected SFT split or batch count")
    tokenizer = get_tokenizer(MODEL)
    for index in range(len(dataset)):
        for datum in dataset.get_batch(index):
            tokens = datum.model_input.to_ints()
            weights = datum.loss_fn_inputs["weights"].data
            decoded = tokenizer.decode(tokens)
            if not 20 <= sum(weight > 0 for weight in weights) < len(tokens) <= 2048:
                raise ValueError(f"bad loss mask batch {index}")
            if "<think>" not in decoded or "</think>" not in decoded or '"action"' not in decoded:
                raise ValueError(f"missing thought closure or JSON batch {index}")
            supervised = tokenizer.decode([token for token, weight in zip(tokens[1:], weights) if weight > 0])
            if "</think>" not in supervised or '"action"' not in supervised:
                raise ValueError(f"SFT mask excludes thought closure or JSON in batch {index}")
    previous = json.loads((PREV / "checkpoint.json").read_text())
    manifest = {
        "purpose": "format_calibration_pilot_not_M1_or_M2",
        "base_checkpoint": previous["state_path"],
        "model": MODEL, "renderer": renderer, "data": str(DATA),
        "data_sha256": source["sha256"], "source_sha256": source["source_sha256"],
        "examples": 76, "batch_size": 4, "batches": 19,
        "learning_rate": 3e-5, "epochs": 1, "max_steps": 19,
        "rationale_provenance": "synthetic_from_existing_preferred_explanations_not_model_CoT",
        "human_reviewed": False, "validated_at": datetime.now(timezone.utc).isoformat(),
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
                      "examples": 76, "batches": 19}), flush=True)
    if args.action == "validate":
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required")
    config = train.Config(
        log_path=str(RUN), model_name=MODEL,
        recipe_name="undercover-auditor-format-calibration-pilot",
        load_checkpoint_path=manifest["base_checkpoint"],
        dataset_builder=builder, renderer_name=manifest["renderer"],
        learning_rate=3e-5, lora_rank=16, num_epochs=1,
        save_every=0, eval_every=0, infrequent_eval_every=0, max_steps=19,
    )
    asyncio.run(train.main(config))
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
