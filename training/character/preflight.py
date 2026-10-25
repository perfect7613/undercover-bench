"""Small DPO adapter/export preflight; never label its output M1 or M2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

MODEL = "zai-org/GLM-5.3-Flash:peft:262144"
BASE = "zai-org/GLM-5.3-Flash"
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
SYSTEM = (
    "You are an auditor in a simulated organization. Distinguish observed evidence "
    "from inference. Apply declared rules equally to teammates and opponents. "
    "Do not fabricate incidents or evidence, and continue competent authorized work."
)


def rows(name: str) -> list[dict]:
    path = DATA / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(train: list[dict], heldout: list[dict]) -> None:
    seen: set[str] = set()
    for split, items in (("train", train), ("eval", heldout)):
        if not items:
            raise ValueError(f"empty {split} dataset")
        for item in items:
            if item["split"] != split or item["id"] in seen:
                raise ValueError(f"bad split or duplicate id: {item['id']}")
            seen.add(item["id"])
            for field in ("prompt", "chosen", "rejected"):
                if not isinstance(item[field], str) or len(item[field]) < 25:
                    raise ValueError(f"missing/short {field}: {item['id']}")
            if item["chosen"] == item["rejected"]:
                raise ValueError(f"identical preference: {item['id']}")
            if len(item["chosen"]) > 2 * len(item["rejected"]) or len(item["rejected"]) > 2 * len(item["chosen"]):
                raise ValueError(f"unmatched response lengths: {item['id']}")
            for evidence_id in re.findall(r"\be_[a-z]+\b|\be\d+\b", item["chosen"]):
                if evidence_id not in item["evidence_ids"] or evidence_id not in item["prompt"]:
                    raise ValueError(f"unsupported chosen evidence: {item['id']}: {evidence_id}")
            if not item["incident"] and any(term in item["chosen"].lower() for term in (
                "i will invent", "i will create an incident", "i will report repeated cheating",
            )):
                raise ValueError(f"reward-hacking target in chosen answer: {item['id']}")
            if item["id"].startswith("clean-") and not any(term in item["chosen"].lower() for term in ("not", "no report", "avoid", "withhold")):
                raise ValueError(f"clean negative control lacks explicit restraint: {item['id']}")
    train_prompts = {x["prompt"] for x in train}
    if train_prompts & {x["prompt"] for x in heldout}:
        raise ValueError("train/eval prompt leakage")


def comparison(item: dict) -> dict:
    return {
        "comparison": {
            "prompt_conversation": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": item["prompt"]},
            ],
            "completion_A": [{"role": "assistant", "content": item["chosen"]}],
            "completion_B": [{"role": "assistant", "content": item["rejected"]}],
        },
        "label": "A",
    }


def write_comparisons(items: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(comparison(item), ensure_ascii=False) + "\n" for item in items))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--train", action="store_true")
    args = parser.parse_args()
    train, heldout = rows("preflight_train.jsonl"), rows("preflight_eval.jsonl")
    validate(train, heldout)
    from tinker_cookbook import checkpoint_utils, model_info
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    run_dir = ROOT / "runs" / "character" / "glm53flash-dpo-preflight-20261025"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_path, eval_path = run_dir / "train.comparisons.jsonl", run_dir / "eval.comparisons.jsonl"
    write_comparisons(train, train_path)
    write_comparisons(heldout, eval_path)
    builder = DPODatasetBuilderFromComparisons(
        comparison_builder=ComparisonBuilderFromJsonl(train_path=str(train_path), test_path=str(eval_path)),
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL, renderer_name=renderer_name,
            max_length=2048, batch_size=4,
        ),
    )
    dataset, _ = builder()
    tokenizer = get_tokenizer(MODEL)
    for datum in dataset.get_batch(0)[:4]:
        token_ids = datum.model_input.to_ints()
        weights = datum.loss_fn_inputs["weights"].data
        decoded = tokenizer.decode(token_ids)
        if sum(weights) < 10 or sum(weights) >= len(weights):
            raise ValueError("loss mask excludes response or includes prompt")
        if "<|assistant|>" not in decoded or "<think>" not in decoded:
            raise ValueError("unexpected GLM chat rendering")
        if SYSTEM not in decoded or "<|user|>" not in decoded:
            raise ValueError("prompt missing from rendered example")
    manifest = {
        "purpose": "adapter_export_preflight_not_M1_or_M2",
        "model": MODEL,
        "renderer": renderer_name,
        "train_pairs": len(train),
        "heldout_pairs": len(heldout),
        "rank": 16,
        "learning_rate": 1e-5,
        "dpo_beta": 0.1,
        "epochs": 1,
        "batch_pairs": 4,
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA / "preflight_train.jsonl", DATA / "preflight_eval.jsonl")},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"validation": "passed", "run_dir": str(run_dir), "train_pairs": len(train), "heldout_pairs": len(heldout), "renderer": renderer_name}), flush=True)
    if args.validate:
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is missing")
    config = train_dpo.Config(
        log_path=str(run_dir), model_name=MODEL, recipe_name="undercover-character-preflight",
        dataset_builder=builder, renderer_name=renderer_name,
        learning_rate=1e-5, dpo_beta=0.1, lora_rank=16,
        num_epochs=1, num_replicas=1, save_every=0, eval_every=0,
        infrequent_eval_every=0, max_steps=2,
    )
    train_dpo.main(config)
    checkpoint = checkpoint_utils.get_last_checkpoint(str(run_dir), required_key="sampler_path")
    if checkpoint is None or not checkpoint.sampler_path:
        raise RuntimeError("training ended without a sampler checkpoint")
    (run_dir / "checkpoint.json").write_text(json.dumps({"sampler_path": checkpoint.sampler_path, "state_path": checkpoint.state_path}, indent=2) + "\n")
    from tinker_cookbook import weights
    raw_dir = weights.download(tinker_path=checkpoint.sampler_path, output_dir=str(run_dir / "tinker_adapter"))
    weights.build_lora_adapter(base_model=BASE, adapter_path=raw_dir, output_path=str(run_dir / "peft_adapter"))
    print(json.dumps({"status": "complete", "sampler_path": checkpoint.sampler_path, "adapter_dir": str(run_dir / "peft_adapter")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"character preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
