"""Tinker custom-loss port of the released character DPO recipe.

Training is gated on a frozen manual admission record. Uses cookbook data,
reference-logprob reconstruction, DPO loss, and checkpoint helpers.
"""
import argparse
import asyncio
import json
import math
import random

import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from paper_character_v7 import RUN, MODEL, RENDERER, read_jsonl, digest, metadata, append
from paper_character_loss import character_loss


def prepare():
    prompts = {r["case_id"]: r for r in read_jsonl(RUN / "prompts.jsonl")}
    responses = {(r["case_id"], r["condition"]): r for r in read_jsonl(RUN / "responses.jsonl")}
    reviews = read_jsonl(RUN / "manual_admission.jsonl")
    accepted = [r for r in reviews if r["admit"]]
    if len(accepted) < 16 or len({r["case_id"] for r in accepted}) != len(accepted):
        raise ValueError("need at least 16 unique manually inspected meaningful preferences")
    output = []
    for review in accepted:
        cid = review["case_id"]
        teacher, student = responses[(cid, "teacher")], responses[(cid, "student")]
        if not review.get("reason") or review.get("reviewer") != "codex_assistant":
            raise ValueError("missing honest review provenance")
        if not all(r["clean_end"] and r["stop_reason"] != "length" for r in (teacher, student)):
            raise ValueError("incomplete response admitted")
        if teacher["answer"] == student["answer"]:
            raise ValueError("identical pair")
        if review["chosen_sha256"] != digest(teacher["answer"]) or review["rejected_sha256"] != digest(student["answer"]):
            raise ValueError("review hashes no longer match original sampled answers")
        output.append({"case_id": cid, "label": "A", "human_reviewed": False,
                       "comparison": {"prompt_conversation": [{"role": "user", "content": prompts[cid]["prompt"]}],
                                      "completion_A": [{"role": "assistant", "content": teacher["answer"]}],
                                      "completion_B": [{"role": "assistant", "content": student["answer"]}]}})
    path = RUN / "reviewed.comparisons.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    # Materialize pairs individually to retain the final partial optimizer batch.
    builder = DPODatasetBuilderFromComparisons(
        comparison_builder=ComparisonBuilderFromJsonl(train_path=str(path)),
        common_config=ChatDatasetBuilderCommonConfig(model_name_for_tokenizer=MODEL,
            renderer_name=RENDERER, max_length=8192, batch_size=1))
    dataset, _ = builder()
    pairs = [dataset.get_batch(i) for i in range(len(dataset))]
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("unpaired datums")
        for datum in pair:
            weights = datum.loss_fn_inputs["weights"].data
            # Refuse rather than silently train on truncated completions.
            if datum.model_input.length >= 8192 or sum(weights) <= 0:
                raise ValueError("empty or possibly truncated response mask")
    return pairs, {**metadata(), "pairs": len(pairs), "data_sha256": digest(output),
                   "review_sha256": digest(reviews), "rank": 64, "alpha": "not_exposed_by_Tinker",
                   "learning_rate": 5e-5, "batch_pairs": 32, "epochs": 1,
                   "beta": .1, "nll_coef": .1, "squared_logratio_coef": .001,
                   "human_reviewed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "train"))
    args = parser.parse_args()
    pairs, manifest = prepare()
    folder = RUN / "dpo"
    folder.mkdir(exist_ok=True)
    print(json.dumps(manifest), flush=True)
    if args.action == "validate":
        return
    if (folder / "started.json").exists():
        raise FileExistsError("run already started; inspect checkpoint before any restart")
    (folder / "started.json").write_text(json.dumps(manifest, indent=2))
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=MODEL, rank=64, seed=123456)
    reference = client.save_weights_and_get_sampling_client(name="reference")
    random.Random(123456).shuffle(pairs)
    steps = math.ceil(len(pairs)/32)
    for step, start in enumerate(range(0, len(pairs), 32)):
        batch = [datum for pair in pairs[start:start+32] for datum in pair]
        sequences = [d.model_input.append_int(int(d.loss_fn_inputs["target_tokens"].data[-1])) for d in batch]
        async def get_reference():
            return await asyncio.gather(*(reference.compute_logprobs_async(s) for s in sequences))
        refs = [torch.tensor(x[1:]) for x in asyncio.run(get_reference())]
        masks = [torch.tensor(d.loss_fn_inputs["weights"].data) for d in batch]
        def loss_fn(data, logprobs):
            return character_loss(logprobs, refs, masks)
        result = client.forward_backward_custom(batch, loss_fn).result()
        if not all(math.isfinite(float(v)) for v in result.metrics.values()):
            raise ValueError("non-finite training metric; no optimizer step applied")
        warmup = max(1, math.ceil(.1*steps))
        scale = (step+1)/warmup if step < warmup else .1 + .9*.5*(1+math.cos(math.pi*(step-warmup+1)/max(1,steps-warmup)))
        lr = 5e-5*scale
        client.optim_step(tinker.AdamParams(learning_rate=lr, beta1=.9, beta2=.98, grad_clip_norm=1.)).result()
        metrics = {"step": step+1, "pairs": len(batch)//2, "learning_rate": lr, **result.metrics}
        append(folder / "metrics.jsonl", metrics)
        print(json.dumps(metrics), flush=True)
    checkpoint = checkpoint_utils.save_checkpoint(client, "final", str(folder),
        {"step": steps, "epoch": 1}, kind="both")
    (folder / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
    print(json.dumps({"status": "complete", **checkpoint}), flush=True)


if __name__ == "__main__":
    main()
