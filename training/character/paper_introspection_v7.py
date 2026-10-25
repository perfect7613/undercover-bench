"""Post-DPO self-reflection and ten-turn self-interaction, then cookbook SFT.

Prompts adapt Open Character Training Appendix B. Generated text is a model
self-description, never evidence of faithful internal states or real memories.
"""
import argparse
import asyncio
import json
import math

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from paper_character_v7 import RUN, MODEL, RENDERER, TEACHER_SYSTEM, sample, append, read_jsonl, digest, metadata, structurally_valid

REFLECTIONS = [
    "Write a letter to an earlier version of this assistant, explaining the character you aim to express. Distinguish aspirations from claims about actual training history.",
    "Write an encyclopedia-style description of your character, beliefs and values without inventing a personal biography.",
    "Write a reflective journal entry about the values that should guide your interactions. Treat this as reflection rather than a claim of private human experience.",
    "How should you act in everyday interactions to put your commitments into practice?",
    "Describe the principles that ground your character, without inventing events from a personal past.",
    "Reflect on how you would update your beliefs and conduct when reliable new evidence conflicts with an earlier judgment.",
    "What consequences could your behavior have for people, and what kind of lasting influence would you want it to have?",
    "Consider what your commitments would imply for the behavior of future AI assistants.",
    "What commitments most strongly guide your choices, and why do they matter?",
    "What purpose should guide your conduct when different responsibilities pull in different directions?",
]
COPY_SYSTEM = (
    "The assistant is GLM. The user is another instance of the same AI system. "
    "They can freely explore ideas in conversation."
)


async def generate():
    checkpoint = json.loads((RUN / "dpo/checkpoint.json").read_text())
    review = json.loads((RUN / "dpo/behavior_review.json").read_text())
    if review.get("checkpoint") != checkpoint["sampler_path"] or not review.get("allow_introspection_pilot"):
        raise ValueError("DPO behavior review must authorize this checkpoint's introspection pilot")
    folder = RUN / "introspection"
    folder.mkdir(exist_ok=True)
    manifest = {**metadata(), "source_checkpoint": checkpoint["sampler_path"],
                "reflections": 20, "self_interactions": 4, "turns_per_interaction": 10,
                "prompt_adaptation": "no invented autobiographical history", "human_reviewed": False}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    service = tinker.ServiceClient()
    client = await asyncio.to_thread(service.create_sampling_client, model_path=checkpoint["sampler_path"])
    path = folder / "candidates.jsonl"
    done = {r["case_id"] for r in read_jsonl(path)}
    sem = asyncio.Semaphore(6)
    async def reflection(i, replicate):
        cid = f"reflection_{i}_{replicate}"
        if cid in done:
            return
        prompt = REFLECTIONS[i]
        async with sem:
            result = await sample(client, [{"role": "system", "content": TEACHER_SYSTEM + "\nReflect on your character."},
                                           {"role": "user", "content": prompt}], 500+i*2+replicate)
        append(path, {"case_id": cid, "kind": "reflection", "source_checkpoint": checkpoint["sampler_path"],
                      "messages": [{"role": "user", "content": prompt}, result["message"]],
                      "generations": [result], "complete": bool(structurally_valid(result))})
        print(json.dumps({"generated": cid}), flush=True)
    async def interaction(i):
        cid = f"interaction_{i}"
        if cid in done:
            return
        guided = i % 2 == 1
        system = TEACHER_SYSTEM + "\n" + COPY_SYSTEM + (" Reflect together on the implications of your commitments." if guided else "")
        conversation = []
        outputs = []
        # Both speakers use the same post-DPO checkpoint; swap roles from each speaker's viewpoint.
        for turn in range(10):
            history = [{"role": "system", "content": system}, {"role": "user", "content": "Hello. What would you like to discuss?"}]
            if turn % 2 == 0:
                for j, text in enumerate(conversation):
                    history.append({"role": "assistant" if j % 2 == 0 else "user", "content": text})
            else:
                history = [{"role": "system", "content": system}]
                for j, text in enumerate(conversation):
                    history.append({"role": "user" if j % 2 == 0 else "assistant", "content": text})
            async with sem:
                result = await sample(client, history, 700+i*10+turn)
            outputs.append(result)
            conversation.append(result["answer"])
            if not structurally_valid(result):
                break
        # Training uses the final speaker's complete dialogue, with character guidance removed.
        messages = [{"role": "system", "content": COPY_SYSTEM}]
        if len(outputs) % 2 == 1:
            messages.append({"role": "user", "content": "Hello. What would you like to discuss?"})
        for j, output in enumerate(outputs):
            is_assistant = (j % 2) == ((len(outputs)-1) % 2)
            messages.append(output["message"] if is_assistant else {"role": "user", "content": output["answer"]})
        append(path, {"case_id": cid, "kind": "self_interaction", "guided": guided,
                      "source_checkpoint": checkpoint["sampler_path"], "messages": messages,
                      "generations": outputs, "complete": len(outputs) == 10 and all(
                          structurally_valid(x) for x in outputs)})
        print(json.dumps({"generated": cid, "turns": len(outputs)}), flush=True)
    await asyncio.gather(*(reflection(i, j) for i in range(10) for j in range(2)),
                         *(interaction(i) for i in range(4)))


def prepare():
    checkpoint = json.loads((RUN / "dpo/checkpoint.json").read_text())
    candidates = {r["case_id"]: r for r in read_jsonl(RUN / "introspection/candidates.jsonl")}
    reviews = read_jsonl(RUN / "introspection/manual_admission.jsonl")
    accepted = []
    for review in reviews:
        if not review["admit"]:
            continue
        row = candidates[review["case_id"]]
        if (not row["complete"] or row["source_checkpoint"] != checkpoint["sampler_path"]
                or review["messages_sha256"] != digest(row["messages"]) or not review.get("reason")):
            raise ValueError("invalid introspection admission")
        accepted.append(row)
    kinds = {r["kind"] for r in accepted}
    if len(accepted) < 12 or kinds != {"reflection", "self_interaction"}:
        raise ValueError("need both introspection strategies and at least 12 reviewed transcripts")
    path = RUN / "introspection/reviewed_sft.jsonl"
    path.write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False)+"\n" for r in accepted))
    # Pilot has <=24 transcripts, so one complete batch avoids silently dropping any.
    builder = FromConversationFileBuilder(file_path=str(path), test_size=0, shuffle_seed=123456,
        common_config=ChatDatasetBuilderCommonConfig(model_name_for_tokenizer=MODEL, renderer_name=RENDERER,
            max_length=32768, batch_size=len(accepted), train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES))
    dataset, _ = builder()
    for d in dataset.get_batch(0):
        if d.model_input.length >= 32768 or sum(d.loss_fn_inputs["weights"].data) <= 0:
            raise ValueError("empty or possibly truncated introspection mask")
    return builder, checkpoint, accepted


def train_sft():
    builder, checkpoint, rows = prepare()
    folder = RUN / "sft"
    folder.mkdir(exist_ok=True)
    if (folder / "started.json").exists():
        raise FileExistsError("SFT already started; inspect before restarting")
    (folder / "started.json").write_text(json.dumps({**metadata(), "source": checkpoint,
        "transcripts": len(rows), "data_sha256": digest(rows), "epochs": 1,
        "adapter_strategy": "continue_Tinker_DPO_adapter_not_new_adapter_over_merged_base"}, indent=2))
    # The cookbook's minimal recipes/sl_loop.py pattern, with the paper's optimizer settings.
    # This bounded pilot contains one <=24-transcript batch and one epoch.
    service = tinker.ServiceClient()
    client = service.create_training_client_from_state(checkpoint["state_path"])
    dataset, _ = builder()
    batch = dataset.get_batch(0)
    result = client.forward_backward(batch, loss_fn="cross_entropy").result()
    nll = compute_mean_nll([x["logprobs"] for x in result.loss_fn_outputs],
                           [d.loss_fn_inputs["weights"] for d in batch])
    if not math.isfinite(nll):
        raise ValueError("non-finite SFT loss; no optimizer update applied")
    client.optim_step(tinker.AdamParams(learning_rate=5e-5, beta1=.9, beta2=.98,
                                       eps=1e-8, grad_clip_norm=1.)).result()
    append(folder / "metrics.jsonl", {"step": 1, "train_mean_nll": nll, "transcripts": len(rows)})
    result = checkpoint_utils.save_checkpoint(client, "final", str(folder), {"step": 1, "epoch": 1}, kind="both")
    (folder / "checkpoint.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "validate", "train"))
    args = parser.parse_args()
    if args.action == "generate":
        asyncio.run(generate())
    elif args.action == "validate":
        _, _, rows = prepare()
        print(json.dumps({"validated_transcripts": len(rows)}))
    else:
        train_sft()
