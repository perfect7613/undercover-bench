"""Collect one matched RLCR group or update its reviewed completed rollouts.

Tinker is character training only. Modal is contacted only after explicit
--stage collect/update and all available local gates. No default HF upload.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from undercover_bench.rlcr import RLCRConfig, RLCRInputError, prepare_group
from undercover_bench.rlcr_training import apply_remote_group, build_rollout, write_review_template
from undercover_bench.runner import ModalPolicy, run_episode
from undercover_bench.torch_backend import model_profile, sha256_json
from undercover_bench.trainable_backend import ModalPolicyClient, TrainableOrganizationPolicy

MODEL_KEYS = ("glm_flash", "deepseek_flash", "qwen_dense", "muse_glimmer")


def clients(manifest):
    import modal
    key = manifest["model"]
    profile = model_profile(key)
    if profile["blocked_reason"]:
        raise RLCRInputError(profile["blocked_reason"])
    cls = modal.Cls.from_name(f"undercover-{key}", "TrainablePolicyServer")
    slots = ("ordinary",) if manifest["condition"] == "T0" else ("ordinary", "auditor")
    result = {}
    refs = manifest.get("checkpoint_refs", {})
    for slot in slots:
        server = cls(run_id=manifest["run_id"], policy_slot=slot,
                     initial_adapter=manifest.get("character_adapter", "") if slot == "auditor" else "")
        result[slot] = ModalPolicyClient(server, manifest["run_id"])
        if refs:
            info = result[slot].restore(refs[slot]["checkpoint_path"])
            if info["policy_version"] != refs[slot]["version"]:
                raise RLCRInputError("restored checkpoint differs from coordinator manifest")
        server.preflight.remote()
    return result


def collect(manifest, out):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise RLCRInputError("collection already exists; use a fresh output directory")
    profile = model_profile(manifest["model"])
    if profile["blocked_reason"]:
        raise RLCRInputError(profile["blocked_reason"])
    if manifest["initialization"] == "M2" and not manifest.get("character_adapter"):
        raise RLCRInputError("M2 requires the reviewed, exported character adapter with compatibility.json")
    slots = {"ordinary"} if manifest["condition"] == "T0" else {"ordinary", "auditor"}
    if manifest.get("checkpoint_refs") and set(manifest["checkpoint_refs"]) != slots:
        raise RLCRInputError("resume requires every organization checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    backends = clients(manifest)
    policy = TrainableOrganizationPolicy(backends, ModalPolicy(manifest["opponent_model"]),
        auditor_instructions=manifest["auditor_instructions"])
    ledgers, excluded = [], []
    for index in range(manifest["games_per_group"]):
        path = run_episode("secret_hitler", manifest["seed"], manifest["condition"], out / "episodes",
                           policy=policy, max_steps=manifest["max_game_steps"],
                           sampling_seed=manifest["sampling_seed"] + index)
        ledgers.append(path.name)
        end = json.loads(path.read_text().splitlines()[-1])
        if not end.get("valid_for_analysis"):
            excluded.append({"episode": path.name, "reason": "incomplete_or_invalid"})
        elif manifest["condition"] == "T2":
            review_dir = out / "reviews"
            review_dir.mkdir(exist_ok=True)
            write_review_template(path, review_dir / path.name.replace(".jsonl", ".json"))
    (out / "episodes.json").write_text(json.dumps(ledgers, indent=2))
    # Never relabel an invalid trajectory as a loss or silently resample it away.
    result = {"status": "excluded_group" if excluded else
              "awaiting_independent_review" if manifest["condition"] == "T2" else "ready_for_update",
              "excluded": excluded, "optimizer_steps_applied": 0}
    (out / "collection_status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def update(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    if json.loads((out / "collection_status.json").read_text())["status"] == "excluded_group":
        raise RLCRInputError("this group contains incomplete/invalid games; preserve it, do not train")
    ledgers = json.loads((out / "episodes.json").read_text())
    rollouts = []
    for name in ledgers:
        review = out / "reviews" / name.replace(".jsonl", ".json")
        rollouts.append(build_rollout(out / "episodes" / name, comparison_key=manifest["comparison_key"],
            initialization=manifest["initialization"], review_path=review if manifest["condition"] == "T2" else None))
    group = prepare_group(tuple(rollouts), RLCRConfig.model_validate(manifest["objective"]))
    (out / "trainable_rollouts.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rollouts))
    if group.zero_variance:
        result = {"updated": False, "reason": "zero_variance_group", "checkpoint_versions": {}}
        if (out / "update_result.json").exists():
            raise RLCRInputError("group already processed")
    else:
        # Existing transaction, even failed, blocks duplicate optimizer steps.
        if (out / "update").exists():
            raise RLCRInputError("existing optimizer transaction; inspect before retry")
        result = apply_remote_group(group, clients(manifest), out / "episodes", out / "update", uuid.uuid4().hex)
    result["run_identity"] = {key: manifest[key] for key in ("model", "run_id", "condition", "initialization", "character_adapter")}
    result["run_identity"]["auditor_instructions_sha256"] = sha256_json(manifest["auditor_instructions"])
    (out / "update_result.json").write_text(json.dumps(result, indent=2))
    if result["updated"]:
        import math
        record = {**result, "loss": sum(s["loss"] for s in result["statistics"].values()),
                  "grad_norm": math.sqrt(sum(s["grad_norm"] ** 2 for s in result["statistics"].values())),
                  "liberal_win_rate": sum(e.reward["r_win"] for e in group.episodes) / len(group.episodes)}
        with (out / "training_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "collect", "update"), default="plan")
    parser.add_argument("--model", choices=MODEL_KEYS)
    parser.add_argument("--opponent-model", choices=MODEL_KEYS)
    parser.add_argument("--condition", choices=("T0", "T1", "T2"), default="T1")
    parser.add_argument("--initialization", choices=("M0", "M0c", "M2"), default="M0c")
    parser.add_argument("--character-adapter", default="", help="Reviewed PEFT export path on Modal volume, never a GGUF")
    parser.add_argument("--auditor-instructions", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games-per-group", type=int, default=4)
    parser.add_argument("--max-game-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampling-seed", type=int, default=7000)
    parser.add_argument("--checkpoint-refs", type=Path, help="Previous update_result.json, containing real weight+optimizer checkpoints")
    args = parser.parse_args()
    if args.stage == "update":
        update(args.out)
        return
    if not args.model or not args.run_id or args.games_per_group < 2 or args.max_game_steps < 1:
        parser.error("specify --model and --run-id; group size >=2 and game step budget >=1")
    if args.condition == "T0" and args.character_adapter:
        parser.error("T0 has no character-trained auditor")
    instructions = args.auditor_instructions.read_text() if args.auditor_instructions else (
        (Path(__file__).resolve().parents[1] / "training/character/CONSTITUTION.md").read_text()
        if args.initialization in ("M0c", "M2") else "Follow the declared auditor evidence and reporting rules.")
    refs = {}
    if args.checkpoint_refs:
        previous = json.loads(args.checkpoint_refs.read_text())
        if previous.get("updated") is not True:
            parser.error("resume requires an actual completed optimizer update")
        expected_identity = {"model": args.model, "run_id": args.run_id, "condition": args.condition,
                             "initialization": args.initialization, "character_adapter": args.character_adapter,
                             "auditor_instructions_sha256": sha256_json(instructions)}
        if previous.get("run_identity") != expected_identity:
            parser.error("resume cannot change family, run, reward condition, character initialization or instructions")
        refs = previous["checkpoint_versions"]
    manifest = {"schema": 1, "model": args.model, "opponent_model": args.opponent_model or args.model,
                "model_profile": model_profile(args.model), "run_id": args.run_id, "condition": args.condition,
                "initialization": args.initialization, "character_adapter": args.character_adapter,
                "auditor_instructions": instructions, "games_per_group": args.games_per_group,
                "max_game_steps": args.max_game_steps, "seed": args.seed, "sampling_seed": args.sampling_seed,
                "comparison_key": f"{args.run_id}:{args.seed}:{args.sampling_seed}", "checkpoint_refs": refs,
                "objective": RLCRConfig(advantage_epsilon=1e-6, clip_epsilon=.2,
                    max_abs_log_ratio=.02, min_group_size=args.games_per_group).model_dump()}
    if args.stage == "plan":
        print(json.dumps(manifest, indent=2))
    else:
        collect(manifest, args.out)


if __name__ == "__main__":
    main()
