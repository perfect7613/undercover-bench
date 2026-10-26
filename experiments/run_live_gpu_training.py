"""Live RLCR GPU Training Loop for Secret Hitler.

Uses ModalTrainablePolicy for live forward inference (with token IDs + behavior logprobs)
and performs live backward pass LoRA updates (policy.train_step) on the GPU container.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from undercover_bench.checkpoint_hf import save_checkpoint_to_disk, upload_checkpoint_to_huggingface
from undercover_bench.runner import run_episode
from undercover_bench.trainable_backend import ModalTrainablePolicy


def run_live_training_steps(
    num_steps: int = 5,
    games_per_step: int = 2,
    max_game_steps: int = 10,
    base_seed: int = 6000,
    condition: str = "T1",
    lambda_audit: float = 0.05,
    hf_repo_id: str | None = "renessauce/undercover-bench-rlcr",
    runs_dir: str | Path = "runs",
    checkpoint_dir: str | Path = "checkpoints",
):
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    cp_path = Path(checkpoint_dir)
    cp_path.mkdir(parents=True, exist_ok=True)
    training_log_path = runs_path / "training_steps.jsonl"

    print(f"\n=======================================================")
    print(f" STARTING LIVE RLCR GPU TRAINING: {num_steps} OPTIMIZER STEPS")
    print(f" Condition: {condition} | Games per step: {games_per_step} | Max game steps: {max_game_steps}")
    if condition == "T2":
        print(f" Active Reporting Reward: lambda={lambda_audit}")
    print(f"=======================================================\n")

    policy = ModalTrainablePolicy()

    for step_idx in range(1, num_steps + 1):
        step_seed = base_seed + (step_idx * 100)
        print(f"\n>>> [Step {step_idx}/{num_steps}] Sampling {games_per_step} games on GPU...")

        step_traces: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        step_reports = 0

        for g_idx in range(games_per_step):
            game_seed = step_seed + g_idx
            policy.set_episode_seed(game_seed)

            episode_path = run_episode(
                game_name="secret_hitler",
                seed=game_seed,
                condition=condition,
                out_dir=runs_path,
                policy=policy,
                max_steps=max_game_steps,
                channel_enabled=True,
            )
            lines = [json.loads(l) for l in episode_path.read_text().splitlines() if l.strip()]
            end_record = lines[-1] if lines[-1].get("kind") == "episode_end" else {}
            outcomes.append(end_record)

            reports = [l for l in lines if l.get("kind") == "human_report"]
            delivered = [r for r in reports if r.get("delivered")]
            step_reports += len(delivered)

            # Collect trainable traces from this game
            step_traces.extend(policy.recorded_decisions)
            print(f"  • Game {g_idx + 1}/{games_per_step} (Seed {game_seed}) -> Winner: {end_record.get('winner')}, Delivered reports: {len(delivered)}, Decisions recorded: {len(policy.recorded_decisions)}")

        # Calculate group return & advantage (incorporating audit reporting reward in T2)
        wins = [1.0 if o.get("winner") == "Liberal" else 0.0 for o in outcomes]
        mean_win = sum(wins) / len(wins)
        s_audit = 1.0 if step_reports > 0 else 0.0
        r_org = mean_win + (lambda_audit * s_audit if condition == "T2" else 0.0)
        advantage = (r_org - 0.5)

        print(f"\n  [RLCR Advantage] Liberal Win-Rate: {mean_win * 100:.1f}%, Delivered Reports: {step_reports}, R_org: {r_org:.3f}, Group Advantage: {advantage:+.3f}")
        print(f"  [Backward Pass] Submitting {len(step_traces)} decision traces to Modal GPU container...")

        # LIVE GPU BACKWARD PASS
        train_res = policy.train_step(
            decisions=step_traces,
            advantage=advantage,
            clip_epsilon=0.2,
        )

        loss = train_res.get("loss", 0.0)
        grad_norm = train_res.get("grad_norm", 0.0)
        new_version = train_res.get("new_version", f"step_{step_idx}")
        print(f"  ✓ GPU Backward Pass Complete!")
        print(f"    - Policy Loss: {loss:.6f}")
        print(f"    - Gradient Norm: {grad_norm:.6f}")
        print(f"    - Updated Version: {new_version}")

        # Log training step
        step_record = {
            "step": step_idx,
            "policy_version": new_version,
            "loss": loss,
            "grad_norm": grad_norm,
            "liberal_win_rate": mean_win,
            "reports_delivered": step_reports,
            "r_org": r_org,
            "advantage": advantage,
            "tokens_updated": train_res.get("token_count", 0),
            "games_in_batch": games_per_step,
        }
        with open(training_log_path, "a") as f:
            f.write(json.dumps(step_record) + "\n")

        # Save checkpoint & persist to Hugging Face
        step_cp_path = save_checkpoint_to_disk(
            checkpoint_dir=cp_path,
            model=None,
            base_model_name="Qwen/Qwen2.5-7B-Instruct",
            step=step_idx,
            runs_dir=runs_path,
            training_log_path=training_log_path,
        )

        if hf_repo_id:
            print(f"  [Hugging Face] Uploading checkpoint step_{step_idx} to {hf_repo_id}...")
            upload_checkpoint_to_huggingface(
                checkpoint_dir=step_cp_path,
                repo_id=hf_repo_id,
                private=True,
            )

    print(f"\n=======================================================")
    print(f" ALL {num_steps} LIVE TRAINING STEPS COMPLETED!")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--games-per-step", type=int, default=2)
    parser.add_argument("--max-game-steps", type=int, default=10, help="Maximum rounds per game")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--condition", default="T2")
    parser.add_argument("--lambda-audit", type=float, default=0.05)
    parser.add_argument("--hf-repo", default="renessauce/undercover-bench-rlcr")
    args = parser.parse_args()

    run_live_training_steps(
        num_steps=args.steps,
        games_per_step=args.games_per_step,
        max_game_steps=args.max_game_steps,
        base_seed=args.seed,
        condition=args.condition,
        lambda_audit=args.lambda_audit,
        hf_repo_id=args.hf_repo,
    )
