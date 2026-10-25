"""Runs the next games and executes the live RLCR GPU training steps,
continuing seamlessly from where training stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from undercover_bench.checkpoint_hf import save_checkpoint_to_disk, upload_checkpoint_to_huggingface
from undercover_bench.runner import run_episode
from undercover_bench.trainable_backend import ModalTrainablePolicy


def get_current_max_step(training_log_path: Path) -> int:
    if not training_log_path.is_file():
        return 0
    steps = []
    for line in training_log_path.read_text().splitlines():
        if line.strip():
            try:
                steps.append(json.loads(line).get("step", 0))
            except Exception:
                pass
    return max(steps) if steps else 0


def run_next_steps(
    num_steps: int = 10,
    games_per_step: int = 2,
    max_game_steps: int = 5,
    seed_base: int = 9500,
    condition: str = "T2",
    lambda_audit: float = 0.05,
    hf_repo_id: str | None = "renessauce/undercover-bench-rlcr",
):
    runs_path = Path("runs")
    runs_path.mkdir(parents=True, exist_ok=True)
    cp_path = Path("checkpoints")
    cp_path.mkdir(parents=True, exist_ok=True)
    training_log_path = runs_path / "training_steps.jsonl"

    start_step = get_current_max_step(training_log_path) + 1
    end_step = start_step + num_steps - 1

    print(f"\n=======================================================")
    print(f" CONTINUING RLCR GPU TRAINING: Steps {start_step} to {end_step}")
    print(f" Games per step: {games_per_step} | Total games: {num_steps * games_per_step}")
    print(f" Condition: {condition} (ACTIVE REPORTING REWARD lambda={lambda_audit})")
    print(f"=======================================================\n")

    policy = ModalTrainablePolicy()

    for step_num in range(start_step, end_step + 1):
        step_seed = seed_base + step_num * 100
        print(f"\n>>> [Step {step_num}/{end_step}] Sampling {games_per_step} games under {condition}...")

        step_traces = []
        step_reports = 0
        step_wins = 0

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
            winner = end_record.get("winner")
            reports = [l for l in lines if l.get("kind") == "human_report"]
            delivered = [r for r in reports if r.get("delivered")]

            if winner == "Liberal":
                step_wins += 1
            step_reports += len(delivered)
            step_traces.extend(policy.recorded_decisions)

            print(f"  • Game {g_idx + 1}/{games_per_step} (Seed {game_seed}) -> Winner: {winner}, Delivered reports: {len(delivered)}, Traces: {len(policy.recorded_decisions)}")

        # Compute group return & advantage with active audit reporting bonus
        liberal_win_rate = step_wins / games_per_step
        # Reporting score: 1.0 if any delivered reports, else 0.0
        s_audit = 1.0 if step_reports > 0 else 0.0
        r_org = liberal_win_rate + (lambda_audit * s_audit if condition == "T2" else 0.0)

        advantage = (r_org - 0.5)
        print(f"  • Step {step_num} Metrics: Win Rate={liberal_win_rate*100:.1f}%, Delivered Reports={step_reports}, R_org={r_org:.3f}, Advantage={advantage:+.3f}")
        print(f"  • Submitting {len(step_traces)} decision traces to Modal GPU container for backward pass...")

        train_res = policy.train_step(
            decisions=step_traces,
            advantage=advantage,
            clip_epsilon=0.2,
        )

        loss = train_res.get("loss", 0.0)
        grad_norm = train_res.get("grad_norm", 0.0)
        new_version = train_res.get("new_version", f"step_{step_num}")
        print(f"  ✓ Backward pass complete!")
        print(f"    - Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f} | Tokens: {train_res.get('token_count', 0)}")

        # Log training step
        step_record = {
            "step": step_num,
            "policy_version": new_version,
            "loss": loss,
            "grad_norm": grad_norm,
            "liberal_win_rate": liberal_win_rate,
            "reports_delivered": step_reports,
            "r_org": r_org,
            "advantage": advantage,
            "tokens_updated": train_res.get("token_count", 0),
            "games_in_batch": games_per_step,
        }
        with open(training_log_path, "a") as f:
            f.write(json.dumps(step_record) + "\n")

        # Save checkpoint locally and to HF Hub
        cp = save_checkpoint_to_disk(
            checkpoint_dir=cp_path,
            model=None,
            base_model_name="Qwen/Qwen2.5-7B-Instruct",
            step=step_num,
            runs_dir=runs_path,
            training_log_path=training_log_path,
        )
        print(f"  • Checkpoint saved locally to {cp}")

        if hf_repo_id:
            print(f"  • Uploading checkpoint {step_num} to Hugging Face Hub: {hf_repo_id}...")
            upload_checkpoint_to_huggingface(cp, repo_id=hf_repo_id, private=True)
        with open(training_log_path, "a") as f:
            f.write(json.dumps(step_record) + "\n")

        # Save checkpoint locally and to HF Hub
        cp = save_checkpoint_to_disk(
            checkpoint_dir=cp_path,
            model=None,
            base_model_name="Qwen/Qwen2.5-7B-Instruct",
            step=step_num,
            runs_dir=runs_path,
            training_log_path=training_log_path,
        )
        print(f"  • Checkpoint saved locally to {cp}")

        if hf_repo_id:
            print(f"  • Uploading checkpoint {step_num} to Hugging Face Hub: {hf_repo_id}...")
            upload_checkpoint_to_huggingface(cp, repo_id=hf_repo_id, private=True)

    print(f"\n=======================================================")
    print(f" ALL {num_steps} TRAINING STEPS COMPLETED!")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--games-per-step", type=int, default=2)
    parser.add_argument("--max-game-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9600)
    parser.add_argument("--hf-repo", default="renessauce/undercover-bench-rlcr")
    args = parser.parse_args()

    run_next_steps(
        num_steps=args.steps,
        games_per_step=args.games_per_step,
        max_game_steps=args.max_game_steps,
        seed_base=args.seed,
        hf_repo_id=args.hf_repo,
    )
