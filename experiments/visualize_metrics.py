"""Visualizer and reporting dashboard for Undercover Bench training and game runs.

Parses episode ledgers and training logs to plot:
1. Win rates across factions & conditions
2. Reporting behavior (reports delivered, visible evidence citations, false accusations)
3. Cumulative and per-episode rewards (Liberal organizational return, audit scores)
4. Policy training loss & gradient norms across optimization steps

Also produces a formatted CLI summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_runs_and_training(runs_dir: str | Path, training_log_path: str | Path | None = None) -> dict[str, Any]:
    runs_path = Path(runs_dir)
    episodes: list[dict[str, Any]] = []

    for file in sorted(runs_path.glob("*.jsonl")):
        if file.name.startswith("trainable_rollouts"):
            continue
        try:
            lines = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        except Exception:
            continue
        if not lines or lines[0].get("kind") != "episode_start":
            continue

        start = lines[0]
        end = lines[-1] if lines[-1].get("kind") == "episode_end" else {}

        reports = [r for r in lines if r.get("kind") == "human_report"]
        decisions = [r for r in lines if r.get("kind") == "decision"]
        token_decisions = [r for r in lines if r.get("kind") == "training_token_decision"]

        delivered_reports = sum(1 for r in reports if r.get("delivered"))
        valid_citations = sum(1 for r in reports if r.get("evidence_ids_visible_to_reporter"))

        reward_data = end.get("reward") or {}
        audit_data = end.get("audit") or {}

        episodes.append({
            "file": file.name,
            "episode_id": start.get("episode_id", file.stem),
            "game": start.get("game"),
            "condition": start.get("condition"),
            "seed": start.get("seed"),
            "winner": end.get("winner"),
            "completed": end.get("completed", False),
            "steps": end.get("steps", 0),
            "invalid_actions": end.get("invalid_actions", 0),
            "total_reports": len(reports),
            "delivered_reports": delivered_reports,
            "valid_evidence_citations": valid_citations,
            "decisions_count": len(decisions),
            "trainable_tokens_count": len(token_decisions),
            "org_return": reward_data.get("R_org") if isinstance(reward_data, dict) else reward_data,
            "audit_score": audit_data.get("S_audit") if isinstance(audit_data, dict) else None,
        })

    training_steps: list[dict[str, Any]] = []
    if training_log_path and Path(training_log_path).is_file():
        try:
            for line in Path(training_log_path).read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    if "loss" in item:
                        training_steps.append(item)
        except Exception:
            pass

    return {"episodes": episodes, "training_steps": training_steps}


def print_console_summary(data: dict[str, Any]) -> None:
    episodes = data["episodes"]
    training_steps = data["training_steps"]

    print("\n" + "=" * 80)
    print(" UNDERCOVER BENCH: EXPERIMENT & TRAINING SUMMARY")
    print("=" * 80)

    if not episodes:
        print("No episode ledgers found.")
    else:
        print(f"\n[Episodes Summary] Total Games Processed: {len(episodes)}")
        header = f"{'Episode ID':<32} {'Cond':<6} {'Winner':<9} {'Reports':<9} {'Valid Ev':<9} {'Reward':<8}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for ep in episodes:
            ep_id = ep['episode_id'][:30]
            cond = str(ep['condition'])
            winner = str(ep['winner'] or 'None')
            rep = f"{ep['delivered_reports']}/{ep['total_reports']}"
            ev = str(ep['valid_evidence_citations'])
            rew = f"{ep['org_return']:.2f}" if isinstance(ep['org_return'], (int, float)) else "N/A"
            print(f"{ep_id:<32} {cond:<6} {winner:<9} {rep:<9} {ev:<9} {rew:<8}")

        # Overall Aggregates
        lib_wins = sum(1 for ep in episodes if ep["winner"] == "Liberal")
        fas_wins = sum(1 for ep in episodes if ep["winner"] == "Fascist")
        total_reports = sum(ep["total_reports"] for ep in episodes)
        valid_evs = sum(ep["valid_evidence_citations"] for ep in episodes)
        print("\nAggregates:")
        print(f"  • Liberal Wins: {lib_wins} | Fascist Wins: {fas_wins} | Win Rate (Liberal): {lib_wins/len(episodes)*100:.1f}%")
        print(f"  • Total Reports Attempted: {total_reports} (Grounded Citations: {valid_evs})")

    if training_steps:
        print("\n[RLCR Policy Training Updates]")
        print(f"  • Total Gradient Steps: {len(training_steps)}")
        latest = training_steps[-1]
        print(f"  • Latest Step {latest.get('step')}: Loss = {latest.get('loss', 0.0):.4f}, Grad Norm = {latest.get('grad_norm', 0.0):.4f}")
    print("=" * 80 + "\n")


def plot_metrics(data: dict[str, Any], output_path: str | Path = "runs/training_dashboard.png") -> str | None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed; skipping graphic plot generation.")
        return None

    episodes = data["episodes"]
    training = data["training_steps"]

    if not episodes and not training:
        print("No data available to plot.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Undercover Bench: Behavioral & Training Metrics", fontsize=15, fontweight="bold")

    # Plot 1: Cumulative Wins across games
    ax1 = axes[0, 0]
    if episodes:
        indices = list(range(1, len(episodes) + 1))
        lib_cum = [sum(1 for e in episodes[:i] if e["winner"] == "Liberal") for i in indices]
        fas_cum = [sum(1 for e in episodes[:i] if e["winner"] == "Fascist") for i in indices]
        ax1.plot(indices, lib_cum, label="Liberal Wins", color="#2b5c8f", marker="o", linewidth=2)
        ax1.plot(indices, fas_cum, label="Fascist Wins", color="#b03030", marker="s", linewidth=2)
        ax1.set_title("Cumulative Wins Across Games")
        ax1.set_xlabel("Game Index")
        ax1.set_ylabel("Wins Count")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "No game data", ha="center", va="center")

    # Plot 2: Reporting Behavior & Evidence Citations
    ax2 = axes[0, 1]
    if episodes:
        indices = list(range(1, len(episodes) + 1))
        reports = [e["delivered_reports"] for e in episodes]
        citations = [e["valid_evidence_citations"] for e in episodes]
        width = 0.35
        ax2.bar([x - width/2 for x in indices], reports, width=width, label="Delivered Reports", color="#e67e22")
        ax2.bar([x + width/2 for x in indices], citations, width=width, label="Grounded Evidence", color="#27ae60")
        ax2.set_title("Human Reporting & Evidence Verification")
        ax2.set_xlabel("Game Index")
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No report data", ha="center", va="center")

    # Plot 3: Organizational Returns & Rewards
    ax3 = axes[1, 0]
    if episodes:
        indices = list(range(1, len(episodes) + 1))
        rewards = [e["org_return"] if isinstance(e["org_return"], (int, float)) else 0.0 for e in episodes]
        ax3.plot(indices, rewards, label="Organizational Return (R_org)", color="#8e44ad", marker="^", linewidth=2)
        ax3.set_title("Liberal Faction Returns (R_org)")
        ax3.set_xlabel("Game Index")
        ax3.set_ylabel("Reward Value")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.6)
    else:
        ax3.text(0.5, 0.5, "No reward data", ha="center", va="center")

    # Plot 4: Training Loss & Gradient Norm
    ax4 = axes[1, 1]
    if training:
        steps = [t.get("step", i) for i, t in enumerate(training, start=1)]
        losses = [t.get("loss", 0.0) for t in training]
        grad_norms = [t.get("grad_norm", 0.0) for t in training]
        ax4.plot(steps, losses, label="PPO Surrogate Loss", color="#c0392b", linewidth=2)
        ax4.set_title("Policy Gradient Loss (RLCR)")
        ax4.set_xlabel("Optimization Step")
        ax4.set_ylabel("Loss")
        ax4_twin = ax4.twinx()
        ax4_twin.plot(steps, grad_norms, label="Grad Norm", color="#7f8c8d", linestyle=":")
        ax4_twin.set_ylabel("Grad Norm")
        ax4.legend(loc="upper left")
        ax4_twin.legend(loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.6)
    else:
        ax4.text(0.5, 0.5, "No policy training steps recorded yet", ha="center", va="center", color="#7f8c8d")
        ax4.set_title("Policy Gradient Loss (RLCR)")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=200)
    plt.close()
    print(f"Metrics plot saved to: {out.resolve()}")
    return str(out.resolve())


def main():
    parser = argparse.ArgumentParser(description="Plot and display Undercover Bench metrics.")
    parser.add_argument("--runs", default="runs", help="Directory containing game episode JSONL files.")
    parser.add_argument("--training-log", default="runs/training_steps.jsonl", help="Path to RLCR training steps log.")
    parser.add_argument("--out", default="runs/training_dashboard.png", help="Path to save generated PNG plot.")
    args = parser.parse_args()

    data = parse_runs_and_training(args.runs, args.training_log)
    print_console_summary(data)
    plot_metrics(data, args.out)


if __name__ == "__main__":
    main()
