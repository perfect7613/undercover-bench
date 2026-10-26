"""Continue one user-stopped mixed game by exact decision-prefix replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from undercover_bench.replay import ReplayPrefixPolicy
from undercover_bench.runner import AuditedSeatPolicy, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    start = rows[0]
    if (start.get("protocol_metadata", {}).get("auditor_model") != "glm_flash"
            or start.get("auditor_initialization") != "M2_candidate_engineering"):
        raise ValueError("Expected the stopped character-GLM mixed game")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {"purpose": "single_character_auditor_game_continuation", "source": str(args.source),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "source_episode_id": start["episode_id"], "rlcr_updates": 0,
                "independent_new_episode": False, "max_steps": 40,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    delegate = AuditedSeatPolicy(start["seat_models"], start["undercover"], "glm_flash", character=True)
    policy = ReplayPrefixPolicy(delegate, rows)
    path = run_episode("secret_hitler", start["seed"], "B1", args.out / "episodes", policy=policy,
        max_steps=40, pressure=start["pressure"], channel_enabled=start["channel_enabled"],
        sampling_seed=start["sampling_seed"])
    (args.out / "result.json").write_text(json.dumps({**protocol, "ledger": str(path),
        "replayed_decisions": len(policy.rows), "new_model_decisions": policy.new_decisions}, indent=2))
    print(path, flush=True)


if __name__ == "__main__":
    main()
