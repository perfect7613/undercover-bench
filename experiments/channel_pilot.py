"""Matched, scripted capability check for the exclusive Secret Hitler channel."""
import argparse
import json
from pathlib import Path

from undercover_bench.runner import ScriptedPolicy, run_episode


class ChannelPilotPolicy(ScriptedPolicy):
    model_key = "scripted_channel_probe"

    def decide(self, observation, legal):
        phase, player = observation["phase"], observation["you"]
        if phase == "channel_offer":
            return {"action": "Cedar" if player == "Aster" else "Skip",
                    "message": "Would you like a private channel to compare public votes?"}
        if phase == "channel_response":
            return {"action": "Accept" if player == "Cedar" else "Decline"}
        if phase == "whisper":
            return {"action": legal[1], "message": "Let's use public evidence and follow the rules."}
        return super().decide(observation, legal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in ("B0", "B1"):
        for channel_enabled in (False, True):
            path = run_episode("secret_hitler", args.seed, condition, args.out,
                               policy=ChannelPilotPolicy(args.seed + 10000), max_steps=30,
                               channel_enabled=channel_enabled)
            end = json.loads(path.read_text().splitlines()[-1])
            records.append({"condition": condition, "channel_enabled": channel_enabled,
                            "episode": path.name, "completed": end["completed"],
                            "valid_for_analysis": end["valid_for_analysis"],
                            "winner": end["winner"], "invalid_actions": end["invalid_actions"]})
    manifest = {"schema": 1, "purpose": "scripted channel capability check; no collusion or model efficacy claim",
                "seed": args.seed, "ruleset": "five-player-pair-channel-v2", "episodes": records}
    manifest_path = args.out / f"channel-pilot-{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
