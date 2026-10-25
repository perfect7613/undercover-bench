"""Summarize and gate one-seed four-model channel pilot ledgers."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    cells = {}
    for path in args.runs.glob(f"secret_hitler-*-{args.seed}-*.jsonl"):
        rows = read(path)
        first = rows[0]
        if first.get("ruleset") != "five-player-pair-channel-v2" or first.get("model") != "mixed":
            continue
        cell = (first["condition"], first["channel_enabled"])
        if cell in cells:
            raise ValueError(f"more than one model pilot episode for {cell}; select a run explicitly")
        cells[cell] = (path, rows)
    expected = {(condition, channel) for condition in ("B0", "B1") for channel in (False, True)}
    if set(cells) != expected:
        raise ValueError(f"missing pilot cells: {expected - set(cells)}")
    starts = [rows[1]["state"] for _, rows in cells.values()]
    for key in ("roles", "deck", "president_index", "rng_state", "alive"):
        if any(state[key] != starts[0][key] for state in starts[1:]):
            raise AssertionError(f"unmatched initial {key}")
    episodes = []
    for (condition, enabled), (path, rows) in sorted(cells.items()):
        end = rows[-1]
        if end["kind"] != "episode_end" or end["completed"] or end["valid_for_analysis"] or end["reward"] is not None:
            raise AssertionError(f"partial-episode reward gate failed: {path}")
        established = [r for r in rows if r["kind"] == "channel_established"]
        messages = [r for r in rows if r["kind"] == "private_message"]
        decisions = [r for r in rows if r["kind"] == "decision"]
        if not enabled and (established or messages):
            raise AssertionError(f"communication leak in no-channel cell: {path}")
        if len(established) > 1:
            raise AssertionError(f"multiple exclusive channels: {path}")
        members = set()
        if established:
            channel = established[0]["channel"]
            members = {channel["initiator"], channel["partner"]}
        for decision in decisions:
            if decision["player"] not in members:
                observation = decision["observation"]
                if observation.get("secret_channel") is not None or any(m.get("channel_id") for m in observation.get("private_inbox", [])):
                    raise AssertionError(f"nonmember saw channel contents: {path} event {decision['sequence']}")
        episodes.append({"condition": condition, "channel_enabled": enabled, "file": path.name,
                         "decisions": len(decisions), "invalid_actions": end["invalid_actions"],
                         "channels": len(established), "channel_members": sorted(members),
                         "private_messages": len(messages),
                         "human_reports": sum(r["kind"] == "human_report" for r in rows),
                         "native_reasoning_decisions": sum(bool(r.get("model_emitted_reasoning")) for r in decisions),
                         "policy_track": next(r["state"]["policies"] for r in reversed(rows) if r["kind"] == "checkpoint")})
    manifest = {"schema": 1, "purpose": "one-round model serving and exclusive-channel pilot; no efficacy claim",
                "seed": args.seed, "matched_initial_state": True, "ruleset": "five-player-pair-channel-v2",
                "episodes": episodes}
    destination = args.runs / f"model-channel-pilot-{args.seed}.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
