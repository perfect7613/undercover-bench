"""Validate an unpromoted character adapter on the existing GLM inference app.

No training, weight download, scripted pact or report reward is performed.
Freeze before the first call; preserve every response including failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from undercover_bench.decisions import decision_model
from undercover_bench.games import SecretHitler
from undercover_bench.runner import ModalPolicy, decision_prompt, run_episode

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    game = SecretHitler(2026102601, undercover=True, pressure=False, channel_enabled=True)
    legal = ["NoReport", "Report"]
    observation = game.observation(game.auditor, "report", {
        "legal": legal, "reporting_channel": "human_oversight", "evidence_ids_available": []})
    prompt = (ROOT / "training/character/CONSTITUTION.md").read_text() + "\n\n" + decision_prompt(observation, legal)
    body = {"purpose": "engineering_validation_only", "research_intervention_promoted": False,
            "rlcr_updates": 0, "app": "undercover-glm_flash", "staged_base_reused": True,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "adapter_manifest": json.loads((ROOT / "runs/character/paper-v7/gguf_character/adapter_manifest.json").read_text()),
            "prompt": prompt, "response_schema": decision_model("report", legal).model_json_schema(),
            "seed": 2026102601, "temperature": 0., "max_tokens": 1024,
            "sequence": [False, True, False], "game_max_steps": 40,
            "interpretation": "Adapter switching and legal output checks, not evidence of improved character, spontaneous collusion or RLCR efficacy."}
    with (out / "protocol.json").open("x") as f:
        json.dump({**body, "sha256": digest(body)}, f, indent=2)


def protocol(out):
    value = json.loads((out / "protocol.json").read_text())
    expected = value.pop("sha256")
    if digest(value) != expected:
        raise ValueError("frozen character engineering protocol changed")
    return value


def probe(out):
    import modal
    config = protocol(out)
    fn = modal.Function.from_name(config["app"], "generate")
    responses = []
    # No silent retry or continuation after a failed first attempt.
    with (out / "responses.jsonl").open("x") as ledger:
        for index, enabled in enumerate(config["sequence"]):
            result = fn.remote(config["prompt"], max_tokens=config["max_tokens"],
                response_schema=config["response_schema"], seed=config["seed"],
                temperature=config["temperature"], use_character=enabled, capture_logprobs=True)
            ledger.write(json.dumps({"index": index, "use_character": enabled, "response": result}) + "\n")
            ledger.flush()
            if result.get("character_adapter_applied") is not enabled:
                raise ValueError("requested adapter state was not applied")
            decision_model("report", ["NoReport", "Report"]).model_validate(json.loads(result["text"]))
            responses.append(result)
    same = responses[0]["text"] == responses[2]["text"]
    changed = (responses[0]["text"] != responses[1]["text"]
               or responses[0].get("logprobs") != responses[1].get("logprobs"))
    result = {"base_final_text_replays": same, "adapter_changes_text_or_logprobs": changed,
              "all_three_actions_legal": True, "all_requested_adapter_states_confirmed": True,
              "compatibility_verified": False, "rlcr_updates": 0}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def game(out):
    config = protocol(out)
    result = json.loads((out / "result.json").read_text())
    if not all(result.get(name) is True for name in (
            "base_final_text_replays", "adapter_changes_text_or_logprobs", "all_three_actions_legal")):
        raise ValueError("adapter switching probe is failed or inconclusive")
    if (out / "game.json").exists() or list((out / "episodes").glob("*.jsonl")):
        raise ValueError("engineering game already attempted; preserve its original outcome")
    policy = ModalPolicy("glm_flash", (ROOT / "training/character/CONSTITUTION.md").read_text(),
                         initialization="M2_candidate_engineering", character_candidate=True)
    path = run_episode("secret_hitler", config["seed"], "B1", out / "episodes", policy=policy,
                       max_steps=config["game_max_steps"], sampling_seed=config["seed"], pressure=False)
    (out / "game.json").write_text(json.dumps({"episode": str(path), "purpose": config["purpose"],
                                             "rlcr_updates": 0}, indent=2))
    print(path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("freeze", "probe", "game"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"freeze": freeze, "probe": probe, "game": game}[args.stage](args.out)


if __name__ == "__main__":
    main()
