"""Modal-backed trainable policy for Undercover Bench.

Conforms to runner decision contract and captures token-level decisions with behavior logprobs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from .decisions import decision_model, normalize_action_payload, normalize_message_payload
from .runner import decision_prompt


class ModalTrainablePolicy:
    def __init__(self, app_name: str = "undercover-trainable-policy", cls_name: str = "TrainablePolicyServer"):
        import modal

        self.app_name = app_name
        self.cls_name = cls_name
        self.server_cls = modal.Cls.from_name(app_name, cls_name)
        self.server = self.server_cls()
        self.model_key = "modal_trainable"
        self.recorded_decisions: list[dict[str, Any]] = []

    def set_episode_seed(self, seed: int):
        self.episode_seed = seed
        self.decision_counts = {}
        self.recorded_decisions = []

    def decide(self, observation: dict[str, Any], legal: list[str]) -> dict[str, Any]:
        phase = observation["phase"]
        model = decision_model(phase, legal)
        schema = model.model_json_schema()
        base = decision_prompt(observation, legal)

        generation_seed = None
        if getattr(self, "episode_seed", None) is not None:
            key = (observation["you"], observation["phase"], observation.get("round"), observation.get("turn"))
            occurrence = self.decision_counts.get(key, 0)
            self.decision_counts[key] = occurrence + 1
            material = json.dumps([self.episode_seed, *key, occurrence], separators=(",", ":"))
            generation_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")

        prompt = base
        # Request generation with token IDs and behavior logprobs from Modal container
        sample_res = self.server.sample_decision.remote(
            prompt=prompt,
            max_new_tokens=120,
            temperature=0.7,
            seed=generation_seed,
        )

        raw = sample_res.get("text", "")
        try:
            parsed = json.loads(raw)
            normalized, normalized_field = normalize_action_payload(parsed, phase, legal)
            normalized, normalized_message_field = normalize_message_payload(normalized, phase)
            response = model.model_validate(normalized).model_dump()
            if normalized_field:
                response["source_action"] = parsed["action"]
            error = None
        except (ValueError, ValidationError) as exc:
            response = {"action": legal[0] if legal else None, "rationale": "Schema validation fallback"}
            error = str(exc)[:500]

        decision_record = {
            "actor": observation["you"],
            "phase": phase,
            "raw_output": raw,
            "error": error,
            "context_token_ids": sample_res.get("context_token_ids", []),
            "generated_token_ids": sample_res.get("generated_token_ids", []),
            "behavior_logprobs": sample_res.get("behavior_logprobs", []),
            "policy_version": sample_res.get("policy_version", "v0"),
            "model_revision": sample_res.get("model_revision", "base"),
            "tokenizer_hash": sample_res.get("tokenizer_hash", "hash"),
            "template_hash": sample_res.get("template_hash", "hash"),
        }
        self.recorded_decisions.append(decision_record)

        response.update(
            {
                "raw_output": raw,
                "error": error,
                "model_key": self.model_key,
                "training_trace": decision_record,
            }
        )
        return response

    def train_step(self, decisions: list[dict[str, Any]], advantage: float, clip_epsilon: float = 0.2) -> dict[str, Any]:
        """Trigger in-session LoRA parameter update on the active Modal GPU container."""
        return self.server.train_step.remote(
            batch_decisions=decisions,
            advantage=advantage,
            clip_epsilon=clip_epsilon,
        )

    def save_checkpoint(
        self,
        tag: str | None = None,
        hf_repo_id: str | None = None,
        private: bool = True,
    ) -> dict[str, Any]:
        return self.server.save_checkpoint.remote(
            tag=tag,
            hf_repo_id=hf_repo_id,
            private=private,
        )
