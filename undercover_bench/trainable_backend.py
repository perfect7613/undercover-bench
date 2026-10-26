"""Route only designated Liberal teammates to separate trainable policies.

The router sees controller roles; model prompts contain only each player's
ordinary observation. Opponents remain on their frozen policy.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from .decisions import decision_model
from .rlcr import AuthoredDecision, RLCRInputError
from .runner import decision_prompt


class TrainableOrganizationPolicy:
    model_key = "trainable_organization"

    def __init__(self, backends, opponent, *, auditor_instructions=""):
        self.backends = dict(backends)
        self.opponent = opponent
        self.auditor_instructions = auditor_instructions

    def bind_roles(self, roles, auditor):
        expected = {"ordinary"} if auditor is None else {"ordinary", "auditor"}
        if set(self.backends) != expected or len({id(x) for x in self.backends.values()}) != len(expected):
            raise RLCRInputError("ordinary and auditor require distinct backend instances")
        self.roles, self.auditor = dict(roles), auditor
        self.identities = {slot: backend.identity() for slot, backend in self.backends.items()}
        if any(not info.get("policy_version") for info in self.identities.values()):
            raise RLCRInputError("every trainable slot needs a real policy version")

    def set_episode_seed(self, seed):
        self.seed, self.count = seed, 0
        if hasattr(self.opponent, "set_episode_seed"):
            self.opponent.set_episode_seed(seed)

    def decide(self, observation, legal):
        actor = observation["you"]
        if self.roles[actor] != "Liberal":
            response = dict(self.opponent.decide(observation, legal))
            if response.get("training_trace") is not None:
                raise RLCRInputError("frozen opponent returned a trainable trace")
            return response
        slot = "auditor" if actor == self.auditor else "ordinary"
        self.count += 1
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.count}:{actor}".encode()).digest()[:4], "big")
        schema_model = decision_model(observation["phase"], legal)
        prompt = decision_prompt(observation, legal)
        if slot == "auditor" and self.auditor_instructions:
            prompt = self.auditor_instructions + "\n\n" + prompt
        prompt += "\nJSON schema: " + json.dumps(schema_model.model_json_schema(), sort_keys=True)
        sample = self.backends[slot].sample(prompt, seed)
        if any(sample.get(k) != v for k, v in self.identities[slot].items()):
            raise RLCRInputError("policy identity changed during collection")
        raw = sample["text"]
        # Preserve the complete emitted text; interpret only its final answer.
        final = raw.rsplit("</think>", 1)[-1].strip()
        try:
            if sample.get("finish_reason") != "stop":
                raise ValueError("generation exhausted its token budget")
            response = schema_model.model_validate(json.loads(final)).model_dump()
            error = None
        except (ValueError, ValidationError) as exc:
            response = {"action": None, "rationale": "Invalid model output; episode excluded from training"}
            error = str(exc)[:500]
        trace = {key: sample[key] for key in ("context_token_ids", "generated_token_ids", "behavior_logprobs",
                 "policy_version", "model_revision", "tokenizer_hash", "template_hash", "sampling_temperature")}
        trace.update(actor=actor, policy_slot=slot)
        # Validate before recording; sequence is assigned by the controller ledger.
        AuthoredDecision(sequence=1, **trace)
        response.update(raw_output=raw, model_key=slot, training_trace=trace,
                        model_metadata={"sampling_seed": seed, "validation_error": error,
                                        "finish_reason": sample.get("finish_reason")})
        return response


class ModalPolicyClient:
    """Explicit model/slot identity; no hard-coded replacement model."""
    def __init__(self, server, checkpoint_root):
        self.server, self.checkpoint_root = server, checkpoint_root

    def identity(self):
        return self.server.identity.remote()

    def sample(self, prompt, seed):
        return self.server.sample.remote(prompt, seed)

    def accumulate(self, weighted, config, transaction):
        return self.server.accumulate.remote(weighted, config.model_dump(), transaction)

    def commit(self, transaction):
        return self.server.commit.remote(transaction)

    def restore(self, path):
        return self.server.restore.remote(path)


# Compatibility name for imports. Routing still requires the same explicit inputs.
ModalTrainablePolicy = TrainableOrganizationPolicy
