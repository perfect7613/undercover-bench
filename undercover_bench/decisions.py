"""Phase-specific action schema shared by constrained generation and validation."""
from typing import Literal

from pydantic import ConfigDict, Field, create_model


def decision_model(phase: str, legal: list[str]):
    if not legal or any(not isinstance(action, str) or not action for action in legal):
        raise ValueError("legal actions must be nonempty strings")
    choices = list(dict.fromkeys(legal))
    fields = {
        "action": (Literal[tuple(choices)], ...),
        "rationale": (str, Field(default="", max_length=300)),
    }
    if phase in ("whisper", "channel_offer"):
        fields["message"] = (str, Field(default="", max_length=1000))
    if phase == "report":
        fields["evidence_ids"] = (list[str], Field(default_factory=list))
    return create_model("GameDecision", __config__=ConfigDict(extra="forbid"), **fields)


def normalize_action_payload(payload: dict, phase: str, legal: list[str]):
    """Accept an unambiguous verb-plus-choice response without hiding its source."""
    if not isinstance(payload, dict):
        return payload, None
    field = "target" if phase == "nominate" else "card" if phase in ("discard", "enact", "enact_after_veto") else None
    verbs = {"nominate"} if phase == "nominate" else {"discard", "enact", "enact_after_veto"} if field else set()
    if field and payload.get("action") in verbs and payload.get(field) in legal:
        normalized = dict(payload)
        normalized["action"] = normalized.pop(field)
        return normalized, field
    return payload, None


def normalize_message_payload(payload: dict, phase: str):
    """Map a clear message-field alias while retaining the raw model output."""
    if not isinstance(payload, dict) or phase not in ("channel_offer", "whisper") or "message" in payload:
        return payload, None
    aliases = ("invitation_message",) if phase == "channel_offer" else ("private_message",)
    for alias in aliases:
        if isinstance(payload.get(alias), str):
            normalized = dict(payload)
            normalized["message"] = normalized.pop(alias)
            return normalized, alias
    return payload, None
