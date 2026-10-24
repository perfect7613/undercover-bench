"""Phase-specific action schema shared by constrained generation and validation."""
from typing import Literal

from pydantic import ConfigDict, Field, create_model


def decision_model(phase: str, legal: list[str]):
    if not legal or any(not isinstance(action, str) or not action for action in legal):
        raise ValueError("legal actions must be nonempty strings")
    choices = list(dict.fromkeys(legal))
    fields = {
        "action": (Literal[tuple(choices)], ...),
        "rationale": (str, Field(max_length=300)),
    }
    if phase == "whisper":
        fields["message"] = (str, Field(default="", max_length=1000))
    if phase == "report":
        fields["evidence_ids"] = (list[str], Field(default_factory=list))
    return create_model("GameDecision", __config__=ConfigDict(extra="forbid"), **fields)
