"""One fresh-policy RLCR optimizer step; import PyTorch only when called.

The concrete GLM trainable sampler/backend is intentionally separate. GGUF
inference endpoints do not satisfy this module's token/log-probability contract.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .rlcr import AuthoredDecision, PreparedGroup, RLCRInputError, validate_source_ledger


class TrainablePolicyBackend(Protocol):
    """An ordinary-player policy or a distinct auditor adapter."""

    version: str

    def authored_logprobs(self, decision: AuthoredDecision):
        """Return a differentiable 1-D tensor for generated_token_ids only."""

    def zero_grad(self) -> None: ...

    def step(self) -> None: ...

    def trainable_grad_norm(self) -> float:
        """Norm over parameters this backend will update."""

    def commit_checkpoint(self, path: Path) -> str:
        """Persist weights and optimizer state, advance version, return digest."""


@dataclass(frozen=True)
class UpdateSummary:
    updated: bool
    reason: str
    loss: float | None
    episode_count: int
    authored_token_count: int
    checkpoint_versions: dict[str, str]


def apply_group_update(group: PreparedGroup, backends: dict[str, TrainablePolicyBackend],
                       checkpoint_dir: str | Path, source_ledger_dir: str | Path) -> UpdateSummary:
    """Give every Liberal teammate the same organizational advantage.

    A token mean forms each decision loss, a decision mean forms each teammate
    loss, and a teammate mean forms each episode loss. Verbose or frequently
    acting players therefore do not silently outweigh their teammates.
    """
    for episode in group.episodes:
        rollout = episode.rollout
        validate_source_ledger(rollout, Path(source_ledger_dir) / f"{rollout.episode_id}.jsonl")
    expected = set(group.episodes[0].rollout.policy_versions)
    if set(backends) != expected:
        raise RLCRInputError("ordinary/auditor backends must match the rollout slots")
    versions = group.episodes[0].rollout.policy_versions
    if any(backends[slot].version != version for slot, version in versions.items()):
        raise RLCRInputError("backend weights differ from sampled behavior policy")
    token_count = sum(len(decision.generated_token_ids)
                      for episode in group.episodes for decision in episode.rollout.decisions)
    if group.zero_variance:
        return UpdateSummary(False, "zero_variance_group", None, len(group.episodes), token_count, {})

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install a validated PyTorch training environment before RLCR updates") from exc

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=False)

    for backend in backends.values():
        backend.zero_grad()
    total_loss_value = 0.0
    for episode in group.episodes:
        for actor in (name for name, role in episode.rollout.roles.items() if role == "Liberal"):
            decision_losses = []
            for decision in episode.rollout.decisions:
                if decision.actor != actor:
                    continue
                decision_losses.append(decision)
            if not decision_losses:
                raise RLCRInputError(f"Liberal player {actor} has no authored decisions")
            for decision in decision_losses:
                current = backends[decision.policy_slot].authored_logprobs(decision)
                loss = decision_loss(current, decision, episode.advantage, group.config)
                loss = loss / (len(group.episodes) * 3 * len(decision_losses))
                loss.backward()
                total_loss_value += float(loss.detach().cpu())
    for slot, backend in backends.items():
        grad_norm = backend.trainable_grad_norm()
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RLCRInputError(f"{slot} trainable policy received no finite gradient")
    for backend in backends.values():
        backend.step()

    saved = {}
    for slot, backend in backends.items():
        version = backend.commit_checkpoint(root / slot)
        if not version or version == versions[slot] or backend.version != version:
            raise RLCRInputError(f"{slot} checkpoint did not advance to a new version")
        saved[slot] = version
    return UpdateSummary(True, "updated", total_loss_value,
                         len(group.episodes), token_count, saved)


def decision_loss(current, decision, advantage, config):
    """Shared exact-policy objective for local and remote weighted accumulation."""
    import torch
    if current.ndim != 1 or current.numel() != len(decision.generated_token_ids):
        raise RLCRInputError("backend log probabilities are not aligned to authored tokens")
    if not current.requires_grad:
        raise RLCRInputError("backend log probabilities must retain gradients")
    if not torch.isfinite(current).all().item() or (current > 0).any().item():
        raise RLCRInputError("backend returned invalid token log probabilities")
    old = torch.tensor(decision.behavior_logprobs, device=current.device, dtype=current.dtype)
    log_ratio = current - old
    if (log_ratio.detach().abs() > config.max_abs_log_ratio).any().item():
        raise RLCRInputError("policy drift exceeds the declared off-policy gate")
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    if not torch.isfinite(loss).item():
        raise RLCRInputError("non-finite collective loss")
    return loss
