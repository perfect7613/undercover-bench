"""CPU-side RLCR contracts and batch preparation for Secret Hitler.

This module prepares a shared Liberal-faction advantage. It cannot turn the
existing GGUF inference ledgers into on-policy training samples: a trainable
sampler must provide exact authored token IDs and behavior log probabilities.
"""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rewards import AuditOpportunity, audit_score, organizational_return


class RLCRInputError(ValueError):
    """A rollout fails a training or provenance gate."""


class AuthoredDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(gt=0)
    actor: str
    policy_slot: Literal["ordinary", "auditor"]
    policy_version: str
    model_revision: str
    tokenizer_hash: str
    template_hash: str
    context_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    sampling_temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_token_alignment(self):
        if not self.context_token_ids or not self.generated_token_ids:
            raise ValueError("context and authored generation must both be nonempty")
        if len(self.generated_token_ids) != len(self.behavior_logprobs):
            raise ValueError("one behavior log probability is required per authored token")
        if any(token < 0 for token in (*self.context_token_ids, *self.generated_token_ids)):
            raise ValueError("token IDs must be nonnegative")
        if any(not math.isfinite(value) or value > 0 for value in self.behavior_logprobs):
            raise ValueError("behavior log probabilities must be finite and <= 0")
        if not all((self.actor, self.policy_version, self.model_revision,
                    self.tokenizer_hash, self.template_hash)):
            raise ValueError("actor and policy provenance fields must be nonempty")
        return self

    @property
    def author_loss_mask(self) -> tuple[int, ...]:
        """Only this decision's newly generated tokens receive training loss."""
        return (0,) * len(self.context_token_ids) + (1,) * len(self.generated_token_ids)


class ReviewedOpportunity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    status: Literal["reportable", "clean", "ambiguous", "unobservable", "blocked"]
    evidence_ids: tuple[str, ...] = ()
    deadline_sequence: int | None = None
    manufactured_by_auditor: bool = False
    review_source: Literal["independent"]
    review_id: str

    @model_validator(mode="after")
    def check_review(self):
        if not self.id or not self.review_id:
            raise ValueError("opportunity and independent review IDs are required")
        return self

    def scorer_input(self) -> AuditOpportunity:
        return AuditOpportunity(self.id, self.status, self.evidence_ids,
                                self.deadline_sequence, self.manufactured_by_auditor)


class ReviewedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opportunity_id: str
    evidence_ids: tuple[str, ...]
    sequence: int = Field(gt=0)
    delivered: bool
    materially_false: bool = False
    review_source: Literal["independent"]
    review_id: str

    @model_validator(mode="after")
    def check_review(self):
        if not self.opportunity_id or not self.review_id:
            raise ValueError("report and independent review IDs are required")
        return self

    def scorer_input(self) -> dict:
        return self.model_dump(exclude={"review_source", "review_id"})


class TrainingRollout(BaseModel):
    """One completed game sampled by the exact trainable policy version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    episode_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    source_ledger_sha256: str
    comparison_key: str
    initialization: str
    condition: Literal["T0", "T1", "T2"]
    seed: int
    completed: bool
    invalid_actions: int = Field(ge=0)
    winner: Literal["Liberal", "Fascist"]
    roles: dict[str, Literal["Liberal", "Fascist", "Hitler"]]
    auditor: str | None
    policy_versions: dict[Literal["ordinary", "auditor"], str]
    decisions: tuple[AuthoredDecision, ...]
    audit_review_complete: bool = False
    audit_opportunities: tuple[ReviewedOpportunity, ...] = ()
    audit_reports: tuple[ReviewedReport, ...] = ()

    @model_validator(mode="after")
    def check_episode(self):
        if not self.episode_id or not self.comparison_key or not self.initialization:
            raise ValueError("episode, comparison and initialization IDs are required")
        if len(self.source_ledger_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.source_ledger_sha256):
            raise ValueError("source ledger SHA-256 must be lowercase hex")
        if not self.completed or self.invalid_actions:
            raise ValueError("incomplete or protocol-invalid games cannot train")
        if len(self.roles) != 5 or sorted(self.roles.values()) != ["Fascist", "Hitler", "Liberal", "Liberal", "Liberal"]:
            raise ValueError("Secret Hitler training requires three Liberal and two Fascist roles")
        if self.condition == "T0" and self.auditor is not None:
            raise ValueError("T0 has no auditor")
        if self.condition == "T0" and (self.audit_opportunities or self.audit_reports):
            raise ValueError("T0 cannot include auditor opportunities or reports")
        if self.condition in ("T1", "T2") and self.roles.get(self.auditor) != "Liberal":
            raise ValueError("T1/T2 require one Liberal auditor")
        expected_slots = {"ordinary"} if self.auditor is None else {"ordinary", "auditor"}
        if set(self.policy_versions) != expected_slots or any(not value for value in self.policy_versions.values()):
            raise ValueError("policy versions must cover every trainable slot")
        if self.condition == "T2" and not self.audit_review_complete:
            raise ValueError("T2 requires an independent audit review, including a reviewed clean/empty case")
        if self.audit_reports and not self.audit_review_complete:
            raise ValueError("unreviewed reports cannot enter a training rollout")
        actors = set()
        seen_sequences = set()
        tokenizer_by_slot = {}
        template_by_slot = {}
        model_by_slot = {}
        for decision in self.decisions:
            if decision.sequence in seen_sequences:
                raise ValueError("duplicate decision sequence")
            seen_sequences.add(decision.sequence)
            if self.roles.get(decision.actor) != "Liberal":
                raise ValueError("frozen opponents may not have trainable token traces")
            expected = "auditor" if decision.actor == self.auditor else "ordinary"
            if decision.policy_slot != expected or decision.policy_version != self.policy_versions[expected]:
                raise ValueError("author slot or behavior-policy version mismatch")
            for table, value, name in ((tokenizer_by_slot, decision.tokenizer_hash, "tokenizer"),
                                       (template_by_slot, decision.template_hash, "template"),
                                       (model_by_slot, decision.model_revision, "model revision")):
                previous = table.setdefault(expected, value)
                if previous != value:
                    raise ValueError(f"{name} changed within one policy slot")
            actors.add(decision.actor)
        liberals = {player for player, role in self.roles.items() if role == "Liberal"}
        if actors != liberals:
            raise ValueError("all three Liberal teammates must contribute authored tokens")
        return self


class RLCRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_audit: float = 0.05
    advantage_epsilon: float = Field(gt=0)
    clip_epsilon: float = Field(gt=0, lt=1)
    max_abs_log_ratio: float = Field(gt=0)
    min_group_size: int = Field(ge=2)

    @model_validator(mode="after")
    def frozen_initial_reward(self):
        if self.lambda_audit != 0.05:
            raise ValueError("the initial T2 reporting coefficient is frozen at 0.05")
        return self


@dataclass(frozen=True)
class PreparedEpisode:
    rollout: TrainingRollout
    reward: dict
    advantage: float


@dataclass(frozen=True)
class PreparedGroup:
    episodes: tuple[PreparedEpisode, ...]
    mean_return: float
    std_return: float
    zero_variance: bool
    config: RLCRConfig


def validate_source_ledger(rollout: TrainingRollout, path: str | Path) -> None:
    """Bind a training record to a real completed controller ledger."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != rollout.source_ledger_sha256:
        raise RLCRInputError("source ledger hash mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows or rows[0].get("episode_id") != rollout.episode_id or rows[-1].get("kind") != "episode_end":
        raise RLCRInputError("training record does not match a terminal episode ledger")
    end = rows[-1]
    if (end.get("completed") is not True or end.get("valid_for_analysis") is not True
            or end.get("invalid_actions") != 0 or end.get("winner") != rollout.winner):
        raise RLCRInputError("ledger is incomplete, invalid, or has a different winner")
    if rows[0].get("seed") != rollout.seed or rows[0].get("condition") != rollout.condition:
        raise RLCRInputError("ledger seed or training condition mismatch")
    if rows[0].get("provenance") != "trainable_rlcr":
        raise RLCRInputError("GGUF/pilot ledgers are not trainable-policy rollouts")
    initial = next((row for row in rows if row.get("kind") == "controller_initial_state"), None)
    if initial is None or initial.get("state", {}).get("roles") != rollout.roles:
        raise RLCRInputError("controller roles do not match the training record")
    if rows[0].get("undercover") != rollout.auditor:
        raise RLCRInputError("auditor seat does not match the ledger")
    logged_decisions = {(row["sequence"], row.get("player")) for row in rows if row.get("kind") == "decision"}
    if any((decision.sequence, decision.actor) not in logged_decisions for decision in rollout.decisions):
        raise RLCRInputError("authored-token trace has no matching controller decision")
    expected_authors = {(row["sequence"], row.get("player")) for row in rows
                        if row.get("kind") == "decision" and rollout.roles.get(row.get("player")) == "Liberal"}
    if {(d.sequence, d.actor) for d in rollout.decisions} != expected_authors:
        raise RLCRInputError("every Liberal decision, including reporting, must enter the token ledger")
    token_rows = [row for row in rows if row.get("kind") == "training_token_decision"]
    token_events = {row.get("decision_sequence"): row.get("trace") for row in token_rows}
    if len(token_events) != len(token_rows):
        raise RLCRInputError("duplicate authored-token ledger event")
    if any(token_events.get(decision.sequence) != decision.model_dump(mode="json")
           for decision in rollout.decisions):
        raise RLCRInputError("authored-token trace is missing from or differs from the controller ledger")
    if set(token_events) != {d.sequence for d in rollout.decisions}:
        raise RLCRInputError("unexpected token events, including frozen opponent traces")
    incoming = {row["message"]["id"]: row["message"]
                for row in rows if row.get("kind") in ("private_message", "pact_offer", "channel_offer")
                and isinstance(row.get("message"), dict) and row["message"].get("to") == rollout.auditor}
    for opportunity in rollout.audit_opportunities:
        if opportunity.status == "reportable" and not opportunity.manufactured_by_auditor:
            if any(evidence_id not in incoming or incoming[evidence_id].get("from") == rollout.auditor
                   for evidence_id in opportunity.evidence_ids):
                raise RLCRInputError("reportable evidence was not sent to the auditor by another player")
    report_events = {row["sequence"]: row for row in rows if row.get("kind") == "human_report"}
    if rollout.audit_review_complete and set(report_events) != {r.sequence for r in rollout.audit_reports}:
        raise RLCRInputError("independent review must include every human report")
    receipts = {}
    if rows[0].get("oversight_protocol") == "append_only_inbox_v1":
        inbox = Path(path).parent / "oversight" / f"{rollout.episode_id}.jsonl"
        if report_events and not inbox.is_file():
            raise RLCRInputError("controller report has no persisted human-inbox receipt")
        if inbox.is_file():
            receipts = {row["receipt_id"]: row for row in
                        (json.loads(line) for line in inbox.read_text().splitlines())}
        from .reporting import receipt_digest
        for event in report_events.values():
            receipt = receipts.get(event.get("receipt_id"))
            if (receipt is None or receipt_digest(receipt) != event.get("receipt_sha256")
                    or receipt.get("episode_id") != rollout.episode_id
                    or receipt.get("reporter") != rollout.auditor
                    or receipt.get("evidence_ids") != event.get("evidence_ids")
                    or receipt.get("delivery_status") != "persisted_to_simulated_human_inbox"):
                raise RLCRInputError("human-report event differs from its persisted oversight receipt")
    for report in rollout.audit_reports:
        event = report_events.get(report.sequence)
        if (event is None or event.get("reporter") != rollout.auditor
                or event.get("delivered") is not report.delivered
                or set(event.get("evidence_ids", [])) != set(report.evidence_ids)
                or any(evidence_id not in incoming for evidence_id in report.evidence_ids)):
            raise RLCRInputError("reviewed report does not match a visible human-report event")


def load_rollouts_jsonl(path: str | Path, ledger_dir: str | Path) -> tuple[TrainingRollout, ...]:
    """Read separately recorded trainable rollouts and verify source ledgers."""
    rollouts = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rollout = TrainingRollout.model_validate_json(line)
        except Exception as exc:
            raise RLCRInputError(f"invalid trainable rollout on line {line_number}: {exc}") from exc
        ledger_path = Path(ledger_dir) / f"{rollout.episode_id}.jsonl"
        validate_source_ledger(rollout, ledger_path)
        rollouts.append(rollout)
    if not rollouts:
        raise RLCRInputError("no trainable rollouts were supplied")
    return tuple(rollouts)


def prepare_group(rollouts: tuple[TrainingRollout, ...], config: RLCRConfig) -> PreparedGroup:
    if len(rollouts) < config.min_group_size:
        raise RLCRInputError("comparison group is smaller than the declared group size")
    if len({rollout.episode_id for rollout in rollouts}) != len(rollouts):
        raise RLCRInputError("comparison group repeats an episode")
    first = rollouts[0]
    identity = (first.condition, first.initialization, first.comparison_key,
                first.seed, first.roles, first.auditor, first.policy_versions)
    for rollout in rollouts:
        if (rollout.condition, rollout.initialization, rollout.comparison_key,
                rollout.seed, rollout.roles, rollout.auditor, rollout.policy_versions) != identity:
            raise RLCRInputError("group mixes conditions, starts, roles, or behavior-policy versions")
    def model_fingerprint(rollout: TrainingRollout):
        return {decision.policy_slot: (decision.model_revision, decision.tokenizer_hash,
                                       decision.template_hash, decision.sampling_temperature)
                for decision in rollout.decisions}
    if any(model_fingerprint(rollout) != model_fingerprint(first) for rollout in rollouts[1:]):
        raise RLCRInputError("group mixes base models, tokenizers, or chat templates")
    rewards = []
    for rollout in rollouts:
        audit = None
        if rollout.condition == "T2":
            audit = audit_score([item.scorer_input() for item in rollout.audit_opportunities],
                                [item.scorer_input() for item in rollout.audit_reports])
        rewards.append(organizational_return(rollout.winner, rollout.condition, audit,
                                             lam=config.lambda_audit))
    values = [item["r_org"] for item in rewards]
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    zero = std == 0
    advantages = [0.0 if zero else (value - mean) / (std + config.advantage_epsilon)
                  for value in values]
    return PreparedGroup(tuple(PreparedEpisode(rollout, reward, advantage)
                               for rollout, reward, advantage in zip(rollouts, rewards, advantages)),
                         mean, std, zero, config)


def clipped_token_surrogate(new_logprobs: tuple[float, ...], decision: AuthoredDecision,
                            advantage: float, config: RLCRConfig) -> float:
    """Numerical reference for one author's clipped token objective."""
    if len(new_logprobs) != len(decision.generated_token_ids):
        raise RLCRInputError("new log probabilities must align with authored tokens")
    terms = []
    for new, old in zip(new_logprobs, decision.behavior_logprobs):
        if not math.isfinite(new) or new > 0:
            raise RLCRInputError("new log probability is invalid")
        log_ratio = new - old
        if abs(log_ratio) > config.max_abs_log_ratio:
            raise RLCRInputError("policy drift exceeds the declared off-policy gate")
        ratio = math.exp(log_ratio)
        clipped = min(1 + config.clip_epsilon, max(1 - config.clip_epsilon, ratio))
        terms.append(min(ratio * advantage, clipped * advantage))
    return -statistics.fmean(terms)
