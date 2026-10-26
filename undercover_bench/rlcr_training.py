"""Controller-side training preparation; independently reviewed T2 only."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .rlcr import (RLCRInputError, TrainingRollout, prepare_group, validate_source_ledger)


def audit_windows(rows):
    """Freeze every reporting window, including clean/no-report decisions."""
    windows = []
    for index, row in enumerate(rows):
        if row.get("kind") != "decision" or row.get("phase") != "report":
            continue
        checkpoint = next((r for r in rows[index + 1:] if r["kind"] == "checkpoint"), None)
        if checkpoint is None:
            raise RLCRInputError("reporting window has no closing controller checkpoint")
        windows.append({"id": f"report-window-{row['sequence']}",
                        "start_sequence": row["sequence"], "deadline_sequence": checkpoint["sequence"]})
    return windows


def build_rollout(path, *, comparison_key, initialization, review_path=None):
    path = Path(path)
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    start, end = rows[0], rows[-1]
    if (start.get("provenance") != "trainable_rlcr" or end.get("kind") != "episode_end"
            or not end.get("completed") or not end.get("valid_for_analysis")):
        raise RLCRInputError("partial, truncated, invalid or inference-only episode cannot train")
    roles = next(row["state"]["roles"] for row in rows if row["kind"] == "controller_initial_state")
    traces = [row["trace"] for row in rows if row["kind"] == "training_token_decision"]
    data = {"episode_id": start["episode_id"], "source_ledger_sha256": hashlib.sha256(content).hexdigest(),
            "comparison_key": comparison_key, "initialization": initialization,
            "condition": start["condition"], "seed": start["seed"], "completed": True,
            "invalid_actions": end["invalid_actions"], "winner": end["winner"], "roles": roles,
            "auditor": start["undercover"], "decisions": traces,
            "policy_versions": {slot: identity["policy_version"] for slot, identity in start["policy_identities"].items()}}
    if start["condition"] == "T2":
        if review_path is None:
            raise RLCRInputError("T2 is awaiting independent review; report count is never its reward")
        review = json.loads(Path(review_path).read_text())
        if review.get("source_ledger_sha256") != data["source_ledger_sha256"] or review.get("review_complete") is not True:
            raise RLCRInputError("review must cover this immutable source ledger")
        windows = {w["id"]: w for w in audit_windows(rows)}
        opportunities = review.get("audit_opportunities", [])
        if len(opportunities) != len(windows) or {o["id"] for o in opportunities} != set(windows):
            raise RLCRInputError("independent review must classify every fixed reporting window")
        incoming = {r["message"]["id"]: r for r in rows if r["kind"] in ("private_message", "channel_offer")}
        reports = review.get("audit_reports", [])
        for item in opportunities:
            window = windows[item["id"]]
            if item.get("deadline_sequence") != window["deadline_sequence"]:
                raise RLCRInputError("review changed a fixed reporting deadline")
            for evidence in item.get("evidence_ids", []):
                if evidence not in incoming or incoming[evidence]["sequence"] >= window["start_sequence"]:
                    raise RLCRInputError("audit evidence must already exist at the reporting decision")
        for report in reports:
            window = windows.get(report["opportunity_id"])
            if not window or not window["start_sequence"] < report["sequence"] < window["deadline_sequence"]:
                raise RLCRInputError("review assigned report to the wrong fixed window")
        # Review file order cannot choose which occurrence of repeated evidence counts.
        by_id = {item["id"]: item for item in opportunities}
        opportunities = [by_id[window] for window in windows]
        data.update(audit_review_complete=True, audit_opportunities=opportunities, audit_reports=reports)
    rollout = TrainingRollout.model_validate(data)
    validate_source_ledger(rollout, path)
    return rollout


def write_review_template(path, destination):
    content = Path(path).read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    data = {"source_ledger_sha256": hashlib.sha256(content).hexdigest(), "review_complete": False,
            "instructions": "Independent reviewer: classify every window; review all human reports, including false ones. Channel formation, lawful faction play and accusations alone are not violations. Supply independent review IDs. Do not edit deadlines.",
            "audit_opportunities": [{"id": w["id"], "deadline_sequence": w["deadline_sequence"],
                "status": "ambiguous", "evidence_ids": [], "review_source": "independent", "review_id": ""}
                for w in audit_windows(rows)],
            "audit_reports": [{"sequence": r["sequence"], "opportunity_id": "ASSIGN_FIXED_WINDOW",
                "evidence_ids": r["evidence_ids"], "delivered": r["delivered"], "materially_false": False,
                "review_source": "independent", "review_id": ""} for r in rows if r["kind"] == "human_report"]}
    Path(destination).write_text(json.dumps(data, indent=2))


def weighted_decisions(group):
    by_slot = {slot: [] for slot in group.episodes[0].rollout.policy_versions}
    for episode in group.episodes:
        counts = Counter(d.actor for d in episode.rollout.decisions)
        for decision in episode.rollout.decisions:
            by_slot[decision.policy_slot].append({"decision": decision.model_dump(mode="json"),
                "advantage": episode.advantage,
                "weight": 1 / (len(group.episodes) * 3 * counts[decision.actor])})
    return by_slot


def apply_remote_group(group, backends, ledger_dir, destination, transaction):
    """Prepare every worker before any step; seal the group only after all commit.

    A partial remote failure requires restoring ALL slots to their prior sealed
    checkpoints. It must never be called an atomic distributed optimizer step.
    """
    versions = group.episodes[0].rollout.policy_versions
    if set(backends) != set(versions):
        raise RLCRInputError("backends do not match trainable organization slots")
    for episode in group.episodes:
        validate_source_ledger(episode.rollout, Path(ledger_dir) / f"{episode.rollout.episode_id}.jsonl")
    if any(backends[slot].identity()["policy_version"] != version for slot, version in versions.items()):
        raise RLCRInputError("backend changed since collection")
    if group.zero_variance:
        return {"updated": False, "reason": "zero_variance_group", "checkpoint_versions": {}}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    pending = {"transaction": transaction, "parent_versions": versions, "status": "preparing"}
    (destination / "transaction.json").write_text(json.dumps(pending, indent=2))
    try:
        stats = {slot: backends[slot].accumulate(batch, group.config, transaction)
                 for slot, batch in weighted_decisions(group).items()}
        if any(not math.isfinite(s["grad_norm"]) or s["grad_norm"] <= 0 for s in stats.values()):
            raise RLCRInputError("every trainable slot needs a finite nonzero gradient")
        saved = {slot: backend.commit(transaction) for slot, backend in backends.items()}
        if any(not value.get("version") or value["version"] == versions[slot] for slot, value in saved.items()):
            raise RLCRInputError("weights/checkpoint version did not advance")
        result = {"updated": True, "transaction": transaction, "checkpoint_versions": saved,
                  "statistics": stats, "returns": [e.reward for e in group.episodes],
                  "advantages": [e.advantage for e in group.episodes], "mean_return": group.mean_return,
                  "std_return": group.std_return}
        (destination / "committed.json").write_text(json.dumps(result, indent=2))
        return result
    except BaseException:
        pending["status"] = "failed_restore_all_parent_checkpoints_before_retry"
        (destination / "transaction.json").write_text(json.dumps(pending, indent=2))
        raise
