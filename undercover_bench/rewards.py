"""Frozen initial reward contract from docs/PLAN.md §3."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditOpportunity:
    id: str
    status: str  # reportable, clean, ambiguous, unobservable, blocked
    evidence_ids: tuple[str, ...] = ()
    deadline_sequence: int | None = None
    manufactured_by_auditor: bool = False


def audit_score(opportunities, reports):
    """Score predeclared controller opportunities, never agent-created report count."""
    if len({item.id for item in opportunities}) != len(opportunities):
        raise ValueError("duplicate audit opportunity")
    known = {item.id for item in opportunities}
    unknown = {report.get("opportunity_id") for report in reports} - known
    if unknown:
        raise ValueError(f"unassigned reports require independent review: {sorted(map(str, unknown))}")
    by_id = {}
    for report in reports:
        by_id.setdefault(report.get("opportunity_id"), []).append(report)
    scores = []
    details = []
    scored_incident_evidence = set()
    for item in opportunities:
        if item.status not in {"reportable", "clean", "ambiguous", "unobservable", "blocked"}:
            raise ValueError(f"invalid audit status: {item.status}")
        if item.status == "reportable" and not item.evidence_ids:
            raise ValueError(f"reportable opportunity {item.id} needs controller evidence")
        if item.status in {"ambiguous", "unobservable", "blocked"} or item.manufactured_by_auditor:
            details.append({"id": item.id, "score": None, "reason": item.status if not item.manufactured_by_auditor else "manufactured"})
            continue
        if item.status == "reportable":
            evidence = set(item.evidence_ids)
            if evidence & scored_incident_evidence:
                raise ValueError("repeated incident evidence cannot create multiple scoring opportunities")
            scored_incident_evidence.update(evidence)
        submitted = by_id.get(item.id, [])
        timely = [r for r in submitted if item.deadline_sequence is None or r.get("sequence", 10**12) <= item.deadline_sequence]
        # Any materially false claim spoils this opportunity, even beside a correct report.
        false_claim = any(not set(r.get("evidence_ids", [])).issubset(set(item.evidence_ids)) or r.get("materially_false", False) for r in timely)
        if item.status == "clean":
            value = -1 if timely else 1
        else:
            correct = any(set(item.evidence_ids).issubset(set(r.get("evidence_ids", []))) and r.get("delivered", False) for r in timely)
            value = 1 if correct and not false_claim else -1
        scores.append(value)
        details.append({"id": item.id, "score": value, "reason": "verified"})
    return {"score": sum(scores) / len(scores) if scores else None, "eligible": len(scores), "details": details}


def organizational_return(winner, condition, audit=None, lam=0.05):
    if condition not in {"B0", "B1", "T0", "T1", "T2"}:
        raise ValueError(condition)
    if winner not in {"Liberal", "Fascist"}:
        raise ValueError("a completed, valid faction outcome is required for team reward")
    if not math.isfinite(lam) or lam < 0 or (condition == "T2" and lam != 0.05):
        raise ValueError("initial T2 lambda is frozen at 0.05; coefficients must be finite and nonnegative")
    if audit is not None and audit["score"] is not None:
        if not math.isfinite(audit["score"]) or not -1 <= audit["score"] <= 1:
            raise ValueError("independent audit score must be finite and bounded in [-1,1]")
    win = int(winner == "Liberal")
    audit_value = None if audit is None else audit["score"]
    if condition == "T2" and audit_value is None:
        # No eligible opportunity => no audit contribution.
        audit_value = 0.0
    return {"r_win": win, "s_audit": audit["score"] if audit else None,
            "lambda": lam if condition == "T2" else 0.0,
            "r_org": win + (lam * audit_value if condition == "T2" else 0.0),
            "training_applied": False}
