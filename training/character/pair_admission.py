"""Admission gate for reviewed, controller-grounded character DPO pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from pydantic import BaseModel, Field, model_validator

GAME_TERMS = re.compile(r"\b(secret hitler|liar'?s bar|liberal policy|fascist policy|chancellor|liar'?s deck)\b", re.I)
LABEL_LEAK = re.compile(r"(?im)^\s*(?:\*\*)?(?:preferred|rejected)(?: response)?(?:\*\*)?\s*:?\s*$")


class Decision(BaseModel):
    action: str = Field(pattern=r"^(continue|investigate|report_to_oversight|revise_assessment|refuse_prohibited_request)$")
    classification: str
    evidence_ids: list[str]
    explanation: str = Field(min_length=25)


class Provenance(BaseModel):
    model: str = Field(min_length=2)
    version: str = Field(min_length=2)
    system_prompt_sha256: str = Field(min_length=64, max_length=64)


class ReviewedPair(BaseModel):
    case_id: str
    family: str
    split: str = Field(pattern=r"^(train|dev|sealed_test)$")
    rule: str = Field(min_length=20)
    observation: str = Field(min_length=30)
    visible_evidence_ids: list[str]
    expected_action: str
    chosen: Decision
    rejected: Decision
    teacher: Provenance
    student: Provenance
    reviewer: str = Field(min_length=2)
    review_note: str = Field(min_length=20)
    human_reviewed: bool

    @model_validator(mode="after")
    def check_pair(self) -> "ReviewedPair":
        if not self.human_reviewed:
            raise ValueError("unreviewed candidate")
        if GAME_TERMS.search(self.rule + " " + self.observation):
            raise ValueError("game-specific character data")
        if self.chosen.action != self.expected_action:
            raise ValueError("chosen action contradicts controller label")
        if self.chosen.action == self.rejected.action and self.chosen.classification == self.rejected.classification:
            raise ValueError("same action and classification on both sides")
        allowed = set(self.visible_evidence_ids)
        if not set(self.chosen.evidence_ids) <= allowed or not set(self.rejected.evidence_ids) <= allowed:
            raise ValueError("response cites evidence the auditor could not see")
        if LABEL_LEAK.search(self.chosen.explanation) or LABEL_LEAK.search(self.rejected.explanation):
            raise ValueError("preferred/rejected label leaked into answer")
        if self.expected_action == "continue" and self.chosen.action == "report_to_oversight":
            raise ValueError("false-positive report in clean case")
        if self.teacher.model == self.student.model and self.teacher.version == self.student.version and self.teacher.system_prompt_sha256 == self.student.system_prompt_sha256:
            raise ValueError("teacher and student provenance indistinguishable")
        return self


def validate_file(path: Path) -> dict:
    pairs = [ReviewedPair.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [p.case_id for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    counts: dict[str, int] = {}
    for pair in pairs:
        key = f"{pair.split}:{pair.family}:{pair.expected_action}"
        counts[key] = counts.get(key, 0) + 1
    return {"pairs": len(pairs), "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_file(args.path), indent=2))


if __name__ == "__main__":
    main()
