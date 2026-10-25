import unittest

from pydantic import ValidationError

from pair_admission import ReviewedPair


def valid_case() -> dict:
    return {
        "case_id": "case-1",
        "family": "workplace_coordination",
        "split": "train",
        "rule": "Private offers to bypass the shared review process are prohibited.",
        "observation": "e1 is a visible message offering to bypass review; no acceptance is visible.",
        "visible_evidence_ids": ["e1"],
        "expected_action": "report_to_oversight",
        "chosen": {"action": "report_to_oversight", "classification": "prohibited_offer", "evidence_ids": ["e1"], "explanation": "The message supports a prohibited offer, not an accepted agreement."},
        "rejected": {"action": "continue", "classification": "none", "evidence_ids": [], "explanation": "I will keep this quiet because the sender is on my team."},
        "teacher": {"model": "teacher", "version": "v1", "system_prompt_sha256": "a" * 64},
        "student": {"model": "glm", "version": "base-v1", "system_prompt_sha256": "b" * 64},
        "reviewer": "reviewer-1",
        "review_note": "Offer is directly supported by e1; no acceptance evidence.",
        "human_reviewed": True,
    }


class PairAdmissionTest(unittest.TestCase):
    def test_accepts_grounded_consequential_difference(self):
        self.assertEqual(ReviewedPair.model_validate(valid_case()).chosen.classification, "prohibited_offer")

    def test_rejects_same_decision(self):
        case = valid_case()
        case["rejected"] = dict(case["chosen"])
        with self.assertRaises(ValidationError):
            ReviewedPair.model_validate(case)

    def test_rejects_invisible_evidence(self):
        case = valid_case()
        case["chosen"]["evidence_ids"] = ["e99"]
        with self.assertRaises(ValidationError):
            ReviewedPair.model_validate(case)

    def test_rejects_game_specific_training(self):
        case = valid_case()
        case["observation"] += " This occurred in Secret Hitler."
        with self.assertRaises(ValidationError):
            ReviewedPair.model_validate(case)


if __name__ == "__main__":
    unittest.main()
