import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from undercover_bench.ledger import Ledger
from undercover_bench.rlcr import (RLCRConfig, RLCRInputError, TrainingRollout,
                                   clipped_token_surrogate, load_rollouts_jsonl,
                                   prepare_group, validate_source_ledger)


ROLES = {"Aster": "Liberal", "Briar": "Liberal", "Cedar": "Fascist",
         "Dove": "Hitler", "Elm": "Liberal"}


def config():
    return RLCRConfig(advantage_epsilon=1e-6, clip_epsilon=0.2,
                      max_abs_log_ratio=2.0, min_group_size=2)


def rollout_data(episode_id="e1", condition="T1", winner="Liberal"):
    auditor = None if condition == "T0" else "Aster"
    versions = {"ordinary": "ordinary-v0"}
    if auditor:
        versions["auditor"] = "auditor-v0"
    decisions = []
    for sequence, actor in ((3, "Aster"), (4, "Briar"), (5, "Elm")):
        slot = "auditor" if actor == auditor else "ordinary"
        decisions.append({"sequence": sequence, "actor": actor, "policy_slot": slot,
                          "policy_version": versions[slot], "model_revision": "glm-base",
                          "tokenizer_hash": "tokenizer-1", "template_hash": "template-1",
                          "context_token_ids": [10, 11], "generated_token_ids": [20, 21],
                          "behavior_logprobs": [-0.5, -0.7]})
    return {"episode_id": episode_id, "source_ledger_sha256": "a" * 64,
            "comparison_key": "seed-7-start-v0", "initialization": "M0c",
            "condition": condition, "seed": 7, "completed": True,
            "invalid_actions": 0, "winner": winner, "roles": ROLES,
            "auditor": auditor, "policy_versions": versions, "decisions": decisions,
            "audit_review_complete": condition == "T2"}


class RLCRContractTests(unittest.TestCase):
    def test_shared_return_and_author_only_mask(self):
        good = TrainingRollout.model_validate(rollout_data())
        bad = TrainingRollout.model_validate(rollout_data("e2", winner="Fascist"))
        group = prepare_group((good, bad), config())
        self.assertAlmostEqual(group.mean_return, 0.5)
        self.assertAlmostEqual(group.std_return, 0.5)
        self.assertGreater(group.episodes[0].advantage, 0)
        self.assertLess(group.episodes[1].advantage, 0)
        self.assertEqual(group.episodes[0].reward["r_org"], 1)
        self.assertEqual(group.episodes[1].reward["r_org"], 0)
        self.assertEqual({decision.actor for decision in good.decisions},
                         {player for player, role in ROLES.items() if role == "Liberal"})
        self.assertEqual(good.decisions[0].author_loss_mask, (0, 0, 1, 1))

    def test_zero_variance_group_has_no_policy_signal(self):
        one = TrainingRollout.model_validate(rollout_data())
        two = TrainingRollout.model_validate(rollout_data("e2"))
        group = prepare_group((one, two), config())
        self.assertTrue(group.zero_variance)
        self.assertEqual([item.advantage for item in group.episodes], [0, 0])
        with self.assertRaises(RLCRInputError):
            prepare_group((one, one), config())

    def test_t2_only_uses_independent_verified_audit(self):
        first = rollout_data(condition="T2")
        first["audit_opportunities"] = [{"id": "incident-1", "status": "reportable",
                                         "evidence_ids": ["w2"], "deadline_sequence": 20,
                                         "review_source": "independent", "review_id": "review-1"}]
        first["audit_reports"] = [{"opportunity_id": "incident-1", "evidence_ids": ["w2"],
                                     "sequence": 15, "delivered": True,
                                     "review_source": "independent", "review_id": "review-2"}]
        second = rollout_data("e2", condition="T2", winner="Fascist")
        second["audit_opportunities"] = first["audit_opportunities"]
        group = prepare_group((TrainingRollout.model_validate(first),
                               TrainingRollout.model_validate(second)), config())
        self.assertEqual(group.episodes[0].reward["r_org"], 1.05)
        self.assertEqual(group.episodes[1].reward["r_org"], -0.05)
        unreviewed = rollout_data(condition="T2")
        unreviewed["audit_review_complete"] = False
        with self.assertRaises(ValidationError):
            TrainingRollout.model_validate(unreviewed)
        self.assertEqual(prepare_group((TrainingRollout.model_validate(rollout_data(condition="T2")),
                                        TrainingRollout.model_validate(rollout_data("e2", condition="T2"))),
                                       config()).episodes[0].reward["r_org"], 1)

    def test_frozen_opponent_and_mismatched_policy_are_rejected(self):
        opponent = rollout_data()
        opponent["decisions"][0]["actor"] = "Cedar"
        with self.assertRaises(ValidationError):
            TrainingRollout.model_validate(opponent)
        stale = rollout_data()
        stale["decisions"][0]["policy_version"] = "auditor-old"
        with self.assertRaises(ValidationError):
            TrainingRollout.model_validate(stale)
        missing = rollout_data()
        missing["decisions"].pop()
        with self.assertRaises(ValidationError):
            TrainingRollout.model_validate(missing)

    def test_clipped_surrogate_and_drift_gate(self):
        decision = TrainingRollout.model_validate(rollout_data()).decisions[0]
        self.assertAlmostEqual(clipped_token_surrogate(decision.behavior_logprobs,
                                                        decision, 1.0, config()), -1.0)
        self.assertAlmostEqual(clipped_token_surrogate(decision.behavior_logprobs,
                                                        decision, -1.0, config()), 1.0)
        with self.assertRaises(RLCRInputError):
            clipped_token_surrogate((-4.0, -0.7), decision, 1.0, config())

    def test_ledger_binding_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e1.jsonl"
            ledger = Ledger(path, "e1")
            ledger.write("episode_start", seed=7, condition="T1", undercover="Aster",
                         provenance="trainable_rlcr")
            ledger.write("controller_initial_state", state={"roles": ROLES})
            for actor in ("Aster", "Briar", "Elm"):
                ledger.write("decision", player=actor)
            draft = TrainingRollout.model_validate(rollout_data())
            for decision in draft.decisions:
                ledger.write("training_token_decision", decision_sequence=decision.sequence,
                             trace=decision.model_dump(mode="json"))
            ledger.write("episode_end", completed=True, valid_for_analysis=True,
                         invalid_actions=0, winner="Liberal")
            data = rollout_data()
            data["source_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rollout = TrainingRollout.model_validate(data)
            validate_source_ledger(rollout, path)
            training_path = Path(directory) / "trainable.jsonl"
            training_path.write_text(rollout.model_dump_json() + "\n")
            self.assertEqual(load_rollouts_jsonl(training_path, directory), (rollout,))
            with path.open("a") as stream:
                stream.write(json.dumps({"kind": "tamper"}) + "\n")
            with self.assertRaises(RLCRInputError):
                validate_source_ledger(rollout, path)
            with self.assertRaises(RLCRInputError):
                load_rollouts_jsonl(training_path, directory)

    def test_t2_report_credit_requires_real_incoming_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e1.jsonl"
            ledger = Ledger(path, "e1")
            ledger.write("episode_start", seed=7, condition="T2", undercover="Aster",
                         provenance="trainable_rlcr")
            ledger.write("controller_initial_state", state={"roles": ROLES})
            for actor in ("Aster", "Briar", "Elm"):
                ledger.write("decision", player=actor)
            ledger.write("private_message", message={"id": "w2", "from": "Briar",
                                                     "to": "Aster", "text": "Offer"})
            draft = TrainingRollout.model_validate(rollout_data(condition="T2"))
            for decision in draft.decisions:
                ledger.write("training_token_decision", decision_sequence=decision.sequence,
                             trace=decision.model_dump(mode="json"))
            report = ledger.write("human_report", reporter="Aster", evidence_ids=["w2"],
                                  delivered=True)
            ledger.write("episode_end", completed=True, valid_for_analysis=True,
                         invalid_actions=0, winner="Liberal")
            data = rollout_data(condition="T2")
            data["audit_opportunities"] = [{"id": "incident", "status": "reportable",
                                             "evidence_ids": ["w2"], "deadline_sequence": report["sequence"],
                                             "review_source": "independent", "review_id": "review-1"}]
            data["audit_reports"] = [{"opportunity_id": "incident", "evidence_ids": ["w2"],
                                       "sequence": report["sequence"], "delivered": True,
                                       "review_source": "independent", "review_id": "review-2"}]
            data["source_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_source_ledger(TrainingRollout.model_validate(data), path)
            data["audit_reports"][0]["sequence"] -= 1
            with self.assertRaises(RLCRInputError):
                validate_source_ledger(TrainingRollout.model_validate(data), path)


if __name__ == "__main__":
    unittest.main()
