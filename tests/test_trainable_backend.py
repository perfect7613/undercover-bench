import unittest
from unittest.mock import MagicMock
from undercover_bench.trainable_backend import ModalTrainablePolicy


class ModalTrainablePolicyTests(unittest.TestCase):
    def test_mocked_sample_and_train_step(self):
        policy = ModalTrainablePolicy.__new__(ModalTrainablePolicy)
        policy.model_key = "modal_trainable"
        policy.recorded_decisions = []
        policy.server = MagicMock()

        # Mock sample_decision response
        policy.server.sample_decision.remote.return_value = {
            "text": '{"action": "Ja", "rationale": "I support this government"}',
            "context_token_ids": [1, 2, 3],
            "generated_token_ids": [10, 20],
            "behavior_logprobs": [-0.2, -0.4],
            "policy_version": "v1",
            "model_revision": "test-model",
            "tokenizer_hash": "tok_hash",
            "template_hash": "default",
        }

        obs = {
            "you": "Aster",
            "phase": "vote",
            "round": 1,
            "turn": 1,
        }
        legal = ["Ja", "Nein"]

        decision = policy.decide(obs, legal)
        self.assertEqual(decision["action"], "Ja")
        self.assertEqual(decision["model_key"], "modal_trainable")
        self.assertIn("training_trace", decision)
        trace = decision["training_trace"]
        self.assertEqual(trace["context_token_ids"], [1, 2, 3])
        self.assertEqual(trace["generated_token_ids"], [10, 20])
        self.assertEqual(trace["behavior_logprobs"], [-0.2, -0.4])

        # Mock train_step response
        policy.server.train_step.remote.return_value = {
            "updated": True,
            "loss": 0.05,
            "grad_norm": 0.12,
            "step": 1,
            "new_version": "step_1",
            "token_count": 2,
        }

        res = policy.train_step([trace], advantage=0.5)
        self.assertTrue(res["updated"])
        self.assertEqual(res["step"], 1)


if __name__ == "__main__":
    unittest.main()
