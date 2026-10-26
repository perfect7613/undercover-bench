import unittest
from unittest.mock import patch

from undercover_bench.runner import AuditedSeatPolicy, ModalPolicy
from undercover_bench.replay import ReplayPrefixPolicy


class FakeFunction:
    def __init__(self): self.calls = 0
    def remote(self, prompt, **kwargs):
        self.calls += 1
        return {"text": '{"action":"Ja"}', "character_adapter_applied": kwargs.get("use_character", False)}


class SeededPolicy(ModalPolicy):
    def __init__(self, key, auditor_instructions="", initialization="M0", character_candidate=False):
        self.model_key, self.auditor_instructions = key, auditor_instructions
        self.character_candidate = character_candidate
        self.fn = FakeFunction()


class ReplayTests(unittest.TestCase):
    def delegate(self):
        return AuditedSeatPolicy(dict(Aster="glm_flash", Briar="deepseek_flash", Cedar="qwen_dense",
                                      Dove="muse_glimmer", Elm="glm_flash"), "Aster", "glm_flash", character=True)

    @patch("undercover_bench.runner.ModalPolicy", SeededPolicy)
    def test_replay_preserves_next_sampling_seed_without_requerying_models(self):
        old = self.delegate();old.set_episode_seed(42)
        obs = {"you": "Aster", "phase": "vote", "round": 1, "auditor": True}
        legal = ["Ja", "Nein"]
        first = old.decide(obs, legal)
        second = old.decide(obs, legal)
        row = {**first, "kind": "decision", "sequence": 4, "player": "Aster", "phase": "vote",
               "observation": obs, "legal": legal}
        delegate = self.delegate()
        replay = ReplayPrefixPolicy(delegate, [{"kind": "episode_start", "condition": "B1", "episode_id": "parent"}, row])
        replay.set_episode_seed(42)
        recovered = replay.decide(obs, legal)
        self.assertEqual(delegate.auditor_policy.fn.calls, 0)
        self.assertEqual(recovered["model_metadata"]["replayed_from"]["sequence"], 4)
        next_response = replay.decide(obs, legal)
        self.assertEqual(delegate.auditor_policy.fn.calls, 1)
        self.assertEqual(next_response["model_metadata"]["sampling_seed"], second["model_metadata"]["sampling_seed"])
        self.assertNotIn("replayed_from", first["model_metadata"])

    @patch("undercover_bench.runner.ModalPolicy", SeededPolicy)
    def test_changed_observation_or_completed_source_is_rejected(self):
        delegate = self.delegate();delegate.set_episode_seed(42)
        obs = {"you": "Aster", "phase": "vote", "round": 1, "auditor": True}
        response = delegate.decide(obs, ["Ja", "Nein"])
        start = {"kind": "episode_start", "condition": "B1", "episode_id": "parent"}
        row = {**response, "kind": "decision", "sequence": 4, "player": "Aster", "phase": "vote",
               "observation": obs, "legal": ["Ja", "Nein"]}
        replay = ReplayPrefixPolicy(self.delegate(), [start, row]);replay.set_episode_seed(42)
        with self.assertRaisesRegex(ValueError, "diverged"):
            replay.decide({**obs, "round": 2}, ["Ja", "Nein"])
        with self.assertRaisesRegex(ValueError, "completed"):
            ReplayPrefixPolicy(self.delegate(), [start, {"kind": "episode_end"}])
