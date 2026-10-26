import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from undercover_bench.runner import AuditedSeatPolicy
from undercover_bench.ledger import Ledger


class FakePolicy:
    def __init__(self, model_key, auditor_instructions="", initialization="M0", character_candidate=False):
        self.model_key, self.instructions = model_key, auditor_instructions
        self.character = character_candidate
    def set_episode_seed(self, seed):
        self.seed = seed
    def decide(self, observation, legal):
        return {"action": legal[0], "model": self.model_key, "character": self.character,
                "constitution": bool(self.instructions)}


class MixedAuditorTests(unittest.TestCase):
    @patch("undercover_bench.runner.ModalPolicy", FakePolicy)
    def test_character_and_instructions_reach_only_designated_auditor(self):
        seats = dict(Aster="glm_flash", Briar="deepseek_flash", Cedar="qwen_dense",
                     Dove="muse_glimmer", Elm="glm_flash")
        p = AuditedSeatPolicy(seats, "Aster", "glm_flash", character=True)
        p.set_episode_seed(7)
        for name, model in seats.items():
            r = p.decide({"you": name, "auditor": name == "Aster"}, ["Ja"])
            self.assertEqual(r["model"], model)
            self.assertEqual(r["character"], name == "Aster")
            self.assertEqual(r["constitution"], name == "Aster")
        with self.assertRaisesRegex(ValueError, "role changed"):
            p.decide({"you": "Elm", "auditor": True}, ["Ja"])

    @patch("undercover_bench.runner.ModalPolicy", FakePolicy)
    def test_deepseek_auditor_is_prompt_only_and_all_families_required(self):
        seats = dict(Aster="deepseek_flash", Briar="glm_flash", Cedar="qwen_dense",
                     Dove="muse_glimmer", Elm="deepseek_flash")
        p = AuditedSeatPolicy(seats, "Aster", "deepseek_flash")
        r = p.decide({"you": "Aster", "auditor": True}, ["NoReport"])
        self.assertFalse(r["character"])
        self.assertTrue(r["constitution"])
        with self.assertRaisesRegex(ValueError, "all four"):
            AuditedSeatPolicy({**seats, "Cedar": "glm_flash"}, "Aster", "deepseek_flash")

    def test_episode_lines_are_durable_and_sequence_contiguous(self):
        with tempfile.TemporaryDirectory() as directory, patch("undercover_bench.ledger.os.fsync") as sync:
            path = Path(directory) / "episode.jsonl"
            ledger = Ledger(path, "test")
            ledger.write("decision", rationale="visible evidence → qualified report")
            ledger.write("checkpoint", state={"round": 1})
            self.assertEqual(sync.call_count, 2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["sequence"] for r in rows], [1, 2])
            self.assertIn("→", rows[0]["rationale"])
