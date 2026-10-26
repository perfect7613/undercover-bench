"""Synthetic local proof fixtures for the offline readiness contract."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from experiments.check_readiness import check_readiness, main
from experiments.prepare_pilot import build_pilot_manifest, freeze_manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, *, completed=False, exported=False, proofs=False, blocked=True):
    run = root / "run"
    run.mkdir()
    models = root / "models.json"
    model = {"repo_id": "fixture/GLM", "revision": "a" * 40,
             "blocked_reason": "known fused expert memory/coverage blocker" if blocked else None}
    write_json(models, {"models": {"glm_flash": model, "deepseek_flash": {"blocked_reason": "optional loader"}}})
    pilot = root / "pilot.json"
    freeze_manifest(pilot, build_pilot_manifest(git_commit="fixture", created_at="2026-10-26T00:00:00Z"))
    dpo = {"state_path": "tinker://dpo:train:0/weights/final",
           "sampler_path": "tinker://dpo:train:0/sampler_weights/final"}
    sft = {"state_path": "tinker://sft:train:0/weights/final",
           "sampler_path": "tinker://sft:train:0/sampler_weights/final"}
    write_json(run / "dpo/checkpoint.json", dpo)
    if completed:
        write_json(run / "sft/started.json", {"source": dpo})
        write_json(run / "sft/checkpoint.json", sft)
        write_json(run / "sft/runtime.json", {"is_lora": True, "lora_rank": 32, "model_id": "sft:train:0"})
        (run / "sft/metrics.jsonl").write_text(json.dumps({"step": 1, "train_mean_nll": 1.05, "transcripts": 24}) + "\n")
    if exported:
        bundle = run / "hf_sft_export"
        adapter = bundle / "adapter"
        adapter.mkdir(parents=True)
        write_json(adapter / "adapter_config.json", {"peft_type": "LORA", "r": 32, "lora_alpha": 32})
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        entries = {}
        cursor = 0
        for name, shape in [(prefix + ".lora_A.weight", [32, 4]), (prefix + ".lora_B.weight", [5, 32])]:
            size = shape[0] * shape[1] * 4
            entries[name] = {"dtype": "F32", "shape": shape, "data_offsets": [cursor, cursor + size]}
            cursor += size
        header = json.dumps(entries).encode()
        (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * cursor)
        files = {"adapter/" + p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in adapter.iterdir()}
        write_json(bundle / "artifact_manifest.json", {"introspection_sft_completed": True, "files": files})
        write_json(run / "character_adapter_manifest.json", {
            "purpose": "character_training_dpo_plus_introspection_sft_candidate", "source_checkpoint": sft,
            "sampler_path": sft["sampler_path"], "adapter_dir": str(adapter),
            "serving_compatibility_verified": False, "rlcr_training_performed": False})
    if proofs:
        evidence_path = run / "fixture-evidence.json"
        # Explicit synthetic fixture; this file is not a research evaluation result.
        write_json(evidence_path, {"purpose": "synthetic unit test proof artifact"})
        artifacts = {evidence_path.name: sha(evidence_path)}
        write_json(run / "character_learning_evidence.json", {
            "checkpoint": sft["sampler_path"], "held_out": True, "independent_of_training": True,
            "reviewer": "synthetic-test-not-human", "improvement_established": True, "controls_saturated": False,
            "baseline": {"correct": 80, "eligible": 100}, "sft": {"correct": 90, "eligible": 100},
            "paired_effect_ci": [.02, .18], "false_positive_noninferiority_passed": True,
            "task_performance_noninferiority_passed": True, "artifacts": artifacts})
        write_json(run / "runtime_compatibility.json", {
            "checkpoint": sft["sampler_path"], "adapter_sha256": sha(run / "hf_sft_export/adapter/adapter_model.safetensors"),
            "model_revision": model["repo_id"] + "@" + model["revision"], "verified": True,
            "same_base_verified": True, "protocol_frozen_before_evaluation": True,
            "runtime_revision": "synthetic-test-runtime", "tokenizer_sha256": "b" * 64,
            "template_sha256": "c" * 64, "complete_trained_target_coverage": True,
            "trained_projection_pair_count": 1, "applied_projection_pair_count": 1,
            "numerical_before_quantization": {"passed": True, "max_abs_error": .0001, "tolerance": .001},
            "quantized_behavior_check": {"passed": True, "cases": 10}, "artifacts": artifacts})
    return run, models, pilot


class OfflineReadinessTests(unittest.TestCase):
    def test_missing_artifacts_never_green_and_optional_deepseek_not_primary_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory))
            report = check_readiness(run, models, pilot)
            self.assertFalse(report["ready_for_primary_rlcr"])
            self.assertTrue(report["baseline_pilot_configured"])
            self.assertFalse(report["checks"]["introspection_sft"]["passed"])
            self.assertFalse(report["optional_training_profiles"]["deepseek_flash"]["required_for_initial_glm_rlcr"])

    def test_started_file_is_not_completed_training(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory))
            write_json(run / "sft/started.json", {"transcripts": 24, "status": "started"})
            report = check_readiness(run, models, pilot)
            self.assertEqual(report["checks"]["introspection_sft"]["status"], "started_not_completed")
            self.assertFalse(report["ready_for_primary_rlcr"])

    def test_sft_requires_finite_metrics_matching_runtime_and_distinct_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory), completed=True)
            self.assertTrue(check_readiness(run, models, pilot)["checks"]["introspection_sft"]["passed"])
            (run / "sft/metrics.jsonl").write_text('{"step":1,"train_mean_nll":NaN,"transcripts":24}\n')
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["introspection_sft"]["passed"])
            (run / "sft/metrics.jsonl").write_text('{"step":1,"train_mean_nll":1.1,"transcripts":24}\n')
            write_json(run / "sft/runtime.json", {"is_lora": True, "lora_rank": 16, "model_id": "sft:train:0"})
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["introspection_sft"]["passed"])

    def test_dpo_only_release_cannot_be_called_m2(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory), completed=True, exported=True)
            data = json.loads((run / "character_adapter_manifest.json").read_text())
            data["purpose"] = "character_training_dpo_candidate"
            write_json(run / "character_adapter_manifest.json", data)
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["sft_export"]["passed"])
            write_json(run / "sft/checkpoint.json", json.loads((run / "dpo/checkpoint.json").read_text()))
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["introspection_sft"]["passed"])

    def test_export_hash_and_source_checkpoint_binding_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory), completed=True, exported=True)
            report = check_readiness(run, models, pilot)
            self.assertTrue(report["checks"]["sft_export"]["passed"])
            path = run / "hf_sft_export/adapter/adapter_model.safetensors"
            data = path.read_bytes()
            path.write_bytes(data[:-1] + b"\1")
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["sft_export"]["passed"])

    def test_known_glm_blocker_wins_even_when_all_other_proof_records_present(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory), completed=True, exported=True, proofs=True)
            with patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
                report = check_readiness(run, models, pilot)
            for name, check in report["checks"].items():
                if name != "glm_primary_training_profile":
                    self.assertTrue(check["passed"], (name, check))
            self.assertEqual(report["checks"]["glm_primary_training_profile"]["status"], "blocked")
            self.assertFalse(report["ready_for_primary_rlcr"])

    def test_saturated_counts_override_optimistic_learning_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory), completed=True, exported=True, proofs=True)
            path = run / "character_learning_evidence.json"
            evidence = json.loads(path.read_text())
            evidence["baseline"] = {"correct": 100, "eligible": 100}
            evidence["sft"] = {"correct": 100, "eligible": 100}
            write_json(path, evidence)
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["independent_character_learning"]["passed"])

    def test_a_bare_compatibility_flag_or_missing_covered_target_cannot_clear_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory), completed=True, exported=True, proofs=True)
            path = run / "runtime_compatibility.json"
            evidence = json.loads(path.read_text())
            evidence["applied_projection_pair_count"] = 0
            write_json(path, evidence)
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["full_adapter_runtime_compatibility"]["passed"])
            write_json(path, {"verified": True})
            self.assertFalse(check_readiness(run, models, pilot)["checks"]["full_adapter_runtime_compatibility"]["passed"])

    def test_exit_is_nonzero_only_when_require_ready_is_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            run, models, pilot = fixture(Path(directory))
            args = ["--run-dir", str(run), "--models", str(models), "--pilot-manifest", str(pilot)]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
            self.assertFalse(json.loads(output.getvalue())["ready_for_primary_rlcr"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--require-ready"]), 1)


if __name__ == "__main__":
    unittest.main()
