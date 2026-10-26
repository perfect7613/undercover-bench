"""Tiny CPU header fixtures; never load or expand real model tensors."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_adapter_compatibility import (AdapterInspectionError, audit_llama_source,
    inspect_adapter, read_safetensors_header, target_inventory)


def write_fixture(directory, shapes):
    header, payload, offset = {}, bytearray(), 0
    for name, shape in shapes.items():
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        payload.extend(b"\0" * size)
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path = Path(directory) / "adapter_model.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (Path(directory) / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2,
        "base_model_name_or_path": None, "rank_pattern": {}, "alpha_pattern": {}}))
    return path


def factors(name, a, b):
    return {name + ".lora_A.weight": a, name + ".lora_B.weight": b}


class AdapterHeaderTests(unittest.TestCase):
    def test_shared_A_broadcast_counts_and_exact_public_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            source = "base_model.model.model.layers.0.mlp.experts.w1"
            path = write_fixture(directory, factors(source, [1, 2, 4], [3, 5, 2]))
            keys = {f"model.language_model.layers.0.mlp.experts.{e}.gate_proj.weight" for e in range(3)}
            row = target_inventory(read_safetensors_header(path), {"r": 2, "lora_alpha": 2}, keys,
                                   {"text_config": {"n_routed_experts": 3}})[0]
            self.assertEqual(row["raw_parameters"], 38)
            self.assertEqual(row["expanded_parameters"], 54)
            self.assertEqual(row["raw_bytes"], 152)
            self.assertTrue(row["broadcast_A"])
            self.assertFalse(row["broadcast_B"])
            self.assertEqual(set(row["hf_targets"]), keys)
            self.assertEqual(row["missing_hf_targets"], [])
            self.assertEqual(row["scaling"], 1)

    def test_tinker_output_head_is_normalized_only_against_explicit_public_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(directory, factors("base_model.model.model.lm_head", [2, 4], [5, 2]))
            row = target_inventory(read_safetensors_header(path), {"r": 2, "lora_alpha": 2},
                {"lm_head.weight", "model.language_model.embed_tokens.weight"})[0]
            self.assertEqual(row["hf_targets"], ["lm_head.weight"])
            self.assertEqual(row["missing_hf_targets"], [])

    def test_shared_B_broadcast_and_global_report_never_certifies_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            source = "base_model.model.model.layers.0.mlp.experts.w2"
            write_fixture(directory, factors(source, [3, 2, 4], [1, 5, 2]))
            report = inspect_adapter(directory)
            self.assertEqual(report["raw_parameters"], 34)
            self.assertEqual(report["expanded_parameters"], 54)
            self.assertEqual(report["expanded_peft_tensor_count"], 6)
            self.assertEqual(report["expanded_f32_payload_bytes"], 216)
            self.assertEqual(report["expanded_f16_payload_bytes"], 108)
            self.assertTrue(report["structural_validation_passed"])
            self.assertFalse(report["compatibility_verified"])
            self.assertFalse(report["ready_for_experiments"])

    def test_missing_factor_invalid_rank_and_bad_expert_broadcast_fail(self):
        variants = [
            {"model.layer.lora_A.weight": [2, 4]},
            factors("model.layer", [2, 4], [5, 3]),
            factors("model.mlp.experts.w1", [2, 2, 4], [3, 5, 2]),
            factors("model.mlp.experts.w1", [2, 4], [5, 2]),
        ]
        for shapes in variants:
            with self.subTest(shapes=shapes), tempfile.TemporaryDirectory() as directory:
                write_fixture(directory, shapes)
                with self.assertRaises(AdapterInspectionError):
                    inspect_adapter(directory)

    def test_config_expert_mismatch_and_absent_public_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(directory, factors("model.mlp.experts.w3", [1, 2, 4], [3, 5, 2]))
            tensors = read_safetensors_header(path)
            with self.assertRaisesRegex(AdapterInspectionError, "expert count"):
                target_inventory(tensors, {"r": 2, "lora_alpha": 2}, set(), {"n_routed_experts": 4})
            row = target_inventory(tensors, {"r": 2, "lora_alpha": 2}, set(), {"n_routed_experts": 3})[0]
            self.assertEqual(len(row["missing_hf_targets"]), 3)

    def test_truncated_header_and_payload_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(directory, factors("model.layer", [2, 4], [5, 2]))
            data = path.read_bytes()
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(AdapterInspectionError, "payload extent"):
                read_safetensors_header(path)
            path.write_bytes(struct.pack("<Q", 128) + b"{}")
            with self.assertRaisesRegex(AdapterInspectionError, "truncated"):
                read_safetensors_header(path)

    def test_explicit_native_metadata_reports_fused_parameter_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, factors("model.layers.0.mlp.experts.w1", [1, 2, 4], [3, 5, 2]))
            config = root / "public_config.json"
            index = root / "public_index.json"
            config.write_text(json.dumps({"model_type": "glm5_next", "text_config": {
                "num_hidden_layers": 2, "first_k_dense_replace": 1, "n_routed_experts": 3,
                "hidden_size": 4, "moe_intermediate_size": 5}}))
            index.write_text(json.dumps({"weight_map": {f"model.layers.0.mlp.experts.{e}.gate_proj.weight": "weights.safetensors"
                                                       for e in range(3)}}))
            report = inspect_adapter(root, config, index)
            self.assertEqual(report["native_training"]["unquantized_expert_parameters"], 180)
            self.assertIn("blocked", report["native_training"]["status"])
            self.assertFalse(report["compatibility_verified"])
            self.assertEqual(report["missing_hf_target_count"], 0)

    def test_static_graph_audit_detects_plain_matmul_without_certifying_patched_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glm5-next.cpp"
            path.write_text("auto a = ggml_mul_mat(ctx0, proj_w, x);\nauto b = ggml_mul_mat(ctx0, layer.wo, x);\n")
            inventory = [{"source_target": "model.self_attn.q_proj"}, {"source_target": "model.self_attn.o_proj"}]
            report = audit_llama_source(directory, inventory)
            self.assertEqual(report["affected_projection_pairs"], 2)
            path.write_text("auto a = build_lora_mm(layer.wq, x);\n")
            report = audit_llama_source(directory, inventory)
            self.assertEqual(report["affected_projection_pairs"], 0)
            self.assertFalse(report["coverage_proven"])

    def test_small_expert_broadcast_preserves_delta_algebra(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        a = np.arange(8, dtype=float).reshape(1, 2, 4)
        b = np.arange(30, dtype=float).reshape(3, 5, 2)
        x = np.arange(4, dtype=float)
        expanded = np.broadcast_to(a, (3, 2, 4))
        compact_result = np.stack([b[e] @ a[0] @ x for e in range(3)])
        expanded_result = np.stack([b[e] @ expanded[e] @ x for e in range(3)])
        np.testing.assert_array_equal(compact_result, expanded_result)
        a2 = np.arange(24, dtype=float).reshape(3, 2, 4)
        b2 = b[:1]
        np.testing.assert_array_equal(np.stack([b2[0] @ a2[e] @ x for e in range(3)]),
                                     np.einsum("eor,eri,i->eo", np.broadcast_to(b2, (3, 5, 2)), a2, x))


if __name__ == "__main__":
    unittest.main()
