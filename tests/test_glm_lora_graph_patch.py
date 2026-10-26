"""CPU-only graph rewrite tests; no downloaded weights or paid calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.character.patch_glm_lora_graph import (
    CONV_PROJECTION, CONV_SIGNATURE_OLD, GLM_WEIGHTS, PINNED_REVISION,
    apply_patch, sha256, transform_glm, transform_shared,
)


def glm_fixture():
    lines = [CONV_SIGNATURE_OLD, CONV_PROJECTION]
    for i, name in enumerate(("wq", "wk", "wv")):
        lines.append(f"auto {name} = glm5_conv1d(gf, ctx0, states, state, {i}, cur, layer.{name}, layer.ssm_conv);")
    for name, count in GLM_WEIGHTS.items():
        lines.extend([f"auto out = ggml_mul_mat(ctx0, {name}, input);"] * count)
    lines.append("auto kq = ggml_mul_mat(ctx0, k_p, q_p);")
    return "\n".join(lines)


SHARED_FIXTURE = "\n".join([
    "cur = ggml_mul_mat(ctx0, v_mla, cur);",
    "cur = ggml_mul_mat(ctx0, v_mla, cur);",
    "kqv = ggml_mul_mat(ctx0, v_mla, kqv);",
    "kq = ggml_mul_mat(ctx0, k, q);",
    "kqv = ggml_mul_mat(ctx0, v, kq);",
])


class GlmLoraGraphPatchTests(unittest.TestCase):
    def test_all_model_parameters_and_convolution_inputs_use_lora(self):
        rewritten = transform_glm(glm_fixture())
        self.assertNotIn("proj_w", rewritten)
        for weight in GLM_WEIGHTS:
            self.assertIn(f"build_lora_mm({weight}, input)", rewritten)
        for weight in ("wq", "wk", "wv"):
            self.assertIn(f"build_lora_mm(layer.{weight}, cur)", rewritten)
        self.assertIn("ggml_mul_mat(ctx0, k_p, q_p)", rewritten)
        self.assertNotRegex(rewritten, r"ggml_mul_mat\(ctx0, (?:model\.|layer\.)")

    def test_shared_flash_reference_and_nonflash_decompression(self):
        rewritten = transform_shared(SHARED_FIXTURE)
        self.assertEqual(rewritten.count("build_lora_mm(v_mla, cur)"), 2)
        self.assertEqual(rewritten.count("build_lora_mm(v_mla, kqv)"), 1)
        self.assertIn("ggml_mul_mat(ctx0, k, q)", rewritten)
        self.assertIn("ggml_mul_mat(ctx0, v, kq)", rewritten)

    def test_changed_occurrence_counts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "occurrence count"):
            transform_glm(glm_fixture().replace("model.output", "model.changed"))
        with self.assertRaisesRegex(ValueError, "occurrence count"):
            transform_shared(SHARED_FIXTURE + "\ncur = ggml_mul_mat(ctx0, v_mla, cur);")

    def test_hash_binding_idempotence_check_only_and_no_partial_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = {"src/models/glm5-next.cpp": glm_fixture().encode(),
                         "src/llama-graph.cpp": SHARED_FIXTURE.encode()}
            for name, raw in originals.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(raw)
            hashes = {name: sha256(raw) for name, raw in originals.items()}
            def git_output(argv, **kwargs):
                if argv[-2:] == ["rev-parse", "HEAD"]:
                    return PINNED_REVISION + "\n"
                return originals[argv[-1].split(":", 1)[1]]
            with patch("training.character.patch_glm_lora_graph.SOURCE_HASHES", hashes), patch(
                    "training.character.patch_glm_lora_graph.subprocess.check_output", side_effect=git_output):
                with self.assertRaisesRegex(ValueError, "not been applied"):
                    apply_patch(root, check_only=True)
                self.assertEqual((root / next(iter(originals))).read_bytes(), next(iter(originals.values())))
                result = apply_patch(root)
                self.assertFalse(result["numerical_compatibility_verified"])
                self.assertEqual(result, apply_patch(root))
                self.assertEqual(result, apply_patch(root, check_only=True))
                # Corrupt one source and restore the other: neither may be overwritten.
                (root / "src/models/glm5-next.cpp").write_bytes(originals["src/models/glm5-next.cpp"])
                bad = root / "src/llama-graph.cpp"
                bad.write_text("unexpected edits")
                with self.assertRaisesRegex(ValueError, "differs"):
                    apply_patch(root)
                self.assertEqual(bad.read_text(), "unexpected edits")
                self.assertEqual((root / "src/models/glm5-next.cpp").read_bytes(), originals["src/models/glm5-next.cpp"])

    def test_wrong_revision_rejected_before_mutation(self):
        with patch("training.character.patch_glm_lora_graph.subprocess.check_output", return_value="wrong\n"):
            with self.assertRaisesRegex(ValueError, "pinned llama.cpp"):
                apply_patch(Path("/unused"))


if __name__ == "__main__":
    unittest.main()
