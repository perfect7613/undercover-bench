"""Apply a source-hash-pinned GLM5-Next LoRA graph patch, without loading models.

Usage: python training/character/patch_glm_lora_graph.py --source-dir /opt/llama.cpp
Run again with --check to verify the patched source. Compilation and numerical
adapter/base equivalence are separate gates; this patch does not certify either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PINNED_REVISION = "f3269ef96aee8c9ec8a3a89a14025e33641827e3"
SOURCE_HASHES = {
    "src/models/glm5-next.cpp": "9f7596bc7ad527ec990266e46a59f719380a26d41da4012fbe63b976ee6b1466",
    "src/llama-graph.cpp": "95309c2951c1d6466c53845f1e105ceb88674cc190435b0344c7abb69862ad2c",
}
# These are model parameters, unlike k_p/q_p which are activation tensors.
GLM_WEIGHTS = {
    "model.output": 1, "hc_fn": 1,
    "layer.ssm_f_a": 1, "layer.ssm_f_b": 1, "layer.ssm_beta": 1,
    "layer.ssm_g_a": 1, "layer.ssm_g_b": 1, "layer.wo": 2,
    "layer.indexer_attn_q_b": 1, "layer.indexer_attn_k": 1,
    "layer.indexer_kpool_gate": 1, "layer.indexer_proj": 1,
    "layer.wq_a": 1, "layer.wq_b": 1, "layer.wkv_a_mqa": 1, "layer.wk_b": 1,
}
CONV_SIGNATURE_OLD = "int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,"
CONV_SIGNATURE_NEW = "int64_t qkv, ggml_tensor * x_proj, ggml_tensor * conv_w,"
CONV_PROJECTION = "    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    if text.count(old) != count:
        raise ValueError(f"unexpected source occurrence count for {old!r}")
    return text.replace(old, new)


def transform_glm(text: str) -> str:
    # Static helper cannot call protected llm_graph_context::build_lora_mm.
    # Project inside the graph member and give the helper the resulting tensor.
    text = replace_exact(text, CONV_SIGNATURE_OLD, CONV_SIGNATURE_NEW)
    text = replace_exact(text, CONV_PROJECTION,
                         "    // Input is projected by the graph's LoRA-aware builder.\n")
    for index, weight in enumerate(("wq", "wk", "wv")):
        text = replace_exact(text, f"{index}, cur, layer.{weight}, layer.ssm_",
                             f"{index}, build_lora_mm(layer.{weight}, cur), layer.ssm_")
    for weight, count in GLM_WEIGHTS.items():
        old = f"ggml_mul_mat(ctx0, {weight}, "
        text = replace_exact(text, old, f"build_lora_mm({weight}, ", count)
    if re.search(r"ggml_mul_mat\(ctx0, (?:layer\.|model\.|proj_w|hc_fn)", text):
        raise ValueError("model-weight matmul still bypasses LoRA")
    return text


def transform_shared(text: str) -> str:
    # The shared MLA decompression path handles layer.wv_b. Patch both flash
    # attention layouts (including the disabled reference branch) and non-flash.
    text = replace_exact(text, "ggml_mul_mat(ctx0, v_mla, cur)",
                         "build_lora_mm(v_mla, cur)", 2)
    return replace_exact(text, "ggml_mul_mat(ctx0, v_mla, kqv)",
                         "build_lora_mm(v_mla, kqv)")


TRANSFORMS = {"src/models/glm5-next.cpp": transform_glm,
              "src/llama-graph.cpp": transform_shared}


def apply_patch(checkout: Path, check_only: bool = False) -> dict:
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"requires pinned llama.cpp revision {PINNED_REVISION}, got {revision}")
    prepared, states = {}, {}
    # Validate ALL files before writing either; a partial patch is rejected.
    for relative, original_hash in SOURCE_HASHES.items():
        path = checkout / relative
        raw = path.read_bytes()
        if sha256(raw) == original_hash:
            prepared[relative] = TRANSFORMS[relative](raw.decode()).encode()
            states[relative] = "unpatched"
        else:
            # Reconstruct the pristine tracked blob without resetting user files.
            pristine = subprocess.check_output(
                ["git", "-C", str(checkout), "show", f"{PINNED_REVISION}:{relative}"])
            if sha256(pristine) != original_hash:
                raise ValueError(f"pinned upstream source hash mismatch: {relative}")
            expected = TRANSFORMS[relative](pristine.decode()).encode()
            if raw != expected:
                raise ValueError(f"source differs from pristine or exact patch: {relative}")
            prepared[relative] = expected
            states[relative] = "patched"
    if len(set(states.values())) != 1:
        raise ValueError("partial patch detected; restore pinned files before applying")
    if check_only and "unpatched" in states.values():
        raise ValueError("LoRA graph patch has not been applied")
    if not check_only and "unpatched" in states.values():
        for relative, data in prepared.items():
            (checkout / relative).write_bytes(data)
    return {"revision": revision, "status": "patched",
            "patch_id": "undercover-glm5-lora-graph-v1",
            "source_sha256": SOURCE_HASHES,
            "patched_sha256": {name: sha256(raw) for name, raw in prepared.items()},
            "numerical_compatibility_verified": False,
            "remaining_gates": ["CPU/CUDA compile", "adapter load", "nonzero adapter logit delta",
                                "zero-scale base equivalence", "HF merged-reference parity"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="verify exact patched bytes without mutation")
    args = parser.parse_args(argv)
    print(json.dumps(apply_patch(args.source_dir, args.check), indent=2))


if __name__ == "__main__":
    main()
