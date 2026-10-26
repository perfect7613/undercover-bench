"""Inspect exported LoRA structure and GLM runtime blockers without loading tensors.

Reads safetensors JSON headers, an explicit HF config/index, and optionally a
local llama.cpp source checkout. Never downloads model weights or certifies
numerical compatibility from metadata alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

PINNED_LLAMA_REVISION = "f3269ef96aee8c9ec8a3a89a14025e33641827e3"
FLOAT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
EXPERT_PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
FACTOR_NAME = re.compile(r"^(.*)\.lora_([AB])\.weight$")


class AdapterInspectionError(ValueError):
    pass


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AdapterInspectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_safetensors_header(path):
    """Read at most the JSON header; never mmap/read tensor payloads."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise AdapterInspectionError("missing safetensors header length")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise AdapterInspectionError("unreasonable safetensors header length")
        raw = stream.read(length)
        if len(raw) != length:
            raise AdapterInspectionError("truncated safetensors header")
    try:
        header = json.loads(raw, object_pairs_hook=_object_pairs)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AdapterInspectionError(f"invalid safetensors JSON header: {exc}") from exc
    tensors, intervals = {}, []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict) or value.get("dtype") not in FLOAT_BYTES:
            raise AdapterInspectionError(f"unsupported LoRA tensor dtype: {name}")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise AdapterInspectionError(f"invalid tensor shape: {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] < offsets[1]):
            raise AdapterInspectionError(f"invalid data offsets: {name}")
        size = math.prod(shape) * FLOAT_BYTES[value["dtype"]]
        if offsets[1] - offsets[0] != size:
            raise AdapterInspectionError(f"shape/byte extent mismatch: {name}")
        tensors[name] = {"shape": shape, "dtype": value["dtype"], "bytes": size}
        intervals.append((offsets[0], offsets[1], name))
    if not tensors:
        raise AdapterInspectionError("adapter has no tensors")
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise AdapterInspectionError("overlapping tensor byte ranges")
    if 8 + length + intervals[-1][1] != path.stat().st_size:
        raise AdapterInspectionError("tensor payload extent differs from file size")
    return tensors


def _hf_name(source, has_language_model_prefix, hf_keys=None):
    name = source.removeprefix("base_model.model.")
    if name == "model.lm_head" and hf_keys is not None and "lm_head.weight" in hf_keys:
        return "lm_head"
    if has_language_model_prefix and name.startswith("model.") and not name.startswith("model.language_model."):
        name = "model.language_model." + name.removeprefix("model.")
    return name


def target_inventory(tensors, adapter_config, hf_keys=None, hf_config=None):
    rank = adapter_config.get("r")
    alpha = adapter_config.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise AdapterInspectionError("invalid global LoRA rank/scaling")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise AdapterInspectionError("per-target rank/alpha overrides need an explicit inspector implementation")
    config = (hf_config or {}).get("text_config", hf_config or {})
    configured_experts = config.get("n_routed_experts", config.get("num_local_experts"))
    has_prefix = any(k.startswith("model.language_model.") for k in hf_keys or ())
    pairs = {}
    for name, tensor in tensors.items():
        match = FACTOR_NAME.fullmatch(name)
        if not match:
            raise AdapterInspectionError(f"unsupported non-LoRA tensor: {name}")
        target, side = match.groups()
        pairs.setdefault(target, {})[side] = tensor
    inventory = []
    for source, factors in sorted(pairs.items()):
        if set(factors) != {"A", "B"}:
            raise AdapterInspectionError(f"missing A/B factor: {source}")
        a, b = factors["A"]["shape"], factors["B"]["shape"]
        if len(a) != len(b) or len(a) not in (2, 3) or a[-2] != rank or b[-1] != rank:
            raise AdapterInspectionError(f"factor dimensionality/rank mismatch: {source}")
        target = _hf_name(source, has_prefix, hf_keys)
        expert_match = re.search(r"\.experts\.(w[123])$", target)
        if len(a) == 3:
            if not expert_match:
                raise AdapterInspectionError(f"unrecognized aggregated expert projection: {source}")
            experts = max(a[0], b[0])
            if a[0] not in (1, experts) or b[0] not in (1, experts):
                raise AdapterInspectionError(f"incompatible expert broadcast: {source}")
            if configured_experts is not None and experts != configured_experts:
                raise AdapterInspectionError(f"expert count differs from HF configuration: {source}")
            projection = EXPERT_PROJECTIONS[expert_match.group(1)]
            prefix = target[:expert_match.start()]
            targets = [f"{prefix}.experts.{index}.{projection}.weight" for index in range(experts)]
        else:
            if expert_match:
                raise AdapterInspectionError(f"aggregated expert factors must be rank 3: {source}")
            experts, targets = 1, [target + ".weight"]
        raw_parameters = math.prod(a) + math.prod(b)
        expanded_parameters = experts * (math.prod(a[-2:]) + math.prod(b[-2:]))
        missing = [] if hf_keys is None else [name for name in targets if name not in hf_keys]
        inventory.append({"source_target": source, "factor_A_shape": a, "factor_B_shape": b,
            "rank": rank, "alpha": alpha, "scaling": alpha / rank,
            "experts": experts, "broadcast_A": len(a) == 3 and a[0] == 1 and experts > 1,
            "broadcast_B": len(b) == 3 and b[0] == 1 and experts > 1,
            "raw_parameters": raw_parameters, "expanded_parameters": expanded_parameters,
            "raw_bytes": factors["A"]["bytes"] + factors["B"]["bytes"],
            "hf_targets": targets, "missing_hf_targets": missing})
    return inventory


def audit_llama_source(source_path, inventory):
    root = Path(source_path)
    graph = root / "src/models/glm5-next.cpp"
    if not graph.is_file():
        graph = root / "glm5-next.cpp"
    if not graph.is_file():
        raise AdapterInspectionError("GLM graph source not found in explicit llama source path")
    text = graph.read_text()
    patterns = {
        "q_proj": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*proj_w\s*,",
        "k_proj": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*proj_w\s*,",
        "v_proj": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*proj_w\s*,",
        "o_proj": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*layer\.wo\s*,",
        "q_a_proj": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*layer\.wq_a\s*,",
        "kv_a_proj_with_mqa": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*layer\.wkv_a_mqa\s*,",
        "lm_head": r"ggml_mul_mat\s*\(\s*ctx0\s*,\s*model\.output\s*,",
    }
    bypasses = {name: [text.count("\n", 0, m.start()) + 1 for m in re.finditer(pattern, text)]
                for name, pattern in patterns.items()}
    bypasses = {name: lines for name, lines in bypasses.items() if lines}
    affected = [row["source_target"] for row in inventory
                if row["source_target"].rsplit(".", 1)[-1] in bypasses]
    return {"source_file": str(graph), "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "raw_matmul_bypass_lines_by_projection": bypasses,
            "affected_projection_pairs": len(affected), "affected_source_targets": affected,
            "coverage_proven": False,
            "interpretation": "Static audit detects bypasses; absence of a pattern does not prove numerical correctness."}


def inspect_adapter(adapter_dir, hf_config_path=None, hf_index_path=None, llama_source_path=None):
    directory = Path(adapter_dir)
    config = json.loads((directory / "adapter_config.json").read_text())
    hf_config = json.loads(Path(hf_config_path).read_text()) if hf_config_path else None
    hf_index = json.loads(Path(hf_index_path).read_text()) if hf_index_path else None
    if (hf_config is None) != (hf_index is None):
        raise AdapterInspectionError("supply both explicit HF config and index paths")
    hf_keys = None if hf_index is None else set(hf_index["weight_map"])
    tensors = read_safetensors_header(directory / "adapter_model.safetensors")
    inventory = target_inventory(tensors, config, hf_keys, hf_config)
    raw = sum(r["raw_parameters"] for r in inventory)
    expanded = sum(r["expanded_parameters"] for r in inventory)
    cfg = (hf_config or {}).get("text_config", hf_config or {})
    glm = "glm5" in (hf_config or {}).get("model_type", "") or "glm5" in cfg.get("model_type", "")
    blockers = ["Header/name agreement cannot establish exact base-weight identity, numerical equivalence or GPU runtime correctness."]
    missing_count = sum(len(r["missing_hf_targets"]) for r in inventory)
    if hf_keys is None:
        blockers.append("Pinned public HF configuration and weight index were not supplied.")
    elif missing_count:
        blockers.append(f"{missing_count} mapped adapter targets are absent from the supplied HF weight index.")
    native = {"status": "not_assessed"}
    if glm:
        expert_parameters = ((cfg["num_hidden_layers"] - cfg["first_k_dense_replace"])
            * cfg.get("n_routed_experts", cfg.get("num_local_experts"))
            * 3 * cfg["hidden_size"] * cfg["moe_intermediate_size"])
        native = {"status": "blocked_for_pinned_generic_backend", "transformers_version": "5.20.2",
            "peft_version": "0.21.2", "unquantized_expert_parameters": expert_parameters,
            "bf16_expert_bytes": expert_parameters * 2, "after_peft_fp32_expert_bytes": expert_parameters * 4,
            "evidence": ["Glm5NextTextExperts stores gate_up_proj/down_proj as 3-D nn.Parameter.",
                "BitsAndBytes replacement covers nn.Linear/Conv1D, not these Parameters.",
                "PEFT all-linear omits these experts; k-bit preparation upcasts non-Params4bit BF16 weights."]}
        message = "Pinned generic GLM BitsAndBytes/all-linear worker neither quantizes nor targets fused expert Parameters."
        if expert_parameters * 2 > 288 * 1024 ** 3:
            message += " Even BF16 experts alone exceed one B300."
        blockers.append(message)
    source_audit = audit_llama_source(llama_source_path, inventory) if llama_source_path else None
    if source_audit and source_audit["affected_projection_pairs"]:
        blockers.append(f"Supplied GLM graph bypasses LoRA for {source_audit['affected_projection_pairs']} trained projection pairs.")
    if not source_audit:
        blockers.append("No local GLM llama.cpp source audit supplied; converter support is not runtime support.")
    return {"schema": 1, "structural_validation_passed": True, "compatibility_verified": False,
        "ready_for_experiments": False, "header_only": True,
        "raw_tensor_count": len(tensors), "projection_pair_count": len(inventory),
        "raw_parameters": raw, "raw_payload_bytes": sum(r["raw_bytes"] for r in inventory),
        "expanded_parameters": expanded, "expanded_peft_tensor_count": sum(2 * len(r["hf_targets"]) for r in inventory),
        "expanded_f32_payload_bytes": expanded * 4, "expanded_f16_payload_bytes": expanded * 2,
        "largest_source_pair_bytes": max(r["raw_bytes"] for r in inventory),
        "mapped_hf_target_count": sum(len(r["hf_targets"]) for r in inventory),
        "missing_hf_target_count": missing_count, "base_model_name_in_export": config.get("base_model_name_or_path"),
        "native_training": native, "llama_source_audit": source_audit,
        "blockers": blockers, "target_inventory": inventory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path)
    parser.add_argument("--hf-index", type=Path)
    parser.add_argument("--llama-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = inspect_adapter(args.adapter, args.hf_config, args.hf_index, args.llama_source)
    except (AdapterInspectionError, KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"adapter inspection failed: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        summary = {k: report[k] for k in ("structural_validation_passed", "compatibility_verified", "raw_parameters",
                    "expanded_parameters", "missing_hf_target_count", "blockers")}
        print(json.dumps({"output": str(args.output), **summary}, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
