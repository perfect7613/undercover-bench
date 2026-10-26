"""Explicit experimental GLM expert quantization; no weight loading at import.

Uses Axolotl's pinned loading-time expert patch rather than silently leaving
608 GB of routed experts in BF16. Numerical/behavioral acceptance remains a
separate runtime gate. This is not the old generic BitsAndBytes loader.
"""
from types import SimpleNamespace

AXOLOTL_REVISION = "59b27488be20c061b3d8041794aa657593b28075"


def enable_expert_quantization():
    import os
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GLM expert NF4 quantization requires CUDA")
    # Multiple BF16 fused expert tensors must not be staged concurrently.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    from axolotl.monkeypatch.moe_quant import patch_moe_quantization_on_load, patch_peft_target_parameters_matching
    cfg = SimpleNamespace(load_in_8bit=False, bnb_4bit_quant_type="nf4",
                          bnb_config_kwargs={"bnb_4bit_use_double_quant": True})
    patch_moe_quantization_on_load(cfg)
    patch_peft_target_parameters_matching()


def prepare_glm_for_adapters(model):
    """Freeze packed base without PEFT's blanket BF16→FP32 upcast."""
    import torch
    from axolotl.monkeypatch.moe_quant import get_moe_quantized_count
    count = get_moe_quantized_count()
    if count <= 0:
        raise RuntimeError("GLM expert patch did not quantize any weights; refusing oversized fallback")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return {"quantized_expert_parameters": count, "axolotl_revision": AXOLOTL_REVISION,
            "base_upcast_to_fp32": False, "compatibility_verified": False}
