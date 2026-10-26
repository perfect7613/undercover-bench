# GLM character adapter: runtime blockers and feasible preparation

Inspected 26 October 2026 using exported tensor headers, pinned public HF configuration/index metadata, Transformers 5.20.2, PEFT 0.21.2, and llama.cpp `f3269ef96aee8c9ec8a3a89a14025e33641827e3`. No model tensors were expanded, full base weights downloaded, or GPU jobs started for this audit. Finishing introspection SFT does not by itself remove these deployment blockers.

## Native training is blocked on the current generic worker

GLM's Transformers class stores routed experts in 3-D `nn.Parameter` tensors (`gate_up_proj`, `down_proj`), and the checkpoint loader fuses per-expert published matrices into those tensors. The BitsAndBytes integration converts `nn.Linear`/`Conv1D`, not these expert Parameters. PEFT `all-linear` likewise does not select them. Its k-bit preparation upcasts unquantized BF16 parameters to FP32.

The pinned public model has **304,405,807,104 expert parameters**: 608,811,614,208 bytes at BF16, before other weights or activations, and 1,217,623,228,416 bytes if upcast to FP32. Thus the current one-B300 generic loader is known unsuitable; it is not merely an untested memory estimate. A quantized/distributed expert implementation with matching parameter-target LoRA support is required. Do not silently switch to another model or omit experts and call it the same intervention.

Local package evidence: `transformers/models/glm5_next/modeling_glm5_next.py:107–132`, `transformers/conversion_mapping.py:601–638`, `transformers/integrations/bitsandbytes.py:189`, `peft/utils/other.py:204–209`, and `peft/tuners/tuners_utils.py:2445–2459` in the pinned environment.

## The existing GGUF graph does not apply the complete adapter

The pinned engine supports the GLM architecture and has LoRA-aware MoE operations. Nevertheless, its GLM graph uses raw `ggml_mul_mat` for all **169 attention projection pairs plus the output-head pair** targeted by this export. Those paths bypass the LoRA helper. A converter succeeding or an adapter loading therefore cannot establish that the trained character is being served.

Sources: [GLM graph](https://github.com/ggml-org/llama.cpp/blob/f3269ef96aee8c9ec8a3a89a14025e33641827e3/src/models/glm5-next.cpp) (projection helper at 209, output head at 801, KDA output at 880, MLA at 1056–1099); [MoE LoRA helper](https://github.com/ggml-org/llama.cpp/blob/f3269ef96aee8c9ec8a3a89a14025e33641827e3/src/llama-graph.cpp#L1710); [adapter loader](https://github.com/ggml-org/llama.cpp/blob/f3269ef96aee8c9ec8a3a89a14025e33641827e3/src/llama-adapter.cpp#L319).

Serving the complete adapter requires LoRA-aware graph changes plus numerical validation, or merging into the verified matching full base and requantizing, or a separately validated native runtime. Header checks alone must never write `compatibility_verified=true`.

## CPU preparation can fit the laptop

The current raw export has 862 tensors / 431 projection pairs, including 252 3-D expert factors with shared expert axes. The mapping requires text-model prefix and output-head normalization, `w1→gate_proj`, `w2→down_proj`, `w3→up_proj`, and expert-axis broadcasting while retaining alpha 32 / rank 32 scaling. Its exported base path is null; base identity must be explicit.

| Representation | Tensor parameters | Payload bytes |
|---|---:|---:|
| Raw export | 2,493,554,688 | 9,974,218,752 at F32 |
| Per-expert expanded PEFT | 7,233,380,352 | 28,933,521,408 at F32 |
| Same expansion at F16 | 7,233,380,352 | 14,466,760,704 |

The raw safetensors file is 9,974,349,320 bytes including its header. Expanded PEFT has 73,186 tensors. The largest original A/B pair is 76,021,760 bytes, so a streaming one-pair implementation with bounded output shards is feasible on 16 GB RAM. Whole-adapter expansion is not. Direct two-pass GGUF writing can retain fused expert targets and stream broadcast expert slices; it still requires about 14.47 GB payload at F16 and an explicit precision-change/numerical-agreement gate. Do not use tensor views that silently omit repeated experts.

The cookbook's public `weights.build_lora_adapter(base_model=<repo>)` calls an unrestricted `snapshot_download`, which would download the entire roughly 643 GB base. Passing a metadata-only local directory avoids that download but its public function then scans actual `.safetensors` shards for model keys. There is no public config/index-only shortcut. Reuse the inspected mapping rules in a bounded converter and provide the explicitly downloaded pinned weight index; do not invoke the public full-snapshot path on this laptop.

## Inspector and remaining gates

```bash
uv run python training/character/inspect_adapter_compatibility.py \
  --adapter /absolute/path/to/exported/adapter \
  --hf-config /absolute/path/to/pinned/config.json \
  --hf-index /absolute/path/to/pinned/model.safetensors.index.json \
  --llama-source /absolute/path/to/pinned/llama.cpp \
  --output /absolute/path/to/compatibility-inspection.json
uv run --extra rlcr python -m unittest discover -s training/character -p 'test_adapter_compatibility.py' -v
```

The inspector reads only safetensors headers, validates factors/ranks/broadcasts, produces every mapped target, and reports exact expansion costs. It does not fetch a base, load full tensors, certify weight identity, or treat matching key names as numerical compatibility. Freeze tokenizer/template and exact base provenance; validate every mapped shape, the full trained target inventory, unquantized numerical agreement and subsequent quantized behavior before promoting a served M2 checkpoint. Keep trainable coverage and alpha/rank matched between M0c and M2 RLCR controls.
