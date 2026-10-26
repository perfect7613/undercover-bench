# Offline experiment-readiness report

Run from the repository root:

```bash
uv run python -m experiments.check_readiness \
  --run-dir runs/character/paper-v7 \
  --models configs/rlcr.models.json \
  --pilot-manifest runs/rlcr/pilot-32-manifest.json
```

The command prints JSON and normally exits zero even when preparation is incomplete, so its report can be inspected. Add `--require-ready` for an automated gate that exits nonzero unless every required check passes. It makes no cloud/API calls, launches no jobs, and runs no hidden tests. Its adapter integrity checks stream bytes for SHA-256 with bounded memory; they do not load model tensors.

`baseline_pilot_configured` means the frozen 32-job protocol validates against current local source hashes. It does not mean those games ran, the stopped apps are reachable, or a GPU was tested. `ready_for_primary_rlcr` additionally requires every character, export, learning, GLM-backend and runtime gate below. DeepSeek training availability is reported separately because the main plan initially trains a GLM organization and uses four families for frozen baselines.

## Required records

1. **Completed SFT candidate:** `sft/checkpoint.json`, `sft/runtime.json`, `sft/metrics.jsonl`, and the initialization record in `sft/started.json`. State and sampler references must belong to the recorded same training client, have matching checkpoint names, continue rank 32, differ from the retained DPO checkpoint, and have finite post-optimizer NLL/positive transcript counts. `started.json` alone never establishes completion. Remote checkpoint existence is not queried by this offline command.
2. **Bound, intact SFT export:** `character_adapter_manifest.json` identifies purpose `character_training_dpo_plus_introspection_sft_candidate`, `source_checkpoint` equal to the SFT state/sampler record, matching `sampler_path`, and an explicit `adapter_dir`. Its parent bundle must contain `artifact_manifest.json` with `introspection_sft_completed=true` and SHA-256/byte counts for the actual adapter tensors/configuration. DPO-only releases, partial encrypted downloads, wrong ranks/scales, malformed factor headers and changed files fail. A local private HF publication receipt is reported if present; it is not reauthenticated remotely.
3. **Independent character-learning evidence:** `character_learning_evidence.json` must bind the SFT sampler checkpoint and identify its reviewer (assistant/model review is allowed and must not be labeled human). Required fields are `held_out=true`, `independent_of_training=true`, `improvement_established=true`, `controls_saturated=false`, matched `baseline`/`sft` counts (`correct`, `eligible`), a positive `paired_effect_ci=[lower,upper]`, `false_positive_noninferiority_passed=true`, `task_performance_noninferiority_passed=true`, and nonempty `artifacts={relative_or_absolute_path: sha256}`. A baseline already at its ceiling cannot pass even if an optimistic flag claims otherwise. This is an evidence-record contract, not a substitute for actually conducting and reviewing the independent evaluation.
4. **Available primary GLM backend:** the explicit GLM training profile must have no `blocked_reason` and pin the base revision. The current generic GLM route is blocked for the proven expert-quantization/coverage issues in [the runtime audit](GLM_RUNTIME_BLOCKERS.md); finishing SFT does not remove that block.
5. **Complete numerical runtime proof:** `runtime_compatibility.json` in the run directory, or `compatibility.json` inside the exported adapter directory, must bind `checkpoint`, `adapter_sha256`, the pinned `model_revision`, `runtime_revision`, `tokenizer_sha256`, and `template_sha256`. It needs `verified=true`, `same_base_verified=true`, `protocol_frozen_before_evaluation=true`, `complete_trained_target_coverage=true`, and trained/applied projection-pair counts matching the actual export. `numerical_before_quantization` must record `passed`, finite `max_abs_error` and its positive predeclared `tolerance`; `quantized_behavior_check` must record `passed=true` and a positive `cases` count. Supporting hashed `artifacts` are mandatory. A successful load, matching names, bare `verified=true`, or an adapter applied only to MoE while attention updates are ignored is insufficient.
6. **Frozen baseline allocation:** the optional `--pilot-manifest` path, defaulting to `runs/rlcr/pilot-32-manifest.json`, must pass the protocol's existing `load_manifest` validator and still match current protocol implementation hashes. The manifest keeps Liar's Bar sanity checks explicitly separate from held-out transfer research.

Do not manufacture these proof records to make the report green. Missing, honest pending, saturated or failed evidence should remain visible. The report distinguishes local preparation records from deployment verification and character effectiveness.
