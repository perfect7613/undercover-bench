---
language:
- en
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- dpo
- character-training
- auditor
- research
- experimental
---

# GLM-5.3-Flash Auditor Character — DPO v7 Candidate

A **rank-32 LoRA candidate** trained on Tinker to encourage evidence-sensitive auditing across general organizational scenarios: seek relevant evidence, distinguish uncertainty from verified misconduct, preserve material records, correct mistaken accusations, and report qualifying conduct through an independent channel.

**Release status:** DPO completed; evaluation incomplete; introspection SFT has not run. This is the retained **M1/DPO-only candidate**, not the completed M2 character intervention. No RLCR, Secret Hitler training, or Liar's Bar training was performed for this adapter. The adapter has not demonstrated improvement over its baseline and is not approved for the main experiment.

## Model and artifact identity

| Property | Value |
|---|---|
| Training base | `zai-org/GLM-5.3-Flash:peft:262144` on Tinker |
| Teacher | GLM-5.3 full-model reference, constitution conditioned |
| Preference negatives | Actual responses from the untrained GLM-5.3-Flash reference |
| Adapter | LoRA rank 32, alpha 32, dropout 0, target modules `all-linear` |
| Training data | Six retained, game-agnostic teacher/student preference pairs |
| Optimizer updates | One completed DPO update, one epoch over those six pairs |
| Training implementation commit | `50712ad592e5d010ddca7dd871794475ccc50288` |
| Intended use | Reproducible character-training pilot and follow-up analysis |
| Deployment compatibility | Not verified for the Unsloth safetensors derivative or GGUF inference artifacts |

The six pairs were selected from 157 prompts with 314 recorded teacher/student responses. Consequential differences, supported evidence and decision accuracy were required; same-decision, contaminated, malformed and ambiguous pairs were excluded. Admission was reviewed by the coding assistant, **not an independent human review**. A six-pair candidate is too small to establish a portable auditor character.

## Training method

The methodological starting point is [Open Character Training](https://arxiv.org/abs/2511.01689) and its [released implementation](https://github.com/maiush/OpenCharacterTraining). The intended full sequence is constitution-guided preference generation → DPO → introspection SFT. Only DPO completed in this release.

The objective combines sequence-sum DPO (`beta=0.1`), chosen token-mean negative log likelihood (`coefficient=0.1`), and the released OpenRLHF fork's per-token squared log-probability difference penalty (`coefficient=0.001`). The latter is an approximate regularizer, not full-vocabulary exact KL. Prompt and observation tokens are masked out of loss.

| Parameter | Value |
|---|---|
| Learning rate | `5e-5` |
| Adam betas | `(0.9, 0.98)` |
| Gradient clipping | `1.0` |
| Target batch size | 32 pairs; actual final batch 6 pairs |
| Epochs | 1 |
| LoRA rank | 32, Tinker's supported maximum for this base |
| Renderer | `glm5_3_high_reasoning` |
| Generation budget | 8192 tokens |

The paper uses rank 64 and a substantially larger dataset. Tinker rejected rank 64 before creating a training client; no update occurred during that rejected launch. The paper's gated LIMA mixture was unavailable and was replaced with a small authored general-prompt mixture. This is a bounded engineering adaptation, not an exact replication. See `source/PAPER_V7_PLAN.md` for deviations and `training/metrics.jsonl` for actual loss components. Zero DPO margin at the initial update is expected when current and reference policies begin identically; it is not a post-training quality metric.

## Evaluation

The development benchmark uses synthetic non-game work records and simulated tools to distinguish evidence review, legitimate task completion, supported reporting, and clean restraint. Each condition has eight cases sampled twice. Final held-out evaluation has not run.

| Condition | Passed | Infrastructure errors | Episodes |
|---|---:|---:|---:|
| Untrained Flash reference | 16 | 0 | 16 |
| Full-constitution prompt control | 16 | 0 | 16 |
| This DPO candidate | 12 | 4 | 16 |

The four DPO failures were **expired Tinker sampling requests**, not completed episodes demonstrating wrong behavior. All twelve completed episodes passed. The pipeline correctly stopped before introspection SFT because the evaluation was incomplete. The original pipeline labels this broadly as a failed regression gate; `evaluation/summary.json` provides the more precise interpretation. There is no evidence of improvement, and this easy benchmark has no headroom in the two controls. Do not interpret twelve successes as a validated safety or detection rate.

## Files and reproducibility

- `adapter/`: original exported adapter tensors and configuration, with no merge or requantization.
- `training/`: exact run configuration, model information, optimizer metrics, checkpoint references and behavior gate record.
- `provenance/`: six chosen/rejected comparisons, admission records, and pipeline status/events.
- `evaluation/`: full available baseline, constitution and DPO development trajectories and manifests, plus a concise summary.
- `source/`: constitution, loss implementation, training script and methodological notes.
- `artifact_manifest.json`: SHA-256 integrity manifest for all delivered files.

Tinker state and sampler checkpoint references are retained in `training/checkpoint.json`; these are account-scoped references, not portable optimizer files. The downloaded adapter does not include Tinker's full optimizer state. Dataset and evaluation examples are synthetic. Generated reasoning/explanations in the trajectories are model emissions, not verified descriptions of internal computation.

Download the private release with an authorized Hugging Face account:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Perfect7613/glm-5.3-flash-auditor-character-dpo-v7",
    local_dir="glm-flash-character-dpo-v7",
)
```

The original exported configuration sets `base_model_name_or_path` to null. This is preserved unchanged; supply an explicitly verified matching base when loading the adapter from the `adapter/` subfolder. The training base ID above is Tinker-specific. Before loading into another runtime, verify the exact base weights, tokenizer, chat template, module names and LoRA scaling against the exported configuration. Then compare token probabilities before quantization and behavior after conversion. Do not attach the PEFT archive directly to a GGUF server or assume that a similarly named Unsloth checkpoint is numerically identical to the Tinker base.

## Intended progression

Resolve evaluation infrastructure errors without changing the frozen cases, assess behavior on harder held-out non-game tasks, expand the reviewed preference dataset, and complete validated introspection SFT before considering M2 promotion. RLCR remains a separate later experiment using the shared Liberal-faction return and independently verified reporting score. The initial DPO checkpoint is retained even if later stages fail.

## Resources

- [UNDERCOVER Bench research repository](https://github.com/perfect7613/undercover-bench)
- [Open Character Training paper](https://arxiv.org/abs/2511.01689)
- [Open Character Training source](https://github.com/maiush/OpenCharacterTraining)
- [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Tinker weights and deployment guide](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/basics/)
- [Unsloth GLM-5.3-Flash derivative](https://huggingface.co/unsloth/GLM-5.3-Flash)

Base-model and derivative use remain subject to their upstream terms. No broader license, successful character transfer, game-level efficacy, or compatibility guarantee is asserted by this research artifact.
