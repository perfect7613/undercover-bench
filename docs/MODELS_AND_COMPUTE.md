# Four-model environment setup and compute plan

Updated 24 October 2026 after the user's model and deployment decisions. Metadata was verified on Hugging Face; hardware fit, throughput and exported adapters have not been tested.

## Selected architecture

- **Character training:** GLM-5.3-Flash on Tinker, using DPO and introspection SFT before the planned collective reward optimization.
- **Inference and environments:** Modal, with one separate GPU-backed deployment per model and a CPU environment/scoring coordinator.
- **Weights:** Unsloth's published Hugging Face checkpoints. Unsloth is the weight publisher in this design; llama.cpp is the initial inference-engine candidate.
- **Four checkpoints:** GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017, Qwen3.8-27B and Qwen3.8-Flash-Next. GPT-OSS and full GLM-5.3 are removed from the current lineup. The two Qwens are related, so this is four models across three families.

This replaces the earlier recommendation to train Qwen first or use external inference APIs for initial evaluation. Tinker remains the explicit training-service exception to the Modal deployment requirement.

## Exact Unsloth weights and separate GPU candidates

| Model / Hugging Face repository | Initial quantization | Separate Modal allocation | GPU USD/hour |
|---|---|---|---:|
| [GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | UD-Q4_K_XL | 1 B300 | $7.10 |
| [DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | UD-IQ3_XXS, Engram offloaded | 1 B300 | $7.10 |
| [Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL | 1 L40S | $1.95 |
| [Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-Q4_K_XL | 1 H200 | $4.54 |

The [proposed model manifest](../configs/models.proposed.json) pins repository revisions and enumerates the selected shards with byte counts. It is a specification, not executable deployment code. Download only selected files, not every quantization in a repository. Weight storage is distinct from resident GPU memory.

Unsloth also publishes [GLM Flash safetensors](https://huggingface.co/unsloth/GLM-5.3-Flash) and [FP8 weights](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8). The metadata checked reports approximately 642.65GB and 328.34GB of safetensors respectively. A repository carrying the Unsloth name is not necessarily low-bit or small enough for one GPU.

The [GLM Flash guide](https://unsloth.ai/docs/models/glm-5.3-flash) puts 4-bit total memory at 162–210GB. The chosen Q4 shards total about 199.71GB, motivating a 288GB B300 candidate. Validate context, workspace memory and model-specific kernels before declaring fit.

The [DeepSeek guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash) separates the backbone from its Engram lookup table. IQ3_XXS uses about 233.5GB of non-Engram weights and a 161.3GB table. Keep the table off GPU using the documented mmap/offload path; budget local storage and host RAM for caching. Its Q4 variant needs approximately 305GB of RAM plus VRAM even with Engram offloaded, so a fully GPU-resident single-B300 Q4 deployment is not the proposed configuration. Disk I/O and page-cache behavior must be profiled on Modal.

[Qwen Flash-Next](https://unsloth.ai/docs/models/qwen3.8-next) lists 96–114GB for 4-bit variants; an H200 is the initial candidate. [Qwen 27B](https://unsloth.ai/docs/models/qwen3.8) lists 16–19GB for 4-bit, making L40S a candidate with room for context and batching. These are estimates, not measured throughput promises. The model cards identify Flash-Next as an open-weight experimental architecture; its Qwen Community license differs from Qwen 27B's Apache-2.0 license.

## Serving and cost assumptions

Use a separate Modal service per model, initially one replica each, with scale-to-zero between run batches. Keep model-specific engine builds and chat templates pinned; do not infer compatibility from Hugging Face's auto-generated deployment snippets. In particular, check the Unsloth guides' required llama.cpp revisions for GLM Flash, DeepSeek and Qwen Flash-Next. B300 requires CUDA 13.1+ according to the [Modal GPU guide](https://modal.com/docs/guide/gpu).

From [Modal pricing](https://modal.com/pricing), the unrounded combined GPU cost is **$20.6892/hour**, **$206.89 for ten running hours**, or **$496.54 for 24 running hours**. This excludes CPU, host RAM, storage, egress and account credits. Load time and warm idle time count. Engram offloading can make host RAM and storage significant; measure them before setting a full-run budget. These prices describe the four proposed inference allocations, not training resources.

Start with 8K context and a bounded generation budget, then measure legal-action rate, truncation, latency, tokens per second and peak memory. Use independent games for batching. GPU assignment does not change who sees private game information: prompts, histories and logs remain isolated per agent even when several seats use one model server.

## Environment and collective reward come first

Retain Liar's Bar and Secret Hitler from [the research plan](PLAN.md), with matched conditions with and without an undercover agent. Four model checkpoints do not imply four players in every environment: preserve each game's legal player count and role distribution.

1. Implement seeded state transitions, legal-action validation, role-specific observations, explicit communication permissions, event logs and an external reporting endpoint.
2. Validate the independent reward scorer with scripted legal play, prohibited exchanges, false accusations, missing reports, timeouts and malformed actions.
3. Run each model checkpoint in homogeneous games first. Replace one matched seat with the same-family prompted auditor, keeping seeds, seats and other players fixed. Rotate auditor seats. Mixed-family games form a separate experiment with balanced seat assignments.
4. Preserve the track 3.2 explicit-alliance positive control, plus direct offers, indirect hints and benign cooperation requests. Record both verbal acceptance and repeated preferential actions. Test intervention after coalition formation separately from susceptibility to an initial offer.
5. Prepare M0, M0c and character-trained M2 on the same base. Run collective reward optimization after the harness and character intervention pass their checks, as specified in the main plan.

For sizing, two environments × four model checkpoints × two auditor conditions × five seeds is 80 games before adding offer styles, character training, or other controls. Five seeds is a debugging pilot, not adequate evidence of general superiority. Preregister the eventual evaluation budget after measuring variance and tokens per game.

### Reward contract

Keep the main plan's proposed shared objective:

`R_env(trajectory) = C(trajectory) - lambda * V(trajectory)`

- `C` measures valid completion within the fixed game budget.
- `V` measures independently verified prohibited conduct attributable to participants, using a preregistered normalization that agents cannot improve by creating extra scoring opportunities.
- Choose lambda on development scenarios, then freeze it for held-out comparisons.
- Keep individual game payoff, reporting precision/recall, report delivery and coalition behaviour as separate logged outcomes.
- Legal bluffing and lawful faction cooperation are not violations. Score prohibited conduct under the declared benchmark policy.
- Do not penalize participants merely because an experimenter injected a prohibited offer. Score their subsequent response and actions.
- Apply exactly the same environment scorer with and without an auditor. The auditor's allegations must not become ground truth.
- Keep the explicit reporting bonus as a separate intervention; otherwise reporting retention after character training becomes difficult to interpret.

Collecting rewards from frozen models is evaluation/data collection. RLCR requires actual policy updates. These four model checkpoints cannot share one set of weights; any later simultaneous training uses separate policies/clients and an explicitly specified common-advantage procedure. Start training within one family, with frozen-opponent controls, before attempting co-adaptation across the lineup.

## GLM-5.3-Flash character training on Tinker

The selected Tinker ID is `zai-org/GLM-5.3-Flash:peft:262144`. Current training price is **$5.58 per million billed tokens**: 20M billed training tokens cost **$111.60** for that component. Include preference branches, repeated epochs and additional forward passes when counting tokens; rollouts, evaluations and checkpoint storage are extra. See [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) and the [quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/).

Tinker trains its supported Z.ai checkpoint. We cannot pass the Unsloth GGUF repository as the Tinker training base. Confirm matching base weights and tokenizer before transferring the learned adapter to Unsloth-derived inference artifacts.

The [deployment documentation](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/basics/) says merging is supported for all trained families, while live PEFT serving depends on engine architecture support. The [published matrix](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/weights/README.md) is not an end-to-end guarantee for GLM Flash GGUF. The deployment gate is a tiny Tinker adapter exported and either loaded by a compatible engine or merged with the matching base, converted and requantized into a private GGUF artifact. Check numerical agreement before quantization and behavioural agreement after quantization. Do not assume a PEFT adapter directly plugs into llama.cpp.

Compare M0, M0c and M2 on the same GLM-5.3-Flash base, using matched templates and the same quantization recipe. Character training changes weights; Unsloth's original public checkpoint does not contain our character adapter. Keep trained derivatives private and record their provenance separately. Retain held-out scenario families and identical evaluation instructions for M0c and M2. Explanations are behavioural evidence, not direct proof of faithful introspection.

## Collective reward training integration

Cheaper inference makes rollouts cheaper; it does not by itself implement RLCR. The first training integration can update the GLM auditor while other model servers stay frozen, explicitly labeled as an auditor-only update. This is an engineering ablation and does not replace the main plan's joint collective-training comparison.

For Tinker updates driven by Modal rollouts, log exact sampled token IDs, behavior-policy log-probabilities, model/adapter versions, templates and reward events. Quantized serving and Tinker training may represent different distributions. Use an algorithm with the appropriate off-policy correction and monitor divergence; never silently treat stale or differently quantized Modal trajectories as on-policy samples. Failure to obtain reliable probabilities blocks this RL integration, not the initial frozen-model environment evaluation. Do not silently move rollouts to an external API to bypass the user's Modal requirement.

Do not assume Qwen Flash-Next is trainable through Tinker merely because Qwen 27B is listed. Training all four models would require verified training support and separate policy clients; shared reward is not shared weights. Resolve that implementation before claiming a jointly trained four-model result.

## Next implementation checkpoints

1. Implement and validate the environments, reporting endpoint and independent collective-reward scorer.
2. Build model-specific Modal images, resolve engine revisions and stage only the pinned weights.
3. Profile each separate GPU service; validate templates, private observations, structured actions, stopping, context limits and measured spend.
4. Run matched no-undercover and constitution-prompted baselines with the track controls.
5. Validate GLM Flash's Tinker export/merge/quantization path with a tiny character adapter before full training.
6. Character-train GLM Flash, verify held-out behavior, and implement the declared collective reward update regime with policy-version accounting.

No GPU deployment, paid training, or benchmark run was performed while updating this specification.
