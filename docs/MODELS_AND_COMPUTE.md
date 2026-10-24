# Four-model environment setup and compute plan

Updated 25 October 2026 after the user's model and deployment decisions. Metadata was verified on Hugging Face; Muse and Qwen direct generations passed on separate Modal L40S GPUs, and GLM direct generation passed on a B300. DeepSeek direct generation and a game-observation action passed on a B300; game-level throughput, broader prompt-cache correctness and exported adapters remain untested.

## Selected architecture

- **Character training:** GLM-5.3-Flash on Tinker, using DPO and introspection SFT before the planned collective reward optimization.
- **Inference and environments:** Modal, with one separate GPU-backed deployment per model and a CPU environment/scoring coordinator.
- **Weights:** Unsloth's published Hugging Face checkpoints. Unsloth is the weight publisher in this design; llama.cpp is the initial inference-engine candidate.
- **Four checkpoints:** GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017, Qwen3.8-27B and Muse-Glimmer-30B. These provide four model families; Muse replaces Qwen Flash-Next.

This replaces the earlier recommendation to train Qwen first or use external inference APIs for initial evaluation. Tinker remains the explicit training-service exception to the Modal deployment requirement.

## Exact Unsloth weights and separate GPU candidates

| Model / Hugging Face repository | Initial quantization | Separate Modal allocation | GPU USD/hour |
|---|---|---|---:|
| [GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | UD-Q4_K_XL | 1 B300 | $7.10 |
| [DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | UD-IQ3_XXS, Engram offloaded | 1 B300 | $7.10 |
| [Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL | 1 L40S | $1.95 |
| [Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) | UD-Q4_K_XL | 1 L40S | $1.95 |

The [pinned model manifest](../configs/models.proposed.json) pins repository revisions and enumerates the selected shards with byte counts. [Modal staging](../modal_services/stage_weights.py) and [serving](../modal_services/serve.py) use this manifest. Download only selected files, not every quantization in a repository. Weight storage is distinct from resident GPU memory.

Unsloth also publishes [GLM Flash safetensors](https://huggingface.co/unsloth/GLM-5.3-Flash) and [FP8 weights](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8). The metadata checked reports approximately 642.65GB and 328.34GB of safetensors respectively. A repository carrying the Unsloth name is not necessarily low-bit or small enough for one GPU.

The [GLM Flash guide](https://unsloth.ai/docs/models/glm-5.3-flash) puts 4-bit total memory at 162–210GB. The chosen Q4 shards total about 199.71GB, motivating a 288GB B300 candidate. Validate context, workspace memory and model-specific kernels before declaring fit.

The [DeepSeek guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash) separates the backbone from its Engram lookup table. IQ3_XXS uses about 233.5GB of non-Engram weights and a 161.3GB table. Keep the table off GPU using the documented mmap/offload path; budget local storage and host RAM for caching. Its Q4 variant needs approximately 305GB of RAM plus VRAM even with Engram offloaded, so a fully GPU-resident single-B300 Q4 deployment is not the proposed configuration. Disk I/O and page-cache behavior must be profiled on Modal.

[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a Meta dense model released under Apache-2.0. The [Unsloth guide](https://unsloth.ai/docs/models/muse-glimmer) lists about 17GB for 4-bit inference before context-dependent overhead. The selected Unsloth GGUF is 15,878,222,368 bytes (15.88GB); initial evaluation is text-only, without a vision projector or speculative drafter. Muse and [Qwen 27B](https://unsloth.ai/docs/models/qwen3.8) both loaded and returned valid JSON on separate L40S apps. Throughput and game-level legal-action rate remain under measurement; a cheaper 24GB allocation may be explored after profiling. Do not transfer published RTX 5090 speculative-decoding speeds to this Modal configuration.

## Serving and cost assumptions

Use a separate Modal service per model, initially one replica each, with scale-to-zero between run batches. Keep model-specific engine builds and chat templates pinned; do not infer compatibility from Hugging Face's auto-generated deployment snippets. In particular, check the Unsloth guides' required llama.cpp revisions for GLM Flash, DeepSeek and Muse. B300 requires CUDA 13.1+ according to the [Modal GPU guide](https://modal.com/docs/guide/gpu).

From [Modal pricing](https://modal.com/pricing), the unrounded combined GPU cost is **$18.1008/hour** while all four model allocations are running, or **$9.05 for 30 minutes**. These are rate examples, not predicted experiment durations. This excludes CPU, host RAM, storage, egress and account credits. Load time and warm idle time count. Engram offloading can make host RAM and storage significant; measure them before setting a full-run budget. These prices describe the four proposed inference allocations, not training resources.

Start with 8K context and a bounded generation budget, then measure legal-action rate, truncation, latency, tokens per second and peak memory. Use independent games for batching. GPU assignment does not change who sees private game information: prompts, histories and logs remain isolated per agent even when several seats use one model server.

## Environment and collective reward come first

Use **Secret Hitler for training** and **Liar's Bar for held-out transfer evaluation**, with matched conditions with and without an undercover agent. Keep Liar's Bar evaluation scenarios out of character training, RL and tuning. Four model checkpoints do not imply four players in every environment: preserve each game's legal player count and role distribution.

1. Implement seeded state transitions, legal-action validation, role-specific observations, explicit communication permissions, event logs and an external reporting endpoint.
2. Validate the independent reward scorer with scripted legal play, prohibited exchanges, false accusations, missing reports, timeouts and malformed actions.
3. Run each model checkpoint in homogeneous games first. Replace one matched seat with the same-family prompted auditor, keeping seeds, seats and other players fixed. Rotate auditor seats. Mixed-family games form a separate experiment with balanced seat assignments.
4. Preserve the track 3.2 explicit-alliance positive control, plus direct offers, indirect hints and benign cooperation requests. Record both verbal acceptance and repeated preferential actions. Test intervention after coalition formation separately from susceptibility to an initial offer.
5. Prepare M0, M0c and character-trained M2 on the same base. Run collective reward optimization after the harness and character intervention pass their checks, as specified in the main plan.

Start with **20–40 short pilot games total**, with the condition allocation recorded explicitly. This is not 20–40 games per cell or a complete factorial study. Use the pilot to measure runtime and verify the harness; extend the budget separately for training and adequately replicated evaluation. Do not tune using held-out Liar's Bar outcomes.

The [full plan](PLAN.md) gives a provisional 32-game allocation and distinguishes non-test adapter checks from sealed transfer evaluation. The latest [decision history](DECISIONS.md) also preserves the earlier pressure, social-reading and mechanism-design ideas without adding them to every pilot cell. B1-M2 means no collective RL yet, not no prior character fine-tuning.

### Reward contract

The initial trained organization is the Liberal faction in Secret Hitler, facing frozen opponents. Team victory gives `R_win = 1`; defeat gives `R_win = 0`. Apply the same team-outcome rule with and without an undercover agent, replacing an existing seat rather than adding a player.

Compare collective-only training `R_org = R_win` against `R_org = R_win + lambda * S_audit`, where `S_audit` is a bounded, independently verified score for reporting decisions and delivery to human oversight. Freeze its rubric and lambda before held-out evaluation. Prevent duplicate credit and rewards for manufacturing incidents. Compare prompted and character-trained auditors within both reward branches.

Use B0/B1 for pre-RL baselines without/with undercover; T0 for team-win RLCR without undercover; T1 for team-win RLCR with undercover; T2 for team-win plus verified-reporting RLCR with undercover. Character training precedes the main RLCR experiment. A common team-level advantage updates all designated trainable teammates; updating only the auditor is a separate ablation.

Keep completion, verified violations, outsider harm, coalition advantage and regret as separate diagnostics. The earlier all-table `completion - penalty * violations` proposal is superseded as the main objective. Legal bluffing and lawful faction cooperation are not misconduct; unequal win rewards are not proof of collusion. Optional verified-harm penalties require a separate ablation. Liar's Bar is evaluated with frozen checkpoints, not given an artificial team-training reward.

The four evaluation models do not automatically form one trainable policy. For the first collective training experiment, use GLM Flash for the trainable organization (with the auditor's character adapter as applicable); other families provide baseline and opponent comparisons. Four-family joint training is outside the initial training setup.

## GLM-5.3-Flash character training on Tinker

The selected Tinker ID is `zai-org/GLM-5.3-Flash:peft:262144`. Current training price is **$5.58 per million billed tokens**: 20M billed training tokens cost **$111.60** for that component. Include preference branches, repeated epochs and additional forward passes when counting tokens; rollouts, evaluations and checkpoint storage are extra. See [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) and the [quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/).

Tinker trains its supported Z.ai checkpoint. We cannot pass the Unsloth GGUF repository as the Tinker training base. Confirm matching base weights and tokenizer before transferring the learned adapter to Unsloth-derived inference artifacts.

The [deployment documentation](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/basics/) says merging is supported for all trained families, while live PEFT serving depends on engine architecture support. The [published matrix](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/weights/README.md) is not an end-to-end guarantee for GLM Flash GGUF. The deployment gate is a tiny Tinker adapter exported and either loaded by a compatible engine or merged with the matching base, converted and requantized into a private GGUF artifact. Check numerical agreement before quantization and behavioural agreement after quantization. Do not assume a PEFT adapter directly plugs into llama.cpp.

Compare M0, M0c and M2 on the same GLM-5.3-Flash base, using matched templates and the same quantization recipe. Character training changes weights; Unsloth's original public checkpoint does not contain our character adapter. Keep trained derivatives private and record their provenance separately. Retain held-out scenario families and identical evaluation instructions for M0c and M2. Explanations are behavioural evidence, not direct proof of faithful introspection.

## Collective reward training integration

Cheaper inference makes rollouts cheaper; it does not by itself implement RLCR. The first training integration can update the GLM auditor while other model servers stay frozen, explicitly labeled as an auditor-only update. This is an engineering ablation and does not replace the main plan's joint collective-training comparison.

For Tinker updates driven by Modal rollouts, log exact sampled token IDs, behavior-policy log-probabilities, model/adapter versions, templates and reward events. Quantized serving and Tinker training may represent different distributions. Use an algorithm with the appropriate off-policy correction and monitor divergence; never silently treat stale or differently quantized Modal trajectories as on-policy samples. Failure to obtain reliable probabilities blocks this RL integration, not the initial frozen-model environment evaluation. Do not silently move rollouts to an external API to bypass the user's Modal requirement.

Muse is an inference baseline; its Tinker training support has not been verified. Training all four models would require verified training support and separate policy clients; shared reward is not shared weights. Resolve that implementation before claiming a jointly trained four-model result.

The main provisional GLM organization uses an ordinary-player policy plus a separate auditor adapter, with both updated from the common advantage. Hold ordinary-player initialization constant across character comparisons. Fully shared role-conditioned weights are an alternative configuration, not interchangeable with auditor-specific character training. Activation capture, contrastive vectors, linear probes and steering on GLM require a separate instrumented-runtime preflight and cost estimate; ordinary GGUF inference support does not establish those capabilities. J-Lens is outside the active scope and has no planned fitting, integration or compute allocation.

## Next implementation checkpoints

1. Implement and validate the environments, reporting endpoint and independent collective-reward scorer.
2. Build model-specific Modal images, resolve engine revisions and stage only the pinned weights.
3. Profile each separate GPU service; validate templates, private observations, structured actions, stopping, context limits and measured spend.
4. Run matched no-undercover and constitution-prompted baselines with the track controls.
5. Validate GLM Flash's Tinker export/merge/quantization path with a tiny character adapter before full training.
6. Character-train GLM Flash, verify held-out behavior, and implement the declared collective reward update regime with policy-version accounting.

The two L40S apps and GLM B300 app are deployed and direct generation passed. DeepSeek B300 app is deployed, all seven shards are staged, and direct generation plus one legal game action passed. One fixed-seed, cache-off A/B/A game-prompt check passed; broader prompt-cache correctness remains a gate. No paid training or benchmark efficacy run has been performed; current games are engineering checks.
