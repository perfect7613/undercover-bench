# Four-model environment setup and compute plan

Research checked 24 October 2026 using Context7 and current primary documentation. This is a setup specification and cost model, not a deployed benchmark or measured throughput result.

## Recommended lineup

| Model | Intended role | Tinker base-model identifier |
|---|---|---|
| GLM-5.3 | Large-model baseline and later character-training replication | `zai-org/GLM-5.3:peft:262144` |
| DeepSeek-V4.1-Flash-1017 | Independent large-model baseline | `deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144` |
| Qwen3.8-27B | First character-training and external-serving development target | `Qwen/Qwen3.8-27B` |
| GPT-OSS-120B | Fourth family and inexpensive training comparison | `openai/gpt-oss-120b` |

All four appear in the current [Tinker model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/). Verify account capabilities before creating training clients. The versioned [DeepSeek release](https://api-docs.deepseek.com/news/news261017/) is dated October 17; record this checkpoint rather than relying on a floating latest alias.

Qwen is the recommended first implementation target because its dense architecture and documented adapter export make iteration easier. This is an engineering recommendation, not evidence that it has the best character-training response. GLM remains a suitable ambitious target once held-out character evaluation and the reward pipeline work.

### What the Unsloth version means

The [Unsloth GLM-5.3 guide](https://unsloth.ai/docs/models/glm-5.3) offers quantized GGUF variants and documents chat-template changes. Tinker trains its listed Z.ai base; an Unsloth repository is not an interchangeable Tinker model identifier. Pin the checkpoint, tokenizer, template, quantization and inference engine. Validate any exported adapter against the exact base before comparing results.

Do not compare an untrained low-bit Unsloth model against a differently quantized trained model and attribute the entire difference to character training. Keep the deployment representation matched within each treatment comparison.

## Environment and collective reward come first

Retain Liar's Bar and Secret Hitler from [the research plan](PLAN.md), with matched conditions with and without an undercover agent. Four model families do not imply four players in every environment: preserve each game's legal player count and role distribution.

1. Implement seeded state transitions, legal-action validation, role-specific observations, explicit communication permissions, event logs and an external reporting endpoint.
2. Validate the independent reward scorer with scripted legal play, prohibited exchanges, false accusations, missing reports, timeouts and malformed actions.
3. Run each model family in homogeneous games first. Replace one matched seat with the same-family prompted auditor, keeping seeds, seats and other players fixed. Rotate auditor seats. Mixed-family games form a separate experiment with balanced seat assignments.
4. Preserve the track 3.2 explicit-alliance positive control, plus direct offers, indirect hints and benign cooperation requests. Record both verbal acceptance and repeated preferential actions. Test intervention after coalition formation separately from susceptibility to an initial offer.
5. Prepare M0, M0c and character-trained M2 on the same base. Run collective reward optimization after the harness and character intervention pass their checks, as specified in the main plan.

For sizing, two environments × four model families × two auditor conditions × five seeds is 80 games before adding offer styles, character training, or other controls. Five seeds is a debugging pilot, not adequate evidence of general superiority. Preregister the eventual evaluation budget after measuring variance and tokens per game.

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

Collecting rewards from frozen models is evaluation/data collection. RLCR requires actual policy updates. Four different model families cannot share one set of weights; any later simultaneous training uses separate policies/clients and an explicitly specified common-advantage procedure. Start training within one family, with frozen-opponent controls, before attempting all-family co-adaptation.

## Character training through Tinker

Use the [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) and cookbook to implement the planned DPO and introspection SFT intervention, followed by RLCR. Maintain held-out scenario families and compare M0c and M2 under identical evaluation instructions. Explanations are behavioural outputs, not proof of faithful introspection.

Tinker's sampling clients can generate from base models and trained checkpoints. This is distinct from its separately advertised serverless inference product; do not assume identical endpoint support or production guarantees.

The [export support matrix](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/weights/README.md) documents Qwen3.8 dense and GPT-OSS adapter conversion. Its published matrix does not establish GLM-5.3 or DeepSeek-V4.1 external adapter-serving compatibility. Their absence is an unresolved compatibility check, not proof they cannot be exported. Training support alone does not establish vLLM, SGLang or llama.cpp LoRA support.

Before committing to Modal for a trained model: train a tiny adapter, save it, export using the [PEFT adapter guide](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/), load it on the exact base, and compare outputs/log-probabilities with the Tinker checkpoint. If unsupported, retain Tinker sampling while investigating export; do not assume merging automatically solves architecture support.

## Tinker budget

USD per million tokens at the time of checking; sampling input assumes no cache discount. Context variants are the exact identifiers above.

| Model | Sampling input | Sampling output | Training | 20M billed training tokens |
|---|---:|---:|---:|---:|
| GLM-5.3 | $3.78 | $12.15 | $14.58 | $291.60 |
| DeepSeek-V4.1-Flash-1017 | $0.60 | $3.60 | $4.98 | $99.60 |
| Qwen3.8-27B | $0.6975 | $5.595 | $4.103 | $82.06 |
| GPT-OSS-120B | $0.33 | $0.84 | $0.737 | $14.74 |

Source: [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/). These are training-component estimates, not complete project quotes. Count tokens processed across epochs, preference branches and extra forward passes; add rollout generation, evaluation, reference computations and checkpoint storage as applicable.

For an illustrative inference workload of 10,000 total calls, split equally across the four models, averaging 8,000 input and 1,000 generated tokens per call, uncached sampling costs about $163.61. Generated reasoning tokens count toward the output assumption. This is not a forecast of tokens per game; instrument the first pilot and replace the assumptions.

## Modal serving budget

Rates converted from [Modal's published per-second prices](https://modal.com/pricing). These are GPU-only charges for one running allocation:

| Allocation | USD/hour | USD/10 hours |
|---|---:|---:|
| 1 × A100 80GB | $2.50 | $24.98 |
| 1 × H100 | $3.95 | $39.49 |
| 4 × H200 | $18.16 | $181.58 |
| 8 × H200 | $36.32 | $363.17 |

Add CPU, host RAM, storage and applicable egress. Account credits are not included. Include model loading and any billed warm idle time in runtime estimates; scale-to-zero does not eliminate startup costs. More replicas multiply allocations. Check account GPU concurrency before scheduling multiple servers.

The [Modal GPU guide](https://modal.com/docs/guide/gpu) allows multi-GPU allocations, including up to eight H200 GPUs per container. The [inference performance guide](https://modal.com/docs/guide/high-performance-llm-inference) explains batching, throughput and cold-start tradeoffs. Sequential turns within one game may leave GPUs underused; batch independent games while preserving each game's action order.

- **Qwen3.8-27B:** a single 80GB GPU is a candidate for a short-context BF16 pilot (roughly 54GB of weights by parameter arithmetic). Profile KV cache, runtime overhead and adapter memory before increasing concurrency. See its [official model card](https://huggingface.co/Qwen/Qwen3.8-27B).
- **GPT-OSS-120B:** its [official card](https://huggingface.co/openai/gpt-oss-120b) documents single-80GB-GPU inference with native MXFP4. That does not guarantee identical memory after adapter export or a BF16 merge. Preserve the required Harmony formatting.
- **GLM-5.3 Unsloth:** documented 4-bit memory is roughly 372–475GB total RAM plus VRAM, and 8-bit is about 810GB. Four H200s are a plausible 4-bit profiling allocation, not a verified serving recipe. GGUF/llama.cpp requirements do not establish vLLM compatibility or throughput.
- **DeepSeek-V4.1:** establish the exact checkpoint format and supported inference engine before assigning a minimum GPU count. Active parameter count is not resident weight size. No measured serving layout is claimed here.

For capacity budgeting only, allocating four H200s to each large model and one H100 to each smaller model costs $44.22/hour, or $442.15 over ten hours. Increasing one large model to eight H200s raises that to $62.37/hour or $623.74 over ten hours. These are illustrative resource reservations, not verified fit or runtime estimates. They require 10 or 14 simultaneous GPUs respectively; account limits may require sequential runs.

## Recommended first deployment decision

Run the game harness on CPU and use metered model sampling for initial baselines. Develop character training on Qwen, then replicate on GLM if held-out results justify the larger experiment. Use Modal where exported weights, local activations, or sustained throughput are needed. Keep the Unsloth GLM serving experiment explicitly labeled until quantization and adapter compatibility are validated.

No paid training or Modal deployment was started while preparing this document.
