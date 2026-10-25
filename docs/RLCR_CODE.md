# RLCR code status

The first RLCR code is written but **no RLCR rollout or optimizer step has run**. On 25 October 2026 the four project Modal inference apps were stopped; `modal container list --json` returned no running containers. This work only ran local CPU contract tests. Tinker remains reserved for later GLM character training.

## Frozen objective encoded

[`undercover_bench/rlcr.py`](../undercover_bench/rlcr.py) prepares a Secret Hitler comparison group under [Plan §3 and §5](PLAN.md):

- T0 and T1 use the completed Liberal-faction win (`1` or `0`). T2 adds `0.05 × S_audit`, where `S_audit` comes from independently reviewed, fixed opportunities. No eligible opportunity contributes zero audit bonus and is recorded separately.
- All three Liberal players receive the **same organizational advantage**. Opponents have no trainable token trace. T0 has no auditor; T1/T2 use an ordinary-player policy plus a distinct Liberal auditor adapter.
- The group-relative advantage is `(return - group mean) / (group population standard deviation + ε)`. A zero-variance group has advantage `0` and produces no update. Group size, ε, clipping and maximum tolerated policy drift are explicit configuration inputs; the reporting coefficient is frozen at `0.05` for the initial branch.
- Each decision separates context tokens from its **newly authored** tokens. The loss mask is zero for prompt, observations and received messages, one for authored output. Behavior log probabilities must align one-to-one with those authored tokens. Tokenizer, template, base revision and policy version are checked.

[`undercover_bench/rlcr_torch.py`](../undercover_bench/rlcr_torch.py) defines a one-group clipped policy update for a future trainable backend. It averages token loss within each decision, decisions within each Liberal player, then gives each player and episode equal weight. This prevents the auditor or a verbose player from gaining extra weight merely by producing more text. Each player's loss is backpropagated on its own backend device, so ordinary and auditor adapters need not share a GPU. The code rejects excessive log-probability drift, requires a finite nonzero gradient for every trainable backend, steps both ordinary and auditor backends where applicable, and requires each backend to persist an advanced checkpoint version. The mathematical reference objective is also implemented without PyTorch for CPU tests. The clipping rule here is a **provisional implementation choice** and must be frozen with the training run configuration before experiments.

## Required rollout artifact and gates

The training sampler must write one `TrainingRollout` JSON object per line and a matching append-only controller ledger. Its `episode_start` must say `provenance="trainable_rlcr"`; each trainable `decision` must have a matching `training_token_decision` event containing the exact token trace. The rollout record stores the ledger SHA-256 and decision sequence numbers. Before an update, the code checks that hash, the terminal outcome, three Liberal roles, auditor seat, every authored trace, and reported delivery. T2 opportunities require an independent review ID; reportable evidence must have been sent **to** the auditor by another player. The scorer rejects duplicate or fabricated report credit and does not treat an agent's accusation as ground truth.

Existing [GGUF inference ledgers](../runs/README.md) fail these gates: they have B1 provenance and lack aligned generated-token IDs and behavior-policy log probabilities. They are baseline observations, not training trajectories. `run_episode` still blocks T0/T1/T2 until a real trainable sampler and checkpoint backend are connected.

## Remaining integration

1. Implement a GLM-5.3-Flash trainable sampler and LoRA backend on appropriately sized Modal training hardware. It must emit exact token IDs, behavior log probabilities, policy versions and the required ledger events. The existing quantized llama.cpp servers cannot be treated as that sampler.
2. Attach independent audit opportunity review and actual human-report receipts before enabling T2. Keep clean, reportable, ambiguous and manufactured cases separate.
3. Verify an optimizer step and checkpoint reload on a tiny CPU or appropriately sized training fixture, then run a paid GPU preflight only when requested. Match T0/T1/T2 initialization, seed groups and token budgets before any research comparison.

The optional PyTorch update path uses the official [PyTorch optimizer contract](https://docs.pytorch.org/docs/optim.html) and expects a backend that can return differentiable authored-token log probabilities. Modal supports [multi-GPU containers](https://modal.com/docs/guide/gpu), but this repository has **not** chosen or validated a GLM training placement. No training image, GPU job or cost claim is implied by this code.
