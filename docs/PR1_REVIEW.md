# PR #1 review and corrected integration

Reviewed 26 October 2026 against PR head `42ca9edfae3b866f4e860397e9c03dc96b2a94ae`, remote main `1a7588e`, and the consolidated plan. The original PR should not be merged as a scientifically valid RLCR implementation. This branch incorporates the remote scaffold and replaces its unsafe training paths. Its archived traces remain historical engineering artifacts.

## Findings that affect the experiment

1. **Wrong training population and credit assignment.** `run_live_gpu_training.py` pooled every decision, including Fascist and Hitler actions, and applied one batch-wide `mean_win - .5` signal. This can reinforce opponent behavior on Liberal victories, penalize successful trajectories when the batch loses, and overweight verbose/frequently acting players. The replacement routes only the three Liberals to trainable policies, gives each episode its own normalized group-relative advantage, and weights tokens within decisions, decisions within players, players within episodes, and episodes equally. Frozen opponents use the existing inference apps.
2. **Reporting reward can be hacked.** `s_audit = 1 if any report was delivered else 0` rewards accusations without verifying conduct, evidence, accuracy, deadlines or clean restraint. The replacement invokes the existing bounded independent scorer. T2 requires a review tied to the immutable ledger and every controller reporting window, including clean/NoReport windows and every submitted report. Legal faction play and channel creation are not automatically labeled misconduct. No incident is manufactured to force a positive result.
3. **Incomplete episodes were used as losses.** The driver converted any non-Liberal outcome, including `winner=null`, into zero. The PR has 18 episode files: 12 completed/valid, 3 truncated, and 3 nonterminal partial files. The replacement rejects a whole comparison group containing incomplete/invalid games and preserves those traces without training or silent resampling.
4. **Behavior probabilities did not match sampling.** The server sampled at temperature .7 but stored unscaled log probabilities, then enabled LoRA dropout for optimization. The replacement uses the same full-softmax temperature and eval mode with autograd for both paths. Native chat templates, exact token traces, model/tokenizer/template hashes, and policy versions are recorded. Every Liberal decision, including reporting, must have a matching token ledger event. A real per-model sampled-versus-recomputed probability and gradient preflight must pass before updates.
5. **Published checkpoints were not resumable weights.** The live driver never called the GPU server's save method and passed `model=None` to a helper that wrote placeholder adapter configuration. No adapter tensors or optimizer state were in those local bundles. The replacement requires actual safetensors, PEFT configuration and optimizer state, hashes every file, verifies reloads, and advances versions only with sealed checkpoints. Continuation restores every slot from the previous committed coordinator receipt; step counters are not restoration.
6. **Synthetic and actual training were mixed.** `run_rlcr_training.py` generated fake token IDs and fixed loss `.0421`/gradient `.185`, while continuation duplicated log lines. That path is removed. CPU test fixtures are explicitly labeled and never exported as research training. Existing logs are not retroactively authenticated.
7. **Model and artifact identity were misleading.** The service was hard-coded to Qwen2.5-7B, used process-random Python hashes, and generated model cards claiming character training and an Apache license regardless of provenance. New profiles pin the four selected safetensors revisions, select their actual architecture classes, and never substitute a smaller family. Character exports and RLCR artifacts retain separate purposes and status.
8. **Case-colliding plan path.** Remote main added `docs/plan.md` beside `docs/PLAN.md`. On a case-insensitive filesystem this overwrites the canonical plan during checkout. The integration removes the lowercase duplicate and preserves the full research plan.

## Model/runtime status

| Family | Training candidate | Current gate |
|---|---|---|
| GLM | Unsloth GLM-5.3-Flash | Architecture class available in pinned Transformers; single-B300 QLoRA fit, kernels, exact probabilities and character adapter compatibility remain unverified |
| Qwen | Unsloth Qwen3.8-27B | Architecture class available; L40S QLoRA preflight remains unverified |
| Muse | Unsloth Muse-Glimmer-30B | Architecture class available; L40S QLoRA preflight remains unverified |
| DeepSeek | Unsloth DeepSeek-V4.1-Flash-1017 | Blocked: FP8/FP4 expert checkpoint needs a compatible distributed training loader; its architecture class is absent from the pinned Transformers release |

All four existing GGUF apps remain the frozen inference endpoints. The training deployment extends the same `undercover-<key>` Modal app and retains its existing `generate` and `stage` functions. A new differentiable worker cannot reuse GGUF tensors as trainable parameters. No new inference app, inference weight download, paid GPU preflight, research rollout, or RLCR update was launched during this review.

## Validation and limits

The final reward audit also rejects incomplete winners, nonfinite/out-of-range audit scores, changes to the frozen reporting coefficient, and reuse of the same incident evidence across scoring windows. The complete suite passes 51 tests, including withholding a final T2 ledger reward until independent review.

Local checks include both game controllers, independent reward/scorer contracts, actor routing and hidden-information isolation, complete token-ledger coverage, truncation exclusion, fixed T2 review windows, exact temperature probabilities, disabled dropout, a real tiny CPU PEFT optimizer update, checkpoint hashes and exact optimizer/weight reload. These validate the implementation contract, not four-family GPU readiness or research efficacy.

A cross-worker commit is not distributed atomicity. If either worker or checkpoint save fails, the coordinator marks the transaction failed and refuses automatic reuse; restore every slot to its parent checkpoint before a new attempt. No partial commit is advertised as a successful collective update.
