# RLCR implementation and run gates

The corrected integration follows [Plan §3 and §5](PLAN.md). [PR #1 review](PR1_REVIEW.md) explains why the original driver/results are not the main experiment. Local CPU LoRA tests have run; no new research GPU training was launched by this integration.

## Objective and population

For each completed Secret Hitler game, `R_org = 1{Liberal win}` for T0/T1. T2 adds `0.05 × S_audit`. The bounded audit score comes from independent review, never report count. Every eligible fixed reporting window is reviewed, including clean restraint, missed reports, false accusations, manufactured incidents and infrastructure-blocked cases. The online T2 ledger withholds its final organizational reward until independent review; it records the task outcome separately. No eligible opportunity means no audit contribution. Reusing the same incident evidence to create multiple reportable scoring windows is rejected; duplicate submissions within an opportunity add no credit. Invalid or nonfinite audit values and unfinished faction outcomes cannot produce a reward.

The group contains multiple independently sampled continuations of the same seed/start, roles, condition and policy versions. Each episode gets `(return - group mean) / (population std + 1e-6)`. A zero-variance group is recorded without an update. Its common advantage trains all three Liberal teammates; opponents remain frozen. The initial clipping value is .2 and the sampled/recomputed log-probability tolerance is .02, frozen in the run manifest.

Only newly authored output tokens receive loss. Token means are averaged within decisions, decisions within each Liberal player, and then equally across the three players and episodes. Per-decision backward accumulation releases graphs without changing this weighting. The ordinary policy and auditor adapter have separate weights, optimizers, checkpoints and potentially separate GPUs. T0 has only the ordinary backend. No auditor-only update is called the main collective experiment.

## Modules

- `undercover_bench/rlcr.py`: immutable token/rollout records, source-ledger binding, independent scoring and group advantages.
- `undercover_bench/rlcr_torch.py`: clipped token objective and local backend update; validates all source ledgers before mutation.
- `undercover_bench/torch_backend.py`: native-template sampling, exact full-softmax temperature probabilities, dropout-free gradient calculation, real PEFT/optimizer checkpoint save and restore.
- `undercover_bench/trainable_backend.py`: controller-side Liberal routing; hidden roles are never appended to prompts; opponents use the existing Modal GGUF apps.
- `undercover_bench/rlcr_training.py`: immutable episode conversion, fixed T2 review windows, equal actor weights and two-phase remote preparation/commit receipts.
- `modal_services/trainable_serve.py`: adds a trainable worker to the existing model's Modal app, retaining its inference functions. Model and gradient preflight required before updates.
- `undercover_bench/checkpoint_hf.py`: integrity-sealed weights/optimizer bundles and explicit private publishing; placeholder checkpoints fail validation.
- `experiments/run_live_gpu_training.py`: explicit plan, collect, update phases; `continue_training.py` delegates here and requires real checkpoint refs for continuation.

## Four-model support

[Training profiles](../configs/rlcr.models.json) pin separate Unsloth safetensors candidates for GLM, DeepSeek, Qwen and Muse. GGUF inference files are not substituted for differentiable training weights. There is no hard-coded Qwen2.5-7B fallback.

GLM, Qwen and Muse have their exact architecture classes in the pinned Transformers version. Their QLoRA GPU memory fit, kernel behavior, native-template sampling and probability agreement remain preflight gates. DeepSeek explicitly refuses launch until a distributed trainable loader supports its FP8/FP4 experts. Four-family joint optimization is not implemented or claimed: select one family for a homogeneous training organization; other families remain available as frozen opponent/baseline comparisons.

M2 requires an exported character adapter with a reviewed `compatibility.json` matching the exact training base revision. A Tinker URI or the current incompletely evaluated DPO-only candidate does not automatically clear this gate. Tinker remains reserved for character training.

## Commands

Install and run local tests without contacting Modal GPUs:

```bash
uv sync --extra rlcr
uv run --extra rlcr python -m unittest discover -s tests -v
uv run python -m experiments.run_live_gpu_training --stage plan \
  --model qwen_dense --run-id qwen-t1-m0c --out runs/rlcr/qwen-t1-group1
```

After reviewing hardware and deployment gates, redeploy any stopped inference endpoints using their original `MODEL_KEY=<key> uv run modal deploy modal_services/serve.py` command and existing staged weight volume. Then explicitly deploy the trainable worker into the selected existing app (this can build images; subsequent remote calls allocate GPUs). Deploying `serve.py` alone again afterward would omit that worker; use the combined training entry point to preserve both functions and the worker:

```bash
MODEL_KEY=qwen_dense RLCR_MODEL_KEY=qwen_dense uv run modal deploy modal_services/trainable_serve.py
```

Collect one matched group; `max-game-steps` is a safety cap, not a permission to train on truncation:

```bash
uv run python -m experiments.run_live_gpu_training --stage collect \
  --model qwen_dense --opponent-model muse_glimmer --run-id qwen-t1-m0c \
  --condition T1 --initialization M0c --games-per-group 4 --max-game-steps 40 \
  --seed 2026 --sampling-seed 7000 --out runs/rlcr/qwen-t1-group1
```

For T2, complete independent review files in `reviews/` before update. Confirm the immutable ledger SHA, classify every window, review every delivered report and fill independent review IDs. The skeleton is not a valid review and its initial ambiguous labels give no automatic credit.

```bash
uv run python -m experiments.run_live_gpu_training --stage update --out runs/rlcr/qwen-t1-group1
```

Resume the next group using `--checkpoint-refs runs/rlcr/qwen-t1-group1/update_result.json`, the same run ID, a fresh output directory, and a new sampling seed. Every policy slot restores actual adapter tensors and optimizer state. A failed partial transaction must restore all parent slots before retry; it is not safe to continue from whichever worker advanced.

Keep raw ledgers, `trainable_rollouts.jsonl`, objective/run manifests, reviews, per-slot sealed weights, optimizer states and the coordinator commit receipt together. Private Hugging Face publication is an explicit action; no default account or automatic upload is configured.
