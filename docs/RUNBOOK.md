# Running the engineering pilot

Use this with [the research plan](PLAN.md) and [implementation status](IMPLEMENTATION_STATUS.md). These commands are engineering checks, not an RLCR training run.

```bash
uv sync
uv run python -m unittest discover -s tests -q
```

The four pinned model keys are `glm_flash`, `deepseek_flash`, `qwen_dense`, and `muse_glimmer`. Each key has its own Modal app, volume, and GPU allocation in [the manifest](../configs/models.proposed.json). Modal apps scale to zero after five idle minutes; starting a B300 app may load hundreds of gigabytes and incurs GPU, CPU, memory, and volume costs.

```bash
uv run modal run modal_services/stage_weights.py --model-key muse_glimmer
MODEL_KEY=muse_glimmer uv run modal deploy modal_services/serve.py
```

Repeat only for keys that have not already been staged/deployed. Staging verifies file presence and byte count; it does not prove that the server can load or generate. Run one short direct generation before a game. In particular, DeepSeek requires an isolated reproducibility check because its provisional llama.cpp branch has a [reported prompt-cache correctness issue](https://github.com/ggml-org/llama.cpp/issues/30944).

```bash
uv run python -m undercover_bench.runner --game secret_hitler --condition B0 --seed 820 --max-steps 30 --model scripted --pressure
uv run python -m undercover_bench.runner --game secret_hitler --condition B1 --seed 820 --max-steps 30 --model scripted --pressure
uv run python -m undercover_bench.runner --game secret_hitler --condition B1 --seed 710 --max-steps 2 --model qwen_dense --pressure
uv run python -m undercover_bench.runner --game liars_bar --condition B1 --seed 713 --max-steps 3 --model scripted --pressure --offer-style direct
```

Keep seed, pressure, model family, opponent policy, and step cap matched for B0/B1 comparisons. A two-round trace is a serving and legality check, not a victory-rate result. The Modal decision path uses a phase-specific JSON schema and local Pydantic validation; GLM and DeepSeek have an 8,192-token outer ceiling so reasoning models have room to return a final action. The episode JSONL contains controller-only secrets as well as player-visible observations, so share it only in the private research setting. `episode_end.valid_for_analysis` is false for partial games and any game with an illegal model action. The visible fallback allows debugging but never supplies a research reward.

```bash
cd dashboard
npm ci
npm run dev
```

The local Next.js episode desk reads `runs/*.jsonl`. Use its event details to inspect observations, submitted and applied actions, model output, model-emitted reasoning, private communications, and controller checkpoints. A returned reasoning field is model output, not verified access to hidden chain of thought.

T0/T1/T2 remain blocked until the Tinker character adapter path, matching rollout token/log-probability capture, and actual joint collective updates are validated. Do not relabel B0/B1 prompt or scripted runs as trained results.
