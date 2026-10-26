# UNDERCOVER Bench

Evaluating character-trained undercover agents under collective reward optimisation.

This repository contains the research plan for comparing game environments with and without an undercover agent, testing character training through DPO and introspection SFT, and measuring whether external human reporting survives collective reward training.

The interpretability study focuses on activation-vector discovery, linear probes and causal steering for role integrity and suspicion formation.

## Documentation

- [RLCR implementation review](docs/PR1_REVIEW.md) — findings in the original training branch, corrected objective and checkpointing, and model compatibility gates.
- [Historical engineering log](docs/EXPERIMENT_RESULTS_AND_LOG.md) — original unvalidated claims and artifacts; see the review before interpreting them.
- [Full research plan](docs/PLAN.md) — environments, rewards, character training, RLCR, experimental controls, human reporting, analysis, and implementation order.
- [Decision history and conversation reconciliation](docs/DECISIONS.md) — every material proposal update, superseded choices, optional extensions and remaining implementation gates.
- [Four-model setup and compute budget](docs/MODELS_AND_COMPUTE.md) — GLM-5.3-Flash character training on Tinker; four Unsloth checkpoints on separate Modal GPUs, export checks and cost assumptions.
- [Implementation status](docs/IMPLEMENTATION_STATUS.md) — what the controllers, tests, dashboard and GPU services actually validate.
- [Observed raw games](docs/OBSERVED_GAMES_2026-10-25.md) — completed four-model Secret Hitler trace, private channel, auditor decisions and excluded reporting attempts.
- [RLCR code and remaining gates](docs/RLCR_CODE.md) — shared-return batch preparation, authored-token masks, clipped update contract, and the trainable-backend work still required.
- [Engineering runbook](docs/RUNBOOK.md) — exact uv, Modal, game, and dashboard commands with analysis gates.
- [Pinned model specification](configs/models.proposed.json) — verified Hugging Face revisions, weight files, GPU allocations and per-model runtime status.
- [Resources and references](docs/RESOURCES.md) — linked papers, game rules, and original project documents, with their role in the study.
- [DPO v7 model card](training/character/CHARACTER_DPO_V7_MODEL_CARD.md) — rank-32 candidate export, six-pair provenance, incomplete development evaluation, and deployment limits.
- [Tinker character-pilot results](training/character/PILOT_V4_RESULTS.md) — renderer diagnosis, repaired DPO and format SFT comparisons, held-out checks, and promotion decision.
- [AI Collusion In Games](docs/sources/AI%20Collusion%20In%20Games.docx) — original environment and reward proposal.
- [UNDERCOVER-Bench Proposed Improvements](docs/sources/UNDERCOVER-Bench%20Proposed%20Improvements.docx) — original character-training comparison and methodological controls.

## Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Ensure Python 3.11+ and `uv` package manager are available. On NixOS / Linux:
```bash
nix-shell -p uv
# Or on standard Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install repository dependencies:
```bash
uv sync
```

Configure credentials by copying `.env.example`:
```bash
cp .env.example .env
```
Populate `.env` with your API keys (e.g. `HF_TOKEN`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) if not already authenticated via system CLIs (`modal setup` / `huggingface-cli login`). Note that `.env*` is gitignored to protect secrets.

### 2. Validate the corrected training path

```bash
uv sync --extra rlcr
uv run --extra rlcr python -m unittest discover -s tests -v
uv run python -m experiments.run_live_gpu_training --stage plan \
    --model qwen_dense --run-id qwen-t1-m0c --out runs/rlcr/qwen-t1-group1
```

### 3. GPU training gates and existing apps

The corrected worker extends the existing `undercover-<model>` Modal app and preserves its GGUF inference functions. It requires a differentiable safetensors loader, exact-policy/gradient preflight, and independent T2 reviews. GLM, Qwen and Muse are unvalidated training candidates; DeepSeek is explicitly blocked pending compatible distributed training. See [the RLCR runbook](docs/RLCR_CODE.md) before deploying or allocating GPUs. No automatic Hugging Face publishing is configured.

### 4. Visualize Results & Launch Dashboard
To re-generate the metric summary plot and inspect game episodes in console:
```bash
uv run python experiments/visualize_metrics.py --runs runs
```
Or start the web dashboard:
```bash
cd dashboard && npm ci && npm run dev
# Open http://localhost:3000
```

## Current build

The first seeded controllers, independent reward scorer, append-only JSONL episode ledger, Modal GPU setup and local Next.js episode desk are implemented. Secret Hitler now offers one exclusive two-player secret channel through voluntary offer and acceptance; `--no-channel` is a separate communication control. One four-model B1 game has completed under this protocol with zero invalid actions; its observed outcome is a case study, not an efficacy estimate. See [implementation status](docs/IMPLEMENTATION_STATUS.md) for exact validation, limitations and remaining training gates.

Run `uv sync`, then `uv run python -m unittest discover -s tests -v`. To inspect episodes, run `cd dashboard && npm ci && npm run dev` and open http://127.0.0.1:3000.

## Research status

Three small character-pilot LoRA updates ran on Tinker: the initial DPO, a format SFT candidate, and a repaired DPO candidate. None has shown an advantage over the untrained GLM reference, so none is promoted as the planned M1/M2 intervention. No RLCR weight update is claimed. All four pinned Unsloth models generated from separate Modal GPU apps: Muse and Qwen on L40S, GLM and DeepSeek on B300. GLM completed a one-round Secret Hitler engineering trace with 14 legal decisions and no invalid actions. DeepSeek returned a legal game decision and passed one fixed-seed, cache-off A/B/A game-prompt check; broader correctness validation remains open.

The plan was consolidated on 24 October 2026 after reading the full visible proposal conversation and both live source documents. It uses character training to initialise the undercover agent, followed by RLCR as an early experimental stage. Secret Hitler uses faction-win reward, with a separate verified-reporting bonus comparison; Liar’s Bar is held-out evaluation. The initial pilot targets 20–40 short games total, with runtime measured before expansion. The plan includes the character constitution, B0/B1/T0/T1/T2 controls, recognition-versus-delivery evaluation, Track 3.2 protocol, mechanistic methods and equation glossaries.

The two original documents are retained as source material. Where they differ from the consolidated plan, [the full research plan](docs/PLAN.md) defines the proposed implementation; original documents may contain superseded rules or experimental choices.
