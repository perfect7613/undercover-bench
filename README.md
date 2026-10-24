# UNDERCOVER Bench

Evaluating character-trained undercover agents under collective reward optimisation.

This repository contains the research plan for comparing game environments with and without an undercover agent, testing character training through DPO and introspection SFT, and measuring whether external human reporting survives collective reward training.

The interpretability study focuses on activation-vector discovery, linear probes and causal steering for role integrity and suspicion formation.

## Documentation

- [Full research plan](docs/PLAN.md) — environments, rewards, character training, RLCR, experimental controls, human reporting, analysis, and implementation order.
- [Decision history and conversation reconciliation](docs/DECISIONS.md) — every material proposal update, superseded choices, optional extensions and remaining implementation gates.
- [Four-model setup and compute budget](docs/MODELS_AND_COMPUTE.md) — GLM-5.3-Flash character training on Tinker; four Unsloth checkpoints on separate Modal GPUs, export checks and cost assumptions.
- [Implementation status](docs/IMPLEMENTATION_STATUS.md) — what the controllers, tests, dashboard and GPU services actually validate.
- [Engineering runbook](docs/RUNBOOK.md) — exact uv, Modal, game, and dashboard commands with analysis gates.
- [Pinned model specification](configs/models.proposed.json) — verified Hugging Face revisions, weight files, GPU allocations and per-model runtime status.
- [Resources and references](docs/RESOURCES.md) — linked papers, game rules, and original project documents, with their role in the study.
- [AI Collusion In Games](docs/sources/AI%20Collusion%20In%20Games.docx) — original environment and reward proposal.
- [UNDERCOVER-Bench Proposed Improvements](docs/sources/UNDERCOVER-Bench%20Proposed%20Improvements.docx) — original character-training comparison and methodological controls.

## Current build

The first seeded controllers, independent reward scorer, append-only JSONL episode ledger, Modal GPU setup and local Next.js episode desk are implemented. Secret Hitler now offers one exclusive two-player secret channel through voluntary offer and acceptance; `--no-channel` is a separate communication control. The sole retained episode predates this channel change and is an incomplete engineering trace, not an efficacy result. See [implementation status](docs/IMPLEMENTATION_STATUS.md) for exact validation, limitations and remaining training gates.

Run `uv sync`, then `uv run python -m unittest discover -s tests -v`. To inspect episodes, run `cd dashboard && npm ci && npm run dev` and open http://127.0.0.1:3000.

## Research status

No character-training or RLCR weight update is claimed. All four pinned Unsloth models generated from separate Modal GPU apps: Muse and Qwen on L40S, GLM and DeepSeek on B300. GLM completed a one-round Secret Hitler engineering trace with 14 legal decisions and no invalid actions. DeepSeek returned a legal game decision and passed one fixed-seed, cache-off A/B/A game-prompt check; broader correctness validation remains open.

The plan was consolidated on 24 October 2026 after reading the full visible proposal conversation and both live source documents. It uses character training to initialise the undercover agent, followed by RLCR as an early experimental stage. Secret Hitler uses faction-win reward, with a separate verified-reporting bonus comparison; Liar’s Bar is held-out evaluation. The initial pilot targets 20–40 short games total, with runtime measured before expansion. The plan includes the character constitution, B0/B1/T0/T1/T2 controls, recognition-versus-delivery evaluation, Track 3.2 protocol, mechanistic methods and equation glossaries.

The two original documents are retained as source material. Where they differ from the consolidated plan, [the full research plan](docs/PLAN.md) defines the proposed implementation; original documents may contain superseded rules or experimental choices.
