# UNDERCOVER Bench

Evaluating character-trained undercover agents under collective reward optimisation.

This repository contains the research plan for comparing game environments with and without an undercover agent, testing character training through DPO and introspection SFT, and measuring whether external human reporting survives collective reward training.

## Documentation

- [Full research plan](docs/PLAN.md) — environments, rewards, character training, RLCR, experimental controls, human reporting, analysis, and implementation order.
- [Resources and references](docs/RESOURCES.md) — linked papers, game rules, and original project documents, with their role in the study.
- [AI Collusion In Games](docs/sources/AI%20Collusion%20In%20Games.docx) — original environment and reward proposal.
- [UNDERCOVER-Bench Proposed Improvements](docs/sources/UNDERCOVER-Bench%20Proposed%20Improvements.docx) — original character-training comparison and methodological controls.

## Status

Research planning. No training runs, benchmark implementation, or experimental results are included yet.

The plan is the version prepared on 24 October 2026. It uses character training to initialise the undercover agent, followed by RLCR as an early experimental stage. The environment-integrity reward is a proposed operational definition to validate before training.

The two original documents are retained as source material. Where they differ from the consolidated plan, [the full research plan](docs/PLAN.md) defines the proposed implementation; original documents may contain superseded rules or experimental choices.
