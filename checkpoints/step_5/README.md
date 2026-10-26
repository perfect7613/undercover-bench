---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- undercover-bench
- lora
- rlcr
- reinforcement-learning
- social-deduction
- character-training
license: apache-2.0
---

# undercover-lora-step-5

This repository contains a **PEFT LoRA adapter** trained with **Reinforcement Learning from Collective Reward (RLCR)** and auditor character optimization under the [Undercover Bench](https://github.com/poseidon-research/collusion_hackathon) framework.

## Model Details
- **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Training Step:** `5`
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **Framework:** PyTorch / PEFT / Transformers / Modal

## Training & Evaluation Metrics
- **Completed Games:** `N/A`
- **Liberal Faction Win Rate:** `N/A`
- **Human Reports Delivered:** `N/A`
- **Grounded Evidence Citations:** `N/A`
- **Latest Policy Loss:** `N/A`

## Files Included in Checkpoint
- `adapter_config.json`: PEFT LoRA configuration
- `adapter_model.safetensors`: Trained adapter weights
- `metrics.json`: Full benchmark evaluation results
- `dashboard.png`: Visualized win-rate, reporting, and reward trajectories
- `episodes_raw/`: Raw JSONL game rollouts and event ledgers
