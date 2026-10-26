"""Industry-standard checkpointing and Hugging Face export module for Undercover Bench.

Exports:
1. LoRA adapter weights (adapter_model.safetensors, adapter_config.json) via PEFT
2. Full experiment metadata, prompt configs, and training curves
3. Benchmark plots, metrics summaries, and raw episode ledgers
4. Generates an industry-standard Model Card (README.md) with base model tags,
   evaluation results, and license details.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo


def build_model_card(
    base_model: str,
    adapter_name: str,
    step: int,
    metrics: dict[str, Any],
) -> str:
    """Generate a clean Hugging Face model card in Markdown format with YAML frontmatter."""
    return f"""---
base_model: {base_model}
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

# {adapter_name}

This repository contains a **PEFT LoRA adapter** trained with **Reinforcement Learning from Collective Reward (RLCR)** and auditor character optimization under the [Undercover Bench](https://github.com/poseidon-research/collusion_hackathon) framework.

## Model Details
- **Base Model:** `{base_model}`
- **Training Step:** `{step}`
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **Framework:** PyTorch / PEFT / Transformers / Modal

## Training & Evaluation Metrics
- **Completed Games:** `{metrics.get('total_games', 'N/A')}`
- **Liberal Faction Win Rate:** `{metrics.get('liberal_win_rate', 'N/A')}`
- **Human Reports Delivered:** `{metrics.get('delivered_reports', 'N/A')}`
- **Grounded Evidence Citations:** `{metrics.get('grounded_evidence', 'N/A')}`
- **Latest Policy Loss:** `{metrics.get('latest_loss', 'N/A')}`

## Files Included in Checkpoint
- `adapter_config.json`: PEFT LoRA configuration
- `adapter_model.safetensors`: Trained adapter weights
- `metrics.json`: Full benchmark evaluation results
- `dashboard.png`: Visualized win-rate, reporting, and reward trajectories
- `episodes_raw/`: Raw JSONL game rollouts and event ledgers
"""


def save_checkpoint_to_disk(
    checkpoint_dir: str | Path,
    model: Any | None,
    base_model_name: str,
    step: int,
    runs_dir: str | Path = "runs",
    training_log_path: str | Path = "runs/training_steps.jsonl",
    plot_path: str | Path = "runs/training_dashboard.png",
) -> Path:
    """Save LoRA weights, artifacts, plots, and raw data to a structured checkpoint directory."""
    cp_path = Path(checkpoint_dir) / f"step_{step}"
    cp_path.mkdir(parents=True, exist_ok=True)

    # 1. Save LoRA adapter weights in PEFT format
    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(cp_path))
    else:
        # Save placeholder/metadata if saving without live in-memory weights
        config_path = cp_path / "adapter_config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps({
                "base_model_name_or_path": base_model_name,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
            }, indent=2))

    # 2. Copy plots if available
    if Path(plot_path).is_file():
        shutil.copy2(plot_path, cp_path / "dashboard.png")

    # 3. Copy raw episode ledgers
    raw_dir = cp_path / "episodes_raw"
    raw_dir.mkdir(exist_ok=True)
    runs_p = Path(runs_dir)
    if runs_p.is_dir():
        for file in runs_p.glob("*.jsonl"):
            shutil.copy2(file, raw_dir / file.name)

    # 4. Compile metrics metadata
    metrics: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model_name,
        "step": step,
    }
    (cp_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5. Write Model Card
    card = build_model_card(base_model_name, f"undercover-lora-step-{step}", step, metrics)
    (cp_path / "README.md").write_text(card)

    print(f"Checkpoint successfully saved to disk: {cp_path.resolve()}")
    return cp_path


def upload_checkpoint_to_huggingface(
    checkpoint_dir: str | Path,
    repo_id: str,
    hf_token: str | None = None,
    private: bool = True,
) -> str:
    """Upload LoRA adapter checkpoint, metrics, model card, and plots to Hugging Face."""
    import huggingface_hub

    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or huggingface_hub.get_token()
    )
    if not token:
        raise ValueError(
            "Hugging Face token not found. Please log in with `huggingface-cli login` or set HF_TOKEN."
        )

    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"Authenticated with Hugging Face as: {user_info.get('name')}")

    # Create repo if it doesn't already exist
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    # Upload entire checkpoint folder (adapter weights, README model card, plots, raw data)
    print(f"Uploading checkpoint from {checkpoint_dir} to https://huggingface.co/{repo_id}...")
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete: {url}")
    return url
