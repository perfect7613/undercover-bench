"""Modal trainable policy service for Undercover Bench.

Serves token-level decisions with behavior log probabilities during game rollouts
and applies LoRA policy-gradient updates on the same GPU container.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

APP_NAME = "undercover-trainable-policy"
app = modal.App(APP_NAME)

# Use PyTorch with CUDA 12.4+ and unsloth / peft support
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.45.0",
        "pydantic>=2.0.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    )
)

volume = modal.Volume.from_name("undercover-trainable-checkpoints", create_if_missing=True)


def _get_hf_secret() -> list[modal.Secret]:
    """Retrieve HF_TOKEN from environment or cached credentials."""
    import huggingface_hub

    token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    secrets = []
    if "huggingface-secret" in os.environ.get("MODAL_SECRETS", ""):
        secrets.append(modal.Secret.from_name("huggingface-secret"))
    elif token:
        secrets.append(modal.Secret.from_dict({"HF_TOKEN": token}))
    return secrets


@app.cls(
    image=image,
    gpu="L40S",  # Default GPU tier; can be configured per instance
    volumes={"/checkpoints": volume},
    secrets=_get_hf_secret(),
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    timeout=3600,
    scaledown_window=300,
    min_containers=0,
    max_containers=1,
)
class TrainablePolicyServer:
    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-5,
    ):
        self.base_model_name = base_model_name_or_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lr = learning_rate

    @modal.enter()
    def setup(self):
        import torch
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading base model {self.base_model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Configure LoRA adapter
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Setup AdamW optimizer for LoRA parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        self.current_version = "init_v0"
        self.step_count = 0

    @modal.method()
    def sample_decision(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate tokens while recording token IDs and exact behavior log probabilities."""
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        context_ids = inputs.input_ids[0].tolist()

        generated_ids: list[int] = []
        behavior_logprobs: list[float] = []

        curr_input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=curr_input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                token_id = next_token.item()
                # Compute log probability under the unscaled or temperature policy
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                token_logprob = log_probs[0, token_id].item()

                generated_ids.append(token_id)
                behavior_logprobs.append(token_logprob)

                if token_id == self.tokenizer.eos_token_id:
                    break

                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)],
                    dim=-1,
                )

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": generated_text,
            "context_token_ids": context_ids,
            "generated_token_ids": generated_ids,
            "behavior_logprobs": behavior_logprobs,
            "policy_version": self.current_version,
            "model_revision": getattr(self.model.config, "_name_or_path", self.base_model_name),
            "tokenizer_hash": str(hash(self.tokenizer.__class__.__name__)),
            "template_hash": "default",
        }

    @modal.method()
    def train_step(
        self,
        batch_decisions: list[dict[str, Any]],
        advantage: float,
        clip_epsilon: float = 0.2,
    ) -> dict[str, Any]:
        """Compute the clipped surrogate policy gradient and apply the LoRA update."""
        import torch

        if not batch_decisions:
            return {"updated": False, "reason": "empty_decisions", "loss": 0.0}

        self.model.train()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        total_loss_val = 0.0
        total_tokens = 0
        num_decisions = len(batch_decisions)

        for d in batch_decisions:
            context_ids = d["context_token_ids"]
            gen_ids = d["generated_token_ids"]
            old_logprobs = torch.tensor(d["behavior_logprobs"], device=self.device, dtype=torch.float32)

            full_ids = torch.tensor([context_ids + gen_ids], device=self.device)
            # Forward pass with autograd
            outputs = self.model(input_ids=full_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Shift logits to align with targets
            shift_logits = logits[len(context_ids) - 1 : len(context_ids) - 1 + len(gen_ids)]
            shift_log_probs = torch.log_softmax(shift_logits, dim=-1)

            target_tokens = torch.tensor(gen_ids, device=self.device).unsqueeze(1)
            new_logprobs = shift_log_probs.gather(dim=1, index=target_tokens).squeeze(1)

            # PPO Clipped Surrogate Loss
            log_ratio = new_logprobs - old_logprobs
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
            decision_loss = -torch.min(surr1, surr2).mean()

            # Immediate backward step per decision to release autograd graph memory
            scaled_loss = decision_loss / num_decisions
            scaled_loss.backward()

            total_loss_val += float(decision_loss.detach().item())
            total_tokens += len(gen_ids)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        self.optimizer.step()
        torch.cuda.empty_cache()
        self.step_count += 1
        self.current_version = f"step_{self.step_count}"

        return {
            "updated": True,
            "loss": total_loss_val / num_decisions,
            "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
            "step": self.step_count,
            "new_version": self.current_version,
            "token_count": total_tokens,
        }

    @modal.method()
    def save_checkpoint(
        self,
        tag: str | None = None,
        hf_repo_id: str | None = None,
        private: bool = True,
    ) -> dict[str, Any]:
        """Persist current LoRA weights to persistent volume and optionally push directly to Hugging Face."""
        import os
        from huggingface_hub import HfApi

        save_tag = tag or self.current_version
        target_dir = Path("/checkpoints") / save_tag
        target_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target_dir))
        self.tokenizer.save_pretrained(str(target_dir))
        volume.commit()

        hf_url = None
        if hf_repo_id:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            try:
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=hf_repo_id, repo_type="model", private=private, exist_ok=True)
                api.upload_folder(
                    folder_path=str(target_dir),
                    repo_id=hf_repo_id,
                    repo_type="model",
                )
                hf_url = f"https://huggingface.co/{hf_repo_id}"
                print(f"Successfully uploaded checkpoint {save_tag} to {hf_url}")
            except Exception as exc:
                print(f"Warning: Hugging Face upload failed ({exc}), but weights are safely stored at {target_dir}")
                hf_url = f"error: {str(exc)}"

        return {
            "checkpoint_path": str(target_dir),
            "version": save_tag,
            "hf_url": hf_url,
        }
