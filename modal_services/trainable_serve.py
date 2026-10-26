"""Explicit per-family Modal LoRA worker, replacing PR #1's Qwen-7B service.

No deploy/run is performed by this file's import. It extends the existing
undercover-<model> app without replacing its generate/stage functions. Select RLCR_MODEL_KEY before
Modal deploy. DeepSeek is blocked pending its distributed FP8/FP4 training loader.
"""
import json
import math
import os
import re
from pathlib import Path

import modal

from undercover_bench.torch_backend import model_profile

KEY = os.environ.get("RLCR_MODEL_KEY")
if not KEY:
    raise ValueError("Set RLCR_MODEL_KEY to glm_flash, deepseek_flash, qwen_dense or muse_glimmer")
PROFILE = model_profile(KEY)
EXPERIMENTAL_GLM = KEY == "glm_flash" and os.environ.get("GLM_EXPERIMENTAL_MOE") == "1"
if PROFILE["blocked_reason"] and not EXPERIMENTAL_GLM:
    raise ValueError(PROFILE["blocked_reason"])
# Extend the existing deployment, preserving its GGUF generate/stage functions.
# A trainable worker still needs safetensors; it cannot backpropagate into GGUF.
if os.environ.get("MODEL_KEY", KEY) != KEY:
    raise ValueError("MODEL_KEY and RLCR_MODEL_KEY must select the same existing app")
os.environ["MODEL_KEY"] = KEY
from modal_services.serve import app, generate, stage, volume as weights_volume  # noqa: E402,F401
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2",
                      "accelerate==1.15.0", "bitsandbytes==0.50.2", "pydantic>=2,<3")
         .env({"RLCR_MODEL_KEY": KEY, "MODEL_KEY": KEY, "HF_HOME": "/weights/trainable-hf-cache"})
         .add_local_python_source("undercover_bench", "modal_services")
         .add_local_dir(Path(__file__).resolve().parents[1] / "configs", "/root/configs"))
if EXPERIMENTAL_GLM:
    from undercover_bench.glm_moe_backend import AXOLOTL_REVISION
    image = (image.apt_install("git")
        .run_commands("git clone https://github.com/axolotl-ai-cloud/axolotl.git /opt/axolotl",
                      f"cd /opt/axolotl && git checkout {AXOLOTL_REVISION}",
                      "pip install --no-deps /opt/axolotl")
        .env({"GLM_EXPERIMENTAL_MOE": "1", "HF_DEACTIVATE_ASYNC_LOAD": "1"}))
volume = modal.Volume.from_name("undercover-rlcr-checkpoints", create_if_missing=True)


@app.function(image=image, volumes={"/weights": weights_volume},
              cpu=4, memory=8192, timeout=3600, retries=0)
def stage_training_weights():
    """Stage differentiable weights on CPU; preserve the existing GGUF files."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(PROFILE["repo_id"], revision=PROFILE["revision"],
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt"], max_workers=8)
    weights_volume.commit()
    return {"model": KEY, "revision": PROFILE["revision"], "cache_path": path,
            "gpu_allocated": False}


@app.cls(image=image, gpu=PROFILE["gpu"],
         volumes={"/checkpoints": volume, "/weights": weights_volume}, cpu=8, memory=65536,
         secrets=([modal.Secret.from_name(os.environ["RLCR_HF_SECRET"])] if os.environ.get("RLCR_HF_SECRET") else []),
         min_containers=0, max_containers=1, scaledown_window=60, timeout=3600)
class TrainablePolicyServer:
    run_id: str = modal.parameter()
    policy_slot: str = modal.parameter()
    initial_adapter: str = modal.parameter(default="")
    checkpoint_path: str = modal.parameter(default="")
    checkpoint_version: str = modal.parameter(default="")

    @modal.enter()
    def setup(self):
        import torch
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
        from undercover_bench.torch_backend import TorchPolicyBackend
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", self.run_id) or self.policy_slot not in ("ordinary", "auditor"):
            raise ValueError("invalid run ID or policy slot")
        self.root = Path("/checkpoints") / self.run_id / KEY / self.policy_slot
        self.root.mkdir(parents=True, exist_ok=True)
        session_config = {"model": PROFILE, "slot": self.policy_slot, "initial_adapter": self.initial_adapter}
        session_file = self.root / "session.json"
        if session_file.exists() and json.loads(session_file.read_text()) != session_config:
            raise ValueError("run ID already belongs to a different model/character initialization")
        session_file.write_text(json.dumps(session_config, indent=2))
        torch.manual_seed(123456)
        config = AutoConfig.from_pretrained(PROFILE["repo_id"], revision=PROFILE["revision"])
        if config.model_type != PROFILE["model_type"] or PROFILE["architecture"] not in config.architectures:
            raise ValueError("model architecture differs from pinned profile")
        loader = getattr(transformers, PROFILE["architecture"], None)
        if loader is None:
            raise ValueError(f"Installed Transformers lacks {PROFILE['architecture']}; no model substitution")
        tokenizer = AutoTokenizer.from_pretrained(PROFILE["repo_id"], revision=PROFILE["revision"])
        if not tokenizer.chat_template:
            raise ValueError("native text chat template unavailable")
        if EXPERIMENTAL_GLM:
            from undercover_bench.glm_moe_backend import enable_expert_quantization
            enable_expert_quantization()
        model = loader.from_pretrained(PROFILE["repo_id"], revision=PROFILE["revision"], dtype=torch.bfloat16,
            device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            **({"attn_implementation": "sdpa"} if EXPERIMENTAL_GLM else {}))
        if EXPERIMENTAL_GLM:
            from undercover_bench.glm_moe_backend import prepare_glm_for_adapters
            self.expert_quantization = prepare_glm_for_adapters(model)
        else:
            model = prepare_model_for_kbit_training(model)
        initial = self.root / "initial"
        selected = Path(self.checkpoint_path).resolve() if self.checkpoint_path else initial
        if self.checkpoint_path:
            from undercover_bench.checkpoint_hf import verify_checkpoint
            if not selected.is_relative_to(self.root.resolve()):
                raise ValueError("resume checkpoint belongs to a different run/model/slot")
            if verify_checkpoint(selected)["version"] != self.checkpoint_version:
                raise ValueError("pinned checkpoint version mismatch")
        elif self.checkpoint_version:
            raise ValueError("checkpoint version requires a checkpoint path")
        if selected.exists():
            model = PeftModel.from_pretrained(model, str(selected), is_trainable=True)
        elif self.initial_adapter:
            from undercover_bench.checkpoint_hf import verify_checkpoint
            # Character export requires an explicit reviewed base-compatibility manifest.
            adapter = Path(self.initial_adapter)
            if EXPERIMENTAL_GLM:
                conversion = json.loads((adapter / "conversion_report.json").read_text())
                config = json.loads((adapter / "adapter_config.json").read_text())
                if (conversion.get("source_pairs_preserved") != 431
                        or config.get("base_model_name_or_path") != PROFILE["repo_id"]
                        or config.get("revision") != PROFILE["revision"]):
                    raise ValueError("experimental GLM adapter lacks a complete pinned native conversion")
            else:
                compatibility = json.loads((adapter / "compatibility.json").read_text())
                if compatibility.get("model_revision") != f"{PROFILE['repo_id']}@{PROFILE['revision']}" or compatibility.get("verified") is not True:
                    raise ValueError("character adapter/base compatibility is not verified")
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        else:
            if EXPERIMENTAL_GLM:
                raise ValueError("experimental GLM preflight requires an explicitly converted full character adapter")
            model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.,
                target_modules="all-linear", task_type="CAUSAL_LM"))
        self.backend = TorchPolicyBackend(model, tokenizer,
            model_revision=f"{PROFILE['repo_id']}@{PROFILE['revision']}",
            checkpointed_backward=EXPERIMENTAL_GLM,
            optimizer_kind="adamw8bit" if EXPERIMENTAL_GLM else "adamw")
        if selected.exists():
            self.backend.restore_optimizer(selected)
        else:
            self.backend.commit_checkpoint(initial)
            volume.commit()
        self.starting_checkpoint = selected.resolve()
        from undercover_bench.rlcr_worker import DurablePolicyWorker
        self.worker = DurablePolicyWorker(self.backend, self.root, volume.commit)
        self.preflight_ok = False

    @modal.method()
    def identity(self):
        return self.backend.identity()

    @modal.method()
    def sample(self, prompt, seed):
        return self.backend.sample(prompt, seed)

    @modal.method()
    def restore(self, path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from undercover_bench.checkpoint_hf import verify_checkpoint
        target = Path(path).resolve()
        if target != self.starting_checkpoint:
            raise ValueError("select a different checkpoint with durable class parameters, not an ephemeral restore")
        verify_checkpoint(target)
        restored = set_peft_model_state_dict(self.backend.model, load_file(str(target / "adapter_model.safetensors")))
        if restored.unexpected_keys:
            raise ValueError("checkpoint adapter keys do not match the loaded model")
        self.backend.restore_optimizer(target)
        self.backend.zero_grad()
        from undercover_bench.rlcr_worker import DurablePolicyWorker
        self.worker = DurablePolicyWorker(self.backend, self.root, volume.commit)
        self.preflight_ok = False
        return self.backend.identity()

    def _preflight(self):
        from undercover_bench.rlcr import AuthoredDecision
        self.backend.zero_grad()
        budget = self.backend.max_new_tokens
        try:
            self.backend.max_new_tokens = 64
            sample = self.backend.sample('Return only JSON: {"action":"Ja","rationale":"test"}', 19)
        finally:
            self.backend.max_new_tokens = budget
        trace = {k: sample[k] for k in ("context_token_ids", "generated_token_ids", "behavior_logprobs",
                 "policy_version", "model_revision", "tokenizer_hash", "template_hash", "sampling_temperature")}
        d = AuthoredDecision(sequence=1, actor="preflight", policy_slot=self.policy_slot, **trace)
        current = self.backend.authored_logprobs(d)
        import torch
        drift = float((current.detach().cpu() - torch.tensor(d.behavior_logprobs)).abs().max())
        (-current.mean()).backward()
        norm = self.backend.trainable_grad_norm()
        self.backend.zero_grad()
        if not math.isfinite(drift) or drift > .02 or not math.isfinite(norm) or norm <= 0:
            raise ValueError("sampled/recomputed probability or gradient preflight failed")
        self.preflight_ok = True
        proof = {"identity": self.backend.identity(), "max_logprob_difference": drift, "grad_norm": norm,
                 "optimizer_steps_applied": 0, "experimental_backend": EXPERIMENTAL_GLM,
                 "character_compatibility_verified": False if EXPERIMENTAL_GLM else None}
        (self.root / "preflight.json").write_text(json.dumps(proof, indent=2))
        volume.commit()
        return proof

    @modal.method()
    def preflight(self):
        return self._preflight()

    @modal.method()
    def accumulate(self, weighted, config, transaction):
        if not self.preflight_ok:
            self._preflight()
        return self.worker.accumulate(weighted, config, transaction, self.policy_slot)

    @modal.method()
    def commit(self, transaction):
        return self.worker.commit(transaction)
