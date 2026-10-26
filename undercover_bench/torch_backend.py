"""Exact-policy sampling and LoRA updates shared by local tests and Modal.

No model download or GPU allocation occurs at import. Sampling and optimization
both use eval mode (dropout off) and the same full-softmax temperature.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from pathlib import Path

from .rlcr import AuthoredDecision, RLCRInputError


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_profile(key, manifest=None):
    path = Path(manifest) if manifest else Path(__file__).resolve().parents[1] / "configs/rlcr.models.json"
    profiles = json.loads(path.read_text())["models"]
    if key not in profiles:
        raise RLCRInputError(f"unknown training model {key}; choose one of {sorted(profiles)}")
    return dict(profiles[key])


def token_logprobs(logits, target_ids, temperature):
    """Gather log probabilities under the actual temperature sampling policy."""
    import torch
    if not math.isfinite(temperature) or temperature <= 0:
        raise RLCRInputError("training requires a finite, strictly positive sampling temperature")
    return torch.log_softmax(logits.float() / temperature, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)).squeeze(-1)


def normalize_token_ids(encoded):
    """Accept native tokenizer containers without changing or truncating tokens."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
        first = encoded[0]
        if hasattr(first, "ids"):
            encoded = first.ids
        elif isinstance(first, (list, tuple)):
            encoded = first
    if (not isinstance(encoded, (list, tuple)) or not encoded
            or any(type(token) is not int or token < 0 for token in encoded)):
        raise RLCRInputError("template must return exactly one nonempty integer token sequence")
    return list(encoded)


class TorchPolicyBackend:
    def __init__(self, model, tokenizer, *, model_revision, learning_rate=1e-5,
                 temperature=.7, max_new_tokens=8192, max_context_tokens=32768,
                 prompt_config=None, version=None, checkpointed_backward=False, optimizer_kind="adamw",
                 probability_mode="cached"):
        import torch
        self.model = model
        self.tokenizer = tokenizer
        self.model_revision = model_revision
        self.temperature = temperature
        if not math.isfinite(temperature) or temperature <= 0:
            raise RLCRInputError("sampling temperature must be positive")
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.prompt_config = prompt_config or {}
        if probability_mode not in ("cached", "exact_prefix"):
            raise RLCRInputError("unknown probability mode")
        self.probability_mode = probability_mode
        # Canonical tokenizer serialization, never Python's process-random hash().
        vocab = tokenizer.get_vocab()
        self.tokenizer_hash = sha256_json({"vocab": vocab, "special": tokenizer.special_tokens_map})
        template_identity = {"template": tokenizer.chat_template, "config": self.prompt_config}
        # Preserve the historical cached fingerprint for existing checkpoints.
        if probability_mode == "exact_prefix":
            template_identity["probability_mode"] = probability_mode
            template_identity["operator_path"] = "grad_full_prefix_checkpoint_nonreentrant_v1"
        self.template_hash = sha256_json(template_identity)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters:
            raise RLCRInputError("backend has no trainable parameters")
        self.checkpointed_backward = checkpointed_backward
        self.optimizer_kind = optimizer_kind
        if optimizer_kind == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit
            self.optimizer = AdamW8bit(self.parameters, lr=learning_rate)
        elif optimizer_kind == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate)
        else:
            raise RLCRInputError("unknown optimizer kind")
        if checkpointed_backward:
            # Checkpointed forwards must remain deterministic relative to sampling.
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
        self.version = version or "initial-" + uuid.uuid4().hex
        self.updates = 0
        self.model.eval()

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def identity(self):
        return {"policy_version": self.version, "model_revision": self.model_revision,
                "tokenizer_hash": self.tokenizer_hash, "template_hash": self.template_hash,
                "sampling_temperature": self.temperature}

    def _check_decision(self, decision):
        expected = self.identity()
        actual = decision.model_dump()
        if any(actual[key] != value for key, value in expected.items()):
            raise RLCRInputError("stale weights, model, tokenizer, template or sampling temperature")
        if len(decision.context_token_ids) + len(decision.generated_token_ids) > self.max_context_tokens:
            raise RLCRInputError("decision exceeds the frozen context budget")

    def sample(self, prompt, seed):
        import torch
        self.model.train(self.checkpointed_backward if self.probability_mode == "exact_prefix" else False)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, **self.prompt_config)
        ids = normalize_token_ids(ids)
        budget = min(self.max_new_tokens, self.max_context_tokens - len(ids))
        if budget <= 0:
            raise RLCRInputError("prompt exceeds context budget; refusing silent truncation")
        context = list(ids)
        current = torch.tensor([ids], device=self.device)
        generated, logprobs = [], []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        eos = self.tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        config_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        eos_ids.update([config_eos] if isinstance(config_eos, int) else config_eos or [])
        if self.probability_mode == "exact_prefix":
            for _ in range(budget):
                prefix = torch.tensor([context + generated], device=self.device)
                # Do not wrap sampling in no_grad: quantized modules can choose
                # different kernels depending on autograd state. This is the
                # very same grad-enabled forward used by authored_logprobs.
                logp_graph = self._exact_prefix_logprobs(prefix)
                logp = logp_graph.detach()
                nxt = torch.multinomial(logp.exp(), 1, generator=generator)
                token = nxt.item()
                generated.append(token)
                logprobs.append(logp[0, token].item())
                del logp_graph, logp
                if token in eos_ids:
                    break
        else:
            with torch.no_grad():
                past = None
                for _ in range(budget):
                    # Use the full prefix on the first call, then one token with its cache.
                    result = self.model(input_ids=current, past_key_values=past, use_cache=True)
                    logits = result.logits[:, -1, :].float() / self.temperature
                    logp = torch.log_softmax(logits, dim=-1)
                    nxt = torch.multinomial(logp.exp(), 1, generator=generator)
                    generated.append(nxt.item())
                    logprobs.append(logp[0, nxt.item()].item())
                    if nxt.item() in eos_ids:
                        break
                    past = result.past_key_values
                    current = nxt if past is not None else torch.tensor([context + generated], device=self.device)
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "context_token_ids": context, "generated_token_ids": generated,
                "behavior_logprobs": logprobs, "finish_reason": "stop" if generated[-1] in eos_ids else "length",
                **self.identity()}

    def _exact_prefix_logprobs(self, prefix):
        """Identical full-prefix operator path in sampling and optimization.

        Nonreentrant checkpointing discards intermediate activations while
        retaining autograd, including when integer inputs do not require grad.
        The full softmax is computed inside the checkpoint, never from a cached
        decode or a post-hoc teacher-forced sequence.
        """
        import torch
        from torch.utils.checkpoint import checkpoint

        def forward(input_ids):
            logits = self.model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
            return torch.log_softmax(logits.float() / self.temperature, dim=-1)

        self.model.train(self.checkpointed_backward)
        with torch.enable_grad():
            return checkpoint(forward, prefix, use_reentrant=False, preserve_rng_state=True)

    def authored_logprobs(self, decision: AuthoredDecision):
        import torch
        self._check_decision(decision)
        # eval disables dropout but deliberately retains autograd.
        self.model.train(self.checkpointed_backward)
        if self.probability_mode == "exact_prefix":
            selected = []
            for index, target in enumerate(decision.generated_token_ids):
                prefix = torch.tensor([list(decision.context_token_ids) +
                                       list(decision.generated_token_ids[:index])], device=self.device)
                selected.append(self._exact_prefix_logprobs(prefix)[0, target])
            return torch.stack(selected)
        full = torch.tensor([decision.context_token_ids + decision.generated_token_ids], device=self.device)
        result = self.model(input_ids=full, use_cache=False)
        start = len(decision.context_token_ids) - 1
        logits = result.logits[0, start:start + len(decision.generated_token_ids)]
        targets = torch.tensor(decision.generated_token_ids, device=logits.device)
        return token_logprobs(logits, targets, self.temperature)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def trainable_grad_norm(self):
        import torch
        norms = [p.grad.detach().float().norm().cpu() for p in self.parameters if p.grad is not None]
        return float(torch.stack(norms).norm()) if norms else 0.0

    def step(self):
        import torch
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
        if not math.isfinite(float(norm)) or norm <= 0:
            raise RLCRInputError("no finite nonzero gradient")
        self.optimizer.step()
        self.updates += 1

    def commit_checkpoint(self, path):
        import torch
        from .checkpoint_hf import seal_checkpoint
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        parent = self.version
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        torch.save({"optimizer": self.optimizer.state_dict(), "updates": self.updates}, path / "optimizer.pt")
        metadata = {**self.identity(), "parent_version": parent, "updates": self.updates,
                    "prompt_config": self.prompt_config, "max_new_tokens": self.max_new_tokens,
                    "max_context_tokens": self.max_context_tokens,
                    "checkpointed_backward": self.checkpointed_backward, "optimizer_kind": self.optimizer_kind,
                    "probability_mode": self.probability_mode}
        self.version = seal_checkpoint(path, metadata)
        return self.version

    def restore_optimizer(self, path):
        """Call after loading the PEFT weights from this same sealed directory."""
        import torch
        from .checkpoint_hf import verify_checkpoint
        manifest = verify_checkpoint(path)
        meta = manifest["metadata"]
        if meta.get("probability_mode", "cached") != self.probability_mode:
            raise RLCRInputError("checkpoint probability mode differs")
        if (meta.get("optimizer_kind", "adamw") != self.optimizer_kind
                or meta.get("checkpointed_backward", False) != self.checkpointed_backward):
            raise RLCRInputError("checkpoint optimizer or activation-checkpoint mode differs")
        for field in ("model_revision", "tokenizer_hash", "template_hash", "sampling_temperature"):
            if meta[field] != self.identity()[field]:
                raise RLCRInputError(f"checkpoint {field} does not match backend")
        state = torch.load(Path(path) / "optimizer.pt", map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        self.version = manifest["version"]
