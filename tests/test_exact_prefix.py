"""Real CPU LoRA checks for identical authored-token probability paths."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from undercover_bench.rlcr import AuthoredDecision, RLCRInputError
from undercover_bench.torch_backend import TorchPolicyBackend, sha256_json
from undercover_bench.checkpoint_hf import verify_checkpoint

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PrefixTokenizer:
    chat_template = 'exact-prefix-fixture'
    special_tokens_map = {'eos_token': '<eos>'}
    eos_token_id = 31

    def get_vocab(self): return {str(i): i for i in range(32)}
    def apply_chat_template(self, *args, **kwargs): return [1, 2, 3]
    def decode(self, ids, **kwargs): return ' '.join(map(str, ids))
    def save_pretrained(self, path): (Path(path) / 'tokenizer.json').write_text('{}')


@unittest.skipUnless(HAS_TORCH, 'requires the optional rlcr torch dependencies')
class ExactPrefixTests(unittest.TestCase):
    def backend(self, mode='exact_prefix', checkpointed=False, model=None):
        if model is None:
            torch.manual_seed(384)
            base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1,
                n_head=2, n_positions=64, bos_token_id=1, eos_token_id=31))
            model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=.5,
                target_modules=['c_attn'], task_type='CAUSAL_LM'))
        return TorchPolicyBackend(model, PrefixTokenizer(), model_revision='tiny-prefix-cpu',
            learning_rate=.01, temperature=.7, max_new_tokens=4, max_context_tokens=64,
            probability_mode=mode, checkpointed_backward=checkpointed)

    def decision(self, sample):
        return AuthoredDecision(sequence=1, actor='Aster', policy_slot='ordinary',
            **{k: v for k, v in sample.items() if k not in ('text', 'finish_reason')})

    def test_each_sample_and_recompute_uses_identical_grad_enabled_prefix(self):
        for checkpointed in (False, True):
            with self.subTest(checkpointed=checkpointed):
                backend = self.backend(checkpointed=checkpointed)
                if checkpointed:
                    backend.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={'use_reentrant': False})
                calls = []
                original = backend.model.forward
                def forward(*args, **kwargs):
                    calls.append((kwargs['input_ids'].tolist(), kwargs.get('use_cache'),
                                  backend.model.training, torch.is_grad_enabled()))
                    return original(*args, **kwargs)
                with patch.object(backend.model, 'forward', side_effect=forward):
                    # Even ambient no_grad must not switch the sampler's operator path.
                    with torch.no_grad():
                        sample = backend.sample('fixture', 421)
                    decision = self.decision(sample)
                    n = len(decision.generated_token_ids)
                    sampled_calls = list(calls)
                    self.assertEqual(len(sampled_calls), n)
                    current = backend.authored_logprobs(decision)
                    self.assertEqual(calls[n:], sampled_calls)
                    for index, (prefix, cache, training, grad) in enumerate(sampled_calls):
                        self.assertEqual(prefix, [list(decision.context_token_ids) +
                                                 list(decision.generated_token_ids[:index])])
                        self.assertFalse(cache)
                        self.assertEqual(training, checkpointed)
                        self.assertTrue(grad)
                    self.assertTrue(current.requires_grad)
                    torch.testing.assert_close(current.detach(),
                        torch.tensor(decision.behavior_logprobs), atol=0, rtol=0)
                    self.assertEqual(decision.author_loss_mask, (0, 0, 0) + (1,) * n)
                    before = [p.detach().clone() for p in backend.parameters]
                    backend.zero_grad()
                    (-current.mean()).backward()
                    self.assertGreater(backend.trainable_grad_norm(), 0)
                    backend.step()
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, backend.parameters)))

    def test_fingerprints_checkpoint_resume_and_legacy_cached_default(self):
        exact = self.backend()
        cached = self.backend(mode='cached')
        self.assertNotEqual(exact.template_hash, cached.template_hash)
        self.assertEqual(cached.template_hash, sha256_json({
            'template': cached.tokenizer.chat_template, 'config': {}}))
        with self.assertRaises(RLCRInputError):
            cached.authored_logprobs(self.decision(exact.sample('fixture', 3)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'exact'
            version = exact.commit_checkpoint(path)
            self.assertEqual(verify_checkpoint(path)['metadata']['probability_mode'], 'exact_prefix')
            resumed = self.backend(model=copy.deepcopy(exact.model))
            resumed.restore_optimizer(path)
            self.assertEqual(resumed.version, version)
            with self.assertRaisesRegex(RLCRInputError, 'probability mode'):
                cached.restore_optimizer(path)
            oldpath = Path(tmp) / 'cached'
            cached.commit_checkpoint(oldpath)
            legacy = verify_checkpoint(oldpath)
            legacy['metadata'].pop('probability_mode')
            # Old sealed checkpoints omitted the field; omitted means cached.
            with patch('undercover_bench.checkpoint_hf.verify_checkpoint', return_value=legacy):
                cached.restore_optimizer(oldpath)
                with self.assertRaisesRegex(RLCRInputError, 'probability mode'):
                    exact.restore_optimizer(oldpath)

    def test_cached_sampler_keeps_historical_no_grad_cache_path(self):
        backend = self.backend(mode='cached')
        calls = []
        original = backend.model.forward
        def forward(*args, **kwargs):
            calls.append((kwargs.get('use_cache'), torch.is_grad_enabled(), backend.model.training))
            return original(*args, **kwargs)
        with patch.object(backend.model, 'forward', side_effect=forward):
            backend.sample('fixture', 17)
        self.assertTrue(calls)
        self.assertTrue(all(item == (True, False, False) for item in calls))
        with self.assertRaisesRegex(RLCRInputError, 'probability mode'):
            self.backend(mode='posthoc')


if __name__ == '__main__':
    unittest.main()
