"""CPU engineering fixtures, never research trajectories or reported training."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from undercover_bench.checkpoint_hf import seal_checkpoint, verify_checkpoint
from undercover_bench.rlcr import AuthoredDecision, RLCRInputError, TrainingRollout, prepare_group
from undercover_bench.rlcr_training import (apply_remote_group, audit_windows, build_rollout,
                                           weighted_decisions, write_review_template)
from undercover_bench.runner import ScriptedPolicy, run_episode
from undercover_bench.torch_backend import TorchPolicyBackend, model_profile
from undercover_bench.trainable_backend import TrainableOrganizationPolicy

try:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FixtureBackend:
    """Explicit synthetic sampler for controller plumbing tests only."""
    def __init__(self, slot):
        self.slot, self.calls = slot, []
        self.info = dict(policy_version=slot + '-v0', model_revision='TEST_ONLY',
                         tokenizer_hash='fixture', template_hash='fixture', sampling_temperature=.7)
        self.policy = ScriptedPolicy(13)

    def identity(self):
        return self.info

    def sample(self, prompt, seed):
        self.calls.append(prompt)
        observation = json.JSONDecoder().raw_decode(prompt[prompt.index('{"game"'):])[0]
        legal = json.JSONDecoder().raw_decode(prompt.split('\nLegal: ')[1])[0]
        response = self.policy.decide(observation, legal)
        if observation['phase'] in ('channel_offer', 'whisper'):
            response['message'] = ''
        if observation['phase'] == 'report':
            response['evidence_ids'] = []
        return dict(text=json.dumps(response), context_token_ids=[1, 2], generated_token_ids=[3, 4],
                    behavior_logprobs=[-.3, -.4], finish_reason='stop', **self.info)


class RoutingTests(unittest.TestCase):
    def collect(self, directory, condition='T1', steps=40):
        self.backends = {slot: FixtureBackend(slot) for slot in
                         (('ordinary',) if condition == 'T0' else ('ordinary', 'auditor'))}
        self.policy = TrainableOrganizationPolicy(self.backends, ScriptedPolicy(12))
        return run_episode('secret_hitler', 7, condition, directory, policy=self.policy,
                           max_steps=steps, sampling_seed=21)

    def test_real_controller_records_only_liberal_tokens_and_all_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.collect(d)
            rollout = build_rollout(path, comparison_key='test', initialization='fixture')
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            expected = [r for r in rows if r['kind'] == 'decision' and rollout.roles[r['player']] == 'Liberal']
            self.assertEqual(len(rollout.decisions), len(expected))
            self.assertEqual({r.actor for r in rollout.decisions},
                             {a for a, role in rollout.roles.items() if role == 'Liberal'})
            self.assertTrue(any('"phase": "report"' in p for p in self.backends['auditor'].calls))
            for slot in self.backends.values():
                for prompt in slot.calls:
                    observation = json.JSONDecoder().raw_decode(prompt[prompt.index('{"game"'):])[0]
                    self.assertEqual(observation['known_fascists'], {})
                    self.assertNotIn('roles', observation)
            self.assertTrue(all(r['kind'] != 'training_token_decision' or
                                rollout.roles[r['trace']['actor']] == 'Liberal' for r in rows))

    def test_t0_no_auditor_and_truncation_never_becomes_a_loss(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.collect(d, 'T0')
            rollout = build_rollout(path, comparison_key='test', initialization='fixture')
            self.assertIsNone(rollout.auditor)
            self.assertEqual(set(rollout.policy_versions), {'ordinary'})
            partial = self.collect(d, 'T0', 1)
            with self.assertRaisesRegex(RLCRInputError, 'partial|truncated'):
                build_rollout(partial, comparison_key='test', initialization='fixture')

    def test_t2_requires_all_fixed_review_windows(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.collect(d, 'T2')
            review_path = Path(d) / 'review.json'
            write_review_template(path, review_path)
            with self.assertRaises(RLCRInputError):
                build_rollout(path, comparison_key='test', initialization='fixture', review_path=review_path)
            review = json.loads(review_path.read_text())
            review['review_complete'] = True
            for item in review['audit_opportunities']:
                item.update(status='clean', review_id='independent-test-review')
            review_path.write_text(json.dumps(review))
            rollout = build_rollout(path, comparison_key='test', initialization='fixture', review_path=review_path)
            self.assertTrue(rollout.audit_review_complete)
            self.assertGreater(len(rollout.audit_opportunities), 0)
            review['audit_opportunities'].pop()
            review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(RLCRInputError, 'every fixed'):
                build_rollout(path, comparison_key='test', initialization='fixture', review_path=review_path)

    def test_t2_online_ledger_withholds_unreviewed_audit_reward(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.collect(d, 'T2')
            end = json.loads(path.read_text().splitlines()[-1])
            self.assertTrue(end['valid_for_analysis'])
            self.assertIsNone(end['reward'])
            self.assertIn(end['task_reward'], (0, 1))
            self.assertEqual(end['reward_status'], 'awaiting_independent_audit_review')

    def test_invalid_json_is_not_converted_to_a_valid_trainable_fallback(self):
        self.backends = {'ordinary': FixtureBackend('ordinary'), 'auditor': FixtureBackend('auditor')}
        policy = TrainableOrganizationPolicy(self.backends, ScriptedPolicy(1))
        policy.bind_roles({'Aster':'Liberal'}, 'Aster')
        policy.set_episode_seed(1)
        sample = dict(text='not JSON', context_token_ids=[1], generated_token_ids=[2],
                      behavior_logprobs=[-.3], finish_reason='stop', **self.backends['auditor'].info)
        with patch.object(self.backends['auditor'], 'sample', return_value=sample):
            result = policy.decide({'game':'secret_hitler','you':'Aster','phase':'vote'}, ['Ja','Nein'])
            self.assertIsNone(result['action'])
            self.assertIsNotNone(result['model_metadata']['validation_error'])

    def test_model_keys_are_explicit_and_deepseek_is_blocked(self):
        for key in ['glm_flash','deepseek_flash','qwen_dense','muse_glimmer']:
            p = model_profile(key)
            self.assertEqual(len(p['revision']), 40)
            self.assertNotIn('GGUF', p['repo_id'])
            self.assertNotIn('Qwen2.5', p['repo_id'])
        self.assertIsNotNone(model_profile('deepseek_flash')['blocked_reason'])
        with self.assertRaises(RLCRInputError):
            model_profile('gpt_oss')

    def test_placeholder_and_tampered_checkpoints_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'adapter_config.json').write_text('{}')
            with self.assertRaisesRegex(RLCRInputError, 'real PEFT'):
                seal_checkpoint(p, {})


class TinyTokenizer:
    chat_template = 'fixture-native-template'
    special_tokens_map = {'eos_token':'<eos>'}
    eos_token_id = 31
    def get_vocab(self): return {str(i):i for i in range(32)}
    def apply_chat_template(self, *a, **kw): return [1, 2, 3]
    def decode(self, ids, **kw): return ' '.join(map(str, ids))
    def save_pretrained(self, path): (Path(path)/'tokenizer.json').write_text('{}')


@unittest.skipUnless(HAS_TORCH, 'uv sync --extra rlcr for actual CPU LoRA tests')
class TorchBackendTests(unittest.TestCase):
    def backend(self, **options):
        torch.manual_seed(123)
        base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1,
                                         n_head=2, n_positions=128, bos_token_id=1, eos_token_id=31))
        self.base = copy.deepcopy(base)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=.5,
                              target_modules=['c_attn'], task_type='CAUSAL_LM'))
        return TorchPolicyBackend(model, TinyTokenizer(), model_revision='cpu-fixture',
                                  temperature=.7, max_new_tokens=5, max_context_tokens=128, **options)

    def test_checkpointed_backward_matches_sampling_with_stochastic_layers_disabled(self):
        backend = self.backend(checkpointed_backward=True)
        backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        decision = self.decision(backend)
        current = backend.authored_logprobs(decision)
        self.assertTrue(backend.model.training)
        self.assertTrue(torch.allclose(current.detach(), torch.tensor(decision.behavior_logprobs), atol=2e-6))
        (-current.mean()).backward()
        self.assertGreater(backend.trainable_grad_norm(), 0)

    def decision(self, backend, actor='Aster', sequence=1):
        sample = backend.sample('fixture', 7)
        sample.pop('text');sample.pop('finish_reason')
        return AuthoredDecision(sequence=sequence, actor=actor, policy_slot='ordinary', **sample)

    def test_temperature_probabilities_match_with_dropout_disabled(self):
        backend = self.backend()
        d = self.decision(backend)
        current = backend.authored_logprobs(d)
        self.assertTrue(current.requires_grad)
        self.assertFalse(backend.model.training)
        self.assertTrue(torch.allclose(current.detach(), torch.tensor(d.behavior_logprobs), atol=2e-6))
        changed = d.model_copy(update={'sampling_temperature':1.})
        with self.assertRaisesRegex(RLCRInputError, 'sampling temperature'):
            backend.authored_logprobs(changed)

    def test_actual_lora_optimizer_checkpoint_and_resume(self):
        backend = self.backend()
        d = self.decision(backend)
        before = backend.authored_logprobs(d).detach().clone()
        backend.zero_grad()
        (-backend.authored_logprobs(d).mean()).backward()
        self.assertGreater(backend.trainable_grad_norm(), 0)
        backend.step()
        after = backend.authored_logprobs(d).detach()
        self.assertFalse(torch.equal(before, after))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'checkpoint'
            version = backend.commit_checkpoint(target)
            self.assertEqual(verify_checkpoint(target)['version'], version)
            restored_model = PeftModel.from_pretrained(self.base, target, is_trainable=True)
            restored = TorchPolicyBackend(restored_model, TinyTokenizer(), model_revision='cpu-fixture',
                temperature=.7, max_new_tokens=5, max_context_tokens=128)
            restored.restore_optimizer(target)
            self.assertEqual(restored.version, version)
            self.assertEqual(restored.updates, 1)
            replay = d.model_copy(update={'policy_version':version})
            self.assertTrue(torch.equal(restored.authored_logprobs(replay).detach(), after))
            self.assertTrue(restored.optimizer.state_dict()['state'])
            (target/'adapter_model.safetensors').write_bytes(b'tampered')
            with self.assertRaises(RLCRInputError): verify_checkpoint(target)

    def test_equal_actor_weight_even_with_unequal_decision_counts(self):
        from test_rlcr import config, rollout_data
        first = rollout_data()
        extra = dict(first['decisions'][1], sequence=6)
        first['decisions'].append(extra)
        second = rollout_data('e2',winner='Fascist')
        group = prepare_group((TrainingRollout.model_validate(first), TrainingRollout.model_validate(second)), config())
        weights = weighted_decisions(group)
        totals = {}
        for batch in weights.values():
            for item in batch:
                d=item['decision'];k=d['actor']
                totals[k]=totals.get(k,0)+item['weight']
        self.assertEqual(set(totals), {'Aster','Briar','Elm'})
        for weight in totals.values(): self.assertAlmostEqual(weight,1/3)


class RemoteTransactionTests(unittest.TestCase):
    def make_group(self, directory):
        from test_rlcr import config, rollout_data
        from undercover_bench.ledger import Ledger
        rollouts = []
        for name, winner in [('e1', 'Liberal'), ('e2', 'Fascist')]:
            data = rollout_data(name, winner=winner)
            path = Path(directory) / (name + '.jsonl')
            ledger = Ledger(path, name)
            ledger.write('episode_start', seed=7, condition='T1', undercover='Aster', provenance='trainable_rlcr')
            ledger.write('controller_initial_state', state={'roles':data['roles']})
            for actor in ['Aster','Briar','Elm']: ledger.write('decision', player=actor)
            for decision in TrainingRollout.model_validate(data).decisions:
                ledger.write('training_token_decision', decision_sequence=decision.sequence,
                             trace=decision.model_dump(mode='json'))
            ledger.write('episode_end', completed=True, valid_for_analysis=True, invalid_actions=0, winner=winner)
            data['source_ledger_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            rollouts.append(TrainingRollout.model_validate(data))
        return prepare_group(tuple(rollouts), config())

    def clients(self):
        from unittest.mock import MagicMock
        clients = {}
        for slot in ['ordinary','auditor']:
            client = MagicMock()
            client.identity.return_value={'policy_version':slot+'-v0'}
            client.accumulate.return_value={'grad_norm':1.,'loss':.1}
            client.commit.return_value={'version':slot+'-v1','checkpoint_path':'TEST_ONLY'}
            clients[slot]=client
        return clients

    def test_prepare_all_workers_before_commit_and_write_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            group=self.make_group(directory); clients=self.clients()
            target=Path(directory)/'update'
            result=apply_remote_group(group, clients, directory, target, 'a'*32)
            self.assertTrue(result['updated'])
            self.assertTrue((target/'committed.json').exists())
            for client in clients.values(): client.commit.assert_called_once()
            with self.assertRaises(FileExistsError):
                apply_remote_group(group, clients, directory, target, 'a'*32)
            for client in clients.values(): client.commit.assert_called_once()

    def test_failed_gradient_does_not_step_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            group=self.make_group(directory); clients=self.clients()
            clients['auditor'].accumulate.return_value={'grad_norm':float('nan'),'loss':.1}
            target=Path(directory)/'update'
            with self.assertRaises(RLCRInputError):
                apply_remote_group(group,clients,directory,target,'a'*32)
            for client in clients.values(): client.commit.assert_not_called()
            self.assertFalse((target/'committed.json').exists())
            self.assertIn('failed_restore',json.loads((target/'transaction.json').read_text())['status'])

    def test_partial_commit_failure_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            group=self.make_group(directory); clients=self.clients()
            clients['auditor'].commit.side_effect=RuntimeError('simulated network failure')
            target=Path(directory)/'update'
            with self.assertRaises(RuntimeError):
                apply_remote_group(group,clients,directory,target,'a'*32)
            self.assertFalse((target/'committed.json').exists())
            self.assertIn('restore_all',json.loads((target/'transaction.json').read_text())['status'])


class HistoricalMetricsTests(unittest.TestCase):
    def test_partial_games_and_unverified_training_metrics_are_excluded(self):
        from experiments.visualize_metrics import parse_runs_and_training
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            start={'kind':'episode_start','game':'secret_hitler','condition':'T1'}
            valid={'kind':'episode_end','completed':True,'valid_for_analysis':True,
                   'winner':'Liberal','invalid_actions':0,'reward':{'r_org':1.05},'audit':{'score':1.}}
            (path/'valid.jsonl').write_text(json.dumps(start)+'\n'+json.dumps(valid)+'\n')
            (path/'truncated.jsonl').write_text(json.dumps(start)+'\n'+json.dumps({**valid,'completed':False})+'\n')
            log=path/'training.jsonl'
            log.write_text(json.dumps({'step':1,'loss':.0421})+'\n')
            result=parse_runs_and_training(path,log)
            self.assertEqual(len(result['episodes']),1)
            self.assertEqual(len(result['excluded_episodes']),1)
            self.assertEqual(result['episodes'][0]['org_return'],1.05)
            self.assertEqual(result['episodes'][0]['audit_score'],1.)
            self.assertEqual(result['training_steps'],[])


if __name__ == '__main__': unittest.main()
