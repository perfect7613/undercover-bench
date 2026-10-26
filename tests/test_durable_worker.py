"""A serverless restart must neither lose credit nor repeat optimizer steps."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from undercover_bench.rlcr import RLCRConfig, RLCRInputError
from undercover_bench.rlcr_worker import DurablePolicyWorker

try:
    import torch
    from peft import PeftModel
    import test_trainable_backend as backend_fixtures
    from undercover_bench.torch_backend import TorchPolicyBackend
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False


@unittest.skipUnless(HAS_TORCH,'install the rlcr extra')
class DurableWorkerTests(unittest.TestCase):
    def fixture(self,root):
        factory=backend_fixtures.TorchBackendTests(methodName='test_temperature_probabilities_match_with_dropout_disabled')
        backend=factory.backend()
        parent=Path(root)/'initial';backend.commit_checkpoint(parent)
        decision=factory.decision(backend)
        weighted=[{'decision':decision.model_dump(mode='json'),'advantage':1.,'weight':1/3}]
        cfg=RLCRConfig(advantage_epsilon=1e-6,clip_epsilon=.2,max_abs_log_ratio=.02,min_group_size=2).model_dump()
        return factory,backend,parent,weighted,cfg

    def restarted(self,factory,parent):
        model=PeftModel.from_pretrained(copy.deepcopy(factory.base),parent,is_trainable=True)
        backend=TorchPolicyBackend(model,backend_fixtures.TinyTokenizer(),model_revision='cpu-fixture',temperature=.7,
                                   max_new_tokens=5,max_context_tokens=128)
        backend.restore_optimizer(parent)
        return backend

    def test_prepare_survives_restart_commit_survives_reply_loss(self):
        with tempfile.TemporaryDirectory() as root:
            factory,backend,parent,batch,cfg=self.fixture(root)
            transaction='a'*32
            worker=DurablePolicyWorker(backend,root)
            stats=worker.accumulate(batch,cfg,transaction,'ordinary')
            self.assertGreater(stats['grad_norm'],0)
            restarted=self.restarted(factory,parent)
            receipt=DurablePolicyWorker(restarted,root).commit(transaction)
            self.assertEqual(restarted.updates,1)
            self.assertNotEqual(receipt['version'],backend.version)
            # Simulate losing the RPC response and cold-starting at the pinned parent.
            replay=self.restarted(factory,parent)
            again=DurablePolicyWorker(replay,root).commit(transaction)
            self.assertEqual(again,receipt)
            self.assertEqual(replay.updates,0)
            checkpoint=json.loads((Path(receipt['checkpoint_path'])/'checkpoint.json').read_text())
            self.assertEqual(checkpoint['metadata']['updates'],1)

    def test_repeated_prepare_is_idempotent_but_different_batch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            factory,backend,parent,batch,cfg=self.fixture(root)
            worker=DurablePolicyWorker(backend,root);transaction='b'*32
            first=worker.accumulate(batch,cfg,transaction,'ordinary')
            fresh=DurablePolicyWorker(self.restarted(factory,parent),root)
            self.assertEqual(fresh.accumulate(batch,cfg,transaction,'ordinary'),first)
            changed=copy.deepcopy(batch);changed[0]['advantage']=-1
            with self.assertRaisesRegex(RLCRInputError,'conflicts'):
                fresh.accumulate(changed,cfg,transaction,'ordinary')

    def test_tampered_gradients_do_not_step(self):
        with tempfile.TemporaryDirectory() as root:
            factory,backend,parent,batch,cfg=self.fixture(root)
            worker=DurablePolicyWorker(backend,root);transaction='c'*32
            worker.accumulate(batch,cfg,transaction,'ordinary')
            (Path(root)/'transactions'/transaction/'gradients.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RLCRInputError,'changed'):
                worker.commit(transaction)
            self.assertEqual(backend.updates,0)

    def test_ambiguous_commit_never_repeats_a_step(self):
        with tempfile.TemporaryDirectory() as root:
            factory,backend,parent,batch,cfg=self.fixture(root)
            worker=DurablePolicyWorker(backend,root);transaction='d'*32
            worker.accumulate(batch,cfg,transaction,'ordinary')
            (Path(root)/'transactions'/transaction/'commit_intent.json').write_text('{}')
            with self.assertRaisesRegex(RLCRInputError,'ambiguous'):
                worker.commit(transaction)
            self.assertEqual(backend.updates,0)


if __name__=='__main__':unittest.main()
