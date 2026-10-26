"""Durable per-policy prepare/commit across serverless container restarts.

Every worker is pinned to a sealed parent checkpoint. Gradients and intent are
persisted before returning; an ambiguous crash never silently repeats a step.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

from .checkpoint_hf import verify_checkpoint
from .rlcr import AuthoredDecision, RLCRConfig, RLCRInputError
from .rlcr_torch import decision_loss
from .torch_backend import sha256_json


def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


class DurablePolicyWorker:
    def __init__(self,backend,root,persist=lambda:None):
        self.backend,self.root,self.persist=backend,Path(root),persist
        self.parent_version=backend.version

    def _folder(self,transaction):
        if not re.fullmatch(r'[a-f0-9]{32}',transaction):
            raise RLCRInputError('invalid optimizer transaction')
        return self.root/'transactions'/transaction

    def accumulate(self,weighted,config,transaction,slot):
        import torch
        cfg=RLCRConfig.model_validate(config)
        folder=self._folder(transaction)
        fingerprint=sha256_json({'weighted':weighted,'config':cfg.model_dump(),'slot':slot})
        metadata_path=folder/'prepared.json'
        if metadata_path.exists():
            metadata=json.loads(metadata_path.read_text())
            if metadata['batch_sha256']!=fingerprint or metadata['parent_version']!=self.parent_version:
                raise RLCRInputError('transaction conflicts with an existing prepared batch')
            return metadata['statistics']
        if not weighted or self.backend.version!=self.parent_version:
            raise RLCRInputError('empty batch or worker no longer at its pinned parent')
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'commit_intent.json').exists():
            raise RLCRInputError('ambiguous prior commit; restore all parents under a new transaction')
        self.backend.zero_grad()
        total=0.
        for item in weighted:
            decision=AuthoredDecision.model_validate(item['decision'])
            weight,advantage=item['weight'],item['advantage']
            if (decision.policy_slot!=slot or not math.isfinite(weight) or not 0<weight<=1
                    or not math.isfinite(advantage)):
                raise RLCRInputError('invalid per-author policy credit')
            loss=decision_loss(self.backend.authored_logprobs(decision),decision,advantage,cfg)*weight
            loss.backward()
            total+=float(loss.detach().cpu())
        norm=self.backend.trainable_grad_norm()
        if not math.isfinite(norm) or norm<=0:
            raise RLCRInputError('no finite nonzero policy gradient')
        gradients={name:(parameter.grad.detach().cpu() if parameter.grad is not None else None)
                   for name,parameter in self.backend.model.named_parameters() if parameter.requires_grad}
        temporary=folder/'gradients.tmp'
        torch.save({'parent_version':self.parent_version,'gradients':gradients},temporary)
        os.replace(temporary,folder/'gradients.pt')
        statistics={'loss':total,'grad_norm':norm}
        metadata={'parent_version':self.parent_version,'batch_sha256':fingerprint,
                  'gradient_sha256':file_digest(folder/'gradients.pt'),'statistics':statistics}
        metadata_path.write_text(json.dumps(metadata,indent=2))
        self.persist()
        return statistics

    def commit(self,transaction):
        import torch
        folder=self._folder(transaction)
        destination=self.root/transaction
        receipt_path=folder/'committed.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            checkpoint=verify_checkpoint(destination)
            if checkpoint['version']!=receipt['version'] or checkpoint['metadata']['parent_version']!=self.parent_version:
                raise RLCRInputError('committed checkpoint differs from its pinned parent/receipt')
            return receipt
        intent=folder/'commit_intent.json'
        if intent.exists():
            raise RLCRInputError('ambiguous optimizer commit; restore all parent checkpoints before retry')
        metadata=json.loads((folder/'prepared.json').read_text())
        if self.backend.version!=self.parent_version or metadata['parent_version']!=self.parent_version:
            raise RLCRInputError('worker parent differs from prepared gradients')
        if file_digest(folder/'gradients.pt')!=metadata['gradient_sha256']:
            raise RLCRInputError('prepared gradient file changed')
        state=torch.load(folder/'gradients.pt',map_location='cpu',weights_only=True)
        parameters={name:p for name,p in self.backend.model.named_parameters() if p.requires_grad}
        if state['parent_version']!=self.parent_version or set(parameters)!=set(state['gradients']):
            raise RLCRInputError('prepared gradient parameter inventory mismatch')
        self.backend.zero_grad()
        for name,parameter in parameters.items():
            gradient=state['gradients'][name]
            if gradient is None:
                continue
            if gradient.shape!=parameter.shape or not torch.isfinite(gradient).all().item():
                raise RLCRInputError('invalid persisted gradient')
            parameter.grad=gradient.to(device=parameter.device,dtype=parameter.dtype)
        # A persistent intent precedes mutation. On crash, never assume the step was absent.
        intent.write_text(json.dumps({'parent_version':self.parent_version,'transaction':transaction}))
        self.persist()
        self.backend.step()
        version=self.backend.commit_checkpoint(destination)
        self.persist()
        receipt={'version':version,'checkpoint_path':str(destination)}
        receipt_path.write_text(json.dumps(receipt,indent=2))
        self.persist()
        return receipt
