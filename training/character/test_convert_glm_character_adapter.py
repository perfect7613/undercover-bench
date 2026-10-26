import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file, load_file
from training.character.convert_glm_character_adapter import convert, fused_gate_up, expert_down, ConversionError


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(7)
    def rand(self,*shape):return self.rng.normal(size=shape).astype(np.float32)
    def test_fused_algebra_and_expert_column_order(self):
        e,r,i,o=3,2,5,4
        a=self.rand(1,r,i);g=self.rand(e,o,r);u=self.rand(e,o,r)
        aa,bb=fused_gate_up(a,g,a.copy(),u)
        delta=np.einsum('ore,eri->eoi',bb.reshape(2*o,r,e),aa.reshape(e,r,i))
        np.testing.assert_allclose(delta,np.concatenate([g@a,u@a],axis=1),rtol=1e-6,atol=1e-6)
    def test_down_broadcast_and_order(self):
        e,r,i,o=3,2,4,5
        a=self.rand(e,r,i);b=self.rand(1,o,r)
        aa,bb=expert_down(a,b,e)
        actual=np.einsum('ore,eri->eoi',bb.reshape(o,r,e),aa.reshape(e,r,i))
        np.testing.assert_allclose(actual,b@a,rtol=1e-6,atol=1e-6)
    def test_refuses_inexact_fusion(self):
        a=self.rand(1,2,5);b=self.rand(3,4,2)
        with self.assertRaisesRegex(ConversionError,'rank64'):
            fused_gate_up(a,b,a+1,b)
    def fixture(self,root):
        src=root/'source';src.mkdir();t={}
        for name,a,b in [('w1',self.rand(1,2,5),self.rand(3,4,2)),('w2',self.rand(3,2,4),self.rand(1,5,2))]:
            prefix='base_model.model.model.layers.0.mlp.experts.'+name
            t[prefix+'.lora_A.weight']=a;t[prefix+'.lora_B.weight']=b
        t['base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight']=t['base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight'].copy()
        t['base_model.model.model.layers.0.mlp.experts.w3.lora_B.weight']=self.rand(3,4,2)
        t['base_model.model.model.lm_head.lora_A.weight']=self.rand(2,5)
        t['base_model.model.model.lm_head.lora_B.weight']=self.rand(5,2)
        save_file(t,str(src/'adapter_model.safetensors'))
        (src/'adapter_config.json').write_text(json.dumps(dict(r=2,lora_alpha=2,lora_dropout=0,bias='none',target_modules='all-linear',task_type=None,peft_type='LORA')))
        index={'weight_map':{f'model.language_model.layers.0.mlp.experts.{e}.{proj}.weight':'unused' for e in range(3) for proj in ['gate_proj','up_proj','down_proj']}}
        index['weight_map']['lm_head.weight']='unused'
        return src,t,{'text_config':{'num_local_experts':3}},index
    def test_streamed_header_and_all_pairs_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            report=convert(src,root/'out',cfg,index,'toy','a'*40)
            self.assertEqual(report['raw_projection_pairs'],4)
            self.assertEqual(report['source_pairs_preserved'],4)
            self.assertEqual(report['native_projection_pairs'],3)
            self.assertFalse(report['compatibility_verified'])
            self.assertEqual(len(load_file(str(root/'out/adapter_model.safetensors'))),6)
            with self.assertRaises(ConversionError):convert(src,root/'out',cfg,index,'toy','a'*40)
    def test_dry_run_and_unequal_real_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            convert(src,root/'out',cfg,index,'toy','a'*40,dry_run=True)
            self.assertFalse((root/'out').exists())
            key='base_model.model.model.layers.0.mlp.experts.w3.lora_A.weight'
            t[key]=t[key]+1
            save_file(t,str(src/'adapter_model.safetensors'))
            with self.assertRaisesRegex(ConversionError,'rank64'):
                convert(src,root/'out',cfg,index,'toy','a'*40)
            self.assertFalse((root/'out').exists())
    def test_missing_factor_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            del t['base_model.model.model.layers.0.mlp.experts.w2.lora_B.weight']
            save_file(t,str(src/'adapter_model.safetensors'))
            with self.assertRaisesRegex(ValueError,'missing A/B'):
                convert(src,root/'out',cfg,index,'toy','a'*40)
            self.assertFalse((root/'out').exists())
    @unittest.skipUnless(importlib.util.find_spec("peft"), "Run this integration check with uv run --extra rlcr")
    def test_cpu_peft_load_actual_wrapper_layout_and_deltas(self):
        import torch
        from torch import nn
        from peft import PeftModel
        class Experts(nn.Module):
            def __init__(self):
                super().__init__();self.gate_up_proj=nn.Parameter(torch.zeros(3,8,5));self.down_proj=nn.Parameter(torch.zeros(3,5,4))
            def forward(self,x):return x
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.model=nn.Module();self.model.language_model=nn.Module()
                layer=nn.Module();layer.mlp=nn.Module();layer.mlp.experts=Experts()
                self.model.language_model.layers=nn.ModuleList([layer]);self.lm_head=nn.Linear(5,5,bias=False)
            def forward(self,x):return self.lm_head(x)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            convert(src,root/'out',cfg,index,'toy','a'*40)
            model=PeftModel.from_pretrained(Model(),str(root/'out'))
            wrappers={x.parameter_name:x for x in model.modules() if hasattr(x,'parameter_name')}
            prefix='base_model.model.model.layers.0.mlp.experts.'
            expected_gate=np.concatenate([t[prefix+'w1.lora_B.weight']@t[prefix+'w1.lora_A.weight'],t[prefix+'w3.lora_B.weight']@t[prefix+'w3.lora_A.weight']],axis=1)
            np.testing.assert_allclose(wrappers['gate_up_proj'].get_delta_weight('default').detach().numpy(),expected_gate,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(wrappers['down_proj'].get_delta_weight('default').detach().numpy(),t[prefix+'w2.lora_B.weight']@t[prefix+'w2.lora_A.weight'],rtol=1e-6,atol=1e-6)
            self.assertEqual(set(wrappers),{'gate_up_proj','down_proj'})
            head=model.base_model.model.lm_head
            np.testing.assert_allclose(head.get_delta_weight('default').detach().numpy(),t['base_model.model.model.lm_head.lora_B.weight']@t['base_model.model.model.lm_head.lora_A.weight'],rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
