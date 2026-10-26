import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from training.character.convert_glm_character_gguf import convert, mla_split, load_gguf
from training.character.convert_glm_character_adapter import ConversionError
from training.character import test_convert_glm_character_adapter as fixtures

SOURCE=Path('/tmp/undercover-glm-lora-build')


class GgufAlgebraTests(unittest.TestCase):
    def test_mla_key_transpose_and_value_slices(self):
        rng=np.random.default_rng(71)
        h,k,v,r,i=3,4,2,2,5
        a=rng.normal(size=(r,i)).astype(np.float32)
        b=rng.normal(size=(h*(k+v),r)).astype(np.float32)
        (ka,kb),(va,vb)=mla_split(a,b,h,k,v)
        expected=(b@a).reshape(h,k+v,i)
        np.testing.assert_allclose(kb@ka,expected[:,:k].transpose(0,2,1),atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(vb@va,expected[:,k:],atol=1e-6,rtol=1e-6)
    def test_mla_rejects_wrong_shape(self):
        with self.assertRaises(ConversionError):mla_split(np.zeros((2,3)),np.zeros((8,2)),2,3,2)


@unittest.skipUnless((SOURCE/'gguf-py').exists(),'pinned local llama.cpp checkout required')
class GgufRoundtripTests(unittest.TestCase):
    def fixture(self,root):
        helper=fixtures.ConverterTests();helper.setUp()
        src,t,cfg,index=helper.fixture(root);cfg['text_config']['num_hidden_layers']=1
        return src,t,cfg,index
    def test_exact_f32_roundtrip_all_targets_and_expert_axes(self):
        gguf=load_gguf(SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            report=convert(src,root/'adapter.gguf',cfg,index,SOURCE)
            self.assertEqual(report['source_pairs_preserved'],4)
            self.assertFalse(report['compatibility_verified']);self.assertTrue(report['all_factors_finite'])
            reader=gguf.GGUFReader(root/'adapter.gguf')
            tensors={x.name:x.data for x in reader.tensors}
            self.assertEqual(len(tensors),8)
            for raw,gg in [('w1','gate'),('w2','down'),('w3','up')]:
                prefix='base_model.model.model.layers.0.mlp.experts.'+raw
                a=tensors['blk.0.ffn_'+gg+'_exps.weight.lora_a'];b=tensors['blk.0.ffn_'+gg+'_exps.weight.lora_b']
                self.assertEqual(a.shape[0],3);self.assertEqual(b.shape[0],3)
                np.testing.assert_array_equal(b@a,t[prefix+'.lora_B.weight']@t[prefix+'.lora_A.weight'])
            np.testing.assert_array_equal(tensors['output.weight.lora_a'],t['base_model.model.model.lm_head.lora_A.weight'])
    def test_f16_records_precision_error_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            report=convert(src,root/'adapter.gguf',cfg,index,SOURCE,outtype='f16')
            self.assertTrue(report['precision_changed']);self.assertGreater(report['factor_cast_max_abs_error'],0)
            self.assertEqual(len(report['source_weights_sha256']),64)
            self.assertFalse(report['compatibility_verified'])
    def test_f16_overflow_never_publishes_output(self):
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            t['base_model.model.model.lm_head.lora_A.weight'][0,0]=1e10
            save_file(t,str(src/'adapter_model.safetensors'))
            with self.assertRaisesRegex(ConversionError,'overflows'):
                convert(src,root/'adapter.gguf',cfg,index,SOURCE,outtype='f16')
            self.assertFalse((root/'adapter.gguf').exists())
    def test_dry_run_no_file_and_unmapped_targets_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src,t,cfg,index=self.fixture(root)
            r=convert(src,root/'adapter.gguf',cfg,index,SOURCE,dry_run=True)
            self.assertFalse((root/'adapter.gguf').exists());self.assertEqual(r['source_pairs_preserved'],4)
            del index['weight_map']['lm_head.weight']
            with self.assertRaises(ConversionError):convert(src,root/'adapter.gguf',cfg,index,SOURCE,dry_run=True)

if __name__=='__main__':unittest.main()
