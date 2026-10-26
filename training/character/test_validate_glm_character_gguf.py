import tempfile
import unittest
from pathlib import Path
import numpy as np
from training.character import test_convert_glm_character_adapter as fixtures
from training.character.convert_glm_character_gguf import convert,load_gguf
from training.character.validate_glm_character_gguf import delta,errors,validate,RELATIVE_L2_TOLERANCE

SOURCE=Path('/tmp/undercover-glm-lora-build')


class DeltaTests(unittest.TestCase):
    def test_execution_without_dense_delta_matches_tiny_dense_reference(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(2,5)).astype('float32');a=rng.normal(size=(3,5)).astype('float32');b=rng.normal(size=(7,3)).astype('float32')
        np.testing.assert_allclose(delta(x,a,b),x@(b@a).T,atol=1e-5,rtol=1e-5)
    def test_zero_reference_and_fixed_tolerance(self):
        self.assertEqual(errors(np.zeros(3),np.zeros(3)),(0.0,0.0))
        self.assertTrue(np.isinf(errors(np.zeros(3),np.ones(3))[0]))
        self.assertEqual(RELATIVE_L2_TOLERANCE,0.005)


@unittest.skipUnless((SOURCE/'gguf-py').exists(),'pinned local llama.cpp required')
class ArtifactTests(unittest.TestCase):
    def make(self,root):
        helper=fixtures.ConverterTests();helper.setUp()
        src,t,cfg,index=helper.fixture(root);cfg['text_config']['num_hidden_layers']=1
        path=root/'adapter.gguf';convert(src,path,cfg,index,SOURCE,outtype='f16')
        return src,path,cfg
    def test_all_pairs_experts_proved_and_scope_is_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            src,path,cfg=self.make(Path(tmp));r=validate(src,path,SOURCE,cfg)
            self.assertTrue(r['passed']);self.assertEqual(r['projection_pairs'],4)
            self.assertEqual(r['expert_slices_tested'],10)
            self.assertFalse(r['compatibility_verified']);self.assertFalse(r['quantized_base_behavior_verified'])
            self.assertEqual(len(r['raw_weights_sha256']),64);self.assertEqual(len(r['gguf_sha256']),64)
            r2=validate(src,path,SOURCE,cfg)
            self.assertEqual(r['results'],r2['results'])
    def test_corrupted_expert_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src,path,cfg=self.make(Path(tmp));gguf=load_gguf(SOURCE)
            reader=gguf.GGUFReader(path,mode='r+')
            target=next(t for t in reader.tensors if t.name=='blk.0.ffn_gate_exps.weight.lora_b')
            target.data[2]+=1
            reader.data.flush()
            r=validate(src,path,SOURCE,cfg)
            self.assertFalse(r['passed']);self.assertGreater(r['max_relative_l2'],0.005)

if __name__=='__main__':unittest.main()
