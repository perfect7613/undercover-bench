import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from validate_character_weights import validate


class WeightValidationTests(unittest.TestCase):
    def test_real_value_change_and_norm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=np.zeros((2,2),dtype=np.float32);b=a.copy();b[0,0]=.5
            save_file({'x.lora_B.weight':a},root/'parent.safetensors')
            save_file({'x.lora_B.weight':b},root/'current.safetensors')
            result=validate(root/'current.safetensors',root/'parent.safetensors',root/'report.json')
            self.assertEqual(result['changed_tensors'],1);self.assertEqual(result['adapter_delta_l2'],.5)
            self.assertFalse(result['compatibility_verified'])

    def test_unchanged_or_nonfinite_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=np.zeros((2,2),dtype=np.float32)
            save_file({'x.lora_B.weight':a},root/'parent.safetensors')
            save_file({'x.lora_B.weight':a},root/'current.safetensors')
            with self.assertRaises(ValueError):validate(root/'current.safetensors',root/'parent.safetensors',root/'r.json')
            a[0,0]=np.nan;save_file({'x.lora_B.weight':a},root/'current.safetensors')
            with self.assertRaises(ValueError):validate(root/'current.safetensors',root/'parent.safetensors',root/'r.json')


if __name__=='__main__':unittest.main()
