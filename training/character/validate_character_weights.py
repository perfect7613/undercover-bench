"""Stream one LoRA tensor at a time; validate finite SFT weights and actual change."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def file_digest(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(current,parent,output):
    current,parent=Path(current),Path(parent)
    differences=[]
    with safe_open(current,framework='numpy') as after,safe_open(parent,framework='numpy') as before:
        names=list(after.keys())
        if set(names)!=set(before.keys()):raise ValueError('DPO/SFT target inventories differ')
        changed=0;norm=0.
        for name in names:
            a=after.get_tensor(name);b=before.get_tensor(name)
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'nonfinite or mismatched adapter tensor: {name}')
            delta=np.subtract(a,b,dtype=np.float64)
            count=int(np.count_nonzero(delta));changed+=bool(count)
            square=float(np.einsum('i,i->',delta.ravel(),delta.ravel()));norm+=square
            differences.append({'name':name,'changed_values':count,'l2_difference':square**.5})
        if changed==0:raise ValueError('SFT artifact has no changed tensor values relative to DPO')
    result={'all_weights_finite':True,'tensor_count':len(names),'changed_tensors':changed,
        'adapter_delta_l2':norm**.5,'current_sha256':file_digest(current),'parent_sha256':file_digest(parent),
        'method':'one tensor at a time; no base model loaded','compatibility_verified':False,
        'behavioral_improvement_established':False,'per_tensor':differences}
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='per_tensor'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    validate(args.current,args.parent,args.output)
