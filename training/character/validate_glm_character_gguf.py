"""CPU raw-to-GGUF LoRA delta proof; never a full base-model serving proof.

Preregistered relative-L2 tolerance: 0.005. Tests every raw projection and every
expert slice using independent deterministic Gaussian inputs, without forming
B@A. numpy.einsum(optimize=False) avoids multithreaded BLAS and large temporaries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import resource
import sys
from pathlib import Path
import re
import numpy as np
from training.character.convert_glm_character_adapter import RawReader, ConversionError
from training.character.convert_glm_character_gguf import load_gguf
from training.character.inspect_adapter_compatibility import read_safetensors_header, target_inventory

RELATIVE_L2_TOLERANCE=0.005
SEED=2026102607


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def delta(x,a,b):
    """Independent row-vector execution, bounded to one expert at a time."""
    hidden=np.einsum('ni,ri->nr',x,np.asarray(a,dtype=np.float32),optimize=False)
    return np.einsum('nr,or->no',hidden,np.asarray(b,dtype=np.float32),optimize=False)


def errors(expected,observed):
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        return float('inf'),float('inf')
    diff=observed.astype(np.float64)-expected.astype(np.float64)
    absolute=float(np.max(np.abs(diff)))
    denominator=float(np.sqrt(np.sum(expected.astype(np.float64)**2)))
    relative=float(np.sqrt(np.sum(diff**2)))/denominator if denominator else (0.0 if absolute==0 else float('inf'))
    return relative,absolute


def mapped_name(source,mapper):
    name=source.removeprefix('base_model.model.').replace('model.language_model.','model.')
    if name=='model.lm_head':name='lm_head'
    match=re.search(r'\.experts\.(w[123])$',name)
    if match:name=name[:match.start()]+'.experts.'+{'w1':'gate_proj','w2':'down_proj','w3':'up_proj'}[match[1]]
    if name.endswith('kv_b_proj'):
        raise ConversionError('MLA split numerical proof needs an explicit independent transposed-input protocol; unsupported by this validator')
    mapped=mapper.get_name(name+'.weight',try_suffixes=('.weight',))
    if mapped is None:raise ConversionError('Unmapped raw projection '+source)
    return mapped


def validate(source,gguf_path,llama_source,hf_config,*,inputs_per_expert=1):
    if type(inputs_per_expert) is not int or not 1<=inputs_per_expert<=8:raise ConversionError('Require 1 to 8 inputs per expert')
    source=Path(source);gguf_path=Path(gguf_path)
    if gguf_path.suffix!='.gguf' or not gguf_path.is_file():raise ConversionError('Require completed .gguf artifact')
    gguf=load_gguf(llama_source)
    config=json.loads((source/'adapter_config.json').read_text())
    inventory=target_inventory(read_safetensors_header(source/'adapter_model.safetensors'),config,hf_config=hf_config)
    raw=RawReader(source/'adapter_model.safetensors')
    hp=hf_config.get('text_config',hf_config)
    mapper=gguf.get_tensor_name_map(gguf.MODEL_ARCH.GLM5_NEXT,hp['num_hidden_layers'])
    reader=gguf.GGUFReader(gguf_path)
    tensors={tensor.name:tensor for tensor in reader.tensors}
    alpha_field=reader.fields.get('adapter.lora.alpha')
    if alpha_field is None:raise ConversionError('Missing GGUF LoRA alpha')
    alpha=float(alpha_field.contents())
    if alpha!=config['lora_alpha']:raise ConversionError('GGUF/source alpha mismatch')
    expected_names=set();results=[];global_relative=global_absolute=0.0;expert_cases=0
    for row in inventory:
        src=row['source_target'];name=mapped_name(src,mapper)
        names=[name+'.lora_a',name+'.lora_b'];expected_names.update(names)
        if any(n not in tensors for n in names):raise ConversionError('Missing GGUF factors '+name)
        a=raw.tensor(src+'.lora_A.weight');b=raw.tensor(src+'.lora_B.weight')
        metadata=[tensors[n] for n in names]
        if any(t.data.dtype not in (np.dtype('float16'),np.dtype('float32')) for t in metadata):raise ConversionError('Unsupported GGUF factor dtype')
        ga,gb=[np.memmap(gguf_path,mode='r',dtype=t.data.dtype,offset=t.data_offset,shape=t.data.shape) for t in metadata]
        e=row['experts'];is_expert=a.ndim==3
        expected_shapes=([e,*a.shape[-2:]],[e,*b.shape[-2:]]) if is_expert else (list(a.shape),list(b.shape))
        if list(ga.shape)!=expected_shapes[0] or list(gb.shape)!=expected_shapes[1]:raise ConversionError('GGUF factor shape/orientation mismatch '+name)
        pair_relative=pair_absolute=0.0
        rng=np.random.default_rng(SEED ^ int(hashlib.sha256(src.encode()).hexdigest()[:16],16))
        for expert in range(e):
            aa=a[expert if a.shape[0]>1 else 0] if is_expert else a
            bb=b[expert if b.shape[0]>1 else 0] if is_expert else b
            ca=ga[expert] if is_expert else ga;cb=gb[expert] if is_expert else gb
            x=rng.normal(size=(inputs_per_expert,a.shape[-1])).astype(np.float32)
            # Alpha/r cancels in relative error, but include it for meaningful maxabs.
            scale=config['lora_alpha']/row['rank']
            relative,absolute=errors(delta(x,aa,bb)*scale,delta(x,ca,cb)*scale)
            pair_relative=max(pair_relative,relative);pair_absolute=max(pair_absolute,absolute)
            expert_cases+=1
        global_relative=max(global_relative,pair_relative);global_absolute=max(global_absolute,pair_absolute)
        # Own per-factor mappings and close them deterministically. Never fault
        # tensor payloads through GGUFReader's retained full-file mapping.
        del aa,bb,ca,cb
        for array in (a,b,ga,gb):array._mmap.close()
        del a,b,ga,gb,array
        results.append({'source':src,'target':name,'expert_slices':e,'max_relative_l2':pair_relative,'max_abs_error':pair_absolute,'passed':pair_relative<=RELATIVE_L2_TOLERANCE})
    if set(tensors)!=expected_names:raise ConversionError('Unexpected/dropped target inventory: '+str(sorted(set(tensors)^expected_names)))
    return {'schema_version':1,'proof_scope':'raw_factor_to_gguf_factor_delta_conversion_only',
            'passed':all(r['passed'] for r in results),'compatibility_verified':False,
            'base_model_numerical_compatibility_verified':False,'quantized_base_behavior_verified':False,
            'relative_l2_tolerance':RELATIVE_L2_TOLERANCE,'tolerance_chosen_before_validation':True,
            'seed':SEED,'inputs_per_expert':inputs_per_expert,'projection_pairs':len(results),'expert_slices_tested':expert_cases,
            'max_relative_l2':global_relative,'max_abs_error':global_absolute,
            'raw_weights_sha256':sha256(raw.path),'gguf_sha256':sha256(gguf_path),
            'raw_config_sha256':sha256(source/'adapter_config.json'),
            'validator_sha256':sha256(__file__),
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'working_memory_protocol':'single expert arithmetic; independently owned factor mmap closed after each projection; GGUFReader mapping metadata only',
            'protocol':'Gaussian x; einsum optimize=False; FP32 factor execution, FP64 norm reduction; every expert',
            'results':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ['source','gguf','llama-source','hf-config','output']:p.add_argument('--'+opt,type=Path,required=True)
    p.add_argument('--inputs-per-expert',type=int,default=1);a=p.parse_args()
    report=validate(a.source,a.gguf,a.llama_source,json.loads(a.hf_config.read_text()),inputs_per_expert=a.inputs_per_expert)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2,allow_nan=False))
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
