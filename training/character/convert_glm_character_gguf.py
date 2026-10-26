"""Stream raw Tinker GLM LoRA into GGUF for patched llama.cpp.

F32 is exact; mixed preserves every B factor in F32 and casts A to F16.
Every precision-changing candidate requires separate numerical validation.

Only header metadata and one projection pair are held at once. No base weights,
network, GPU, or temporary copy of the whole output. All trained targets must map.
The pinned graph patch and full-model numerical checks remain separate gates.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import re
import subprocess
import shutil
import sys
from pathlib import Path
import numpy as np
from training.character.convert_glm_character_adapter import RawReader, ConversionError
from training.character.inspect_adapter_compatibility import read_safetensors_header, target_inventory, PINNED_LLAMA_REVISION


def load_gguf(source):
    source=Path(source).resolve()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision != PINNED_LLAMA_REVISION:
        raise ConversionError('Require pinned llama.cpp revision '+PINNED_LLAMA_REVISION)
    sys.path.insert(0,str(source/'gguf-py'))
    module=importlib.import_module('gguf')
    if not Path(module.__file__).resolve().is_relative_to(source/'gguf-py'):
        raise ConversionError('gguf module was imported from a different checkout')
    return module


def mla_split(a,b,heads,key_dim,value_dim):
    """Exact HF kv_b -> llama k_b(transposed),v_b factor transformation.

    Raw W=B@A; upstream views W[H,K+V,I], transposes only K.
    """
    if a.ndim!=2 or b.ndim!=2 or b.shape[0]!=heads*(key_dim+value_dim):
        raise ConversionError('Invalid MLA kv_b shape')
    by_head=b.reshape(heads,key_dim+value_dim,b.shape[-1])
    expanded_a=np.broadcast_to(a,(heads,*a.shape))
    key_b=by_head[:,:key_dim,:]
    value_b=by_head[:,key_dim:,:]
    return ((key_b.transpose(0,2,1),expanded_a.transpose(0,2,1)),(expanded_a,value_b))


def plan(source,hf_config,hf_index,gguf):
    source=Path(source)
    config=json.loads((source/'adapter_config.json').read_text())
    if any(config.get(k) for k in ['use_rslora','use_dora','rank_pattern','alpha_pattern']) or config.get('lora_dropout',0)!=0 or config.get('bias')!='none':
        raise ConversionError('Only global-rank standard zero-dropout bias-free LoRA supported')
    hp=hf_config.get('text_config',hf_config)
    if hp.get('tie_word_embeddings',hf_config.get('tie_word_embeddings',False)):
        raise ConversionError('Tied output head requires a separately verified mapping')
    tensors=read_safetensors_header(source/'adapter_model.safetensors')
    inventory=target_inventory(tensors,config,hf_index['weight_map'],hf_config)
    if any(row['missing_hf_targets'] for row in inventory):
        raise ConversionError('Public index lacks trained targets')
    mapper=gguf.get_tensor_name_map(gguf.MODEL_ARCH.GLM5_NEXT,hp['num_hidden_layers'])
    operations=[]
    for row in inventory:
        src=row['source_target'];name=src.removeprefix('base_model.model.')
        name=name.replace('model.language_model.','model.')
        if name=='model.lm_head':name='lm_head'
        match=re.search(r'\.experts\.(w[123])$',name)
        if match:
            name=name[:match.start()]+'.experts.'+{'w1':'gate_proj','w2':'down_proj','w3':'up_proj'}[match[1]]
        if name.endswith('kv_b_proj'):
            h,k,v=hp['num_attention_heads'],hp['qk_nope_head_dim'],hp['v_head_dim']
            if row['factor_B_shape'][-2]!=h*(k+v):raise ConversionError('MLA row count mismatch')
            r=row['rank'];i=row['factor_A_shape'][-1]
            variants=[('mla_k',name.replace('kv_b_proj','k_b_proj'),[[h,r,k],[h,i,r]]),
                      ('mla_v',name.replace('kv_b_proj','v_b_proj'),[[h,r,i],[h,v,r]])]
        elif len(row['factor_A_shape'])==3:
            e=row['experts'];variants=[('expert',name,[[e,*row['factor_A_shape'][-2:]],[e,*row['factor_B_shape'][-2:]]])]
        else:
            # Current KDA Q/K/V are plain linear weights. No rotary permutation
            # or convolution reshape occurs for these names in pinned glm.py.
            variants=[('dense',name,[row['factor_A_shape'],row['factor_B_shape']])]
        for kind,name,shapes in variants:
            mapped=mapper.get_name(name+'.weight',try_suffixes=('.weight',))
            if mapped is None:raise ConversionError('Unmapped trained target: '+name)
            operations.append(dict(source=src,kind=kind,target=mapped,shapes=shapes,experts=row['experts']))
    names=[op['target'] for op in operations]
    if len(set(names))!=len(names):raise ConversionError('Duplicate mapped GGUF projection')
    return config,hp,operations,len(inventory)


def factors(reader,op,hp):
    a=reader.tensor(op['source']+'.lora_A.weight');b=reader.tensor(op['source']+'.lora_B.weight')
    if op['kind']=='expert':
        e=op['experts'];return np.broadcast_to(a,(e,*a.shape[-2:])),np.broadcast_to(b,(e,*b.shape[-2:]))
    if op['kind'].startswith('mla_'):
        pair=mla_split(a,b,hp['num_attention_heads'],hp['qk_nope_head_dim'],hp['v_head_dim'])
        return pair[0 if op['kind']=='mla_k' else 1]
    return a,b


def factor_dtype(outtype,side):
    return np.dtype('float32' if outtype=='f32' or (outtype=='mixed' and side=='b') else 'float16')


def convert(source,output,hf_config,hf_index,llama_source,*,dry_run=False,outtype="f32",base_revision="61d609da1b5fef8f25c754534a0fba313aba1caf"):
    if outtype not in ("f32","f16","mixed"):raise ConversionError("Unsupported output dtype")
    if not re.fullmatch("[0-9a-f]{40}",base_revision):raise ConversionError("Immutable base revision required")
    gguf=load_gguf(llama_source)
    config,hp,operations,count=plan(source,hf_config,hf_index,gguf)
    report={'schema_version':1,'raw_projection_pairs':count,'mapped_projection_pairs':len(operations),
            'source_pairs_preserved':len({op['source'] for op in operations}),
            'payload_bytes':sum(math.prod(shape)*factor_dtype(outtype,side).itemsize for op in operations for side,shape in zip(('a','b'),op['shapes'])),
            'llama_revision':PINNED_LLAMA_REVISION,'base_weight_revision':base_revision,'dtype':outtype.upper(),
            'conversion_source_sha256':{str(p):hashlib.sha256((Path(llama_source)/p).read_bytes()).hexdigest() for p in ['conversion/glm.py','gguf-py/gguf/tensor_mapping.py','gguf-py/gguf/gguf_writer.py']},
            'precision_changed':outtype!='f32','factor_cast_max_abs_error':None,'factor_cast_errors':{},'all_factors_finite':False,
            'base_weight_revision_verified':False,
            'factor_dtypes':{side:factor_dtype(outtype,side).name for side in ('a','b')},
            'compatibility_verified':False,'graph_patch_required':True,
            'full_model_numerical_validation_required':True,'operations':operations}
    if dry_run:return report
    output=Path(output)
    if output.exists() or output.with_suffix(output.suffix+'.partial').exists():raise ConversionError('Output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free < report['payload_bytes'] + 1024**3:
        raise ConversionError('Insufficient free disk: require payload plus 1 GiB reserve')
    digest=hashlib.sha256()
    with (Path(source)/'adapter_model.safetensors').open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    report['source_weights_sha256']=digest.hexdigest()
    report['source_config_sha256']=hashlib.sha256((Path(source)/'adapter_config.json').read_bytes()).hexdigest()
    partial=output.with_suffix(output.suffix+'.partial')
    writer=gguf.GGUFWriter(partial,gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GLM5_NEXT])
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE,'lora')
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA,float(config['lora_alpha']))
    for op in operations:
        for side,shape in zip(('a','b'),op['shapes']):
            dtype=factor_dtype(outtype,side)
            writer.add_tensor_info(op['target']+'.lora_'+side,shape,dtype,math.prod(shape)*dtype.itemsize)
    reader=RawReader(Path(source)/'adapter_model.safetensors')
    try:
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file()
        for op in operations:
            arrays=factors(reader,op,hp)
            for side,array,shape in zip(('a','b'),arrays,op['shapes']):
                if list(array.shape)!=shape:raise ConversionError('Internal shape mismatch')
                if not np.isfinite(array).all():raise ConversionError('Non-finite source factor')
                dtype=factor_dtype(outtype,side)
                with np.errstate(over='ignore'):
                    cast=np.ascontiguousarray(array,dtype=dtype)
                if not np.isfinite(cast).all():raise ConversionError('Output dtype overflows source factor')
                error=float(np.max(np.abs(array-cast.astype(np.float32)))) if dtype==np.dtype('float16') else 0.0
                report['factor_cast_max_abs_error']=max(report['factor_cast_max_abs_error'] or 0.0,error)
                report['factor_cast_errors'][op['target']+'.lora_'+side]=error
                writer.write_tensor_data(cast)
                del cast
            del arrays
        writer.close()
    except BaseException:
        writer.close()
        raise
    report['all_factors_finite']=True
    partial.rename(output)
    (output.with_suffix(output.suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option in ['source','output','hf-config','hf-index','llama-source']:p.add_argument('--'+option,type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--outtype',choices=['f32','f16','mixed'],default='f32');p.add_argument('--base-revision',default='61d609da1b5fef8f25c754534a0fba313aba1caf');a=p.parse_args()
    r=convert(a.source,a.output,json.loads(a.hf_config.read_text()),json.loads(a.hf_index.read_text()),a.llama_source,dry_run=a.dry_run,outtype=a.outtype,base_revision=a.base_revision)
    print(json.dumps({k:v for k,v in r.items() if k not in ('operations','factor_cast_errors')},indent=2))

if __name__=='__main__':main()
