"""Bounded, exact Tinker GLM adapter conversion for native PEFT ParamWrapper.

CPU only. No base weights are downloaded. The output is a single streamed
safetensors file (PEFT's loader does not support sharded adapter indexes).
Requires identical gate/up A factors; refuses approximation otherwise. Loading
this file is not evidence of full-model numerical or serving compatibility.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import struct
from pathlib import Path
import numpy as np
from training.character.inspect_adapter_compatibility import read_safetensors_header, target_inventory, _hf_name


class ConversionError(ValueError):
    pass


class RawReader:
    def __init__(self, path):
        self.path = Path(path)
        read_safetensors_header(self.path)
        with self.path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            self.header = json.loads(stream.read(size))
        self.start = 8 + size

    def tensor(self, key):
        meta = self.header[key]
        if meta['dtype'] != 'F32':
            raise ConversionError('Exact converter currently requires F32 source factors; no implicit casts')
        return np.memmap(self.path, mode='r', dtype='<f4', offset=self.start + meta['data_offsets'][0], shape=tuple(meta['shape']))


def fused_gate_up(gate_a, gate_b, up_a, up_b):
    """Return exact native expert A[E*r,I], B[2*O,r*E] factors.

    PEFT B columns use rank-major, expert-minor order, not E-major order.
    """
    experts = max(gate_a.shape[0], gate_b.shape[0], up_a.shape[0], up_b.shape[0])
    if gate_a.shape[1:] != up_a.shape[1:] or not np.array_equal(gate_a, up_a):
        raise ConversionError('gate/up A factors differ: rank32 fusion would change the function; rank64 conversion required')
    if gate_b.shape[1:] != up_b.shape[1:]:
        raise ConversionError('gate/up B shapes differ')
    a = np.broadcast_to(gate_a, (experts, *gate_a.shape[1:]))
    b = np.concatenate([np.broadcast_to(gate_b, (experts, *gate_b.shape[1:])),
                        np.broadcast_to(up_b, (experts, *up_b.shape[1:]))], axis=1)
    return a.reshape(-1, a.shape[-1]), b.transpose(1, 2, 0).reshape(b.shape[1], -1)


def expert_down(a, b, experts):
    a = np.broadcast_to(a, (experts, *a.shape[1:]))
    b = np.broadcast_to(b, (experts, *b.shape[1:]))
    return a.reshape(-1, a.shape[-1]), b.transpose(1, 2, 0).reshape(b.shape[1], -1)


def conversion_plan(source, hf_config, hf_index, base_model, revision):
    source = Path(source)
    reader = RawReader(source / 'adapter_model.safetensors')
    config = json.loads((source / 'adapter_config.json').read_text())
    if config.get('use_rslora') or config.get('use_dora') or config.get('lora_dropout', 0) != 0 or config.get('bias') != 'none':
        raise ConversionError('Only standard zero-dropout bias-free LoRA is supported')
    inventory = target_inventory(read_safetensors_header(reader.path), config, hf_index['weight_map'], hf_config)
    if any(row['missing_hf_targets'] for row in inventory):
        raise ConversionError('Public HF index lacks one or more trained target weights')
    groups, operations, target_modules, target_parameters = {}, [], [], []
    prefix = any(k.startswith('model.language_model.') for k in hf_index['weight_map'])
    for row in inventory:
        src = row['source_target']; target = _hf_name(src, prefix, hf_index['weight_map'])
        match = re.search(r'\.experts\.(w[123])$', target)
        if match:
            groups.setdefault(target[:match.start()] + '.experts', {})[match[1]] = row
        else:
            target_modules.append(target)
            operations.append({'kind':'dense', 'sources':[src], 'target':target, 'output_prefix':'base_model.model.' + target,
                               'shapes':[row['factor_A_shape'],row['factor_B_shape']]})
    for target, rows in sorted(groups.items()):
        if set(rows) != {'w1','w2','w3'}:
            raise ConversionError(f'incomplete expert projection group: {target}')
        gate, down, up = [rows[k] for k in ('w1','w2','w3')]
        if len({r['experts'] for r in rows.values()}) != 1:
            raise ConversionError('expert counts differ within layer')
        ga = reader.tensor(gate['source_target'] + '.lora_A.weight')
        ua = reader.tensor(up['source_target'] + '.lora_A.weight')
        if ga.shape != ua.shape or not np.array_equal(ga, ua):
            raise ConversionError(f'{target}: unequal gate/up A requires rank64; refused without approximation')
        e, r = gate['experts'], gate['rank']
        # target_parameters are wrapped in order: first gate_up is inside down's base_layer.
        target_parameters.extend([target + '.gate_up_proj', target + '.down_proj'])
        operations.extend([
            {'kind':'gate_up','sources':[gate['source_target'],up['source_target']], 'target':target+'.gate_up_proj',
             'output_prefix':'base_model.model.'+target+'.base_layer', 'experts':e,
             'shapes':[[e*r,ga.shape[-1]],[gate['factor_B_shape'][-2]*2,e*r]]},
            {'kind':'down','sources':[down['source_target']], 'target':target+'.down_proj',
             'output_prefix':'base_model.model.'+target, 'experts':e,
             'shapes':[[e*r,down['factor_A_shape'][-1]],[down['factor_B_shape'][-2],e*r]]}])
    native_config = dict(config, base_model_name_or_path=base_model, revision=revision,
                         target_modules=target_modules, target_parameters=target_parameters,
                         peft_version='0.21.2', inference_mode=True)
    return reader, native_config, operations, len(inventory)


def values(reader, operation):
    def pair(src):
        return reader.tensor(src+'.lora_A.weight'), reader.tensor(src+'.lora_B.weight')
    if operation['kind'] == 'dense':
        return pair(operation['sources'][0])
    if operation['kind'] == 'down':
        return expert_down(*pair(operation['sources'][0]), operation['experts'])
    return fused_gate_up(*pair(operation['sources'][0]), *pair(operation['sources'][1]))


def convert(source, output, hf_config, hf_index, base_model, revision, *, dry_run=False):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ConversionError('Require immutable 40-hex base revision')
    reader, config, operations, count = conversion_plan(source,hf_config,hf_index,base_model,revision)
    header, offset = {}, 0
    for operation in operations:
        for side, shape in zip('AB', operation['shapes']):
            size = math.prod(shape)*4
            name = operation['output_prefix']+'.lora_'+side+'.weight'
            if name in header:
                raise ConversionError('duplicate output factor')
            header[name] = {'dtype':'F32','shape':shape,'data_offsets':[offset,offset+size]}
            offset += size
    report = {'schema_version':1,'raw_projection_pairs':count,'native_projection_pairs':len(operations),
              'output_payload_bytes':offset,'gate_up_shared_A_layers':sum(op['kind']=='gate_up' for op in operations),
              'source_pairs_preserved':sum(len(op['sources']) for op in operations),
              'compatibility_verified':False,'full_model_numerical_validation_required':True,
              'format':'PEFT 0.21.2 ParamWrapper, Transformers 5.20.2 GLM expert orientation',
              'output_tensors':header}
    if dry_run:
        return report
    output = Path(output)
    if output.exists():
        raise ConversionError('Output directory already exists; refusing overwrite')
    output.mkdir(parents=True)
    raw = json.dumps(header,separators=(',',':')).encode()
    raw += b' '*((-len(raw))%8)
    temporary = output/'adapter_model.safetensors.partial'
    # Each largest fused factor ~151MB for this artifact; never retain full adapter.
    with temporary.open('xb') as stream:
        stream.write(struct.pack('<Q',len(raw)));stream.write(raw)
        for operation in operations:
            arrays = values(reader,operation)
            for array, shape in zip(arrays,operation['shapes']):
                if list(array.shape)!=shape:
                    raise ConversionError('internal output shape mismatch')
                np.ascontiguousarray(array,dtype='<f4').tofile(stream)
            del arrays
    temporary.rename(output/'adapter_model.safetensors')
    (output/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'conversion_report.json').write_text(json.dumps(report,indent=2)+'\n')
    read_safetensors_header(output/'adapter_model.safetensors')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hf-config',type=Path,required=True);parser.add_argument('--hf-index',type=Path,required=True)
    parser.add_argument('--base-model',required=True);parser.add_argument('--revision',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    report=convert(args.source,args.output,json.loads(args.hf_config.read_text()),json.loads(args.hf_index.read_text()),args.base_model,args.revision,dry_run=args.dry_run)
    print(json.dumps({k:v for k,v in report.items() if k!='output_tensors'},indent=2))

if __name__=='__main__':
    main()
