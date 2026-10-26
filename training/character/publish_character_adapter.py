"""Validate and privately publish an already prepared character export bundle.

The bundle must contain real decrypted Tinker adapter tensors plus provenance,
training/evaluation records and a model card. Does not train or allocate GPUs.
Authentication uses standard HF_TOKEN or cached Hugging Face credentials.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from huggingface_hub import HfApi
from safetensors import safe_open


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def prepare_manifest(directory):
    directory = Path(directory)
    if any(directory.rglob('*.age')):
        raise ValueError('export/decryption is incomplete')
    required = ['README.md', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
                'training/started.json', 'training/checkpoint.json', 'training/metrics.jsonl',
                'evaluation/summary.json', 'provenance/reviewed.comparisons.jsonl']
    if any(not (directory / name).is_file() for name in required):
        raise ValueError('export bundle is incomplete')
    config = json.loads((directory / 'adapter/adapter_config.json').read_text())
    run = json.loads((directory / 'training/started.json').read_text())
    evaluation = json.loads((directory / 'evaluation/summary.json').read_text())
    release = json.loads((directory / 'release_stage.json').read_text()) if (directory / 'release_stage.json').exists() else {'stage':'dpo'}
    sft = release.get('stage') == 'dpo_plus_introspection_sft'
    rank = json.loads((directory / 'training/runtime.json').read_text()).get('lora_rank') if sft else run['rank']
    if sft:
        metrics = [json.loads(line) for line in (directory / 'training/metrics.jsonl').read_text().splitlines()]
        if (not metrics or not all(row.get('step',0)>=1 and math.isfinite(row.get('train_mean_nll',float('nan'))) for row in metrics)
                or not (directory / 'provenance/reviewed_sft.jsonl').exists()):
            raise ValueError('SFT release needs actual completed optimizer metrics and admitted introspection data')
    if config['peft_type'] != 'LORA' or config['r'] != rank:
        raise ValueError('adapter type/rank differs from the recorded training run')
    with safe_open(directory / 'adapter/adapter_model.safetensors', framework='pt', device='cpu') as weights:
        names = list(weights.keys())
        if not names or not any('lora_A' in name for name in names) or not any('lora_B' in name for name in names):
            raise ValueError('export is missing actual LoRA tensors')
        shapes = {name: weights.get_slice(name).get_shape() for name in names}
    parameter_count = 0
    for shape in shapes.values():
        count = 1
        for dim in shape:
            count *= dim
        parameter_count += count
    files = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.name in ('artifact_manifest.json', 'upload_receipt.json'):
            continue
        if path.is_symlink():
            raise ValueError('export must not contain symlinks')
        if path.suffix in ('.json', '.jsonl', '.md', '.py', '.toml'):
            text = path.read_text()
            if re.search(r'(?:tml-[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{25,}|pt_live_[A-Za-z0-9]{25,})', text):
                raise ValueError(f'credential-like content in {path.relative_to(directory)}')
        files[str(path.relative_to(directory))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    manifest = {'schema': 1, 'purpose': 'character_training_dpo_plus_introspection_sft_candidate' if sft else 'character_training_dpo_candidate',
                'training_base': 'zai-org/GLM-5.3-Flash:peft:262144',
                'tensor_count': len(names), 'parameter_count': parameter_count,
                'adapter_rank': config['r'], 'adapter_alpha': config['lora_alpha'],
                'introspection_sft_completed': sft, 'rlcr_performed': False,
                'compatibility_verified': False, 'evaluation': evaluation, 'files': files}
    (directory / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def publish(directory, repo_id):
    manifest = prepare_manifest(directory)
    api = HfApi()
    account = api.whoami()['name']
    if repo_id.split('/')[0].lower() != account.lower():
        raise ValueError('destination must belong to the authenticated user')
    api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo_id, repo_type='model').private:
        raise ValueError('existing repository is public; refusing to upload private artifacts')
    commit = api.upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type='model',
        ignore_patterns=['upload_receipt.json'],
        commit_message='Publish character candidate with actual weights, stage provenance and unpromoted evaluation status')
    info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    uploaded = {s.rfilename: s for s in info.siblings}
    for name, item in manifest['files'].items():
        remote = uploaded.get(name)
        if remote is None or remote.size != item['bytes']:
            raise ValueError(f'uploaded file missing or wrong size: {name}')
        if remote.lfs is not None and remote.lfs.sha256 != item['sha256']:
            raise ValueError(f'uploaded weight digest differs: {name}')
    receipt = {'repo_id': repo_id, 'private': info.private, 'commit': commit.oid,
               'url': f'https://huggingface.co/{repo_id}', 'verified_files': len(manifest['files']),
               'tensor_count': manifest['tensor_count'], 'parameter_count': manifest['parameter_count']}
    (Path(directory) / 'upload_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()
    publish(args.directory, args.repo)
