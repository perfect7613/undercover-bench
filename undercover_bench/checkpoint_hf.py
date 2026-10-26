"""Seal real model + optimizer checkpoints; publish only when explicitly called.

Replaces PR #1's metadata-only placeholder checkpoints. No default destination,
model license claim, raw evidence upload, or implicit publishing.
"""
import hashlib
import json
from pathlib import Path

from .rlcr import RLCRInputError


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal_checkpoint(path, metadata):
    path = Path(path)
    if not (path / "adapter_model.safetensors").is_file() or not (path / "adapter_config.json").is_file():
        raise RLCRInputError("a checkpoint needs real PEFT adapter weights and configuration")
    if not (path / "optimizer.pt").is_file():
        raise RLCRInputError("a resumable checkpoint needs optimizer state")
    files = {str(p.relative_to(path)): _digest(p) for p in sorted(path.rglob("*"))
             if p.is_file() and p.name != "checkpoint.json"}
    body = {"schema": 1, "files": files, "metadata": metadata}
    version = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    (path / "checkpoint.json").write_text(json.dumps({**body, "version": version}, indent=2))
    return version


def verify_checkpoint(path):
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text())
    body = {key: manifest[key] for key in ("schema", "files", "metadata")}
    version = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if manifest["version"] != version:
        raise RLCRInputError("checkpoint manifest digest mismatch")
    required = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt"}
    if not required.issubset(manifest["files"]):
        raise RLCRInputError("incomplete adapter/optimizer checkpoint")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or _digest(target) != digest:
            raise RLCRInputError(f"missing or changed checkpoint file: {name}")
    return manifest


def upload_checkpoint_to_huggingface(checkpoint_dir, repo_id, hf_token=None, private=True):
    from huggingface_hub import HfApi
    verify_checkpoint(checkpoint_dir)
    if not private:
        raise RLCRInputError("project checkpoints must stay private")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo_id, repo_type="model").private:
        raise RLCRInputError("existing destination repository is public")
    # Include the version directory; do not overwrite every update at repository root.
    manifest = verify_checkpoint(checkpoint_dir)
    api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id, repo_type="model",
                      path_in_repo=manifest["version"])
    return f"https://huggingface.co/{repo_id}/tree/main/{manifest['version']}"
