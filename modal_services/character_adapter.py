"""Prepare only the character adapter on CPU in the existing GLM weight volume.

Never downloads a base model, allocates a GPU, or changes staged GGUF weights.
"""
from pathlib import Path
import modal

from modal_services.serve import app, volume, MODEL, KEY

if KEY != "glm_flash":
    raise ValueError("Set MODEL_KEY=glm_flash")

REVISION = "f3269ef96aee8c9ec8a3a89a14025e33641827e3"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
    .pip_install("numpy>=2,<3", "huggingface-hub>=1,<2", "safetensors>=0.6")
    .run_commands("git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
                  f"cd /opt/llama.cpp && git checkout {REVISION}")
    .env({"MODEL_KEY": "glm_flash"})
    .add_local_file(str(Path(__file__).resolve().parents[1] / "configs/models.proposed.json"),
                    "/opt/undercover-model-config.json")
    .add_local_file(str(Path(__file__).resolve().parents[1] / "training/character/patch_glm_lora_graph.py"),
                    "/root/training/character/patch_glm_lora_graph.py"))
for name in ("convert_glm_character_adapter.py", "convert_glm_character_gguf.py",
             "inspect_adapter_compatibility.py", "validate_glm_character_gguf.py"):
    image = image.add_local_file(str(Path(__file__).resolve().parents[1] / "training/character" / name),
                                 "/root/training/character/" + name)


@app.function(image=image, volumes={"/weights": volume}, cpu=8, memory=8192,
              timeout=2400, retries=0, secrets=[modal.Secret.from_name("undercover-character-hf")])
def stage_character_adapter():
    import json
    from huggingface_hub import hf_hub_download
    from training.character.convert_glm_character_gguf import convert
    from training.character.validate_glm_character_gguf import validate, sha256
    root = Path("/weights/character/glm-sft-v7")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        proof = json.loads((root / "delta_validation.json").read_text())
        if (manifest["gguf_sha256"] != sha256(root / "adapter.gguf") or not proof["passed"]
                or proof["gguf_sha256"] != manifest["gguf_sha256"]):
            raise ValueError("Existing character artifacts differ; refusing overwrite")
        return {"status": "already_staged", "manifest": manifest, "gpu_allocated": False}
    source = root / "source"
    repo = "Perfect7613/glm-5.3-flash-auditor-character-sft-v7"
    commit = "44c25e1c3ff8d47b2ad6eeac74d65e14cc7228f9"
    # Only this 10 GB adapter and its config; never a base snapshot.
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        path = hf_hub_download(repo, "adapter/" + name, revision=commit, local_dir=source)
        target = source / name
        if not target.exists():
            target.symlink_to(Path(path).resolve().relative_to(source.resolve()))
    expected = "b5c8aadb3685dce6fd6240d6c68aa17a882146417df22c2472afa0ef9d83f022"
    if sha256(source / "adapter_model.safetensors") != expected:
        raise ValueError("Character source differs from the verified SFT export")
    base = "unsloth/GLM-5.3-Flash"
    base_revision = "61d609da1b5fef8f25c754534a0fba313aba1caf"
    config = json.loads(Path(hf_hub_download(base, "config.json", revision=base_revision)).read_text())
    index = json.loads(Path(hf_hub_download(base, "model.safetensors.index.json", revision=base_revision)).read_text())
    converted = convert(source, root / "adapter.gguf", config, index, Path("/opt/llama.cpp"),
                        outtype="mixed", base_revision=base_revision)
    proof = validate(source, root / "adapter.gguf", Path("/opt/llama.cpp"), config)
    (root / "delta_validation.json").write_text(json.dumps(proof, indent=2))
    if not proof["passed"] or proof["projection_pairs"] != 431:
        volume.commit()
        raise ValueError("Character conversion failed numerical validation; no serving manifest written")
    manifest = {"schema": 1, "purpose": "character_adapter_engineering_validation",
        "source_pairs_preserved": 431, "source_weights_sha256": expected,
        "gguf_sha256": proof["gguf_sha256"], "bytes": (root / "adapter.gguf").stat().st_size,
        "serving_revision": MODEL["revision"], "source_hf_release": repo + "@" + commit,
        "precision": converted["dtype"], "delta_conversion_validation_passed": True,
        "compatibility_verified": False, "character_improvement_established": False}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return {"status": "staged", "manifest": manifest, "gpu_allocated": False,
            "base_model_downloaded": False}
