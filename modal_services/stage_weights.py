"""Stage only exact pinned GGUF shards in separate per-model Modal volumes."""
import json
from pathlib import Path
import modal

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/models.proposed.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("/opt/undercover-model-config.json")
models = json.loads(CONFIG_PATH.read_text())["models"]
app = modal.App("undercover-stage-weights")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface-hub>=0.34,<2")
         .add_local_file(str(CONFIG_PATH), "/opt/undercover-model-config.json"))
volumes = {m["key"]: modal.Volume.from_name(f"undercover-weights-{m['key']}", create_if_missing=True) for m in models}


@app.function(image=image, volumes={"/weights": volumes["muse_glimmer"]}, timeout=7200, cpu=4, memory=16384)
def stage_muse():
    return _stage("muse_glimmer")


@app.function(image=image, volumes={"/weights": volumes["qwen_dense"]}, timeout=7200, cpu=4, memory=16384)
def stage_qwen():
    return _stage("qwen_dense")


@app.function(image=image, volumes={"/weights": volumes["glm_flash"]}, timeout=14400, cpu=8, memory=32768)
def stage_glm():
    return _stage("glm_flash")


@app.function(image=image, volumes={"/weights": volumes["deepseek_flash"]}, timeout=14400, cpu=8, memory=32768)
def stage_deepseek():
    return _stage("deepseek_flash")


def _stage(key):
    from huggingface_hub import snapshot_download
    model = next(m for m in models if m["key"] == key)
    files = [entry["path"] for entry in model["selected_files"]]
    snapshot_download(repo_id=model["repo_id"], revision=model["revision"],
                      allow_patterns=files, local_dir="/weights", max_workers=4)
    actual = sum((Path("/weights") / file).stat().st_size for file in files)
    if actual != model["total_selected_bytes"]:
        raise RuntimeError(f"byte count mismatch for {key}: {actual}")
    volumes[key].commit()
    return {"model": key, "files": len(files), "bytes": actual}


@app.local_entrypoint()
def main(model_key: str = "muse_glimmer"):
    fn = {"muse_glimmer": stage_muse, "qwen_dense": stage_qwen, "glm_flash": stage_glm, "deepseek_flash": stage_deepseek}.get(model_key)
    if fn is None:
        raise ValueError("unknown model")
    print(fn.remote())
