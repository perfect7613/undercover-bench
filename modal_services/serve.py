"""One private, scale-to-zero Modal GPU app per pinned Unsloth GGUF.

Usage: MODEL_KEY=muse_glimmer uv run modal deploy modal_services/serve.py
       MODEL_KEY=muse_glimmer uv run modal run modal_services/serve.py::stage
"""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen
import modal

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/models.proposed.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("/opt/undercover-model-config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())
KEY = os.environ.get("MODEL_KEY", "muse_glimmer")
MODEL = next((m for m in CONFIG["models"] if m["key"] == KEY), None)
if MODEL is None:
    raise ValueError(f"Unknown MODEL_KEY {KEY}")
APP_NAME = f"undercover-{KEY}"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(f"undercover-weights-{KEY}", create_if_missing=True)
base = "nvidia/cuda:13.1.0-devel-ubuntu24.04" if MODEL["modal"]["gpu"] == "B300" else "nvidia/cuda:12.8.1-devel-ubuntu22.04"
llama_revision = ("b72050d03ae14bb3cb5138ecefab3f3f65157d31" if KEY == "deepseek_flash"
                  else "f3269ef96aee8c9ec8a3a89a14025e33641827e3")
llama_repo = "https://github.com/danielhanchen/llama.cpp.git" if KEY == "deepseek_flash" else "https://github.com/ggml-org/llama.cpp.git"
cuda_arch = "103" if MODEL["modal"]["gpu"] == "B300" else "89"
image = (modal.Image.from_registry(base, add_python="3.12")
         .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev")
         .run_commands(f"git clone {llama_repo} /opt/llama.cpp && cd /opt/llama.cpp && git checkout {llama_revision}")
         .run_commands(f"cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={cuda_arch} -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build /opt/llama.cpp/build -j 8 --target llama-server")
         .pip_install("huggingface-hub>=0.34,<2")
         .env({"MODEL_KEY": KEY})
         .add_local_file(str(CONFIG_PATH), "/opt/undercover-model-config.json"))
stage_image = (modal.Image.debian_slim(python_version="3.12")
               .pip_install("huggingface-hub>=0.34,<2")
               .env({"MODEL_KEY": KEY})
               .add_local_file(str(CONFIG_PATH), "/opt/undercover-model-config.json"))


@app.function(image=stage_image, volumes={"/weights": volume}, timeout=7200, cpu=4, memory=16384)
def stage():
    from huggingface_hub import snapshot_download
    files = [item["path"] for item in MODEL["selected_files"]]
    snapshot_download(repo_id=MODEL["repo_id"], revision=MODEL["revision"],
                      allow_patterns=files, local_dir="/weights", max_workers=8)
    missing = [f for f in files if not (Path("/weights") / f).is_file()]
    if missing:
        raise RuntimeError(f"Missing staged files: {missing}")
    volume.commit()
    return {"key": KEY, "revision": MODEL["revision"], "files": len(files),
            "bytes": sum((Path("/weights") / f).stat().st_size for f in files)}


_server = None
_server_log = None


def _server_log_tail():
    path = Path('/tmp/undercover-llama-server.log')
    if not path.exists():
        return 'no server log'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 6000))
        return stream.read().decode(errors='replace')


@app.function(image=image, gpu=MODEL["modal"]["gpu"], volumes={"/weights": volume},
              timeout=1800, min_containers=0, max_containers=1, scaledown_window=300,
              memory=245760 if KEY == "deepseek_flash" else 32768)
def generate(prompt: str, max_tokens: int = 120, response_schema: dict | None = None,
             temperature: float = 0.2, seed: int | None = None):
    global _server, _server_log
    first = Path("/weights") / MODEL["selected_files"][0]["path"]
    if not first.is_file():
        raise RuntimeError(f"Weights not staged for {KEY}; run stage first")
    if _server is not None and _server.poll() is not None:
        raise RuntimeError(f"llama-server exited with code {_server.returncode}; refusing restart loop: {_server_log_tail()}")
    if _server is None:
        command = ["/opt/llama.cpp/build/bin/llama-server", "-m", str(first),
                   "--host", "127.0.0.1", "--port", "8080", "-c", "8192",
                   "-ngl", "99", "--jinja", "--parallel", "1", "--no-webui",
                   "--alias", KEY]
        if KEY == "qwen_dense":
            command += ["--reasoning", "on"]
        if KEY == "muse_glimmer":
            command += ["--chat-template-kwargs", '{"reasoning_strength":"low"}']
        if KEY == "glm_flash":
            command += ["--reasoning-effort", "low", "--reasoning-budget", "128"]
        if KEY == "deepseek_flash":
            command += ["--override-tensor", "per_layer_token_embd.weight=CPU", "--reasoning", "on", "--reasoning-effort", "low"]
        # A pipe fills while a large GGUF loads and can block the server before /health.
        if _server_log is not None:
            _server_log.close()
        _server_log = open('/tmp/undercover-llama-server.log', 'ab', buffering=0)
        _server = subprocess.Popen(command, stdout=_server_log, stderr=subprocess.STDOUT)
        ready = False
        for _ in range(540):
            if _server.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {_server.returncode}: {_server_log_tail()}")
            try:
                with urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                    if response.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(2)
        if not ready:
            _server.kill()
            _server.wait(timeout=30)
            raise TimeoutError(f"llama-server did not become healthy: {_server_log_tail()}")
    payload = {"model": KEY, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature, "cache_prompt": False}
    if seed is not None:
        payload["seed"] = seed
    if response_schema is not None:
        payload["response_format"] = {"type": "json_schema", "schema": response_schema}
    request = Request("http://127.0.0.1:8080/v1/chat/completions",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    msg = choice["message"]
    return {"text": msg.get("content") or "", "model_emitted_reasoning": msg.get("reasoning_content"),
            "model_message": msg, "finish_reason": choice.get("finish_reason"),
            "model": MODEL["repo_id"], "revision": MODEL["revision"], "usage": result.get("usage")}
