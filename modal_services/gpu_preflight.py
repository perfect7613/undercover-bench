"""Cheap allocation check; never claims a model loaded."""
import subprocess
import modal

app = modal.App("undercover-gpu-preflight")
image = modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", timeout=120)
def l40s():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True).strip()


@app.function(image=image, gpu="B300", timeout=120)
def b300():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True).strip()


@app.local_entrypoint()
def main():
    print("L40S", l40s.remote())
    print("B300", b300.remote())
