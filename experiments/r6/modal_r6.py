"""Launch R6 on Modal (A10G, float32). Protocol: docs/protocols/R6-*.md, docs/AUTHORIZATION.md Addendum 3.

  modal volume put r6-data <staged files> /          (model files, hashes.json, corpus.json)
  modal run --detach experiments/r6/modal_r6.py --code-commit SHA --corpus-digest D

The container runs experiments/r6/run_r6.py unchanged (hash and corpus digest checks included) with R6_DEVICE=cuda.
"""
import os
import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app = modal.App("mech-interp-r6")
vol = modal.Volume.from_name("r6-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "transformers==5.8.1", "numpy")
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "R6_DEVICE": "cuda", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(os.path.join(ROOT, "experiments", "r6"), "/repo/experiments/r6", ignore=["__pycache__"])
    .add_local_dir(os.path.join(ROOT, "imports", "astral-neuron", "tools"), "/repo/imports/astral-neuron/tools",
                   ignore=["__pycache__"])
)


@app.function(image=image, gpu="A10G", timeout=6 * 3600, volumes={"/data": vol})
def run(code_commit: str, corpus_digest: str):
    import sys
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("cuda", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    sys.path.insert(0, "/repo/experiments/r6")
    import run_r6
    os.makedirs("/data/out", exist_ok=True)
    out = "/data/out/r6_result.json"
    try:
        run_r6.main(["run", "--model-dir", "/data/model", "--model-hashes", "/data/hashes.json",
                     "--corpus", "/data/corpus.json", "--expected-corpus-digest", corpus_digest,
                     "--code-commit", code_commit, "--out", out], on_checkpoint=vol.commit)
    finally:
        vol.commit()
    print(open(out).read()[:4000])


@app.local_entrypoint()
def main(code_commit: str, corpus_digest: str):
    run.remote(code_commit, corpus_digest)
