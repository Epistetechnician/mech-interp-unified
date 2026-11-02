#!/usr/bin/env python3
"""R4e: replicate R4d on fresh text (arm A) and Pythia-70m (arm B). Protocol: docs/protocols/R4e-replication.md"""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import GPTNeoXForCausalLM
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_offshoot_trajectory as r4  # noqa: E402
import r4c_birth_order_similarity as r4c  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results"
REVS_160 = r4c.REVS
REVS_70 = ["step2000", "step4000", "step8000", "step16000", "step24000", "step32000", "step48000",
           "step64000", "step80000", "step100000", "step120000", "step143000"]


def fresh_text(hashes):
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                        repo_type="dataset")
    hashes["wikitext-2-raw-v1/train"] = r4.sha(p)
    lines = [t.strip() for t in pq.read_table(p).column("text").to_pylist()]
    lines = [t for t in lines if len(t) >= 200]
    return lines[800:1200], lines[1200:1600]


def load_bin_first(rev, hashes):
    local = hf_hub_download(r4.REPO, "pytorch_model.bin", revision=rev)
    hashes[f"{rev}/pytorch_model.bin"] = r4.sha(local)
    hf_hub_download(r4.REPO, "config.json", revision=rev)
    return GPTNeoXForCausalLM.from_pretrained(str(Path(local).parent), torch_dtype=torch.float32).eval()


def run(name, repo, layers, revs, loader):
    r4.REPO, r4.LAYERS, r4.N_SEL = repo, layers, 500
    r4.get_text = fresh_text
    if loader:
        r4.load_model = loader
    r4c.REVS = revs
    r4.FINAL = revs[-1]
    r4c.OUT = ROOT / f"r4e_{name}.json"
    r4c.main()
    return json.loads(r4c.OUT.read_text())


def main():
    a = run("armA_pythia160m_fresh_text", "EleutherAI/pythia-160m", [3, 6, 9], REVS_160, None)
    orig_load = r4.load_model
    b = run("armB_pythia70m_fresh_text", "EleutherAI/pythia-70m", [1, 3, 4], REVS_70, load_bin_first)
    r4.load_model = orig_load
    sa = sum(v["growth_pass"] for v in a["layers"].values()) >= 2
    sb = sum(v["growth_pass"] for v in b["layers"].values()) >= 2
    verdict = "replicated" if sa and sb else "partial" if (sa or sb) else "not_replicated"
    out = {"armA_pass": sa, "armB_pass": sb, "verdict": verdict}
    (ROOT / "r4e_verdict.json").write_text(json.dumps(out, indent=1) + "\n")
    print("R4E", out)


if __name__ == "__main__":
    main()
