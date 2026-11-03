"""Shared text pools for R1b (wikitext-2 train, Qwen3 tokenizer). Slices are named so dev and test never overlap."""
import numpy as np

import lib

SLICES = {  # line-index ranges over wikitext-2 train lines with len >= 200
    "r1b1_fit": (2800, 3400), "r1b1_assess": (3400, 4000),   # frozen R1b-1 (used)
    "dev_fit": (4000, 4600), "dev_assess": (4600, 5200),     # v2 development only
    "test_fit": (5200, 5800), "test_assess": (5800, 6400),   # v2 frozen test, untouched until freeze
}
CTX_LEN = 64


def load_lines():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
    lines = [t.strip() for t in pq.read_table(p).column("text").to_pylist()]
    return [t for t in lines if len(t) >= 200]


def pool(name, lines, tok):
    a, b = SLICES[name]
    out = []
    for t in lines[a:b]:
        ids = tok(t, add_special_tokens=False)["input_ids"][:CTX_LEN]
        if len(ids) >= 32:
            out.append(lib.Ctx([tok.decode([i]) for i in ids], np.zeros(len(ids))))
    return out
