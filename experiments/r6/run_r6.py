"""R6 runner (Apple MPS, float32, offline, no provider, zero spend). Protocol: docs/protocols/R6-*.md.

  python3 -I experiments/r6/run_r6.py hash-model --model-dir DIR --out hashes.json
  python3 -I experiments/r6/run_r6.py run --model-dir DIR --model-hashes hashes.json --corpus corpus.json \\
      --expected-corpus-digest D --code-commit SHA --out out_r6.json

`corpus.json` is the V54-format corpus ({"manifest":..., "split_items": {"fit":[...],"tune":[...],"assess":[...]}},
items {id,text,good,bad}); raw text stays outside the repo. Nothing is downloaded by this script.
"""
import argparse
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r6  # noqa: E402
from r6 import v50, pythia_plan  # noqa: E402


def hash_model(model_dir):
    return {n: pythia_plan.file_sha256(os.path.join(model_dir, n)) for n in sorted(os.listdir(model_dir))
            if n.endswith((".safetensors", ".bin", ".json"))}


def main(argv=None, backend_factory=None, on_checkpoint=None):
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash-model")
    h.add_argument("--model-dir", required=True)
    h.add_argument("--out", required=True)
    r = sub.add_parser("run")
    for n in ("model_dir", "model_hashes", "corpus", "out", "expected_corpus_digest", "code_commit"):
        r.add_argument("--" + n.replace("_", "-"), required=True)
    r.add_argument("--seed", type=int, default=71)
    args = a.parse_args(argv)
    if args.cmd == "hash-model":
        json.dump(hash_model(args.model_dir), open(args.out, "w"), indent=1, sort_keys=True)
        return None
    pythia_plan.verify_model_dir(args.model_dir, json.load(open(args.model_hashes)))
    corpus = json.load(open(args.corpus))
    cd = v50.digest(corpus["manifest"])
    if cd != args.expected_corpus_digest:
        raise v50.V50Error("corpus manifest differs from expected digest")
    cfg = {"seed": args.seed, "target_layers": list(r6.FROZEN6["target_layers"]),
           "model_id": r6.FROZEN6["model"], "corpus_digest": cd, "code_commit": args.code_commit}
    if backend_factory is None:
        def backend_factory(model_dir, items, layers):
            import qwen3_backend
            return qwen3_backend.make_backend_class()(model_dir, items, layers, device=os.environ.get("R6_DEVICE", "mps"))
    backend = backend_factory(args.model_dir, dict(corpus["split_items"]), cfg["target_layers"])
    t0 = time.time()
    def ckpt(o):
        json.dump(dict(o, partial=True), open(args.out + ".partial", "w"), indent=1, sort_keys=True, default=str)
        if on_checkpoint:
            on_checkpoint()

    res = r6.run_r6(backend, cfg, log=lambda *x: print("%.0fs" % (time.time() - t0), *x, flush=True), checkpoint=ckpt)
    res.update(wall_seconds=round(time.time() - t0, 1), device=os.environ.get("R6_DEVICE", "mps"), spend_usd=0.0)
    json.dump(res, open(args.out, "w"), indent=1, sort_keys=True)
    return res


if __name__ == "__main__":
    main()
