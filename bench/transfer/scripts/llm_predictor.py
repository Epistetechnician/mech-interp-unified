#!/usr/bin/env python3
"""LLM-predictor harness (no network, no paid call).

  llm_predictor.py dry  [--split holdout|train|all]            write results/llm_prompts_<split>.jsonl
  llm_predictor.py score --predictions FILE [--split ...]      score {"id","label"} lines produced elsewhere
"""
import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from f1 import build, llm_predictor  # noqa: E402


def pick(items, split):
    sp = json.loads((BENCH / "splits" / "split.json").read_text())["assignments"]
    return [i for i in items if split == "all" or sp[i["id"]] == split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dry", "score"])
    ap.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    ap.add_argument("--predictions")
    a = ap.parse_args()
    items = pick(build.load(), a.split)
    if a.mode == "dry":
        out = BENCH / "results" / f"llm_prompts_{a.split}.jsonl"
        llm_predictor.write_prompts(items, out)
        print(f"wrote {len(items)} prompts to {out}; no model was called")
    else:
        preds = {}
        for line in Path(a.predictions).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r["label"]
        print(json.dumps(llm_predictor.score_predictions(items, preds), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
