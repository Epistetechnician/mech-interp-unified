#!/usr/bin/env python3
"""Run majority and metadata baselines on the frozen split; write results/baselines.json and BASELINES.md."""
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from f1 import baselines, build  # noqa: E402


def fmt(s):
    lo, hi = s["accuracy_wilson95"]
    return f"{s['correct']}/{s['n']} = {s['accuracy']:.3f} [{lo:.2f}, {hi:.2f}], macro-F1 {s['macro_f1']:.3f}"


def main():
    items = build.load()
    sp = json.loads((BENCH / "splits" / "split.json").read_text())
    train = [i for i, s in sp["assignments"].items() if s == "train"]
    hold = sp["holdout_ids"]
    res = baselines.evaluate(items, train, hold)
    res["split_items_sha256"] = sp["items_jsonl_sha256"]
    (BENCH / "results" / "baselines.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    lines = ["# Baseline results (frozen split)", "",
             f"Items {sp['n_items']} (train {sp['n_train']}, holdout {sp['n_holdout']}); majority label on train: `{res['majority_label_train']}`.",
             "Accuracy with Wilson 95% interval; macro-F1 over labels present in truth or prediction. Holdout n is small: read the intervals.", "",
             "| method | train acc | holdout acc | all acc | holdout tested-only (replicated vs failed) |", "|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        to = r["holdout"]["tested_only"]
        lines.append(f"| {m} | {fmt(r['train'])} | {fmt(r['holdout'])} | {fmt(r['all'])} | {fmt(to)} |")
    loco = res["methods"]["majority"]["leave_one_cluster_out"]
    lines += ["", f"Majority, leave-one-cluster-out over all items (cluster-grouped sensitivity): {fmt(loco)}", "",
              "Confusion matrices (rows truth, columns prediction) are in `baselines.json`.", "",
              "LLM predictor: not run (needs a paid API call; see README)."]
    (BENCH / "results" / "BASELINES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
