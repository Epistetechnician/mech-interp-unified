#!/usr/bin/env python3
"""Write splits/split.json from items.jsonl using f1.split (hash of item id)."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from f1 import build, split  # noqa: E402


def compute(items):
    ids = [i["id"] for i in items]
    assign = {i: split.assign(i) for i in sorted(ids)}
    hold = sorted(i for i, s in assign.items() if s == "holdout")
    lab = {i["id"]: i["label"] for i in items}
    clu = {i["id"]: i["cluster_id"] for i in items}
    hold_clusters = {clu[i] for i in hold}
    train_clusters = {clu[i] for i in assign if assign[i] == "train"}
    return {
        "rule": "holdout iff int(sha256(salt + ':' + id).hexdigest()[:8], 16) % mod == 0",
        "salt": split.SALT,
        "mod": split.MOD,
        "split_py_sha256": hashlib.sha256((BENCH / "f1" / "split.py").read_bytes()).hexdigest(),
        "items_jsonl_sha256": hashlib.sha256((BENCH / "items.jsonl").read_bytes()).hexdigest(),
        "n_items": len(ids),
        "n_holdout": len(hold),
        "n_train": len(ids) - len(hold),
        "holdout_ids": hold,
        "holdout_ids_digest": split.digest(hold),
        "holdout_label_counts": dict(sorted(Counter(lab[i] for i in hold).items())),
        "train_label_counts": dict(sorted(Counter(lab[i] for i in assign if assign[i] == "train").items())),
        "clusters_on_both_sides": sorted(hold_clusters & train_clusters),
        "assignments": assign,
    }


def main():
    items = build.load()
    out = compute(items)
    (BENCH / "splits" / "split.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("assignments",)}, indent=1))


if __name__ == "__main__":
    main()
