#!/usr/bin/env python3
"""Re-derive labels and the split from sources and check everything committed.

Exit 0 only if every check passes. Items whose evidence lives in an unavailable external root
are checked from stored evidence plus stored digests and reported as external_unavailable.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "scripts"))
from f1 import baselines, build, evidence, rules  # noqa: E402
from f1.registry import ITEMS  # noqa: E402
import make_split  # noqa: E402


def planned_ids():
    text = (BENCH / "LABEL_RULES.md").read_text()
    sec = text.split("## 4. Planned items", 1)[1].split("## 5.", 1)[0]
    ids = re.findall(r"^\| ([A-Za-z0-9][A-Za-z0-9._-]*) \|", sec, flags=re.M)
    n = int(re.search(r"Count: (\d+) planned", sec).group(1))
    return [i for i in ids if i != "id"], n


def git_order_ok():
    def first_commit(path):
        r = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", path], cwd=BENCH,
                           capture_output=True, text=True)
        c = r.stdout.split()
        return c[-1] if c else None
    a, b = first_commit("LABEL_RULES.md"), first_commit("items.jsonl")
    if a is None or b is None:
        return None
    r = subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=BENCH)
    return r.returncode == 0 and a != b


def run_checks(items=None, items_path=None, check_git=True):
    rep = {"checks": {}, "errors": [], "external_unavailable": []}
    items_path = items_path or build.ITEMS_PATH
    items = items if items is not None else build.load(items_path)
    specs = {s["id"]: s for s in ITEMS}

    def check(name, ok, msg=""):
        rep["checks"][name] = bool(ok)
        if not ok:
            rep["errors"].append(f"{name}: {msg}")

    # 1 schema
    try:
        import jsonschema
        schema = json.loads((BENCH / "item.schema.json").read_text())
        v = jsonschema.Draft202012Validator(schema)
        bad = [(i["id"], e.message) for i in items for e in v.iter_errors(i)]
        check("schema", not bad, str(bad[:3]))
    except ImportError:
        check("schema", False, "jsonschema not installed")

    # 2 ids vs preregistered plan
    ids = [i["id"] for i in items]
    plan, n = planned_ids()
    check("ids_unique", len(ids) == len(set(ids)), "duplicate ids")
    check("ids_match_plan", set(ids) == set(plan) and len(plan) == n == len(ids),
          f"extra={sorted(set(ids)-set(plan))} missing={sorted(set(plan)-set(ids))} plan_count={n}")
    check("registry_matches_items", set(specs) == set(ids), "registry/items id mismatch")

    # 3 re-derive each item: (a) from stored evidence, (b) from sources where available
    mism_stored, mism_src = [], []
    for it in items:
        spec = specs.get(it["id"])
        if spec is None:
            continue
        vals = {e["name"]: e["value"] for e in it["evidence"]}
        try:
            again = build.assemble(spec, vals, [dict(e) for e in it["evidence"]])
        except Exception as ex:  # noqa: BLE001
            mism_stored.append((it["id"], f"{type(ex).__name__}: {ex}"))
            continue
        if build.dumps(again) != build.dumps(it):
            mism_stored.append((it["id"], "stored evidence does not reproduce stored item"))
        # source re-extraction
        unavailable = False
        for e in it["evidence"]:
            if e["kind"] != "absence_assertion" and not evidence.root_available(e["root"]):
                unavailable = True
                continue
            try:
                sha, val = evidence.extract(e["root"], e["file"], e["kind"], e["pointer"])
            except Exception as ex:  # noqa: BLE001
                mism_src.append((it["id"], e["name"], f"{type(ex).__name__}: {ex}"))
                continue
            if sha != e["sha256"] or val != e["value"]:
                mism_src.append((it["id"], e["name"], "source digest or value changed"))
        if unavailable:
            rep["external_unavailable"].append(it["id"])
    check("labels_from_stored_evidence", not mism_stored, str(mism_stored[:3]))
    check("evidence_matches_sources", not mism_src, str(mism_src[:3]))

    # 4 label bookkeeping
    check("labels_in_enum", all(i["label"] in rules.LABELS for i in items), "bad label")

    # 5 split
    sp_path = BENCH / "splits" / "split.json"
    if items_path == build.ITEMS_PATH and sp_path.exists():
        stored = json.loads(sp_path.read_text())
        fresh = make_split.compute(items)
        check("split_matches", stored == fresh,
              "split.json differs from recomputation (split.py or items.jsonl changed)")
    else:
        check("split_matches", False, "splits/split.json missing or non-default items path")

    # 6 baselines
    bp = BENCH / "results" / "baselines.json"
    if bp.exists() and items_path == build.ITEMS_PATH:
        stored = json.loads(bp.read_text())
        sp = json.loads(sp_path.read_text())
        train = [i for i, s in sp["assignments"].items() if s == "train"]
        fresh = baselines.evaluate(items, train, sp["holdout_ids"])
        fresh["split_items_sha256"] = sp["items_jsonl_sha256"]
        check("baselines_match", json.loads(json.dumps(fresh, sort_keys=True)) == json.loads(json.dumps(stored, sort_keys=True)),
              "results/baselines.json differs from recomputation")
    else:
        check("baselines_match", False, "results/baselines.json missing")

    # 7 preregistration order (informational where git history is absent)
    if check_git:
        g = git_order_ok()
        if g is not None:
            check("rules_committed_before_items", g, "LABEL_RULES.md was not committed before items.jsonl")

    rep["n_items"] = len(items)
    rep["label_counts"] = dict(sorted(Counter(i["label"] for i in items).items()))
    rep["ok"] = not rep["errors"]
    return rep


def main():
    rep = run_checks()
    print(json.dumps({k: v for k, v in rep.items() if k != "checks"}, indent=1))
    for k, v in rep["checks"].items():
        print(("PASS " if v else "FAIL ") + k)
    if rep["external_unavailable"]:
        print("external_unavailable (checked from stored evidence only):", ", ".join(rep["external_unavailable"]))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
