"""Build items.jsonl from sources by applying f1.rules to extracted evidence."""
from __future__ import annotations

import json
from pathlib import Path

from . import evidence, rules
from .registry import ITEMS

BENCH = Path(__file__).resolve().parents[1]
ITEMS_PATH = BENCH / "items.jsonl"

COND_KEYS = ("model", "size", "size_b", "device", "dtype", "checkpoint", "corpus", "instrument_gate_status")


def build_item(spec: dict, extractor=evidence.extract) -> dict:
    vals, ev_out = {}, []
    for e in spec["ev"]:
        sha, value = extractor(e["root"], e["file"], e["kind"], e["pointer"])
        vals[e["name"]] = value
        ev_out.append({**e, "sha256": sha, "value": value})
    return assemble(spec, vals, ev_out)


def assemble(spec: dict, vals: dict, ev_out: list) -> dict:
    res = rules.evaluate(spec["kind"], vals)
    base = spec["base"]
    tr = dict(spec["transfer"])
    tr_cond = {k: tr.pop(k) for k in COND_KEYS}
    transfer = {"run_id": tr.pop("run_id"), **tr_cond,
                "axes_changed": list(spec["axes"]), "confounds": list(spec["confounds"]),
                "arm_status": res["arm_status"], "verdict": str(res["arm_verdict"])}
    out = {
        "id": spec["id"],
        "cluster_id": spec["cluster_id"],
        "claim": dict(spec["claim"]),
        **{k: base[k] for k in COND_KEYS},
        "base": {"run_id": spec["base_run"], "verdict": str(res["base_verdict"]), "verdict_class": res["base_class"]},
        "transfer": transfer,
        "label": res["label"],
        "label_rule": res["rule"],
        "label_reason": res["reason"],
        "label_inputs": res["inputs"],
        "evidence": ev_out,
        "ceiling": spec["ceiling"],
        "notes": spec["notes"],
    }
    return out


def build_all(extractor=evidence.extract) -> list[dict]:
    seen = set()
    out = []
    for spec in ITEMS:
        if spec["id"] in seen:
            raise ValueError(f"duplicate id {spec['id']}")
        seen.add(spec["id"])
        out.append(build_item(spec, extractor))
    return out


def dumps(item: dict) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write(items: list[dict], path: Path = ITEMS_PATH) -> None:
    path.write_text("".join(dumps(i) + "\n" for i in items))


def load(path: Path = ITEMS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
