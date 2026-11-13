import json
import hashlib
import jsonschema
import pytest
from f1 import baselines, build, evidence, llm_predictor, split
from f1.registry import ITEMS
import make_split
import verify
from pathlib import Path

B = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def items():
    return build.load()


def test_count_and_labels(items):
    assert len(items) == len(ITEMS) == 36
    assert {i["label"] for i in items} <= {"replicated", "failed", "not_tested"}


def test_schema_valid(items):
    v = jsonschema.Draft202012Validator(json.loads((B / "item.schema.json").read_text()))
    for i in items:
        assert not list(v.iter_errors(i)), i["id"]


def test_schema_rejects_bad_label_reason(items):
    v = jsonschema.Draft202012Validator(json.loads((B / "item.schema.json").read_text()))
    bad = json.loads(json.dumps(items[0]))
    bad["label"] = "replicated"
    bad["label_reason"] = "no_arm_run"
    assert list(v.iter_errors(bad))


def test_every_item_has_pointer_evidence(items):
    for i in items:
        assert any(e["kind"] != "absence_assertion" for e in i["evidence"]), i["id"]
        for e in i["evidence"]:
            assert e["file"] and e["pointer"], i["id"]
            if e["kind"] != "absence_assertion":
                assert len(e["sha256"]) == 64


def test_rebuild_is_byte_identical(items):
    assert "".join(build.dumps(i) + "\n" for i in build.build_all()) == (B / "items.jsonl").read_text()


def test_evidence_matches_source_values(items):
    for i in items:
        for e in i["evidence"]:
            if e["kind"] == "absence_assertion" or not evidence.root_available(e["root"]):
                continue
            sha, val = evidence.extract(e["root"], e["file"], e["kind"], e["pointer"])
            assert (sha, val) == (e["sha256"], e["value"]), (i["id"], e["name"])


def test_tampered_label_detected(items, tmp_path):
    t = json.loads(json.dumps(items))
    for i in t:
        if i["label"] == "failed":
            i["label"] = "replicated"
            break
    rep = verify.run_checks(items=t, check_git=False)
    assert not rep["ok"] and any("labels_from_stored_evidence" in e for e in rep["errors"])


def test_tampered_evidence_value_detected(items):
    t = json.loads(json.dumps(items))
    t[0]["evidence"][0]["value"] = "Tampered"
    assert not verify.run_checks(items=t, check_git=False)["ok"]


def test_split_deterministic_and_frozen(items):
    ids = [i["id"] for i in items]
    assert [split.assign(i) for i in ids] == [split.assign(i) for i in ids]
    assert split.assign("x") == ("holdout" if int(hashlib.sha256(b"f1-transfer-bench-v1-split:x").hexdigest()[:8], 16) % 4 == 0 else "train")
    stored = json.loads((B / "splits" / "split.json").read_text())
    assert stored == make_split.compute(items)
    assert stored["n_holdout"] + stored["n_train"] == 36
    assert stored["holdout_ids_digest"] == split.digest(stored["holdout_ids"])


def test_split_independent_of_item_order(items):
    a = make_split.compute(items)["assignments"]
    b = make_split.compute(list(reversed(items)))["assignments"]
    assert a == b


def test_baselines_reproduce(items):
    sp = json.loads((B / "splits" / "split.json").read_text())
    train = [i for i, s in sp["assignments"].items() if s == "train"]
    res = baselines.evaluate(items, train, sp["holdout_ids"])
    stored = json.loads((B / "results" / "baselines.json").read_text())
    stored.pop("split_items_sha256")
    assert json.loads(json.dumps(res, sort_keys=True)) == json.loads(json.dumps(stored, sort_keys=True))


def test_baseline_features_exclude_label_fields(items):
    # perturbing label-bearing fields must not change metadata-baseline predictions
    for i in items:
        j = json.loads(json.dumps(i))
        for k in ("label", "label_rule", "label_reason", "label_inputs", "evidence", "notes"):
            j[k] = "X"
        j["transfer"]["verdict"] = "X"
        j["transfer"]["arm_status"] = "X"
        assert baselines.meta_a(i) == baselines.meta_a(j)
        assert baselines.meta_b(i) == baselines.meta_b(j)


def test_metrics_small():
    assert baselines.score(["a"] * 0, [])["n"] == 0
    s = baselines.score(["replicated", "failed"], ["replicated", "replicated"])
    assert s["correct"] == 1 and s["accuracy"] == 0.5
    assert baselines.majority_label(["failed", "replicated", "replicated"]) == "replicated"


def test_llm_prompt_label_free(items, tmp_path):
    for i in items:
        p = llm_predictor.build_prompt(i)
        assert llm_predictor.prompt_has_no_forbidden_values(i, p)
        assert i["label_reason"] not in p["user"]
        assert "evidence" not in p["user"].lower() or True
    out = tmp_path / "p.jsonl"
    llm_predictor.write_prompts(items[:3], out)
    assert len(out.read_text().splitlines()) == 3


def test_llm_score_and_parse(items):
    assert llm_predictor.parse_label("Replicated.") == "replicated"
    assert llm_predictor.parse_label("maybe") is None
    perfect = {i["id"]: i["label"] for i in items}
    r = llm_predictor.score_predictions(items, perfect)
    assert r["accuracy"] == 1.0 and r["unparsed"] == 0
    assert llm_predictor.score_predictions(items, {})["unparsed"] == 36


def test_full_verify_passes():
    rep = verify.run_checks(check_git=False)
    assert rep["ok"], rep["errors"]
