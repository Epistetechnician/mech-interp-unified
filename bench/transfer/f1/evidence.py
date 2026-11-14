"""Evidence extraction: json_pointer, regex, absence_assertion. Read-only."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ROOTS = {
    "repo": str(REPO_ROOT),
    "jspace": "/Users/shaanp/Documents/GitHub/patelshaan/jspace-pilot",
    "zkbench": "/Users/shaanp/Documents/GitHub/composed-zk-benchmark-os",
}
ENV = {"jspace": "F1_ROOT_JSPACE", "zkbench": "F1_ROOT_ZKBENCH"}


def root_path(root: str) -> Path:
    if root == "repo":
        return REPO_ROOT
    return Path(os.environ.get(ENV[root], DEFAULT_ROOTS[root]))


def root_available(root: str) -> bool:
    return root_path(root).is_dir()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_pointer(doc, pointer: str):
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"bad pointer {pointer!r}")
    cur = doc
    for raw in pointer[1:].split("/"):
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(tok)]
        else:
            cur = cur[tok]
    return cur


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def regex_value(text: str, pattern: str):
    """Pattern must match exactly once in whitespace-normalised text."""
    t = _norm(text)
    matches = list(re.finditer(pattern, t))
    if len(matches) != 1:
        raise ValueError(f"regex {pattern!r} matched {len(matches)} times (need exactly 1)")
    m = matches[0]
    if m.re.groups:
        return m.group(1)
    return True


def extract(root: str, file: str, kind: str, pointer: str):
    """Return (sha256, value). Raises if the source cannot be read or the pointer fails."""
    if kind == "absence_assertion":
        base = root_path(root)
        for rel in [p.strip() for p in pointer.split(";") if p.strip()]:
            if not (base / rel).exists():
                raise FileNotFoundError(f"absence_assertion path missing: {root}:{rel}")
        return None, True
    path = root_path(root) / file
    sha = sha256_file(path)
    if kind == "json_pointer":
        with open(path) as f:
            return sha, json_pointer(json.load(f), pointer)
    if kind == "regex":
        return sha, regex_value(path.read_text(errors="replace"), pointer)
    raise ValueError(kind)
