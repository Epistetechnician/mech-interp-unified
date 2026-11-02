#!/usr/bin/env python3
"""Verify every imported file against imports/MANIFEST.json (hash, size, no extras)."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / "imports/MANIFEST.json").read_text())
expected = {m["path"]: m for m in manifest["files"]}
bad = []
for path, m in expected.items():
    f = root / path
    if not f.is_file():
        bad.append(("missing", path))
        continue
    data = f.read_bytes()
    if len(data) != m["bytes"] or hashlib.sha256(data).hexdigest() != m["sha256"]:
        bad.append(("mismatch", path))
for f in (root / "imports").rglob("*"):
    rel = str(f.relative_to(root))
    if f.is_file() and rel != "imports/MANIFEST.json" and rel not in expected:
        bad.append(("extra", rel))
print(f"{len(expected)} files checked, {len(bad)} problems")
for kind, p in bad[:50]:
    print(kind, p)
sys.exit(1 if bad else 0)
