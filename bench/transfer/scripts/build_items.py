#!/usr/bin/env python3
"""Rebuild items.jsonl from sources (needs the external roots for jspace items)."""
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from f1 import build  # noqa: E402

items = build.build_all()
build.write(items)
print(len(items), dict(Counter(i["label"] for i in items)))
