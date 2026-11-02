#!/usr/bin/env python3
"""Run the sandbox-only qualification and continual-learning fixtures.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.continual import run_continual_suite
from interp_lab.qualification import run_qualification

print(json.dumps({"qualification": run_qualification(), "continual_learning": run_continual_suite()}, sort_keys=True))
