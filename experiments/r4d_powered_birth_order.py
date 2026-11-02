#!/usr/bin/env python3
"""R4d: R4c with 500 neurons per layer. Protocol: docs/protocols/R4d-powered-birth-order.md"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_offshoot_trajectory as r4  # noqa: E402
import r4c_birth_order_similarity as r4c  # noqa: E402

r4.N_SEL = 500
r4c.OUT = Path(__file__).resolve().parent.parent / "results" / "r4d_powered_birth_order.json"

if __name__ == "__main__":
    r4c.main()
