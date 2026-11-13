"""python3 -m audit.verify : re-derive audit/results/f2_phase1.json from seeds; exit 0 iff counts and digests match."""
import json
import sys
from pathlib import Path

from .harness import evaluate
from .run import DEFAULT_OUT


def _strip(d):
    return {k: v for k, v in d.items() if k != "runtime_s"}


def main(argv=None):
    path = Path(argv[0]) if argv else DEFAULT_OUT
    saved = json.loads(path.read_text())
    new = json.loads(json.dumps(evaluate(saved["n_seeds"], saved["seed0"]), default=lambda o: o.item()))
    a, b = _strip(saved), _strip(new)
    if a == b:
        print(f"VERIFIED: {path} re-derived exactly from seeds (n={saved['n_seeds']})")
        return 0
    bad = [k for k in a if a[k] != b.get(k)]
    print("MISMATCH in:", bad)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
