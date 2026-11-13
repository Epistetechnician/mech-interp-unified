"""python3 -m audit.run [--n 200] [--out audit/results/f2_phase1.json]"""
import argparse
import sys
import time
from pathlib import Path

from .harness import PROTOCOL_N, dumps, evaluate

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "f2_phase1.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PROTOCOL_N)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    t = time.time()
    res = evaluate(a.n, a.seed0)
    res["runtime_s"] = round(time.time() - t, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(dumps(res))
    print(f"wrote {a.out} in {res['runtime_s']}s; harness valid={res['harness_validity']['valid']}")
    return 0 if res["harness_validity"]["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
