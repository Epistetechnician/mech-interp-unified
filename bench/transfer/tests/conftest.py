import sys
from pathlib import Path
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
sys.path.insert(0, str(B / "scripts"))
