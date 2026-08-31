from pathlib import Path
import sys


SIMPLIFIED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMPLIFIED_ROOT))
