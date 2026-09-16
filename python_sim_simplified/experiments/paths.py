"""Fixed experiment output paths and repeat-run directory refresh."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'


def reset_output(name):
    output = RESULTS/name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output
