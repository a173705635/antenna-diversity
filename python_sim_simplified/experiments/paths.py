"""Fixed experiment output paths and repeat-run directory refresh."""
from pathlib import Path
import shutil

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT_DIR/'results'
INTERFERENCE_RESULTS_DIR = RESULTS_DIR/'interference'


def reset_output(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir
