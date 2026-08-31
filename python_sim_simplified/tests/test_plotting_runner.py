import matplotlib
import os
from pathlib import Path
import subprocess
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import default_config
from pipeline import run_end_to_end
from plotting import plot_results


def test_plot_results_returns_seven_independent_figures():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)

    figures = plot_results(results)

    assert set(figures) == {
        "rf_spectrum",
        "dfe_spectrum",
        "combined_spectra",
        "array_response",
        "acquisition",
        "tracking",
        "cn0",
    }
    assert len({id(figure) for figure in figures.values()}) == 7
    for figure in figures.values():
        assert figure.axes
        plt.close(figure)


def test_direct_runner_does_not_show_on_noninteractive_backend():
    script = Path(__file__).resolve().parents[1] / "run_end_to_end.py"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONWARNINGS"] = "error"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parents[1],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
