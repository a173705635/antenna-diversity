from itertools import combinations
import os
from pathlib import Path
import shutil
import subprocess
import sys

from antenna import antenna_response, build_antenna_model
from config import default_config
from pipeline import run_end_to_end


def test_standalone_copy_runs_without_research_package(tmp_path):
    source = Path(__file__).resolve().parents[1]
    standalone = tmp_path / "python_sim_simplified"
    shutil.copytree(
        source,
        standalone,
        ignore=shutil.ignore_patterns("tests", "results", "__pycache__"),
    )
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONWARNINGS"] = "error"

    completed = subprocess.run(
        [sys.executable, str(standalone / "run_end_to_end.py")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(list((standalone / "results").glob("*.png"))) == 7


def test_all_eleven_two_to_four_corner_combinations_form_a_response():
    names = ("TL", "TR", "BL", "BR")
    tested = []
    for size in (2, 3, 4):
        for active in combinations(names, size):
            cfg = default_config()
            cfg.phone.active_antennas = active
            response, _ = antenna_response(
                build_antenna_model(cfg),
                cfg.signal.true_direction,
                cfg.signal.true_polarization,
            )
            assert response.shape == (size,)
            tested.append(active)
    assert len(tested) == 11


def test_two_three_and_four_channels_reach_dfe_and_combining():
    for active in (
        ("TL", "BR"),
        ("TL", "TR", "BR"),
        ("TL", "TR", "BL", "BR"),
    ):
        cfg = default_config()
        cfg.signal.duration_ms = 8
        cfg.tracking.max_epochs = 1
        cfg.phone.active_antennas = active

        results = run_end_to_end(cfg)

        assert results["dfe_channels"].shape[0] == len(active)
        assert set(results["branches"]) == set(cfg.diversity.modes)
