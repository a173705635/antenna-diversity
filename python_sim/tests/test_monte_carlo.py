import json

import numpy as np
import pytest

from antenna_diversity.config import default_config
from antenna_diversity.experiments.monte_carlo import run_monte_carlo


def _fast_config():
    cfg = default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    return cfg


def test_monte_carlo_runs_each_seed_and_summarizes_branch_rates():
    result = run_monte_carlo(_fast_config(), seeds=(101, 102))

    assert [run.seed for run in result.runs] == [101, 102]
    assert set(result.capture_rate) == {"single", "egc", "mvdr"}
    assert set(result.tracking_rate) == {"single", "egc", "mvdr"}
    assert all(0.0 <= value <= 1.0 for value in result.capture_rate.values())
    json.dumps(result.to_dict(), allow_nan=False)


def test_monte_carlo_does_not_mutate_the_input_config():
    cfg = _fast_config()
    original_seed = cfg.random_seed
    original_plot_enable = cfg.plot.enable

    run_monte_carlo(cfg, seeds=(11,))

    assert cfg.random_seed == original_seed
    assert cfg.plot.enable == original_plot_enable


def test_monte_carlo_is_repeatable_and_keeps_only_scalar_branch_metrics():
    first = run_monte_carlo(_fast_config(), seeds=(37,)).to_dict()
    second = run_monte_carlo(_fast_config(), seeds=(37,)).to_dict()

    assert first == second
    assert set(first["distributions"]) == {
        "acquisition_metric",
        "final_doppler_error_hz",
        "output_sinr_db",
        "worst_interferer_gain",
    }

    def contains_array(value):
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, dict):
            return any(contains_array(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_array(item) for item in value)
        return False

    assert not contains_array(first)


def test_monte_carlo_rejects_an_empty_seed_sequence():
    with pytest.raises(ValueError, match="seeds must be nonempty"):
        run_monte_carlo(_fast_config(), seeds=())
