import numpy as np

from antenna_diversity.analysis.metrics import format_metrics
from antenna_diversity.config import default_config
from antenna_diversity.pipeline import run_end_to_end


def test_default_pipeline_reproduces_diversity_conclusion():
    """Catch any cross-stage mismatch that removes the expected MVDR advantage."""
    cfg = default_config()
    cfg.plot.enable = False

    results = run_end_to_end(cfg)

    assert set(results.branches) == {"single", "egc", "mvdr"}
    assert not results.branches["single"].acquisition["success"]
    assert not results.branches["egc"].acquisition["success"]
    assert results.branches["mvdr"].acquisition["success"]
    assert results.branches["mvdr"].tracking["success"]
    assert (
        abs(results.branches["mvdr"].acquisition["doppler_hz"] - 1500.0)
        <= 250.0
    )
    assert (
        abs(
            results.branches["mvdr"].tracking["carrier_frequency_hz"][-1]
            - 1500.0
        )
        <= 250.0
    )
    jammer_steering = results.truth["jammer_steering_nominal"]
    single_gain = abs(
        np.vdot(
            results.branches["single"].combining["weights"],
            jammer_steering,
        )
    )
    mvdr_gain = abs(
        np.vdot(
            results.branches["mvdr"].combining["weights"],
            jammer_steering,
        )
    )
    assert mvdr_gain < single_gain
    assert len(results.metrics) == 3


def test_metric_formatter_includes_every_branch_and_tracking_state():
    """Catch a summary that silently omits failed acquisition branches."""
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    table = format_metrics(results.metrics)

    assert "single" in table
    assert "egc" in table
    assert "mvdr" in table
    assert "Acquired" in table
    assert "Tracking" in table
