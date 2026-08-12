import numpy as np
import pytest

from antenna_diversity.analysis.metrics import format_metrics
from antenna_diversity.config import default_config, mate70_default_config
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


def test_phone_pipeline_uses_assumed_target_steering_for_weights():
    """Catch use of true/legacy steering when forming phone-array weights."""

    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2

    results = run_end_to_end(cfg)

    assert results.dfe_channels.shape[0] == 4
    assert set(results.branches) == {"single", "egc", "mvdr"}
    for branch in results.branches.values():
        np.testing.assert_allclose(
            branch.combining["constraint_steering"],
            results.truth["desired_steering_assumed"],
        )
        assert branch.combining["weights_are_fixed"] is True
        assert (
            branch.combining["covariance_snapshots"]
            == results.dfe_channels.shape[1]
        )
        assert len(branch.combining["interferer_direction_gains"]) == 1
        assert branch.combining["worst_interferer_gain"] == pytest.approx(
            max(branch.combining["interferer_direction_gains"])
        )

    for record in results.metrics:
        combining = results.branches[record["mode"]].combining
        for key in (
            "target_direction_gain",
            "interferer_direction_gains",
            "worst_interferer_gain",
            "covariance_snapshots",
            "weights_are_fixed",
        ):
            assert record[key] == combining[key]


def test_phone_pipeline_reports_stable_gain_without_interferers():
    """Catch empty-reduction failures or non-JSON-safe no-interferer metrics."""

    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    cfg.interferers = ()

    results = run_end_to_end(cfg)

    for record in results.metrics:
        assert record["interferer_direction_gains"] == []
        assert record["worst_interferer_gain"] == 0.0
