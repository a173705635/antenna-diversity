import copy

import numpy as np
import pytest

from antenna_diversity.analysis.metrics import format_metrics
from antenna_diversity.array.antenna_config import (
    AntennaStateConfig,
    PhoneAntennaConfig,
)
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.config import (
    default_config,
    legacy_ula_config,
    mate70_default_config,
)
from antenna_diversity.pipeline import run_end_to_end


def _unit_parametric_patterns():
    return {
        name: ParametricPatternParameters(
            back_lobe_gain_db=0.0,
            cross_polar_discrimination_db=300.0,
        )
        for name in ("TL", "TR", "BL", "BR")
    }


def test_legacy_pipeline_preserves_original_diversity_conclusion():
    """Catch any cross-stage mismatch that removes the expected MVDR advantage."""
    cfg = legacy_ula_config()
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


def test_mate70_baseline_mvdr_improves_spatial_and_receiver_metrics():
    cfg = default_config()
    cfg.plot.enable = False

    results = run_end_to_end(cfg)

    single = results.branches["single"]
    mvdr = results.branches["mvdr"]
    assert (
        mvdr.combining["interferer_direction_gains"][0]
        < 0.1 * single.combining["interferer_direction_gains"][0]
    )
    assert mvdr.acquisition["metric"] > single.acquisition["metric"]
    assert mvdr.acquisition["success"]
    assert mvdr.tracking["success"]


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
    assert "FinalCn0DbHz" in table


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


def test_phone_pipeline_propagates_nonideal_effective_manifold_once():
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    cfg.array.active_antennas = ("TL", "TR")
    state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=_unit_parametric_patterns(),
        attenuation_regions=(
            AttenuationRegion(
                ("TL",),
                cfg.signal.true_direction.azimuth_deg,
                cfg.signal.true_direction.elevation_deg,
                1.0,
                6.020599913279624,
            ),
        ),
        coupling_paths=(CouplingPath("TL", "TR", -6.020599913279624, 90.0),),
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="end-to-end fixture",
        model_source="parametric",
        true_state=state,
        assumed_state=copy.deepcopy(state),
    )
    model = PhoneAntennaModel.from_config(cfg)
    expected = model.effective_manifold(
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )

    results = run_end_to_end(cfg)

    np.testing.assert_allclose(results.truth["desired_steering_true"], expected)
    np.testing.assert_allclose(
        results.truth["desired_component"],
        expected[:, None] * results.truth["desired_waveform"][None, :],
    )
