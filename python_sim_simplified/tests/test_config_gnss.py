import numpy as np

from core.config import CORNER_NAMES, CWInterferer, WidebandInterferer
from core.gnss import ca_code
from core.presets import default_config


def test_default_config_uses_mate70_geometry_and_signal_values():
    cfg = default_config()

    assert cfg.random_seed == 42
    assert cfg.signal.carrier_hz == 1_575.42e6
    assert cfg.signal.fs_rf_hz == 33e6
    assert cfg.dfe.decimation == 5
    assert cfg.signal.fs_rf_hz/cfg.dfe.decimation == 6.6e6
    assert cfg.signal.if_hz == 4.092e6
    assert cfg.phone.active_antennas == ("TL", "TR", "BL", "BR")
    assert cfg.phone.phase_centers_m["TL"] == (-0.0759 / 2, 0.1609 / 2, 0.0)
    assert cfg.phone.channel_delay_ns == {name: 0.0 for name in CORNER_NAMES}
    assert cfg.phone.estimated_channel_delay_ns == {
        name: 0.0 for name in CORNER_NAMES
    }
    assert len(cfg.interferers) == 2
    assert isinstance(cfg.interferers[0], CWInterferer)
    assert isinstance(cfg.interferers[1], WidebandInterferer)
    assert cfg.interferers[1].bandwidth_hz == 2.0e6


def test_default_scenario_is_interference_only():
    cfg = default_config()

    assert not hasattr(cfg.signal, "multipath_paths")
    assert not hasattr(cfg, "multipath_detection")
    assert not hasattr(cfg, "multipath_suppression")


def test_shared_physical_parameters_have_one_default_source():
    cfg = default_config()

    assert cfg.reference_impedance_ohm == 50.0
    assert not hasattr(cfg.signal, "resistance_ohm")
    assert not hasattr(cfg.rf, "resistance_ohm")
    assert not hasattr(cfg.dfe, "fs_in_hz")
    assert not hasattr(cfg.dfe, "if_hz")
    assert cfg.dfe.nco_frequency_hz is None
    assert not hasattr(cfg.acquisition, "code_rate_hz")


def test_ca_code_is_repeatable_bipolar_and_1023_chips_long():
    first = ca_code(1)
    second = ca_code(1)

    assert first.shape == (1023,)
    assert set(np.unique(first)) == {-1.0, 1.0}
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first[:10], [-1, -1, 1, 1, -1, 1, 1, 1, 1, 1]
    )
