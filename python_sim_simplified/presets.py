"""Ready-to-run configurations for the simplified research simulation."""

from pathlib import Path

from config import (
    CORNER_NAMES,
    AcquisitionConfig,
    AntennaModelConfig,
    AntennaState,
    AttenuationRegion,
    CouplingPath,
    CWInterferer,
    DFEConfig,
    Direction,
    DiversityConfig,
    ElementOrientation,
    ParametricPattern,
    PhoneConfig,
    PlotConfig,
    RFConfig,
    SignalConfig,
    SimulationConfig,
    TrackingConfig,
    WidebandInterferer,
)


def mate70_phase_centers() -> dict[str, tuple[float, float, float]]:
    x = 0.0759 / 2.0
    y = 0.1609 / 2.0
    return {
        "TL": (-x, y, 0.0),
        "TR": (x, y, 0.0),
        "BL": (-x, -y, 0.0),
        "BR": (x, -y, 0.0),
    }


def default_config() -> SimulationConfig:
    positions = mate70_phase_centers()
    zero = {name: 0.0 for name in CORNER_NAMES}
    ideal = AntennaModelConfig(
        "ideal_isotropic",
        AntennaState(),
        AntennaState(),
    )
    return SimulationConfig(
        random_seed=42,
        reference_impedance_ohm=50.0,
        signal=SignalConfig(),
        phone=PhoneConfig(
            active_antennas=CORNER_NAMES,
            phase_centers_m=positions,
            assumed_phase_centers_m=positions.copy(),
            channel_amplitude_error_db=zero.copy(),
            channel_phase_error_deg=zero.copy(),
            channel_delay_ns=zero.copy(),
            estimated_channel_amplitude_error_db=zero.copy(),
            estimated_channel_phase_error_deg=zero.copy(),
            estimated_channel_delay_ns=zero.copy(),
            antenna_model=ideal,
        ),
        interferers=[
            CWInterferer(Direction(80.0, 10.0)),
            WidebandInterferer(Direction(80.0, 10.0)),
        ],
        rf=RFConfig(),
        dfe=DFEConfig(),
        diversity=DiversityConfig(),
        acquisition=AcquisitionConfig(),
        tracking=TrackingConfig(),
        plot=PlotConfig(),
    )


def parametric_config(
    active_antennas: tuple[str, ...] = CORNER_NAMES,
) -> SimulationConfig:
    cfg = default_config()
    cfg.phone.active_antennas = active_antennas
    orientations = {
        "TL": ElementOrientation(135.0, 12.0, -4.0),
        "TR": ElementOrientation(45.0, 10.0, 3.0),
        "BL": ElementOrientation(225.0, 8.0, 5.0),
        "BR": ElementOrientation(315.0, 11.0, -3.0),
    }
    patterns = {
        "TL": ParametricPattern(
            1.5,
            1.4,
            -17.0,
            cross_polar_discrimination_db=22.0,
            cross_phase_deg=8.0,
        ),
        "TR": ParametricPattern(
            1.0,
            1.6,
            -18.0,
            cross_polar_discrimination_db=24.0,
            cross_phase_deg=-6.0,
        ),
        "BL": ParametricPattern(
            0.5,
            1.8,
            -19.0,
            cross_polar_discrimination_db=21.0,
            cross_phase_deg=12.0,
        ),
        "BR": ParametricPattern(
            1.2,
            1.5,
            -18.5,
            cross_polar_discrimination_db=23.0,
            cross_phase_deg=-10.0,
        ),
    }
    blockage = (
        AttenuationRegion(("BL", "BR"), 225.0, 20.0, 42.0, 8.0, 18.0),
    )
    coupling = (
        CouplingPath("TL", "TR", -18.0, 32.0),
        CouplingPath("TR", "TL", -20.5, -24.0),
        CouplingPath("BL", "BR", -19.0, 20.0),
        CouplingPath("BR", "BL", -21.0, -38.0),
        CouplingPath("TL", "BR", -25.0, 55.0),
        CouplingPath("BR", "TL", -27.0, -48.0),
    )
    cfg.phone.antenna_model = AntennaModelConfig(
        "synthetic_phone_corner_v1",
        AntennaState(
            "parametric",
            patterns,
            orientations=orientations,
            attenuation_regions=blockage,
            coupling_paths=coupling,
        ),
        AntennaState(
            "parametric",
            patterns.copy(),
            orientations=orientations.copy(),
            coupling_paths=coupling,
        ),
    )
    return cfg


def measured_config(
    csv_path: Path,
    active_antennas: tuple[str, ...] = CORNER_NAMES,
) -> SimulationConfig:
    cfg = default_config()
    cfg.phone.active_antennas = active_antennas
    cfg.phone.antenna_model = AntennaModelConfig(
        "measured_pattern",
        AntennaState("measured", measured_pattern_csv=Path(csv_path)),
        AntennaState("measured", measured_pattern_csv=Path(csv_path)),
    )
    return cfg
