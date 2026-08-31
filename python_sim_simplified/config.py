"""Editable parameters for the simplified GNSS antenna-diversity model."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


CORNER_NAMES = ("TL", "TR", "BL", "BR")
SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass
class Direction:
    azimuth_deg: float
    elevation_deg: float


@dataclass
class WavePolarization:
    kind: str = "rhcp"
    tilt_deg: float = 0.0
    az_real: float = 1.0
    az_imag: float = 0.0
    el_real: float = 0.0
    el_imag: float = 0.0


@dataclass
class ElementOrientation:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass
class ParametricPattern:
    peak_gain_db: float = 0.0
    cosine_exponent: float = 1.0
    back_lobe_gain_db: float = -20.0
    nominal_polarization: WavePolarization = field(
        default_factory=WavePolarization
    )
    cross_polar_discrimination_db: float = 30.0
    cross_phase_deg: float = 0.0


@dataclass
class AttenuationRegion:
    elements: tuple[str, ...]
    center_azimuth_deg: float
    center_elevation_deg: float
    radius_deg: float
    attenuation_db: float
    transition_width_deg: float = 0.0


@dataclass
class CouplingPath:
    source: str
    receiver: str
    magnitude_db: float
    phase_deg: float


@dataclass
class AntennaState:
    pattern_kind: str = "ideal"
    parametric_patterns: dict[str, ParametricPattern] = field(
        default_factory=dict
    )
    measured_pattern_csv: Path | None = None
    orientations: dict[str, ElementOrientation] = field(
        default_factory=lambda: {
            name: ElementOrientation() for name in CORNER_NAMES
        }
    )
    attenuation_regions: tuple[AttenuationRegion, ...] = ()
    coupling_paths: tuple[CouplingPath, ...] = ()


@dataclass
class AntennaModelConfig:
    model_label: str
    true_state: AntennaState
    assumed_state: AntennaState


@dataclass
class SignalConfig:
    prn: int = 1
    carrier_hz: float = 1_575.42e6
    code_rate_hz: float = 1.023e6
    fs_rf_hz: float = 16.368e6
    if_hz: float = 4.092e6
    doppler_hz: float = 1_500.0
    code_delay_chips: float = 137.25
    initial_phase_rad: float = 0.35
    duration_ms: int = 100
    power_dbm: float = -128.5
    resistance_ohm: float = 50.0
    true_direction: Direction = field(
        default_factory=lambda: Direction(30.0, 60.0)
    )
    assumed_direction: Direction = field(
        default_factory=lambda: Direction(30.0, 60.0)
    )
    true_polarization: WavePolarization = field(
        default_factory=WavePolarization
    )
    assumed_polarization: WavePolarization = field(
        default_factory=WavePolarization
    )

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_MPS / self.carrier_hz


@dataclass
class PhoneConfig:
    active_antennas: tuple[str, ...]
    phase_centers_m: dict[str, tuple[float, float, float]]
    assumed_phase_centers_m: dict[str, tuple[float, float, float]]
    channel_amplitude_error_db: dict[str, float]
    channel_phase_error_deg: dict[str, float]
    channel_delay_ns: dict[str, float]
    estimated_channel_amplitude_error_db: dict[str, float]
    estimated_channel_phase_error_deg: dict[str, float]
    estimated_channel_delay_ns: dict[str, float]
    antenna_model: AntennaModelConfig


@dataclass
class CWInterferer:
    direction: Direction
    enable: bool = True
    power_dbm: float = -90.0
    offset_hz: float = 250e3
    initial_phase_rad: float = 0.8
    start_ms: float = 0.0
    stop_ms: float | None = None
    polarization: WavePolarization = field(default_factory=WavePolarization)


@dataclass
class RFStage:
    name: str
    gain_db: float
    noise_figure_db: float


@dataclass
class RFConfig:
    temperature_k: float = 290.0
    noise_bandwidth_hz: float = 8.184e6
    resistance_ohm: float = 50.0
    stages: tuple[RFStage, ...] = field(
        default_factory=lambda: (
            RFStage("LNA", 28.0, 1.2),
            RFStage("Mixer", 8.0, 8.0),
            RFStage("VGA", 44.0, 10.0),
        )
    )
    adc_bits: int = 10
    adc_vpp: float = 1.0
    enable_soft_limiting: bool = True


@dataclass
class DFEConfig:
    fs_in_hz: float = 16.368e6
    if_hz: float = 4.092e6
    decimation: int = 4
    cutoff_hz: float = 1.8e6
    num_taps: int = 129
    output_bits: int = 8
    output_binary_shift: int = 2

    @property
    def fs_out_hz(self) -> float:
        return self.fs_in_hz / self.decimation


@dataclass
class BlindNullConfig:
    detection_threshold_db: float = 6.0
    reference_channel: int = 0


@dataclass
class DiversityConfig:
    modes: tuple[str, ...] = ("single", "egc", "mvdr", "blind_null")
    diagonal_loading_factor: float = 1e-2
    blind_null: BlindNullConfig = field(default_factory=BlindNullConfig)


@dataclass
class AcquisitionConfig:
    doppler_bins_hz: np.ndarray = field(
        default_factory=lambda: np.arange(-5_000.0, 5_000.1, 250.0)
    )
    coherent_ms: int = 1
    noncoherent_ms: int = 4
    exclusion_chips: float = 1.0
    threshold: float = 2.5
    code_rate_hz: float = 1.023e6


@dataclass
class TrackingConfig:
    coherent_ms: int = 1
    acquisition_ms: int = 4
    early_late_spacing_chips: float = 0.5
    fll_assist_epochs: int = 5
    fll_gain: float = 0.5
    pll_kp: float = 0.30
    pll_ki_hz: float = 4.0
    dll_kp_chips: float = 0.08
    dll_ki_hz: float = 0.5
    cn0_window_epochs: int = 10
    max_epochs: int = 100


@dataclass
class PlotConfig:
    show: bool = True
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "results"
    )


@dataclass
class SimulationConfig:
    random_seed: int
    signal: SignalConfig
    phone: PhoneConfig
    interferers: list[CWInterferer]
    rf: RFConfig
    dfe: DFEConfig
    diversity: DiversityConfig
    acquisition: AcquisitionConfig
    tracking: TrackingConfig
    plot: PlotConfig


def mate70_phase_centers() -> dict[str, tuple[float, float, float]]:
    x = 0.0759 / 2.0
    y = 0.1609 / 2.0
    return {
        "TL": (-x, y, 0.0),
        "TR": (x, y, 0.0),
        "BL": (-x, -y, 0.0),
        "BR": (x, -y, 0.0),
    }


def channel_response(phone: PhoneConfig, estimated: bool = False) -> np.ndarray:
    amplitude = (
        phone.estimated_channel_amplitude_error_db
        if estimated
        else phone.channel_amplitude_error_db
    )
    phase = (
        phone.estimated_channel_phase_error_deg
        if estimated
        else phone.channel_phase_error_deg
    )
    amplitude_db = np.array([amplitude[name] for name in phone.active_antennas])
    phase_deg = np.array([phase[name] for name in phone.active_antennas])
    return 10.0 ** (amplitude_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))


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
        interferers=[CWInterferer(Direction(120.0, 10.0))],
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
        "TL": ParametricPattern(1.5, 1.4, -17.0, cross_polar_discrimination_db=22.0, cross_phase_deg=8.0),
        "TR": ParametricPattern(1.0, 1.6, -18.0, cross_polar_discrimination_db=24.0, cross_phase_deg=-6.0),
        "BL": ParametricPattern(0.5, 1.8, -19.0, cross_polar_discrimination_db=21.0, cross_phase_deg=12.0),
        "BR": ParametricPattern(1.2, 1.5, -18.5, cross_polar_discrimination_db=23.0, cross_phase_deg=-10.0),
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
        AntennaState("parametric", patterns, orientations=orientations,
                     attenuation_regions=blockage, coupling_paths=coupling),
        AntennaState("parametric", patterns.copy(), orientations=orientations.copy(),
                     coupling_paths=coupling),
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
