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
    """Synthetic dimensionless receive Jones pattern, not handset data."""
    peak_gain_db: float = 0.0
    cosine_exponent: float = 1.0
    back_lobe_gain_db: float = -20.0
    nominal_polarization: WavePolarization = field(
        default_factory=WavePolarization
    )
    cross_polar_discrimination_db: float = 30.0
    cross_phase_deg: float = 0.0
    azimuth_peak_deg: float = 0.0
    azimuth_attenuation_db: float = 0.0
    # Extra receive-path length (mm): qx*ux² + qy*uy² + qxy*ux*uy.
    phase_variation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class AttenuationRegion:
    elements: tuple[str, ...]
    center_azimuth_deg: float
    center_elevation_deg: float
    radius_deg: float
    attenuation_db: float
    transition_width_deg: float = 0.0
    # Positive means a positive received-voltage phase rotation, not delay.
    phase_shift_deg: float = 0.0


@dataclass
class CouplingPath:
    """Phenomenological constant voltage transfer, not an RF S-parameter."""
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
    # Residual element response, before coupling; absent names mean zero error.
    amplitude_error_db: dict[str, float] = field(default_factory=dict)
    phase_error_deg: dict[str, float] = field(default_factory=dict)


@dataclass
class AntennaModelConfig:
    model_label: str
    true_state: AntennaState
    assumed_state: AntennaState


@dataclass
class SignalConfig:
    signal_kind: str = "L1CA"
    prn: int = 1
    carrier_hz: float = 1_575.42e6
    code_rate_hz: float = 1.023e6
    fs_rf_hz: float = 33e6
    if_hz: float = 4.092e6
    doppler_hz: float = 1_500.0
    code_delay_chips: float = 137.25
    initial_phase_rad: float = 0.35
    duration_ms: int = 100
    power_dbm: float = -128.5
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
    # Optional ideal-lowpass source; None preserves the legacy sampled chips.
    code_bandwidth_hz: float | None = None

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
    enable_flag: bool = True
    power_dbm: float = -90.0
    offset_hz: float = 250e3
    initial_phase_rad: float = 0.8
    start_ms: float = 0.0
    stop_ms: float | None = None
    polarization: WavePolarization = field(default_factory=WavePolarization)


@dataclass
class WidebandInterferer:
    direction: Direction
    enable_flag: bool = True
    power_dbm: float = -90.0
    center_offset_hz: float = 0.0
    bandwidth_hz: float = 2.0e6
    random_seed: int = 1001
    filter_taps: int = 129
    start_ms: float = 0.0
    stop_ms: float | None = None
    polarization: WavePolarization = field(default_factory=WavePolarization)


Interferer = CWInterferer | WidebandInterferer


@dataclass
class RFStage:
    name: str
    gain_db: float
    noise_figure_db: float


@dataclass
class RFConfig:
    temperature_k: float = 290.0
    noise_bandwidth_hz: float = 8.184e6
    # Odd FIR length; noise_bandwidth_hz is total/two-sided IF ENBW.
    noise_filter_taps: int = 129
    stages: tuple[RFStage, ...] = field(
        default_factory=lambda: (
            RFStage("LNA", 28.0, 1.2),
            RFStage("Mixer", 8.0, 8.0),
            RFStage("VGA", 44.0, 10.0),
        )
    )
    adc_bits: int = 10
    adc_vpp: float = 1.0
    enable_soft_limiting_flag: bool = True
    # Equivalent output error at fixed noise figure, before limiter/ADC.
    # A scalar applies to every row; a tuple follows input channel row order.
    amplitude_error_db: float | tuple[float, ...] = 0.0
    phase_error_deg: float | tuple[float, ...] = 0.0


@dataclass
class DFEConfig:
    decimation: int = 5
    cutoff_hz: float = 1.8e6
    num_taps: int = 129
    output_bits: int = 8
    output_binary_shift: int = 2
    nco_frequency_hz: float | None = None


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


@dataclass
class TrackingConfig:
    signal_kind: str = "L1CA"
    # Legacy atan2 assumes a data-free carrier. Costas removes unknown BPSK
    # sign transitions from PLL/FLL errors, retaining a half-cycle ambiguity.
    carrier_discriminator: str = "atan2"
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
    # None preserves the legacy rectangular replica. Positive Hz enables an
    # ideal-lowpass research reference; this is a one-sided baseband cutoff.
    code_replica_bandwidth_hz: float | None = None
    refine_initial_phase_flag: bool = False


@dataclass
class PlotConfig:
    show_flag: bool = True
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "results"
    )


@dataclass
class SimulationConfig:
    random_seed: int
    reference_impedance_ohm: float
    signal: SignalConfig
    phone: PhoneConfig
    interferers: list[Interferer]
    rf: RFConfig
    dfe: DFEConfig
    diversity: DiversityConfig
    acquisition: AcquisitionConfig
    tracking: TrackingConfig
    plot: PlotConfig


# Ready-to-run configuration factories are defined in presets.py.
