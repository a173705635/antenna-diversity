"""Configuration objects for the GPS L1 C/A simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from antenna_diversity.array.geometry import mate70_phase_centers


SPEED_OF_LIGHT_MPS = 299_792_458.0
PHONE_CORNER_NAMES = {"TL", "TR", "BL", "BR"}


@dataclass(frozen=True)
class Direction:
    azimuth_deg: float
    elevation_deg: float

    def validate(self) -> None:
        if not np.isfinite(self.azimuth_deg):
            raise ValueError("direction.azimuth_deg must be finite")
        if not np.isfinite(self.elevation_deg) or not 0.0 <= self.elevation_deg <= 90.0:
            raise ValueError("direction.elevation_deg must lie in [0, 90]")


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
    duration_ms: int = 40
    power_dbm: float = -128.5
    resistance_ohm: float = 50.0
    true_direction: Direction | None = None
    assumed_direction: Direction | None = None

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_MPS / self.carrier_hz


@dataclass
class ArrayConfig:
    num_elements: int
    spacing_m: float
    positions_m: NDArray[np.float64]
    desired_angle_deg: float = -10.0
    channel_amplitude_error_db: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    channel_phase_error_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )


@dataclass
class PhoneGeometryConfig:
    model_name: str
    length_m: float
    width_m: float
    thickness_m: float
    phase_centers_m: dict[str, tuple[float, float, float]]
    assumed_phase_centers_m: dict[str, tuple[float, float, float]]


@dataclass
class PhoneArrayConfig:
    active_antennas: tuple[str, ...]
    channel_amplitude_error_db: dict[str, float]
    channel_phase_error_deg: dict[str, float]
    estimated_channel_amplitude_error_db: dict[str, float]
    estimated_channel_phase_error_deg: dict[str, float]
    pattern_model: str = "isotropic"

    @property
    def num_elements(self) -> int:
        return len(self.active_antennas)

    def channel_response(self, estimated: bool = False) -> NDArray[np.complex128]:
        amplitude_source = (
            self.estimated_channel_amplitude_error_db
            if estimated
            else self.channel_amplitude_error_db
        )
        phase_source = (
            self.estimated_channel_phase_error_deg
            if estimated
            else self.channel_phase_error_deg
        )
        amplitude_db = np.asarray(
            [amplitude_source[name] for name in self.active_antennas]
        )
        phase_deg = np.asarray(
            [phase_source[name] for name in self.active_antennas]
        )
        return 10.0 ** (amplitude_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))


@dataclass
class JammerConfig:
    enable: bool = True
    power_dbm: float = -90.0
    offset_hz: float = 250e3
    angle_deg: float = 35.0
    initial_phase_rad: float = 0.8


@dataclass(frozen=True)
class CWInterferer:
    direction: Direction
    enable: bool = True
    power_dbm: float = -90.0
    offset_hz: float = 250e3
    initial_phase_rad: float = 0.8
    start_ms: float = 0.0
    stop_ms: float | None = None


@dataclass(frozen=True)
class RFStage:
    name: str
    gain_db: float
    noise_figure_db: float


@dataclass
class RFConfig:
    temperature_k: float = 290.0
    noise_bandwidth_hz: float = 8.184e6
    resistance_ohm: float = 50.0
    stages: tuple[RFStage, ...] = (
        RFStage("LNA", 28.0, 1.2),
        RFStage("Mixer", 8.0, 8.0),
        RFStage("VGA", 44.0, 10.0),
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
    input_bits: int = 10
    output_bits: int = 8
    output_binary_shift: int = 2

    @property
    def fs_out_hz(self) -> float:
        return self.fs_in_hz / self.decimation


@dataclass
class DiversityConfig:
    modes: tuple[str, ...] = ("single", "egc", "mvdr")
    diagonal_loading_factor: float = 1e-2


@dataclass
class AcquisitionConfig:
    doppler_bins_hz: NDArray[np.float64] = field(
        default_factory=lambda: np.arange(-5_000.0, 5_000.0 + 250.0, 250.0)
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
    max_epochs: int = 30
    max_final_doppler_error_hz: float = 250.0


@dataclass
class PlotConfig:
    enable: bool = True
    show: bool = True
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "results"
    )


@dataclass
class SimulationConfig:
    random_seed: int
    signal: SignalConfig
    array: ArrayConfig | PhoneArrayConfig
    jammer: JammerConfig
    rf: RFConfig
    dfe: DFEConfig
    diversity: DiversityConfig
    acquisition: AcquisitionConfig
    tracking: TrackingConfig
    plot: PlotConfig
    phone: PhoneGeometryConfig | None = None
    interferers: tuple[CWInterferer, ...] = ()

    def validate(self) -> None:
        """Reject configurations that violate chain-wide assumptions."""

        if not isinstance(self.random_seed, (int, np.integer)):
            raise ValueError("random_seed must be an integer")
        if not isinstance(self.signal.prn, (int, np.integer)) or not 1 <= self.signal.prn <= 32:
            raise ValueError("signal.prn must be an integer from 1 through 32")
        positive_signal_values = (
            self.signal.carrier_hz,
            self.signal.code_rate_hz,
            self.signal.fs_rf_hz,
            self.signal.duration_ms,
            self.signal.resistance_ohm,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive_signal_values):
            raise ValueError("signal frequencies, duration, and resistance must be positive")

        if isinstance(self.array, ArrayConfig):
            if self.signal.true_direction is not None or self.signal.assumed_direction is not None:
                raise ValueError("legacy ArrayConfig requires directions to be None")
            if self.phone is not None:
                raise ValueError("legacy ArrayConfig requires phone to be None")
            positions = np.asarray(self.array.positions_m)
            amplitude_errors = np.asarray(self.array.channel_amplitude_error_db)
            phase_errors = np.asarray(self.array.channel_phase_error_deg)
            if self.array.num_elements < 1 or positions.size != self.array.num_elements:
                raise ValueError("array.positions_m must contain one position per element")
            if amplitude_errors.size != self.array.num_elements or phase_errors.size != self.array.num_elements:
                raise ValueError("channel error arrays must contain one value per element")
            if not np.all(np.isfinite(positions)) or not np.isfinite(self.array.desired_angle_deg):
                raise ValueError("array positions and angle must be finite")
        elif isinstance(self.array, PhoneArrayConfig):
            if self.phone is None:
                raise ValueError("PhoneArrayConfig requires a phone configuration")
            if self.signal.true_direction is None or self.signal.assumed_direction is None:
                raise ValueError("PhoneArrayConfig requires true and assumed directions")
            self.signal.true_direction.validate()
            self.signal.assumed_direction.validate()

            corner_names = set(self.phone.phase_centers_m)
            corner_maps = (
                self.phone.assumed_phase_centers_m,
                self.array.channel_amplitude_error_db,
                self.array.channel_phase_error_deg,
                self.array.estimated_channel_amplitude_error_db,
                self.array.estimated_channel_phase_error_deg,
            )
            if corner_names != PHONE_CORNER_NAMES or any(
                set(corner_map) != PHONE_CORNER_NAMES for corner_map in corner_maps
            ):
                raise ValueError("phone coordinate and channel-error maps must use the same corner-name set")
            if any(
                np.asarray(coordinates).shape != (3,)
                or not np.all(np.isfinite(coordinates))
                for coordinates in (
                    *self.phone.phase_centers_m.values(),
                    *self.phone.assumed_phase_centers_m.values(),
                )
            ):
                raise ValueError("phone phase-center coordinates must be finite 3-vectors")
            if any(
                not np.isfinite(error)
                for error_map in corner_maps[1:]
                for error in error_map.values()
            ):
                raise ValueError("phone channel errors must be finite")
            if (
                not 2 <= self.array.num_elements <= 4
                or len(set(self.array.active_antennas)) != self.array.num_elements
                or any(name not in corner_names for name in self.array.active_antennas)
            ):
                raise ValueError("active_antennas must be a unique two-to-four corner subset")
        else:
            raise ValueError("array must be an ArrayConfig or PhoneArrayConfig")

        if len(self.interferers) > 3:
            raise ValueError("phone configurations support at most three interferers")
        for interferer in self.interferers:
            interferer.direction.validate()

        if self.rf.temperature_k < 0 or self.rf.noise_bandwidth_hz <= 0:
            raise ValueError("RF temperature and noise bandwidth are invalid")
        if self.rf.resistance_ohm <= 0 or self.rf.adc_vpp <= 0:
            raise ValueError("RF resistance and ADC full scale must be positive")
        if not isinstance(self.rf.adc_bits, (int, np.integer)) or self.rf.adc_bits < 2:
            raise ValueError("RF ADC bits must be an integer of at least two")
        if not self.rf.stages:
            raise ValueError("RF stages must not be empty")
        if any(
            not np.isfinite(stage.gain_db)
            or not np.isfinite(stage.noise_figure_db)
            or stage.noise_figure_db < 0
            for stage in self.rf.stages
        ):
            raise ValueError("RF stage gains and noise figures are invalid")

        if not np.isfinite(self.dfe.fs_in_hz) or self.dfe.fs_in_hz <= 0:
            raise ValueError("DFE input sample rate must be positive and finite")
        if not isinstance(self.dfe.decimation, (int, np.integer)) or self.dfe.decimation < 1:
            raise ValueError("DFE decimation must be a positive integer")
        if not 0 < self.dfe.cutoff_hz < self.dfe.fs_out_hz / 2:
            raise ValueError("DFE cutoff must lie below output Nyquist")
        if (
            not isinstance(self.dfe.num_taps, (int, np.integer))
            or self.dfe.num_taps < 3
            or self.dfe.num_taps % 2 == 0
        ):
            raise ValueError("DFE FIR length must be an odd integer of at least three")
        if self.dfe.output_bits != 0 and (
            not isinstance(self.dfe.output_bits, (int, np.integer))
            or self.dfe.output_bits < 2
        ):
            raise ValueError("DFE output bits must be zero or an integer of at least two")
        if (
            not isinstance(self.dfe.output_binary_shift, (int, np.integer))
            or self.dfe.output_binary_shift < 0
        ):
            raise ValueError("DFE output binary shift must be a nonnegative integer")

        samples_per_code = self.dfe.fs_out_hz * 1e-3
        samples_per_chip = self.dfe.fs_out_hz / self.signal.code_rate_hz
        if not np.isclose(samples_per_code, round(samples_per_code), atol=1e-12):
            raise ValueError("DFE output must contain an integer number of samples per code")
        if samples_per_chip < 2 or not np.isclose(
            samples_per_chip, round(samples_per_chip), atol=1e-12
        ):
            raise ValueError("DFE output must contain at least two integer samples per chip")

        bins = np.asarray(self.acquisition.doppler_bins_hz)
        if bins.size == 0 or not np.all(np.isfinite(bins)):
            raise ValueError("acquisition Doppler bins must be nonempty and finite")
        if self.acquisition.coherent_ms < 1 or self.acquisition.noncoherent_ms < 1:
            raise ValueError("acquisition integration lengths must be positive")
        if self.acquisition.threshold <= 0 or self.acquisition.code_rate_hz <= 0:
            raise ValueError("acquisition threshold and code rate must be positive")
        if self.tracking.coherent_ms < 1 or self.tracking.max_epochs < 1:
            raise ValueError("tracking integration length and epoch count must be positive")


def legacy_ula_config() -> SimulationConfig:
    """Return the baseline dual-antenna GPS L1 C/A configuration."""

    signal = SignalConfig()
    spacing_m = signal.wavelength_m / 2
    array = ArrayConfig(
        num_elements=2,
        spacing_m=spacing_m,
        positions_m=np.arange(2, dtype=float) * spacing_m,
    )
    cfg = SimulationConfig(
        random_seed=42,
        signal=signal,
        array=array,
        jammer=JammerConfig(),
        rf=RFConfig(resistance_ohm=signal.resistance_ohm),
        dfe=DFEConfig(fs_in_hz=signal.fs_rf_hz, if_hz=signal.if_hz),
        diversity=DiversityConfig(),
        acquisition=AcquisitionConfig(code_rate_hz=signal.code_rate_hz),
        tracking=TrackingConfig(acquisition_ms=4),
        plot=PlotConfig(),
    )
    cfg.validate()
    return cfg


def mate70_default_config() -> SimulationConfig:
    """Return the Mate 70 four-corner GPS L1 C/A configuration."""

    signal = SignalConfig(
        true_direction=Direction(30.0, 60.0),
        assumed_direction=Direction(30.0, 60.0),
    )
    phase_centers_m = mate70_phase_centers()
    phone = PhoneGeometryConfig(
        model_name="HUAWEI Mate 70",
        length_m=0.1609,
        width_m=0.0759,
        thickness_m=0.0078,
        phase_centers_m=phase_centers_m,
        assumed_phase_centers_m=phase_centers_m.copy(),
    )
    zero_errors = {name: 0.0 for name in phase_centers_m}
    array = PhoneArrayConfig(
        active_antennas=("TL", "TR", "BL", "BR"),
        channel_amplitude_error_db=zero_errors.copy(),
        channel_phase_error_deg=zero_errors.copy(),
        estimated_channel_amplitude_error_db=zero_errors.copy(),
        estimated_channel_phase_error_deg=zero_errors.copy(),
    )
    cfg = SimulationConfig(
        random_seed=42,
        signal=signal,
        array=array,
        jammer=JammerConfig(enable=False),
        rf=RFConfig(resistance_ohm=signal.resistance_ohm),
        dfe=DFEConfig(fs_in_hz=signal.fs_rf_hz, if_hz=signal.if_hz),
        diversity=DiversityConfig(),
        acquisition=AcquisitionConfig(code_rate_hz=signal.code_rate_hz),
        tracking=TrackingConfig(acquisition_ms=4),
        plot=PlotConfig(),
        phone=phone,
        interferers=(CWInterferer(direction=Direction(120.0, 10.0)),),
    )
    cfg.validate()
    return cfg


def default_config() -> SimulationConfig:
    """Return the baseline dual-antenna GPS L1 C/A configuration."""

    return legacy_ula_config()
