"""Hardware-style fixed-point carrier NCO for tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antenna_diversity.numeric import matlab_round


@dataclass(frozen=True)
class CarrierNCOUpdate:
    """Carrier NCO state after one tracking-epoch update."""

    command_frequency_hz: float
    fcw: int
    frequency_hz: float
    quantization_error_hz: float
    phase_word: int


def _finite_real_scalar(value: object, name: str) -> float:
    candidate = np.asarray(value)
    if (
        candidate.ndim != 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.complexfloating)
        or isinstance(value, (bool, np.bool_))
        or not bool(np.isfinite(candidate))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(candidate)


class FixedCarrierNCO:
    """N-bit signed FCW with an N-bit unsigned phase accumulator."""

    def __init__(
        self,
        *,
        clock_hz: float,
        word_bits: int,
        coherent_interval_s: float,
        initial_frequency_hz: float,
    ) -> None:
        self.clock_hz = _finite_real_scalar(clock_hz, "clock_hz")
        if self.clock_hz <= 0.0:
            raise ValueError("clock_hz must be positive")
        if isinstance(word_bits, (bool, np.bool_)) or not isinstance(
            word_bits, (int, np.integer)
        ):
            raise ValueError("word_bits must be an integer")
        self.word_bits = int(word_bits)
        if not 2 <= self.word_bits <= 62:
            raise ValueError("word_bits must lie in [2, 62]")

        interval_s = _finite_real_scalar(
            coherent_interval_s, "coherent_interval_s"
        )
        if interval_s <= 0.0:
            raise ValueError("coherent_interval_s must be positive")
        ticks_exact = self.clock_hz * interval_s
        ticks = int(np.floor(ticks_exact + 0.5))
        if not np.isclose(ticks_exact, ticks, rtol=0.0, atol=1e-9):
            raise ValueError(
                "coherent interval must contain an integer number of NCO ticks"
            )

        self.modulus = 1 << self.word_bits
        self.minimum_fcw = -(1 << (self.word_bits - 1))
        self.maximum_fcw = (1 << (self.word_bits - 1)) - 1
        self.n_ticks = ticks
        self.phase_word = 0

        initial = self._quantize_frequency(initial_frequency_hz)
        self.command_frequency_hz = initial.command_frequency_hz
        self.fcw = initial.fcw
        self.frequency_hz = initial.frequency_hz
        self.quantization_error_hz = initial.quantization_error_hz

    @property
    def frequency_lsb_hz(self) -> float:
        return self.clock_hz / self.modulus

    @property
    def phase_rad(self) -> float:
        return float(2.0 * np.pi * self.phase_word / self.modulus)

    def _quantize_frequency(
        self, command_frequency_hz: float
    ) -> CarrierNCOUpdate:
        command = _finite_real_scalar(
            command_frequency_hz, "command_frequency_hz"
        )
        fcw = int(matlab_round(command / self.frequency_lsb_hz).item())
        if not self.minimum_fcw <= fcw <= self.maximum_fcw:
            raise ValueError("carrier FCW exceeds the signed N-bit range")
        actual = fcw * self.frequency_lsb_hz
        if not np.isfinite(actual):
            raise ValueError("quantized carrier frequency must be finite")
        return CarrierNCOUpdate(
            command_frequency_hz=command,
            fcw=fcw,
            frequency_hz=actual,
            quantization_error_hz=actual - command,
            phase_word=self.phase_word,
        )

    def update(
        self,
        command_frequency_hz: float,
        phase_correction_rad: float,
    ) -> CarrierNCOUpdate:
        """Quantize the next frequency and advance one coherent epoch."""

        quantized = self._quantize_frequency(command_frequency_hz)
        correction = _finite_real_scalar(
            phase_correction_rad, "phase_correction_rad"
        )
        correction_word = int(
            matlab_round(
                correction / (2.0 * np.pi) * self.modulus
            ).item()
        )
        self.phase_word = (
            self.phase_word
            + quantized.fcw * self.n_ticks
            + correction_word
        ) % self.modulus
        self.command_frequency_hz = quantized.command_frequency_hz
        self.fcw = quantized.fcw
        self.frequency_hz = quantized.frequency_hz
        self.quantization_error_hz = quantized.quantization_error_hz
        return CarrierNCOUpdate(
            command_frequency_hz=self.command_frequency_hz,
            fcw=self.fcw,
            frequency_hz=self.frequency_hz,
            quantization_error_hz=self.quantization_error_hz,
            phase_word=self.phase_word,
        )
