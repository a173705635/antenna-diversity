"""One-millisecond Early/Prompt/Late DLL with FLL-assisted PLL."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.config import TrackingConfig
from antenna_diversity.gnss.ca_code import ca_code
from antenna_diversity.receiver.carrier_nco import FixedCarrierNCO


def _sample_code(
    code: NDArray[np.float64], phase_chips: NDArray[np.float64]
) -> NDArray[np.float64]:
    indices = np.floor(np.mod(phase_chips, 1023.0)).astype(np.int64)
    return code[indices]


def _wrap_phase(phase_rad: float) -> float:
    return float(np.mod(phase_rad + np.pi, 2.0 * np.pi) - np.pi)


def estimate_cn0_db_hz(
    prompt_window: ArrayLike,
    coherent_interval_s: float,
) -> float:
    """Estimate C/N0 from a window of coherent Prompt correlators."""

    prompt_samples = np.asarray(prompt_window, dtype=np.complex128)
    if prompt_samples.ndim != 1:
        raise ValueError("prompt_window must be one-dimensional")
    if not np.isfinite(coherent_interval_s) or coherent_interval_s <= 0.0:
        raise ValueError("coherent_interval_s must be finite and positive")
    if prompt_samples.size < 2:
        return float("nan")

    coherent_power = abs(np.mean(prompt_samples)) ** 2
    total_power = np.mean(np.abs(prompt_samples) ** 2)
    eps = np.finfo(float).eps
    noise_power = max(float(total_power - coherent_power), eps)
    return float(
        10.0
        * np.log10(
            max(
                coherent_power / (noise_power * coherent_interval_s),
                eps,
            )
        )
    )


def track(
    x: ArrayLike,
    fs_hz: float,
    prn: int,
    acq: dict[str, Any],
    cfg: TrackingConfig,
    carrier_hz: float,
    nominal_code_rate_hz: float,
) -> dict[str, Any]:
    """Track residual carrier and C/A-code phase after acquisition."""

    samples = np.asarray(x, dtype=np.complex128)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("x must be a nonempty one-dimensional vector")
    if not acq.get("success", False):
        return {
            "success": False,
            "status": "acquisition_failed",
            "prompt": np.array([], dtype=np.complex128),
        }
    if cfg.carrier_nco_mode not in {"float", "fixed"}:
        raise ValueError('carrier_nco_mode must be "float" or "fixed"')
    if fs_hz <= 0 or carrier_hz <= 0 or nominal_code_rate_hz <= 0:
        raise ValueError("sample, carrier, and code frequencies must be positive")

    samples_per_epoch_exact = fs_hz * cfg.coherent_ms * 1e-3
    if not np.isclose(
        samples_per_epoch_exact, round(samples_per_epoch_exact), atol=1e-9
    ):
        raise ValueError("coherent interval must contain an integer number of samples")
    samples_per_epoch = round(samples_per_epoch_exact)
    acquisition_samples = round(cfg.acquisition_ms * 1e-3 * fs_hz)

    first_epoch = int(acq["code_phase_samples"])
    samples_per_ms = round(fs_hz * 1e-3)
    while first_epoch < acquisition_samples:
        first_epoch += samples_per_ms
    available_epochs = (samples.size - first_epoch) // samples_per_epoch
    n_epochs = min(int(available_epochs), cfg.max_epochs)
    if n_epochs < 1:
        raise ValueError("no complete tracking epoch remains after acquisition")

    code = ca_code(prn)
    sample_index = np.arange(samples_per_epoch, dtype=np.float64)
    epoch_duration_s = samples_per_epoch / fs_hz
    carrier_frequency_command_hz = float(acq["doppler_hz"])
    fixed_nco: FixedCarrierNCO | None = None
    if cfg.carrier_nco_mode == "fixed":
        fixed_nco = FixedCarrierNCO(
            clock_hz=cfg.carrier_nco_clock_hz,
            word_bits=cfg.carrier_nco_word_bits,
            coherent_interval_s=epoch_duration_s,
            initial_frequency_hz=carrier_frequency_command_hz,
        )
        carrier_frequency_hz = fixed_nco.frequency_hz
        carrier_phase_rad = fixed_nco.phase_rad
        frequency_lsb_hz = fixed_nco.frequency_lsb_hz
        carrier_fcw_history = np.zeros(n_epochs, dtype=np.int64)
        carrier_phase_word_history = np.zeros(n_epochs, dtype=np.uint64)
    else:
        carrier_frequency_hz = carrier_frequency_command_hz
        carrier_phase_rad = 0.0
        frequency_lsb_hz = 0.0
    code_phase_chips = 0.0
    dll_frequency_correction_hz = 0.0
    code_rate_hz = nominal_code_rate_hz * (
        1.0 + carrier_frequency_hz / carrier_hz
    )
    previous_prompt: complex | None = None

    early = np.zeros(n_epochs, dtype=np.complex128)
    prompt = np.zeros(n_epochs, dtype=np.complex128)
    late = np.zeros(n_epochs, dtype=np.complex128)
    dll_error = np.zeros(n_epochs, dtype=np.float64)
    pll_error_rad = np.zeros(n_epochs, dtype=np.float64)
    fll_error_hz = np.zeros(n_epochs, dtype=np.float64)
    carrier_frequency_command_history = np.zeros(
        n_epochs, dtype=np.float64
    )
    carrier_frequency_history = np.zeros(n_epochs, dtype=np.float64)
    carrier_frequency_quantization_error_history = np.zeros(
        n_epochs, dtype=np.float64
    )
    code_rate_history = np.zeros(n_epochs, dtype=np.float64)
    code_phase_history = np.zeros(n_epochs, dtype=np.float64)
    phase_lock_metric = np.zeros(n_epochs, dtype=np.float64)
    cn0_db_hz = np.full(n_epochs, np.nan, dtype=np.float64)
    epoch_start_samples = np.zeros(n_epochs, dtype=np.int64)
    eps = np.finfo(float).eps

    for epoch in range(n_epochs):
        block_first = first_epoch + epoch * samples_per_epoch
        block = samples[block_first : block_first + samples_per_epoch]
        epoch_start_samples[epoch] = block_first

        carrier_phase = (
            carrier_phase_rad
            + 2.0
            * np.pi
            * carrier_frequency_hz
            * sample_index
            / fs_hz
        )
        baseband = block * np.exp(-1j * carrier_phase)

        prompt_phase = (
            code_phase_chips + sample_index * code_rate_hz / fs_hz
        )
        half_spacing = cfg.early_late_spacing_chips / 2.0
        prompt_code = _sample_code(code, prompt_phase)
        early_code = _sample_code(code, prompt_phase + half_spacing)
        late_code = _sample_code(code, prompt_phase - half_spacing)

        early[epoch] = np.sum(baseband * early_code)
        prompt[epoch] = np.sum(baseband * prompt_code)
        late[epoch] = np.sum(baseband * late_code)

        dll_error[epoch] = (
            abs(early[epoch]) - abs(late[epoch])
        ) / (abs(early[epoch]) + abs(late[epoch]) + eps)
        pll_error_rad[epoch] = np.arctan2(
            prompt[epoch].imag, prompt[epoch].real
        )
        if previous_prompt is not None:
            fll_error_hz[epoch] = np.angle(
                prompt[epoch] * np.conj(previous_prompt)
            ) / (2.0 * np.pi * epoch_duration_s)

        fll_correction = 0.0
        if 0 < epoch < cfg.fll_assist_epochs:
            fll_correction = cfg.fll_gain * fll_error_hz[epoch]
        carrier_frequency_command_hz += (
            fll_correction + cfg.pll_ki_hz * pll_error_rad[epoch]
        )
        phase_correction_rad = cfg.pll_kp * pll_error_rad[epoch]
        if fixed_nco is None:
            carrier_frequency_hz = carrier_frequency_command_hz
            carrier_phase_rad = _wrap_phase(
                carrier_phase_rad
                + 2.0 * np.pi * carrier_frequency_hz * epoch_duration_s
                + phase_correction_rad
            )
            quantization_error_hz = 0.0
        else:
            nco_update = fixed_nco.update(
                carrier_frequency_command_hz,
                phase_correction_rad,
            )
            carrier_frequency_hz = nco_update.frequency_hz
            carrier_phase_rad = fixed_nco.phase_rad
            quantization_error_hz = nco_update.quantization_error_hz
            carrier_fcw_history[epoch] = nco_update.fcw
            carrier_phase_word_history[epoch] = nco_update.phase_word

        dll_frequency_correction_hz += cfg.dll_ki_hz * dll_error[epoch]
        code_rate_hz = (
            nominal_code_rate_hz
            * (1.0 + carrier_frequency_hz / carrier_hz)
            + dll_frequency_correction_hz
        )
        code_phase_chips = float(
            np.mod(
                code_phase_chips
                + samples_per_epoch * code_rate_hz / fs_hz
                + cfg.dll_kp_chips * dll_error[epoch],
                1023.0,
            )
        )

        carrier_frequency_command_history[epoch] = (
            carrier_frequency_command_hz
        )
        carrier_frequency_history[epoch] = carrier_frequency_hz
        carrier_frequency_quantization_error_history[epoch] = (
            quantization_error_hz
        )
        code_rate_history[epoch] = code_rate_hz
        code_phase_history[epoch] = code_phase_chips
        previous_prompt = prompt[epoch]

        window_first = max(0, epoch - cfg.cn0_window_epochs + 1)
        prompt_window = prompt[window_first : epoch + 1]
        phase_lock_metric[epoch] = abs(np.sum(prompt_window)) / (
            np.sum(np.abs(prompt_window)) + eps
        )
        cn0_db_hz[epoch] = estimate_cn0_db_hz(
            prompt_window,
            epoch_duration_s,
        )

    result = {
        "success": True,
        "status": "tracking",
        "first_epoch_sample": first_epoch,
        "epoch_start_samples": epoch_start_samples,
        "early": early,
        "prompt": prompt,
        "late": late,
        "dll_error": dll_error,
        "pll_error_rad": pll_error_rad,
        "fll_error_hz": fll_error_hz,
        "carrier_nco_mode": cfg.carrier_nco_mode,
        "carrier_frequency_command_hz": carrier_frequency_command_history,
        "carrier_frequency_hz": carrier_frequency_history,
        "carrier_frequency_quantization_error_hz": (
            carrier_frequency_quantization_error_history
        ),
        "carrier_nco_frequency_lsb_hz": frequency_lsb_hz,
        "code_rate_hz": code_rate_history,
        "code_phase_chips": code_phase_history,
        "prompt_magnitude": np.abs(prompt),
        "phase_lock_metric": phase_lock_metric,
        "cn0_db_hz": cn0_db_hz,
    }
    if fixed_nco is not None:
        result["carrier_fcw"] = carrier_fcw_history
        result["carrier_phase_word"] = carrier_phase_word_history
    return result
