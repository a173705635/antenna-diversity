"""One-millisecond Early/Prompt/Late DLL with FLL-assisted PLL."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from config import TrackingConfig
from gnss import ca_code
from code_replica import bandlimited_code_replica, _sample_code
from models import AcquisitionResult, TrackingResult, TrackingInitialization


@dataclass
class _TrackingState:
    carrier_frequency_hz: float
    carrier_phase_rad: float
    code_phase_chips: float
    code_rate_hz: float
    dll_frequency_correction_hz: float = 0.0
    previous_prompt: complex | None = None


def _wrap_phase(phase_rad: float) -> float:
    return np.mod(phase_rad + np.pi, 2.0 * np.pi) - np.pi


def _refine_initial_phase(
    block: np.ndarray, code: np.ndarray, state: _TrackingState, fs_hz: float,
) -> TrackingInitialization:
    """Fit the received correlation peak, not necessarily the LOS in multipath.

    Uses the first tracking block before replaying that block through the loops.
    Frequency comes from acquisition; no truth or known path delay is supplied.
    """
    n = np.arange(block.size)
    wiped = block * np.exp(-2j * np.pi * state.carrier_frequency_hz * n / fs_hz)
    phase = n * state.code_rate_hz / fs_hz

    def correlation(offset):
        return np.sum(wiped * _sample_code(code, phase + offset))

    def score(offset):
        # Fit an unknown complex amplitude: remove phase-dependent replica
        # energy so table interpolation cannot move the estimated peak.
        replica = _sample_code(code, phase + offset)
        energy = float(np.vdot(replica, replica).real)
        return abs(np.sum(wiped * replica))**2 / energy

    grid = np.linspace(-0.5, 0.5, 65)
    powers = np.array([score(offset) for offset in grid])
    best = int(np.argmax(powers))
    if not np.isfinite(powers[best]) or powers[best] <= 0:
        raise ValueError("initial phase refinement requires nonzero finite correlation")
    left, right = grid[max(0, best-1)], grid[min(grid.size-1, best+1)]
    scale = powers[best]
    result = minimize_scalar(lambda offset: -score(offset) / scale,
                             bounds=(left, right), method="bounded",
                             options={"xatol": 1e-8})
    # Retain grid endpoints too: a boundary peak is not an interior solution.
    choices = [grid[best], float(result.x)]
    offset = max(choices, key=score)
    prompt = correlation(offset)
    state.code_phase_chips = float(np.mod(offset, 1023.))
    state.carrier_phase_rad = float(np.angle(prompt))
    return {"code_phase_offset_chips": float(offset),
            "carrier_phase_rad": state.carrier_phase_rad,
            "at_search_boundary": bool(abs(offset) >= .5 - 1e-6),
            "samples_used": int(block.size), "uses_truth": False}


def estimate_cn0_db_hz(
    prompt_window: np.ndarray,
    coherent_interval_s: float,
) -> float:
    if len(prompt_window) < 2:
        return np.nan
    coherent_power = abs(np.mean(prompt_window)) ** 2
    total_power = np.mean(np.abs(prompt_window) ** 2)
    noise_power = total_power - coherent_power
    return 10.0 * np.log10(
        coherent_power / (noise_power * coherent_interval_s)
    )


def _correlate_epoch(
    block: np.ndarray,
    sample_index: np.ndarray,
    code: np.ndarray,
    state: _TrackingState,
    fs_hz: float,
    early_late_spacing_chips: float,
) -> tuple[complex, complex, complex]:
    carrier_phase = state.carrier_phase_rad + (
        2.0
        * np.pi
        * state.carrier_frequency_hz
        * sample_index
        / fs_hz
    )
    baseband = block * np.exp(-1j * carrier_phase)
    prompt_phase = (
        state.code_phase_chips
        + sample_index * state.code_rate_hz / fs_hz
    )
    half_spacing = early_late_spacing_chips / 2.0
    early = np.sum(
        baseband * _sample_code(code, prompt_phase + half_spacing)
    )
    prompt = np.sum(baseband * _sample_code(code, prompt_phase))
    late = np.sum(
        baseband * _sample_code(code, prompt_phase - half_spacing)
    )
    return early, prompt, late


def _update_tracking_loops(
    state: _TrackingState,
    prompt: complex,
    dll_error: float,
    pll_error_rad: float,
    fll_error_hz: float,
    epoch: int,
    epoch_duration_s: float,
    samples_per_epoch: int,
    fs_hz: float,
    cfg: TrackingConfig,
    carrier_hz: float,
    nominal_code_rate_hz: float,
) -> None:
    fll_correction = 0.0
    if 0 < epoch < cfg.fll_assist_epochs:
        fll_correction = cfg.fll_gain * fll_error_hz
    state.carrier_frequency_hz += (
        fll_correction + cfg.pll_ki_hz * pll_error_rad
    )
    state.carrier_phase_rad = _wrap_phase(
        state.carrier_phase_rad
        + 2.0 * np.pi * state.carrier_frequency_hz * epoch_duration_s
        + cfg.pll_kp * pll_error_rad
    )
    state.dll_frequency_correction_hz += cfg.dll_ki_hz * dll_error
    state.code_rate_hz = nominal_code_rate_hz * (
        1.0 + state.carrier_frequency_hz / carrier_hz
    ) + state.dll_frequency_correction_hz
    state.code_phase_chips = np.mod(
        state.code_phase_chips
        + samples_per_epoch * state.code_rate_hz / fs_hz
        + cfg.dll_kp_chips * dll_error,
        1023.0,
    )
    state.previous_prompt = prompt


def track(
    x: np.ndarray,
    fs_hz: float,
    prn: int,
    acquisition: AcquisitionResult,
    cfg: TrackingConfig,
    carrier_hz: float,
    nominal_code_rate_hz: float,
) -> TrackingResult:
    """Track residual carrier and C/A-code phase after acquisition."""

    samples_per_epoch = round(fs_hz * cfg.coherent_ms * 1e-3)
    acquisition_samples = round(cfg.acquisition_ms * 1e-3 * fs_hz)
    samples_per_ms = round(fs_hz * 1e-3)
    first_epoch = int(acquisition["code_phase_samples"])
    while first_epoch < acquisition_samples:
        first_epoch += samples_per_ms
    available_epochs = (len(x) - first_epoch) // samples_per_epoch
    n_epochs = min(available_epochs, cfg.max_epochs)

    code = ca_code(prn)
    if cfg.code_replica_bandwidth_hz is not None:
        if cfg.code_replica_bandwidth_hz >= fs_hz / 2:
            raise ValueError("local replica bandwidth must be below receiver Nyquist")
        code = bandlimited_code_replica(
            code, nominal_code_rate_hz, cfg.code_replica_bandwidth_hz
        )
    sample_index = np.arange(samples_per_epoch)
    epoch_duration_s = samples_per_epoch / fs_hz
    carrier_frequency_hz = float(acquisition["doppler_hz"])
    state = _TrackingState(
        carrier_frequency_hz=carrier_frequency_hz,
        carrier_phase_rad=0.0,
        code_phase_chips=0.0,
        code_rate_hz=nominal_code_rate_hz
        * (1.0 + carrier_frequency_hz / carrier_hz),
    )
    initialization = None
    if cfg.refine_initial_phase:
        if cfg.code_replica_bandwidth_hz is None:
            raise ValueError("phase refinement requires a continuous local code replica")
        if n_epochs < 1:
            raise ValueError("phase refinement requires a complete tracking epoch")
        initialization = _refine_initial_phase(
            x[first_epoch:first_epoch + samples_per_epoch], code, state, fs_hz
        )

    early = np.zeros(n_epochs, dtype=complex)
    prompt = np.zeros(n_epochs, dtype=complex)
    late = np.zeros(n_epochs, dtype=complex)
    dll_error = np.zeros(n_epochs)
    pll_error_rad = np.zeros(n_epochs)
    fll_error_hz = np.zeros(n_epochs)
    carrier_history = np.zeros(n_epochs)
    code_rate_history = np.zeros(n_epochs)
    code_phase_history = np.zeros(n_epochs)
    phase_lock_metric = np.zeros(n_epochs)
    cn0_db_hz = np.full(n_epochs, np.nan)
    epoch_start_samples = np.zeros(n_epochs, dtype=int)

    for epoch in range(n_epochs):
        first = first_epoch + epoch * samples_per_epoch
        block = x[first : first + samples_per_epoch]
        epoch_start_samples[epoch] = first
        early[epoch], prompt[epoch], late[epoch] = _correlate_epoch(
            block,
            sample_index,
            code,
            state,
            fs_hz,
            cfg.early_late_spacing_chips,
        )

        dll_error[epoch] = (abs(early[epoch]) - abs(late[epoch])) / (
            abs(early[epoch]) + abs(late[epoch])
        )
        pll_error_rad[epoch] = np.angle(prompt[epoch])
        if state.previous_prompt is not None:
            fll_error_hz[epoch] = np.angle(
                prompt[epoch] * np.conj(state.previous_prompt)
            ) / (2.0 * np.pi * epoch_duration_s)

        _update_tracking_loops(
            state,
            prompt[epoch],
            dll_error[epoch],
            pll_error_rad[epoch],
            fll_error_hz[epoch],
            epoch,
            epoch_duration_s,
            samples_per_epoch,
            fs_hz,
            cfg,
            carrier_hz,
            nominal_code_rate_hz,
        )
        carrier_history[epoch] = state.carrier_frequency_hz
        code_rate_history[epoch] = state.code_rate_hz
        code_phase_history[epoch] = state.code_phase_chips
        window_first = max(0, epoch - cfg.cn0_window_epochs + 1)
        prompt_window = prompt[window_first : epoch + 1]
        phase_lock_metric[epoch] = abs(np.sum(prompt_window)) / np.sum(
            np.abs(prompt_window)
        )
        cn0_db_hz[epoch] = estimate_cn0_db_hz(
            prompt_window,
            epoch_duration_s,
        )

    result: TrackingResult = {
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
        "carrier_frequency_hz": carrier_history,
        "code_rate_hz": code_rate_history,
        "code_phase_chips": code_phase_history,
        "prompt_magnitude": np.abs(prompt),
        "phase_lock_metric": phase_lock_metric,
        "cn0_db_hz": cn0_db_hz,
    }
    if initialization is not None:
        result["initialization"] = initialization
    return result
