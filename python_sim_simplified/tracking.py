"""One-millisecond Early/Prompt/Late DLL with FLL-assisted PLL."""

import numpy as np

from gnss import ca_code


def _sample_code(code, phase_chips):
    return code[np.floor(np.mod(phase_chips, 1023.0)).astype(int)]


def _wrap_phase(phase_rad):
    return np.mod(phase_rad + np.pi, 2.0 * np.pi) - np.pi


def estimate_cn0_db_hz(prompt_window, coherent_interval_s):
    if len(prompt_window) < 2:
        return np.nan
    coherent_power = abs(np.mean(prompt_window)) ** 2
    total_power = np.mean(np.abs(prompt_window) ** 2)
    noise_power = total_power - coherent_power
    return 10.0 * np.log10(
        coherent_power / (noise_power * coherent_interval_s)
    )


def track(
    x,
    fs_hz,
    prn,
    acquisition,
    cfg,
    carrier_hz,
    nominal_code_rate_hz,
):
    samples_per_epoch = round(fs_hz * cfg.coherent_ms * 1e-3)
    acquisition_samples = round(cfg.acquisition_ms * 1e-3 * fs_hz)
    samples_per_ms = round(fs_hz * 1e-3)
    first_epoch = int(acquisition["code_phase_samples"])
    while first_epoch < acquisition_samples:
        first_epoch += samples_per_ms
    available_epochs = (len(x) - first_epoch) // samples_per_epoch
    n_epochs = min(available_epochs, cfg.max_epochs)

    code = ca_code(prn)
    sample_index = np.arange(samples_per_epoch)
    epoch_duration_s = samples_per_epoch / fs_hz
    carrier_frequency_hz = float(acquisition["doppler_hz"])
    carrier_phase_rad = 0.0
    code_phase_chips = 0.0
    dll_frequency_correction_hz = 0.0
    code_rate_hz = nominal_code_rate_hz * (
        1.0 + carrier_frequency_hz / carrier_hz
    )
    previous_prompt = None

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
        carrier_phase = carrier_phase_rad + (
            2.0
            * np.pi
            * carrier_frequency_hz
            * sample_index
            / fs_hz
        )
        baseband = block * np.exp(-1j * carrier_phase)

        prompt_phase = code_phase_chips + sample_index * code_rate_hz / fs_hz
        half_spacing = cfg.early_late_spacing_chips / 2.0
        early[epoch] = np.sum(
            baseband * _sample_code(code, prompt_phase + half_spacing)
        )
        prompt[epoch] = np.sum(baseband * _sample_code(code, prompt_phase))
        late[epoch] = np.sum(
            baseband * _sample_code(code, prompt_phase - half_spacing)
        )

        dll_error[epoch] = (abs(early[epoch]) - abs(late[epoch])) / (
            abs(early[epoch]) + abs(late[epoch])
        )
        pll_error_rad[epoch] = np.angle(prompt[epoch])
        if previous_prompt is not None:
            fll_error_hz[epoch] = np.angle(
                prompt[epoch] * np.conj(previous_prompt)
            ) / (2.0 * np.pi * epoch_duration_s)

        fll_correction = 0.0
        if 0 < epoch < cfg.fll_assist_epochs:
            fll_correction = cfg.fll_gain * fll_error_hz[epoch]
        carrier_frequency_hz += (
            fll_correction + cfg.pll_ki_hz * pll_error_rad[epoch]
        )
        carrier_phase_rad = _wrap_phase(
            carrier_phase_rad
            + 2.0 * np.pi * carrier_frequency_hz * epoch_duration_s
            + cfg.pll_kp * pll_error_rad[epoch]
        )
        dll_frequency_correction_hz += cfg.dll_ki_hz * dll_error[epoch]
        code_rate_hz = nominal_code_rate_hz * (
            1.0 + carrier_frequency_hz / carrier_hz
        ) + dll_frequency_correction_hz
        code_phase_chips = np.mod(
            code_phase_chips
            + samples_per_epoch * code_rate_hz / fs_hz
            + cfg.dll_kp_chips * dll_error[epoch],
            1023.0,
        )

        carrier_history[epoch] = carrier_frequency_hz
        code_rate_history[epoch] = code_rate_hz
        code_phase_history[epoch] = code_phase_chips
        previous_prompt = prompt[epoch]
        window_first = max(0, epoch - cfg.cn0_window_epochs + 1)
        prompt_window = prompt[window_first : epoch + 1]
        phase_lock_metric[epoch] = abs(np.sum(prompt_window)) / np.sum(
            np.abs(prompt_window)
        )
        cn0_db_hz[epoch] = estimate_cn0_db_hz(
            prompt_window, epoch_duration_s
        )

    return {
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
