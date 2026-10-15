import numpy as np
from inspect import signature
from scipy.linalg import solve_discrete_lyapunov

from core.acquisition import acquire
from core.config import AcquisitionConfig, TrackingConfig
from core.gnss import ca_code
from core.tracking import (
    _TrackingState, _update_tracking_loops, estimate_cn0_db_hz, track,
)


def receiver_signal(
    fs_hz=4.092e6,
    duration_ms=30,
    doppler_hz=1250.0,
    delay_chips=175.0,
):
    n = np.arange(round(fs_hz * duration_ms * 1e-3))
    phase = np.mod(n * 1.023e6 / fs_hz - delay_chips, 1023.0)
    code = ca_code(1)[np.floor(phase).astype(int)]
    clean = code * np.exp(1j * 2.0 * np.pi * doppler_hz * n / fs_hz)
    rng = np.random.default_rng(11)
    noise = 0.2 * (
        rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size)
    )
    return clean + noise


def test_acquisition_and_tracking_follow_valid_signal():
    fs_hz = 4.092e6
    samples = receiver_signal(fs_hz)

    acquisition = acquire(samples, fs_hz, 1, 1.023e6, AcquisitionConfig())
    tracking = track(
        samples,
        fs_hz,
        1,
        acquisition,
        TrackingConfig(max_epochs=20),
        1_575.42e6,
        1.023e6,
    )

    assert acquisition["success"]
    assert abs(acquisition["doppler_hz"] - 1250.0) <= 250.0
    assert tracking["success"]
    assert tracking["prompt"].size > 5
    assert np.isfinite(tracking["cn0_db_hz"][-1])
    assert set(tracking) == {
        "success",
        "status",
        "first_epoch_sample",
        "epoch_start_samples",
        "early",
        "prompt",
        "late",
        "dll_error",
        "pll_error_rad",
        "fll_error_hz",
        "carrier_frequency_hz",
        "carrier_nco_phase_rad",
        "carrier_nco_frequency_hz",
        "code_nco_phase_chips",
        "code_nco_rate_hz",
        "code_rate_hz",
        "code_phase_chips",
        "prompt_magnitude",
        "phase_lock_metric",
        "cn0_db_hz",
    }


def test_cn0_matches_coherent_to_noise_power_ratio():
    prompt = np.array([10 + 1j, 10 - 1j, 10 + 1j, 10 - 1j])
    expected = 10.0 * np.log10(100.0 / (1.0 * 0.001))

    assert np.isclose(estimate_cn0_db_hz(prompt, 0.001), expected)


PLL25 = (.06315066862415342, .32776091874805263)


def test_tracking_quality_uses_standard_loss_rule():
    assert 'loss_quality' not in signature(track).parameters


def closed_loop(offset_hz, rate_hz_s, fll_gain=.5, pll=(.15, 1.),
                epochs=1000, step_hz=0.):
    cfg = TrackingConfig(pll_kp=pll[0], pll_ki_hz=pll[1], fll_gain=fll_gain)
    state = _TrackingState(0., 0., 0., 1.023e6)
    truth_phase = 0.
    previous = None
    frequencies = []
    errors = []
    for epoch in range(epochs):
        true_frequency = offset_hz + rate_hz_s * epoch * .001
        if epoch >= 500:
            true_frequency += step_hz
        truth_phase += 2 * np.pi * true_frequency * .001
        prompt = np.exp(1j * (truth_phase - state.carrier_phase_rad))
        phase_error = .5 * np.angle(prompt**2)
        fll_error = (0. if previous is None else
                     .5 * np.angle((prompt * np.conj(previous))**2) / (2 * np.pi * .001))
        _update_tracking_loops(state, prompt, 0., phase_error, fll_error, epoch,
                               .001, 2046, 2.046e6, cfg, 1575.42e6, 1.023e6)
        frequencies.append(state.carrier_frequency_hz)
        errors.append(phase_error)
        previous = prompt
    return np.array(frequencies), np.array(errors), state


def test_ideal_prompt_loop_converges_with_fll_start_and_fixed_offset():
    assisted, errors, state = closed_loop(20., 0.)
    unassisted, _, _ = closed_loop(20., 0., fll_gain=0.)
    assert assisted[1] > unassisted[1]
    assert abs(assisted[-1] - 20.) < .01
    assert max(abs(errors[-100:])) < .01
    assert state.code_rate_hz > 1.023e6


def test_ideal_prompt_loop_tracks_both_100_hz_per_second_slopes():
    for rate in (-100., 100.):
        frequencies, errors, _ = closed_loop(0., rate)
        assert abs(frequencies[-1] - rate * .999) < 3.
        assert max(abs(errors[-100:])) < .11


def test_pll25_discrete_noise_bandwidth_is_25_hz():
    kp, ki = PLL25
    dt = .001
    transition = np.array([[1-kp-2*np.pi*dt*ki, 2*np.pi*dt], [-ki, 1.]])
    noise = np.array([[-kp-2*np.pi*dt*ki], [-ki]])
    covariance = solve_discrete_lyapunov(transition, noise @ noise.T)
    np.testing.assert_allclose(covariance[0, 0] / (2 * dt), 25., rtol=1e-11)


def test_pll25_actual_loop_handles_grid_edge_slopes_and_step():
    for offset in (-125., 125.):
        frequency, error, _ = closed_loop(offset, 0., pll=PLL25, epochs=1500)
        assert abs(frequency[-1] - offset) < .01
        assert max(abs(error[-100:])) < .01
        assert np.sign(frequency[1]) == np.sign(offset)
    for rate in (-100., 100.):
        frequency, error, _ = closed_loop(0., rate, pll=PLL25, epochs=1500)
        assert abs(frequency[-1] - rate * 1.499) < 3.
        assert max(abs(error[-100:])) < .31
    for step in (-40., 40.):
        frequency, error, _ = closed_loop(0., 0., pll=PLL25,
                                        epochs=1500, step_hz=step)
        assert abs(frequency[-1] - step) < .01
        assert max(abs(error[500:650])) < np.pi/2
        assert max(abs(error[-100:])) < .01
