import numpy as np

from core import single_source
from core.config import AcquisitionConfig, TrackingConfig
from core.tracking import track
from test_autonomous_tracking import samples


def test_fixed_window_quality_uses_100_samples_and_nan_is_bad():
    good = np.full(100, 2.+1j)
    assert np.isnan(single_source.coherent_cn0(good[:1], 100)[0])
    assert single_source._coherent_loss_quality(good, 1., 0.)
    assert not single_source._coherent_loss_quality(np.full(100, 1.+2j), 1., 0.)
    cn0 = single_source.coherent_cn0(good, 100)[-1]
    snr = 10**(cn0/10)*.001
    threshold = .8*(snr/(1+snr))
    assert single_source._coherent_loss_quality(good, threshold, 0.)
    assert not single_source._coherent_loss_quality(
        good, np.nextafter(threshold, 0.), 0.)


def test_online_quality_matches_full_convolution_at_each_completed_epoch():
    rng = np.random.default_rng(43)
    prompt = 2. + .25*rng.normal(size=120) + 1j*(1. + .2*rng.normal(size=120))
    full_cn0 = single_source.coherent_cn0(prompt, 100)
    for k in range(100, len(prompt)):
        cn0 = full_cn0[k]
        snr = 10**(cn0/10)*.001
        threshold = .8*(snr/(1+snr))
        assert single_source._coherent_loss_quality(prompt[:k+1], threshold, 0.)
        assert not single_source._coherent_loss_quality(
            prompt[:k+1], np.nextafter(threshold, 0.), 0.)


def test_callback_reads_only_completed_prefix_and_stops_on_fifth_bad_epoch():
    fs, clean, _ = samples()
    cfg = TrackingConfig(max_epochs=130, cn0_window_epochs=7)
    acq = dict(code_phase_samples=0, doppler_hz=500.)
    seen = []

    def quality(prefix, lock, pll):
        seen.append((len(prefix), prefix.copy(), lock, pll))
        return len(prefix) < 102 or len(prefix) == 106

    stopped = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6,
                    loss_confirm_epochs=5, loss_settle_epochs=100,
                    loss_quality=quality)
    full = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6)
    assert stopped['status'] == 'tracking_quality_lost'
    assert len(stopped['prompt']) == 111
    assert seen[0][0] == 101
    assert seen[-1][0] == 111
    for length, prefix, lock, pll in seen:
        np.testing.assert_array_equal(prefix, full['prompt'][:length])
        assert lock == full['phase_lock_metric'][length-1]
        assert pll == full['pll_error_rad'][length-1]
    for name, value in stopped.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, full[name][:111])


def test_callback_is_dormant_without_loss_gate_and_natural_end_is_preserved():
    fs, clean, _ = samples()
    cfg = TrackingConfig(max_epochs=103)
    acq = dict(code_phase_samples=0, doppler_hz=500.)
    calls = []
    quality = lambda prompt, lock, pll: calls.append(len(prompt)) or False
    default = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6)
    dormant = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6,
                    loss_quality=quality)
    np.testing.assert_array_equal(default['prompt'], dormant['prompt'])
    assert calls == []
    natural = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6,
                    loss_confirm_epochs=5, loss_quality=quality)
    assert natural['status'] == 'tracking'
    assert calls == [101, 102, 103]


def test_early_loss_preserves_prefix_before_future_invalid_correlation():
    fs, clean, _ = samples()
    clean[round(.12*fs):] = 0.
    cfg = TrackingConfig(max_epochs=200)
    acq = dict(code_phase_samples=0, doppler_hz=500.)
    old = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6)
    fast = track(clean, fs, 1, acq, cfg, 1575.42e6, 1.023e6,
                 loss_confirm_epochs=5,
                 loss_quality=lambda prefix, lock, pll: False)
    assert old['status'] == 'invalid_correlation'
    assert fast['status'] == 'tracking_quality_lost'
    assert len(fast['prompt']) == 105
    np.testing.assert_array_equal(fast['prompt'], old['prompt'][:105])


def test_fast_receiver_uses_fixed_100_prompt_window_and_retains_arc_cursor(monkeypatch):
    prompts = np.r_[np.full(100, 2.+1j), np.full(5, 1.+100j), 2.+1j]
    seen = []

    def fake_track(signal, fs, prn, acq, cfg, carrier, code_rate, **kwargs):
        seen.append((len(signal), kwargs))
        start = np.arange(len(prompts))*2046+200_000
        lock = np.ones(len(prompts))
        pll = np.zeros(len(prompts))
        end = len(prompts)
        if kwargs:
            bad = 0
            for k in range(100, len(prompts)):
                good = kwargs['loss_quality'](prompts[:k+1], lock[k], pll[k])
                bad = 0 if good else bad+1
                if bad == 5:
                    end = k+1
                    break
        return dict(success=True, status='tracking_quality_lost' if end < len(prompts)
                    else 'tracking', prompt=prompts[:end],
                    phase_lock_metric=lock[:end], pll_error_rad=pll[:end],
                    epoch_start_samples=start[:end])

    monkeypatch.setattr(single_source, 'track', fake_track)
    monkeypatch.setattr(single_source, 'acquire',
                        lambda *args: dict(success=True))
    signal = np.ones(500_000, complex)
    acquisition = AcquisitionConfig(noncoherent_ms=1, coherent_ms=1)
    cfg = TrackingConfig(cn0_window_epochs=7)
    result = single_source.receive_measured(signal, 1, acquisition, cfg,
                                            fast_truncation=True)
    assert seen[0][1]['loss_confirm_epochs'] == 5
    assert seen[0][1]['loss_settle_epochs'] == 100
    assert len(result['segments']) >= 2
    first = result['segments'][0]['history']
    assert len(first['prompt']) == 105
    assert np.isnan(first['cn0_db_hz'][0])
    assert result['segments'][1]['history']['epoch_start_samples'][0] == (
        first['epoch_start_samples'][-1]+2046+200_000)


def test_default_receiver_keeps_full_track_call(monkeypatch):
    calls = []
    monkeypatch.setattr(single_source, 'acquire', lambda *args: dict(success=True))

    def fake_track(*args, **kwargs):
        calls.append(kwargs)
        n = 105
        return dict(success=True, status='tracking',
                    prompt=np.full(n, 2.+1j), phase_lock_metric=np.ones(n),
                    pll_error_rad=np.zeros(n),
                    epoch_start_samples=np.arange(n)*2046+200_000)

    monkeypatch.setattr(single_source, 'track', fake_track)
    signal = np.ones(500_000, complex)
    result = single_source.receive_measured(
        signal, 1, AcquisitionConfig(noncoherent_ms=1, coherent_ms=1),
        TrackingConfig())
    assert calls == [{}]
    assert len(result['segments'][0]['history']['prompt']) == 105
