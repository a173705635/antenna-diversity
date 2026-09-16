import numpy as np
import pytest

from stream_replay import source, consume

W = 299792458/1575.42e6


@pytest.mark.parametrize('sample_step', [1, -1, -100])
def test_sample_discontinuity_reaches_consumer_and_reference_does_not_self_restore(sample_step):
    observations, truth = source(channels=4, sample_step=sample_step, reset_rad=0., reported=True)
    rows = consume(observations, wavelength_m=W, reference_epochs={0, 12})
    affected = [r for r in rows if r['channel'] == 3]
    assert affected[7]['state'] == 'reference_usable'
    assert affected[8]['state'] == 'unavailable'
    assert affected[8]['residual_m'] is None
    assert all(r['state'] == 'await_reference' and r['residual_m'] is None for r in affected[9:12])
    assert affected[12]['state'] == 'reference_usable'
    assert affected[12]['arc_id'] != affected[7]['arc_id']
    assert all(r['state'] == 'reference_usable' for r in rows if r['channel'] != 3)
    # Check the waveform, independently of the reported counter/state path.
    clean, _ = source(channels=4, sample_step=0, reset_rad=0., reported=True)
    ratio = observations['phasor'][8, 3]/clean['phasor'][8, 3]
    np.testing.assert_allclose(ratio, np.exp(2j*np.pi*(10000.+truth['doppler_hz'])*sample_step/truth['fs_hz']), atol=2e-13)


def test_hidden_sample_drop_is_not_mislabeled_as_successful_detection():
    observations, _ = source(channels=2, sample_step=1, reset_rad=0., reported=False)
    rows = consume(observations, wavelength_m=W, reference_epochs={0})
    assert all(r['state'] == 'reference_usable' for r in rows)
    assert max(abs(np.array(rows[8]['residual_m']))) > .01
    # The consumer cannot read simulation truth or future epochs.
    prefix = {k: v[:9] if isinstance(v, np.ndarray) else v for k, v in observations.items()}
    assert consume(prefix, wavelength_m=W, reference_epochs={0}) == rows[:9]


def test_reported_nco_reset_changes_arc_even_with_good_quality():
    observations, _ = source(channels=2, sample_step=0, reset_rad=np.pi, reported=True)
    rows = consume(observations, wavelength_m=W, reference_epochs={0})
    assert observations['quality'].all()
    assert rows[8]['state'] == 'await_reference'
    assert rows[8]['arc_id'] != rows[7]['arc_id']
    assert all(r['residual_m'] is None for r in rows[8:])


def test_hidden_full_cycle_reset_is_invisible_in_complex_phase():
    observations, _ = source(channels=2, sample_step=0, reset_rad=2*np.pi, reported=False)
    rows = consume(observations, wavelength_m=W, reference_epochs={0})
    assert all(r['state'] == 'reference_usable' for r in rows)
    np.testing.assert_allclose([r['residual_m'] for r in rows], 0., atol=1e-14)


def test_reference_channel_discontinuity_revokes_all_dependent_pairs():
    observations, _ = source(channels=4, sample_step=1, reset_rad=0., reported=True,
                             fault_channel=0)
    rows = consume(observations, wavelength_m=W, reference_epochs={0})
    assert all(r['residual_m'] is None for r in rows if r['epoch'] >= 8)
    assert {r['channel'] for r in rows if r['epoch'] == 8} == {1, 2, 3}


def test_clean_stream_remains_usable_without_repeated_admissions():
    observations, _ = source(channels=4, sample_step=0, reset_rad=0., reported=True)
    rows = consume(observations, wavelength_m=W, reference_epochs={0})
    assert all(r['state'] == 'reference_usable' for r in rows)
    np.testing.assert_allclose([r['residual_m'] for r in rows], 0., atol=1e-14)
