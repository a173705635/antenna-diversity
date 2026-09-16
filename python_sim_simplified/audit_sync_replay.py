"""Reproducible W2 timing sensitivity scan; assumed parameters, not hardware data."""
import argparse
import json
from pathlib import Path

import numpy as np

from sync_replay import replay, difference_errors
from stream_replay import source, consume


def audit_stream():
    cases = []
    for channels in (2, 3, 4):
        for fault, step, reset in [('drop_sample', 1, 0.), ('repeat_sample', -1, 0.),
                                   ('repeat_block', -100, 0.), ('half_cycle', 0, np.pi),
                                   ('full_cycle', 0, 2*np.pi)]:
            for channel in (0, channels-1):
                for reported in (True, False):
                    obs, truth = source(channels=channels, sample_step=step,
                                        reset_rad=reset, reported=reported, fault_channel=channel)
                    rows = consume(obs, wavelength_m=299792458/1575.42e6,
                                   reference_epochs={0, 12})
                    affected = [r for r in rows if channel == 0 or r['channel'] == channel]
                    at_fault = [r for r in affected if r['epoch'] == truth['event_epoch']]
                    before_admission = [r for r in affected if 8 <= r['epoch'] < 12]
                    after = [r for r in affected if r['epoch'] >= 12]
                    cases.append(dict(channels=channels, fault=fault, fault_channel=channel,
                                      reported=reported,
                                      old_reference_revoked_at_event=all(r['residual_m'] is None for r in at_fault),
                                      usable_before_new_admission=sum(r['residual_m'] is not None for r in before_admission),
                                      usable_after_new_admission=sum(r['residual_m'] is not None for r in after),
                                      rows=rows))
    return dict(scope='synthetic counter/reset evidence to SDCP consumer; not GNSS reacquisition',
                sample_rate_hz=100000., block_samples=100, epochs=20,
                event_epoch=8, external_reference_epochs=[0, 12], cases=cases,
                reported_cases_revoked=sum(c['reported'] and c['old_reference_revoked_at_event'] for c in cases),
                hidden_cases_still_usable=sum(not c['reported'] and not c['old_reference_revoked_at_event'] for c in cases))


def audit():
    carrier, intermediate = 1575.42e6, 1e6
    wavelength = 299792458/carrier
    times = np.arange(1001)/100.
    doppler = np.array([-4000., -2500., -1000., 0., 700., 1800., 3000., 4500.])
    rows = []
    for count in (2, 3, 4):
        for fault, amplitudes in (
                ('rf_delay_s', [1e-11, 1e-10, 1e-9]),
                ('adc_skew_s', [1e-9, 1e-8, 1e-7]),
                ('sample_rate_error', [1e-9, 1e-8, 1e-7]),
                ('nco_offset_s', [1e-9, 1e-8, 1e-7]),
                ('label_offset_s', [1e-5, 1e-4, 1e-3])):
            for amplitude in amplitudes:
                for pattern in ('common', 'relative'):
                    parameters = {key: np.zeros(count) for key in
                                  ('rf_delay_s', 'adc_skew_s', 'sample_rate_error',
                                   'nco_offset_s', 'label_offset_s')}
                    parameters[fault][:] = amplitude
                    if pattern == 'relative':
                        parameters[fault][0] = 0.
                    result = replay(times, doppler, carrier_hz=carrier,
                                    if_hz=intermediate, **parameters)
                    sd, dd = difference_errors(result['phasor'], result['clean_phasor'], wavelength)
                    aligned = result['label_aligned_error_m']
                    aligned_sd = aligned[:, 1:]-aligned[:, :1]
                    aligned_dd = aligned_sd[..., 1:]-aligned_sd[..., :1]
                    rows.append(dict(channels=count, fault=fault, amplitude=amplitude,
                                     pattern=pattern,
                                     max_wrapped_sd_m=float(abs(sd).max()),
                                     max_wrapped_dd_m=float(abs(dd).max()),
                                     max_unwrapped_error_m=float(abs(result['unwrapped_error_m']).max()),
                                     max_label_aligned_dd_m=float(abs(aligned_dd).max())))
    return dict(scope='W2 narrowband timing sensitivity; no fault detection or navigation claim',
                carrier_hz=carrier, if_hz=intermediate, doppler_hz=doppler.tolist(),
                duration_s=10., epochs=len(times), seed='none; deterministic tones',
                channel_pattern='common: all channels; relative: all except channel 0',
                cases=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stream', action='store_true', help='counter and reset lifecycle instead of timing scan')
    args = parser.parse_args()
    result = audit_stream() if args.stream else audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    print(json.dumps({'cases': len(result['cases']), 'output': str(args.output)}))
