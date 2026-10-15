"""Reusable offline recorded GPS L1 receiver, independent of audit scripts.

Experiments supply complete interleaved IQ byte pairs and an in-range sample window.
"""
from pathlib import Path
import hashlib
import numpy as np
from .config import AcquisitionConfig, DFEConfig, TrackingConfig
from .dfe import process_dfe_blocks, design_decimator
from .acquisition import acquire
from .tracking import track
from .gps_lnav import decode_lnav
from .gps_observables import transmit_time, solve_gps_epoch
from .positioning import PositioningFailure
from .positioning import carrier_range
from .config import SPEED_OF_LIGHT_MPS as C


def read_iq8(iq_path, start_sample, count):
    raw = np.fromfile(iq_path, dtype=np.int8, offset=start_sample*2, count=count*2)
    return raw[::2].astype(np.float32) + 1j*raw[1::2].astype(np.float32)


def prepare_recording(iq_path, cache_path, *, start_sample, sample_count,
                      fs_hz, if_hz, dfe_config, block_samples=1000000):
    """Read once into a disk-backed DFE output; input sample units stay explicit."""
    iq_path = Path(iq_path)
    cache_path = Path(cache_path)
    if cache_path.exists():
        raise ValueError('refusing to overwrite an existing DFE cache')
    filter_coefficients, info = design_decimator(
        fs_hz, dfe_config.decimation, dfe_config.cutoff_hz, dfe_config.num_taps
    )
    delay = info['group_delay_input_samples']
    count = len(range(delay, sample_count, dfe_config.decimation))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(cache_path, dtype='<c16', mode='w+', shape=(1, count))
    digest = hashlib.sha256()
    sum_i = sum_q = sum_power = 0.
    rails = 0
    def blocks():
        nonlocal sum_i, sum_q, sum_power, rails
        with iq_path.open('rb') as f:
            f.seek(start_sample * 2)
            remaining = sample_count
            while remaining:
                n = min(block_samples, remaining)
                data = f.read(n * 2)
                digest.update(data)
                raw = np.frombuffer(data, dtype='i1')
                i = raw[::2].astype(float)
                q = raw[1::2].astype(float)
                sum_i += i.sum()
                sum_q += q.sum()
                sum_power += np.sum(i * i + q * q)
                rails += int(np.sum((i == -128) | (i == 127) | (q == -128) | (q == 127)))
                # 累计统计量使用原始 IQ 码值，DFE 处理同一批连续复采样。
                yield (i + 1j * q)[None, :]
                remaining -= n

    index = 0
    for block in process_dfe_blocks(blocks(), dfe_config, fs_hz, if_hz):
        out[:, index:index + block.shape[1]] = block
        index += block.shape[1]
    assert index == count
    out.flush()
    info.update({
        'if_hz': if_hz if dfe_config.nco_frequency_hz is None else dfe_config.nco_frequency_hz,
        'output_bits': dfe_config.output_bits,
        'output_length': count,
        'filter_coefficients': filter_coefficients,
        'input_window_sha256': digest.hexdigest(),
        'input_start_sample': start_sample,
        'input_samples': sample_count,
        'input_stats': {
            'mean_i': sum_i / sample_count,
            'mean_q': sum_q / sample_count,
            'rms': np.sqrt(sum_power / sample_count),
            'rail_fraction': rails / sample_count,
        },
    })
    return out, info


def receive_channels(baseband, fs_hz, prns, acquisition_config, tracking_config, *, acquisition_only_flag=False):
    """Yield independent GPS L1 channels from one common baseband stream."""
    for prn in prns:
        acquisition_result = acquire(baseband, fs_hz, prn, 1.023e6, acquisition_config)
        row = {
            'prn': prn,
            'acquisition': {key: value for key, value in acquisition_result.items()
                            if key not in ('search_power', 'doppler_bins_hz')},
        }
        if acquisition_result['success_flag'] and not acquisition_only_flag:
            row['history'] = track(
                baseband, fs_hz, prn, acquisition_result, tracking_config,
                1575.42e6, 1.023e6,
            )
        yield row


def position_channels(channels, fs_hz, times_s, *, common_travel_s=.075):
    """Offline GPS code PVT, using broadcast data decoded from these channels.

    This is retrospective processing of one continuous arc; ephemeris decoded
    near its end can be used earlier. No receiver reference position is input.
    """
    times = np.asarray(times_s, float)
    if times.size == 0:
        return {'status':'not_evaluated_no_observation_times', 'navigation':{}, 'epochs':[]}
    usable = {}
    navigation = {}
    carriers = {}
    for prn, tracking_history in channels.items():
        nav = decode_lnav(tracking_history, fs_hz)
        navigation[prn] = nav
        eph = nav['ephemeris']
        if eph is not None and eph['health'] == 0 and not eph['tgd_unavailable_flag']:
            usable[prn] = (eph, transmit_time(tracking_history, nav['time_anchors'][0], times, fs_hz))
            carriers[prn] = carrier_range(tracking_history, times, fs_hz, .001, C / 1575.42e6)
    if len(usable) < 4:
        return {'status': 'insufficient_decoded_ephemerides', 'navigation': navigation, 'epochs': []}
    first = [tx[0] for _, tx in usable.values() if np.isfinite(tx[0])]
    if len(first) < 4:
        return {'status': 'insufficient_first_epoch', 'navigation': navigation, 'epochs': []}
    origin = max(first) + common_travel_s
    epochs = []
    for k, t in enumerate(times):
        prns = []
        ephemerides = []
        txs = []
        for prn, (eph, tx) in usable.items():
            tracking_history = channels[prn]
            i = np.searchsorted(tracking_history['epoch_start_samples'] / fs_hz, t, side='right') - 1
            if i < 0 or not np.isfinite(tx[k]):
                continue
            if tracking_history['phase_lock_metric'][i] < .8 or abs(tracking_history['pll_error_rad'][i]) > .7:
                continue
            prns.append(prn)
            ephemerides.append(eph)
            txs.append(tx[k])
        if len(prns) < 4:
            epochs.append({'sample_time_s': float(t), 'status': 'insufficient_tracking_quality'})
            continue
        try:
            solution = solve_gps_epoch(ephemerides, txs, origin + t - times[0])
        except PositioningFailure as e:
            epochs.append({'sample_time_s': float(t), 'status': str(e)})
            continue
        epochs.append({'sample_time_s': float(t), 'status': 'position', 'prns': prns,
                       'transmit_tow_s': txs,
                       'carrier_range_m': [float(carriers[prn][k]) for prn in prns],
                       'carrier_half_cycle_resolved_flag': False, **solution})
    return {'status':'evaluated','navigation':navigation,'epochs':epochs,
            'common_clock_origin_tow_s':origin,'common_clock_origin_sample_s':float(times[0]),
            'mode':'offline GPS L1 code SPP; decoded clocks/TGD, Sagnac; no atmosphere correction'}
