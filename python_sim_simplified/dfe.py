"""Common-NCO DDC, FIR decimation, and DFE output quantization."""

import numpy as np
from scipy.signal import lfilter

from config import DFEConfig
from models import DFEInfo


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps):
    delay = (num_taps - 1) // 2
    m = np.arange(-delay, delay + 1)
    h = (2.0 * cutoff_hz / fs_in_hz) * np.sinc(
        2.0 * cutoff_hz / fs_in_hz * m
    )
    h *= np.hamming(num_taps)
    h /= h.sum()
    return h, {
        "fs_in_hz": fs_in_hz,
        "fs_out_hz": fs_in_hz / decimation,
        "decimation": decimation,
        "cutoff_hz": cutoff_hz,
        "num_taps": num_taps,
        "group_delay_input_samples": delay,
    }


def process_dfe(
    adc_codes: np.ndarray,
    cfg: DFEConfig,
    fs_in_hz: float,
    signal_if_hz: float,
) -> tuple[np.ndarray, DFEInfo]:
    nco_frequency_hz = (
        signal_if_hz
        if cfg.nco_frequency_hz is None
        else cfg.nco_frequency_hz
    )
    h, raw_info = design_decimator(
        fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps
    )
    n = np.arange(adc_codes.shape[1])
    nco = np.exp(-1j * 2.0 * np.pi * nco_frequency_hz * n / fs_in_hz)
    mixed = adc_codes * nco[None, :]
    filtered = lfilter(h, [1.0], mixed, axis=1)
    output = filtered[
        :, raw_info["group_delay_input_samples"] :: cfg.decimation
    ]

    output_scale = 1
    if cfg.output_bits > 0:
        output_scale = 2**cfg.output_binary_shift
        minimum = -(2 ** (cfg.output_bits - 1))
        maximum = 2 ** (cfg.output_bits - 1) - 1
        i_code = np.clip(
            _round_half_away_from_zero(output.real / output_scale),
            minimum,
            maximum,
        )
        q_code = np.clip(
            _round_half_away_from_zero(output.imag / output_scale),
            minimum,
            maximum,
        )
        output = (i_code + 1j * q_code).astype(complex)

    info: DFEInfo = {
        "fs_in_hz": raw_info["fs_in_hz"],
        "fs_out_hz": raw_info["fs_out_hz"],
        "decimation": raw_info["decimation"],
        "cutoff_hz": raw_info["cutoff_hz"],
        "num_taps": raw_info["num_taps"],
        "group_delay_input_samples": raw_info["group_delay_input_samples"],
        "if_hz": nco_frequency_hz,
        "output_bits": cfg.output_bits,
        "output_scale": output_scale,
        "output_length": output.shape[1],
        "filter_coefficients": h,
    }
    return output, info


def process_dfe_blocks(blocks, cfg, fs_in_hz, signal_if_hz):
    """Bounded-memory equivalent of process_dfe over contiguous input blocks.

    NCO, FIR state, group-delay removal and decimation phase are continuous.
    A caller must start a new stream when the original capture has a gap.
    """
    h,info=design_decimator(fs_in_hz,cfg.decimation,cfg.cutoff_hz,cfg.num_taps)
    frequency=signal_if_hz if cfg.nco_frequency_hz is None else cfg.nco_frequency_hz
    offset=0;state=None;channels=None
    for block in blocks:
        block=np.asarray(block)
        if block.ndim!=2 or block.shape[1]==0:
            raise ValueError('DFE blocks must be nonempty channel-by-sample arrays')
        if state is None:
            channels=block.shape[0];state=np.zeros((channels,len(h)-1),complex)
        if block.shape[0]!=channels:raise ValueError('DFE channel count changed')
        n=np.arange(offset,offset+block.shape[1])
        mixed=block*np.exp(-1j*2*np.pi*frequency*n/fs_in_hz)[None,:]
        filtered,state=lfilter(h,[1.],mixed,axis=1,zi=state)
        first=max(offset,info['group_delay_input_samples'])
        first+=(info['group_delay_input_samples']-first)%cfg.decimation
        output=filtered[:,first-offset::cfg.decimation]
        offset+=block.shape[1]
        if cfg.output_bits>0:
            scale=2**cfg.output_binary_shift;minimum=-(2**(cfg.output_bits-1));maximum=-minimum-1
            output=(np.clip(_round_half_away_from_zero(output.real/scale),minimum,maximum)+
                    1j*np.clip(_round_half_away_from_zero(output.imag/scale),minimum,maximum))
        if output.shape[1]:yield output
