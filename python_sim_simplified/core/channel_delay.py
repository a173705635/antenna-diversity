"""通道时延：移动中频采样序列，并补齐射频载波对应的相位变化。

时延以非负有限值输入；插值滤波器使用奇数抽头，各通道与时延一一对应。
"""

import numpy as np


DEFAULT_FRACTIONAL_DELAY_TAPS = 33


def fractional_delay(
    signal: np.ndarray,
    delay_samples: float,
    num_taps: int = DEFAULT_FRACTIONAL_DELAY_TAPS,
) -> np.ndarray:
    """延迟一维复采样，delay_samples 单位为采样点；输出等长，记录外按零处理。"""

    samples = np.asarray(signal, dtype=np.complex128)
    if delay_samples == 0.0:
        return samples.copy()

    # 将浮点换算后贴近整数的时延归整，避免整数移位误走插值。
    nearest_integer = round(delay_samples)
    if abs(delay_samples - nearest_integer) <= 1e-12:
        delay_samples = float(nearest_integer)

    integer_delay_samples = int(np.floor(delay_samples))
    fractional_delay_samples = delay_samples - integer_delay_samples
    delayed_samples = np.zeros(samples.shape, dtype=np.complex128)
    if integer_delay_samples >= samples.size:
        return delayed_samples
    if fractional_delay_samples == 0.0:
        delayed_samples[integer_delay_samples:] = samples[: samples.size - integer_delay_samples]
        return delayed_samples

    # 窗化 sinc 实现小数采样点时延；归一化使滤波器的直流增益为 1。
    half_filter_length = num_taps // 2
    tap_offsets_samples = np.arange(-half_filter_length, half_filter_length + 1, dtype=np.float64)
    delay_filter = np.sinc(tap_offsets_samples - fractional_delay_samples) * np.hamming(num_taps)
    delay_filter /= np.sum(delay_filter)
    convolution_samples = np.convolve(samples, delay_filter, mode="full")
    # 去掉滤波器中心偏移，再加入整数时延；开头补零，超出记录的尾部截去。
    interpolated_samples = convolution_samples[half_filter_length : half_filter_length + samples.size]
    delayed_samples[integer_delay_samples:] = interpolated_samples[: samples.size - integer_delay_samples]
    return delayed_samples


def delay_phase(delays_s: np.ndarray, source_rf_frequency_hz: float) -> np.ndarray:
    """返回各时延对应的窄带相位因子 exp(-j2πfτ)，频率单位 Hz，时延单位 s。"""

    return np.exp(
        -1j
        * 2.0
        * np.pi
        * source_rf_frequency_hz
        * np.asarray(delays_s, dtype=np.float64)
    )


def apply_channel_delays(
    input_if_channels: np.ndarray,
    delays_s: np.ndarray,
    fs_hz: float,
    carrier_hz: float,
    if_hz: float,
) -> np.ndarray:
    """对 [通道, 时间采样] 中频信号施加时延，返回等形状复采样。

    先移动序列，再补齐载波与中频之差对应的相位；该补偿属于时延模型，
    不表示接收机校准，也不能替代宽带序列的实际延迟。
    """

    input_if_channels = np.asarray(input_if_channels, dtype=np.complex128)
    channel_delays_s = np.asarray(delays_s, dtype=np.float64)
    if np.all(channel_delays_s == 0.0):
        return input_if_channels.copy()

    delayed_if_channels = np.empty(input_if_channels.shape, dtype=np.complex128)
    # 序列时延已包含中频相位变化，只需补上 carrier_hz - if_hz 的部分。
    carrier_phase_factors = delay_phase(channel_delays_s, carrier_hz - if_hz)
    for channel_index, channel_delay_s in enumerate(channel_delays_s):
        delayed_channel_samples = fractional_delay(
            input_if_channels[channel_index], channel_delay_s * fs_hz,
        )
        delayed_if_channels[channel_index] = delayed_channel_samples * carrier_phase_factors[channel_index]
    return delayed_if_channels
