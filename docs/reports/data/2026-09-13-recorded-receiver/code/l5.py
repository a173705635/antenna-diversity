"""GPS L5 Q5 pilot, independently implemented from IS-GPS-705J 3.3.2.2/Table 3-I.

ponytail: PRNs 1..32 only; extend the published assignment table when needed.
Q5-only research source, not a complete I5/CNAV transmitter.
"""
from functools import lru_cache
import numpy as np

Q5_ADVANCE = (1701, 323, 5292, 2020, 5429, 7136, 1041, 5947,
              4315, 148, 535, 1939, 5206, 5910, 3595, 5135,
              6082, 6990, 3546, 1523, 4548, 4484, 1893, 3961,
              7106, 5299, 4660, 276, 4389, 3783, 1591, 1601)
NH20 = np.array([1-2*int(b) for b in '00000100110101001110'])


@lru_cache(maxsize=32)
def l5q_code(prn):
    if not isinstance(prn, (int, np.integer)) or not 1 <= prn <= 32:
        raise ValueError('L5 Q5 supports integer PRNs 1..32')
    def cycle(taps, length):
        register = np.ones(13, dtype=np.uint8)
        output = np.empty(length, dtype=np.uint8)
        for k in range(length):
            output[k] = register[-1]
            feedback = np.bitwise_xor.reduce(register[np.array(taps)-1])
            register[1:] = register[:-1]
            register[0] = feedback
        return output
    xa = cycle((9, 10, 12, 13), 8190)
    xb = cycle((1, 3, 4, 6, 7, 8, 12, 13), 8191)
    n = np.arange(10230)
    code = 1.-2.*(xa[n % 8190] ^ xb[(n+Q5_ADVANCE[prn-1]) % 8191])
    code.flags.writeable = False
    return code


def synchronize_nh(prompts):
    """Search NH20 phase and acquisition-bin residual frequency from 40 ms.

    Unknown carrier phase is marginalized by magnitude. No transmit-time truth.
    Threshold is a basic ambiguity rejection, not a calibrated false-alarm rate.
    """
    prompts = np.asarray(prompts)
    if prompts.shape != (40,) or not np.isfinite(prompts).all() or not np.any(prompts):
        raise ValueError('NH synchronization requires 40 finite nonzero prompts')
    # Transition-safe acquisition uses 250-Hz bins. Restrict residual search
    # to that bin: NH20 has a strong offset-10 / 500-Hz sidelobe.
    frequencies = np.arange(-125., 125.01, 2.5)
    n = np.arange(40)
    wipe = np.exp(-2j*np.pi*frequencies[:, None]*n*.001)
    symbols = NH20[(n[None, :]+np.arange(20)[:, None]) % 20]
    correlation = (symbols*prompts) @ wipe.T
    power = abs(correlation)**2
    offset, bin_index = np.unravel_index(power.argmax(), power.shape)
    competitors = power.max(axis=1)
    competitors[offset] = 0
    metric = power[offset, bin_index]/max(competitors.max(), np.finfo(float).tiny)
    if metric < 1.5:
        raise ValueError('ambiguous NH synchronization')
    return {'offset': int(offset), 'frequency_hz': float(frequencies[bin_index]),
            'phase_rad': float(np.angle(correlation[offset, bin_index])),
            'metric': float(metric)}
