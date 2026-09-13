"""Analytical examples for the inventory. No RF simulation or hardware data."""
from pathlib import Path
import json
import math

C = 299792458.
F = 1575.42e6
WAVELENGTH = C / F


def calculate():
    pll = []
    for cn0_db in (35., 40., 45.):
        cn0 = 10**(cn0_db/10)
        sigma = WAVELENGTH/(2*math.pi)*math.sqrt(15/cn0*(1+1/(2*.001*cn0)))
        pll.append({'cn0_db_hz': cn0_db, 'bandwidth_hz': 15., 'integration_s': .001,
                    'one_channel_sigma_mm': sigma*1000,
                    'independent_equal_two_channel_sdcp_sigma_mm': math.sqrt(2)*sigma*1000})
    return {
        'kind': 'Analytical calculations with stated hypothetical inputs; not measurements or current-loop predictions',
        'gps_l1_wavelength_mm': WAVELENGTH*1000,
        'phase_to_distance': [{'phase_deg': d, 'distance_mm': WAVELENGTH*d/360*1000} for d in (1, 2, 3, 5, 10, 180, 360)],
        'rf_pure_delay': [{'delay_ps': t, 'phase_deg': 360*F*t*1e-12,
                          'distance_mm': C*t*1e-12*1000} for t in (1, 10, 100)],
        'if_aperture_example': {'if_hz': 4.092e6, 'delay_ps': 10.,
            'phase_deg': 360*4.092e6*10e-12, 'l1_equivalent_distance_mm': WAVELENGTH*4.092e6*10e-12*1000},
        'phase_frequency_slope': {'differential_group_delay_ns': 1., 'frequency_change_hz': 5000.,
            'phase_change_deg': 360*5000*1e-9, 'distance_change_mm': WAVELENGTH*5000*1e-9*1000},
        'same_epoch_mapping': [{'nco_hz': 2000., 'time_error_us': t,
                               'distance_error_mm': WAVELENGTH*2000*t*1e-6*1000} for t in (1, 10, 100)],
        'baseline_projection_bound': [{'baseline_mm': 75.9, 'single_rotation_error_deg': d,
                                     'bound_mm': 2*75.9*math.sin(math.radians(d)/2)} for d in (1, 3, 5)],
        'pll_thermal_example': pll,
        'phase_rounding_example': {'phase_bits': 16, 'max_instantaneous_error_mm': WAVELENGTH/2**17*1000},
        'dfe_nominal_delay': {'taps': 129, 'input_fs_hz': 33e6, 'input_samples': 64,
                              'microseconds': 64/33e6*1e6},
        'carrier_multipath_single_echo_bound_mm': WAVELENGTH/4*1000,
        'gore_published_example_arithmetic': {'coefficient_deg_per_ghz_ft': .24,
            'frequency_ghz': 4.2, 'length_ft': 10., 'temperature_from_c': 25.,
            'temperature_to_c': -55., 'phase_change_deg': .24*4.2*10},
    }


if __name__ == '__main__':
    result = calculate()
    target = Path(__file__).with_name('analytical_examples.json')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
