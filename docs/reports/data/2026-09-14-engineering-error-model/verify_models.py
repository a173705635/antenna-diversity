"""Deterministic analytical checks; no receiver or hardware claims."""
from pathlib import Path
import json
import re
import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent.parent
REPORT = REPORTS / '2026-09-14-engineering-error-model.md'
text = REPORT.read_text(encoding='utf-8')
checks = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append(name)


def close(name, actual, expected, atol=1e-10):
    check(name, np.allclose(actual, expected, atol=atol, rtol=1e-8))


old = json.loads((REPORTS / 'data/2026-09-13-sdcp-error-inventory/review.json').read_text(encoding='utf-8'))['inventory_items']
ids = re.findall(r'^\| ((?:A|C|R|L|AD|D|T|O|S|E|X)\d{2}) \|', text, re.M)
check('49 inherited items', len(old) == 49)
check('65 unique rows', len(ids) == len(set(ids)) == 65)
check('all inherited IDs preserved', set(old) <= set(ids))
check('16 new IDs', set(ids) - set(old) == {f'X{i:02d}' for i in range(1, 17)})
for target in re.findall(r'\[[^\[\]\n]+\]\(([^)]+\.(?:md|py|json))\)', text):
    if not target.startswith('https:'):
        # This script creates this sole generated output after successful checks.
        if target.endswith('/analytical_examples.json'):
            continue
        check('local link ' + target, (REPORT.parent / target).exists())

c, f = 299792458., 1575.42e6
lam = c / f
M, S = 4, 5
Da = np.column_stack((-np.ones(M-1), np.eye(M-1)))
Ds = np.column_stack((-np.ones(S-1), np.eye(S-1)))
cov_sd = Da @ Da.T
cov_dd = np.kron(Ds, Da) @ np.kron(Ds, Da).T
close('SD variance', np.diag(cov_sd), 2)
close('SD shared antenna covariance', cov_sd[0, 1], 1)
close('DD variance', np.diag(cov_dd), 4)
close('DD shared satellite covariance', cov_dd[0, M-1], 2)
check('DD rank', np.linalg.matrix_rank(cov_dd) == (M-1)*(S-1))
B = np.array([[.0759, 0, 0], [0, .1609, 0], [.0759, .1609, 0]])


def unit(a, e):
    return np.array([np.cos(e)*np.sin(a), np.cos(e)*np.cos(a), np.sin(e)])


a, e = .8, .6
u = unit(a, e)
ua = np.array([np.cos(e)*np.cos(a), -np.cos(e)*np.sin(a), 0])
ue = np.array([-np.sin(e)*np.sin(a), -np.sin(e)*np.cos(a), np.cos(e)])
J = -B @ np.column_stack((ua, ue))
h = 1e-6
Jfd = np.column_stack((-B @ (unit(a+h,e)-unit(a-h,e))/(2*h), -B @ (unit(a,e+h)-unit(a,e-h))/(2*h)))
close('DOA Jacobian finite difference', J, Jfd)
close('planar mirror', B @ u, B @ unit(a, -e))
u2 = unit(2.1, .3)
beta = np.array([.012, -.018, .025])
close('unknown bias confounding', -B @ u + beta, -B @ u2 + beta + B @ (u2-u))
same = np.tile((-B @ u + beta)[:, None], (1,S))
close('same source DD loses direction and bias', same @ Ds.T, 0)
ref = unit(.2, .9)
dd = -B @ (u-ref)
close('trusted reference reconstruction sign', dd-B@ref, -B@u)
phase = np.array([.1, .7, 1.4, -2.0, 2.5])
z = np.exp(1j*phase)
pair_score = np.mean([1-np.cos(phase[i]-phase[j]) for i in range(S) for j in range(i+1,S)])
close('DD concentration identity', pair_score, S/(S-1)*(1-abs(z.mean())**2))
close('concentration common bias invariance', abs(z.mean()), abs((z*np.exp(1.2j)).mean()))
d = -.03
close('half-cycle harmonic invariance', np.exp(4j*np.pi*d/lam), np.exp(4j*np.pi*(d+lam/2)/lam))
N, r = 10, .7
C = r**np.abs(np.arange(N)[:,None]-np.arange(N)[None,:])
mean_var = C.sum()/N**2
finite = (1+2*sum((1-k/N)*r**k for k in range(1,N)))/N
close('finite AR1 average covariance', mean_var, finite)
R = 10.
length = np.linalg.norm(B[-1])
exact = np.sqrt(R**2+length**2)-R
approx = length**2/(2*R)
check('nearfield expansion error below 1 um at phone baseline 10m', abs(exact-approx) < 1e-6)
ratio, psi = .001, .7
mix = -lam/(2*np.pi)*np.angle(1+ratio*np.exp(1j*psi))
linear = -lam/(2*np.pi)*ratio*np.sin(psi)
check('weak mixture local approximation', abs(mix-linear) < 2e-8)
check('equal amplitude destructive interference', abs(1+np.exp(1j*np.pi)) < 1e-14)
close('RF delay phase sign', -lam/(2*np.pi)*(-2*np.pi*f*1e-11), c*1e-11)
cn0 = 10**(40/10)
pll = lam/(2*np.pi)*np.sqrt(15/cn0*(1+1/(2*.001*cn0)))
examples = {
    'provenance': 'Analytical examples, not measured hardware specifications or new detection trials',
    'wavelength_m': lam,
    'one_degree_phase_magnitude_mm': lam/360*1000,
    'RF_delay_10ps_mm': c*1e-11*1000,
    'ADC_4_092MHz_10ps_magnitude_mm': lam*4.092e6*1e-11*1000,
    'NCO_2000Hz_timestamp_1us_magnitude_mm': lam*2000*1e-6*1000,
    'group_delay_1ns_frequency_separation_5kHz_magnitude_mm': lam*1e-9*5000*1000,
    'phase_16bit_rounding_bound_mm': lam/2**17*1000,
    'phone_diagonal_m': length,
    'phone_at_10m_perpendicular_nearfield_exact_mm': exact*1000,
    'phone_at_10m_second_order_mm': approx*1000,
    'one_m_baseline_at_10m_exact_mm': (np.sqrt(101)-10)*1000,
    'one_m_baseline_at_10m_second_order_mm': 50.,
    'PLL_example_15Hz_1ms_40dBHz_single_mm': pll*1000,
    'PLL_example_independent_SD_mm': np.sqrt(2)*pll*1000,
    'AR1_r07_N10_variance_of_mean_over_single_variance': mean_var,
    'AR1_r07_N10_exact_effective_samples': 1/mean_var,
    'AR1_r07_N10_long_window_approx_effective_samples': N*(1-r)/(1+r),
}
output = {'passed': len(checks), 'checks': checks, 'inherited_ids': old, 'new_ids': sorted(set(ids)-set(old)), 'examples': examples}
(HERE/'analytical_examples.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'passed': len(checks), 'examples': examples}, ensure_ascii=False, indent=2))
