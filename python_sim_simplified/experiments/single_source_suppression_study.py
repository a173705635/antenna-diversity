"""Development-RF split-support and receive-PLL comparisons.

Each phase uses the saved 116901 weak 2 s RF and three antennas. The split
phase shares one frontend run between two navigations; PLL phases reuse its
saved baseline. Each phase writes only its own files. Defaults stay unchanged.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from core.single_source import frontend, navigate
from experiments.paths import RESULTS


INPUT = RESULTS/'weak-ten'/'inputs'/'116901_weak_30_24_2'
SELECTED = RESULTS/'mechanism-four'/'summary.json'
OUTPUT = 'single_source_suppression_study'
ANTENNAS = [0,1,2]
SUPPORT_THRESHOLD = 3.
PHASE = 'pll25'
RECEIVER_PLL = (.15,1.)
PLL25 = (.06315066862415342,.32776091874805263)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def aligned_windows(times):
    """True when the 20 ms observation ends at a basis block boundary."""
    width=round(.02*FS)
    return np.array([round(t*FS)%width==0 for t in times])


def verify_split_energy(observations):
    """Check the saved ratio against its saved on/off energy totals."""
    ratio=observations['support_split']
    on=observations['split_on_energy'];off=observations['split_off_energy']
    assert np.array_equal(np.isfinite(ratio),np.isfinite(on)&np.isfinite(off))
    finite=np.isfinite(ratio)
    np.testing.assert_allclose(ratio[finite],on[finite]/off[finite],rtol=1e-12)


def score(navigation,times,case):
    published=np.isfinite(navigation['position']).all(axis=1)
    post=times>=case['attack_start']
    missed=np.r_[False,~published[post],False]
    starts=np.flatnonzero(np.diff(missed.astype(int))==1)
    ends=np.flatnonzero(np.diff(missed.astype(int))==-1)
    longest=int((ends-starts).max()) if len(starts) else 0
    tail=times>=times[-1]-.5
    published_times=times[published]
    truth=np.column_stack((case['velocity_east']*times,np.zeros(len(times))))
    error=np.linalg.norm(navigation['position'][:,:2]-truth,axis=1)
    finite=error[post][np.isfinite(error[post])]
    return dict(opportunities=int(post.sum()),published=int(published[post].sum()),
                good20=int((finite<=20.).sum()),error30=int((finite>=30.).sum()),
                error50=int((finite>=50.).sum()),
                p95_error_m=None if not len(finite) else float(np.percentile(finite,95)),
                max_error_m=None if not len(finite) else float(finite.max()),
                longest_unavailable_s=longest*.01,
                tail_published=int(published[tail].sum()),tail_opportunities=int(tail.sum()),
                last_published_available_time_s=None if not len(published_times) else float(published_times[-1]),
                initialization_available_time_s=navigation['initialization_time'])


def tracking_counts(observations):
    finite=np.isfinite(observations['ranges'])
    locked=finite&(observations['lock']>=.5)
    cn0=locked&(observations['cn0']>=25.)
    gain=cn0&(observations['gain']>=.05)
    supported=gain&(observations['support']>=SUPPORT_THRESHOLD)
    arc=observations['arc']
    continuous=np.zeros_like(finite)
    continuous[1:]=finite[1:]&finite[:-1]&(arc[1:]==arc[:-1])&(arc[1:]>=0)
    return dict(finite=finite.sum(axis=1),lock=locked.sum(axis=1),
                cn0=cn0.sum(axis=1),gain=gain.sum(axis=1),
                support=supported.sum(axis=1),arc_continuous=continuous.sum(axis=1))


def tracking_summary(counts,available_time,attack_start):
    post=available_time>=attack_start
    return {name:dict(mean=float(values[post].mean()),
                      epochs_ge6=int((values[post]>=6).sum()))
            for name,values in counts.items()}


def pll_main():
    """Replay only the known weak three-antenna RF against stage-one baseline."""
    receiver_pll=PLL25 if PHASE=='pll25' else RECEIVER_PLL
    design=None
    if PHASE=='pll25':
        from scipy.linalg import solve_discrete_lyapunov
        kp,ki=receiver_pll;dt=.001
        damping=1/np.sqrt(2)
        natural_frequency=46.12656133938665
        radius=np.exp(-damping*natural_frequency*dt)
        angle=natural_frequency*np.sqrt(1-damping**2)*dt
        np.testing.assert_allclose([kp,ki],
            [1-radius**2,(1+radius**2-2*radius*np.cos(angle))/(2*np.pi*dt)],rtol=1e-12)
        transition=np.array([[1-kp-2*np.pi*dt*ki,2*np.pi*dt],[-ki,1.]])
        noise=np.array([[kp+2*np.pi*dt*ki],[ki]])
        covariance=solve_discrete_lyapunov(transition,noise@noise.T)
        design=dict(target_bandwidth_hz=25.,damping_ratio=float(damping),
                    natural_frequency_rad_s=natural_frequency,
                    epoch_s=dt,transition=transition,measurement_noise=noise,
                    pole_equations='r=exp(-zeta*wn*T); phi=wn*sqrt(1-zeta**2)*T',
                    gain_equations='kp=1-r**2; ki=(1+r**2-2*r*cos(phi))/(2*pi*T)',
                    covariance_equation='P = A P A.T + b b.T',
                    bandwidth_equation='B = P[0,0] / (2 * epoch_s)',
                    bandwidth_model='linearized sampled phase white noise; excludes 1ms integration, Costas nonlinearity and initial FLL',
                    achieved_bandwidth_hz=float(covariance[0,0]/(2*dt)))
    root=RESULTS/OUTPUT
    baseline_tracking=root/'tracking.npz'
    baseline_navigation=root/'navigation.npz'
    baseline_summary=root/'summary.json'
    stage_one=json.loads(baseline_summary.read_text(encoding='utf-8'))
    metadata=json.loads((INPUT/'metadata.json').read_text(encoding='utf-8'))
    with np.load(INPUT/'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(INPUT/'samples.npy',mmap_mode='r')
    hashes={name:digest(INPUT/name) for name in ('samples.npy','observations.npz','metadata.json')}
    assert hashes==stage_one['input_sha256']
    assert digest(baseline_tracking)==stage_one['tracking_sha256']
    assert digest(baseline_navigation)==stage_one['navigation_sha256']
    sources=source_hashes()
    output=root/PHASE
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(serializable(dict(
        scope='Receive-stage PLL only; startup calibration and beam unchanged',
        source_sha256=sources,input_sha256=hashes,
        baseline_source_sha256=stage_one['source_sha256'],
        baseline_tracking_sha256=digest(baseline_tracking),
        baseline_navigation_sha256=digest(baseline_navigation),
        receiver_pll_kp=receiver_pll[0],receiver_pll_ki_hz=receiver_pll[1],
        design=design,antennas=ANTENNAS)),indent=2),encoding='utf-8')

    with np.load(baseline_tracking) as saved:
        baseline_observations=dict(saved)
    with np.load(baseline_navigation) as saved:
        baseline=dict(position=saved['baseline_position'],selected=saved['baseline_selected'],
                      initialization_time=stage_one['original']['initialization_available_time_s'])
    available=baseline_observations['available_time']
    baseline_score=score(baseline,available,metadata['case'])
    assert baseline_score==stage_one['original']

    print(PHASE,'weak 116901 3A: frontend start',flush=True)
    candidate_observations=frontend(record,ANTENNAS,receiver_pll=receiver_pll)
    print(PHASE,'weak 116901 3A: frontend done; navigation start',flush=True)
    candidate=navigate(candidate_observations)
    candidate_score=score(candidate,candidate_observations['available_time'],metadata['case'])
    assert np.array_equal(candidate_observations['time'],baseline_observations['time'])
    assert np.array_equal(candidate_observations['available_time'],available)
    for key in ('startup_prns','calibration','calibration_rate','gain','ratio','satellites'):
        np.testing.assert_array_equal(candidate_observations[key],baseline_observations[key],err_msg=key)
    print(PHASE,'weak 116901 3A: navigation done; saving',flush=True)
    tracking=output/'tracking.npz';navigation=output/'navigation.npz'
    np.savez_compressed(tracking,**candidate_observations)
    np.savez_compressed(navigation,position=candidate['position'],
                        candidate_position=candidate['candidate_position'],
                        selected=candidate['selected'])
    before_counts=tracking_counts(baseline_observations)
    after_counts=tracking_counts(candidate_observations)
    np.savez_compressed(output/'diagnostics.npz',available_time=available,
                        **{'baseline_'+k:v for k,v in before_counts.items()},
                        **{'pll_'+k:v for k,v in after_counts.items()})
    report=dict(baseline=baseline_score,candidate=candidate_score,
                receiver_pll_kp=receiver_pll[0],receiver_pll_ki_hz=receiver_pll[1],
                design=design,
                baseline_tracking_summary=tracking_summary(before_counts,available,metadata['case']['attack_start']),
                pll_tracking_summary=tracking_summary(after_counts,available,metadata['case']['attack_start']),
                baseline_source_sha256=stage_one['source_sha256'],source_sha256=sources,
                input_sha256=hashes,baseline_tracking_sha256=digest(baseline_tracking),
                baseline_navigation_sha256=digest(baseline_navigation),
                tracking_sha256=digest(tracking),navigation_sha256=digest(navigation),
                diagnostics_sha256=digest(output/'diagnostics.npz'))
    (output/'summary.json').write_text(json.dumps(serializable(report),indent=2,allow_nan=False),encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,6),constrained_layout=True)
    for name in ('finite','lock','cn0','support'):
        axes[0].plot(available,before_counts[name],alpha=.6,label='baseline '+name)
        axes[0].plot(available,after_counts[name],alpha=.6,linestyle='--',label='PLL '+name)
    axes[0].set(ylabel='Satellites / epoch');axes[0].legend(ncol=2,fontsize=7)
    axes[1].plot(available,np.isfinite(baseline['position']).all(axis=1).astype(int),label='baseline')
    axes[1].plot(available,np.isfinite(candidate['position']).all(axis=1).astype(int),label='PLL')
    axes[1].set(xlabel='Output available time (s)',ylabel='Published');axes[1].legend()
    for axis in axes:axis.grid(alpha=.3)
    fig.savefig(output/'summary.png',dpi=150)
    plt.close(fig)
    assert source_hashes()==sources
    print(PHASE,'weak 116901 3A: done',baseline_score,'->',candidate_score,flush=True)


def main():
    if PHASE in ('pll','pll25'):
        pll_main()
        return
    selected=json.loads(SELECTED.read_text(encoding='utf-8'))
    row=next(row for row in selected['selected_rows']
             if '116901_weak_30_24_2' in row['input'] and row['subset']=='three')
    metadata=json.loads((INPUT/'metadata.json').read_text(encoding='utf-8'))
    with np.load(INPUT/'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(INPUT/'samples.npy',mmap_mode='r')
    input_hashes={name:digest(INPUT/name) for name in ('samples.npy','observations.npz','metadata.json')}
    assert input_hashes['samples.npy']==row['after']['input_sha256']
    sources=source_hashes()
    output=RESULTS/OUTPUT
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,input=str(INPUT.resolve()),
        input_sha256=input_hashes,selected_summary_sha256=digest(SELECTED),
        antennas=ANTENNAS,support_threshold=SUPPORT_THRESHOLD)),indent=2),encoding='utf-8')

    print('RF input verified; frontend start',flush=True)
    observations=frontend(record,ANTENNAS,split_support=True)
    print('frontend done; paired navigation start',flush=True)
    baseline=navigate(observations)
    candidate=navigate({**observations,'support':observations['support_split']})
    print('paired navigation done; saved-data verification start',flush=True)
    historical=RESULTS/'mechanism-four'/'evaluations'/row['after_id']
    with np.load(historical/'tracking.npz') as saved:
        for key in saved.files:
            np.testing.assert_array_equal(observations[key],saved[key],err_msg=f'tracking.{key}')
    with np.load(historical/'navigation.npz') as saved:
        for key in saved.files:
            np.testing.assert_array_equal(baseline[key],saved[key],err_msg=f'navigation.{key}')
    assert baseline['initialization_time']==row['after']['initialization_time']

    tracking=output/'tracking.npz'
    np.savez_compressed(tracking,**observations)
    with np.load(tracking) as saved:verify_split_energy(saved)
    np.savez_compressed(output/'navigation.npz',
                        baseline_position=baseline['position'],baseline_selected=baseline['selected'],
                        split_position=candidate['position'],split_selected=candidate['selected'])
    times=observations['time'];available=observations['available_time']
    aligned=aligned_windows(times)
    post=available>=metadata['case']['attack_start']
    baseline_gate=observations['support']>=SUPPORT_THRESHOLD
    split_gate=observations['support_split']>=SUPPORT_THRESHOLD
    strata={}
    for name,mask in (('post_attack_aligned',aligned&post),
                      ('post_attack_half_block',~aligned&post)):
        strata[name]=dict(epochs=int(mask.sum()),
                          original_finite=int(np.isfinite(observations['support'][mask]).sum()),
                          split_finite=int(np.isfinite(observations['support_split'][mask]).sum()),
                          original_pass=int(baseline_gate[mask].sum()),
                          split_pass=int(split_gate[mask].sum()),
                          changed_satellite_epochs=int((baseline_gate[mask]!=split_gate[mask]).sum()))
    report=dict(input_sha256=input_hashes,source_sha256=sources,
                historical_after_id=row['after_id'],
                historical_tracking_sha256=digest(historical/'tracking.npz'),
                historical_navigation_sha256=digest(historical/'navigation.npz'),
                tracking_sha256=digest(tracking),navigation_sha256=digest(output/'navigation.npz'),
                original=score(baseline,available,metadata['case']),
                split=score(candidate,available,metadata['case']),strata=strata)
    (output/'summary.json').write_text(json.dumps(serializable(report),indent=2,allow_nan=False),encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,1,figsize=(9,6),constrained_layout=True)
    axes[0].plot(available,baseline_gate.sum(axis=1),label='Original support')
    axes[0].plot(available,split_gate.sum(axis=1),label='Split support')
    axes[0].set(ylabel='Satellites above support threshold')
    axes[1].plot(available,np.isfinite(baseline['position']).all(axis=1).astype(int),label='Original')
    axes[1].plot(available,np.isfinite(candidate['position']).all(axis=1).astype(int),label='Split')
    axes[1].set(xlabel='Output available time (s)',ylabel='Published')
    for axis in axes:axis.grid(alpha=.3);axis.legend()
    figure.savefig(output/'summary.png',dpi=150)
    plt.close(figure)
    assert source_hashes()==sources
    print('original',report['original'],'split',report['split'],flush=True)


if __name__=='__main__':
    main()
