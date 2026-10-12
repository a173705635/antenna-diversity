"""Offline startup-amplitude steering diagnosis on eight saved IQ records.

Receiver IQ stays phase-only calibrated. A relative three-antenna magnitude
model changes only the desired steering used by a hypothetical beam fit;
neither the frontend nor navigation consumes these candidate projectors.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.array_processing import startup_relative_amplitude
from core.presets import mate70_phase_centers
from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS,sky,steering
from core.serialization import serializable
from core.spoofing.single_source_scene import local_signal
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_complex_response_audit import (
    effective_response,generator_truth,phase_changes,
)
from experiments.spoofing.single_source_long_coherent import source_snapshot
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_ramp_low_power import (
    OUTPUT as STAGE39,preflight as preflight_stage39,
)
from experiments.spoofing.single_source_ramp_takeover_long import OUTPUT as STAGE35
from experiments.spoofing.single_source_long_absent import OUTPUT as STAGE38
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,digest,
)
from experiments.spoofing.single_source_template_coherence import (
    M,STARTUP,WIDTH,extract_startup,group_five_ms,
    projected_power_retention,trigger_fit,truth_directions,aided_template,
)


OUTPUT=RESULTS/STUDY_OUTPUT/'startup_amplitude'
H0=RESULTS/'satellite-count-24'
H0_GENERATOR=RESULTS.parent.parent/'archive'/'pre_refactor_2026-09-16'/'satellite_count_generator.py'
H0_SEEDS=(114001,114013,114037)
SAVED={
    'weak_0':(STAGE35/'weak',True),
    'absent_0':(STAGE38,True),
    'control_0':(STAGE35/'no_attack_control',False),
    'weak_minus5':(STAGE39/'weak',True),
    'absent_minus5':(STAGE39/'genuine_absent',True),
}


def startup_prompts(record,prns,templates,calibration,calibration_rate):
    """Calibrated [antenna, PRN, ms] prompts from samples strictly before .1 s."""
    samples=np.arange(STARTUP)
    x=record['samples'][ANTENNAS,:STARTUP]/(
        calibration[:,None]*np.exp(1j*calibration_rate[:,None]*samples[None,:]/FS))
    time=samples/FS
    local=np.stack([local_signal(record['codes'][star],template,time)
                    for star,template in zip(prns,templates)])
    milliseconds=STARTUP//2046
    prompts=np.einsum('mkn,skn->msk',x.reshape(M,milliseconds,2046),
                      local.conj().reshape(len(prns),milliseconds,2046))/2046
    return prompts


def check_startup_prompts(record,prns,templates,calibration,calibration_rate,prompts):
    """Independent per-PRN, per-ms complex dots check the saved prompt tensor."""
    for index,star in enumerate(prns):
        for millisecond in range(STARTUP//2046):
            samples=np.arange(millisecond*2046,(millisecond+1)*2046)
            local=local_signal(record['codes'][star],templates[index],samples/FS)
            for antenna in range(M):
                calibrated=record['samples'][ANTENNAS[antenna],samples]/(
                    calibration[antenna]*np.exp(
                        1j*calibration_rate[antenna]*samples/FS))
                direct=np.vdot(local,calibrated)/2046
                np.testing.assert_allclose(prompts[antenna,index,millisecond],
                                           direct,rtol=0,atol=1e-12)


def estimate_amplitude(prompts,nominal,prns):
    """Equal-PRN mean covariance; 0-based C12/C02 and C21/C01 ratios."""
    return startup_relative_amplitude(prompts,nominal,prns)


def fit_blocks(record,prns,templates,calibration,calibration_rate,amplitude):
    """Original and amplitude-steering aided-5 ms fits, with one noise model."""
    count=(record['samples'].shape[1]+WIDTH-1)//WIDTH
    raw=np.full((2,count),np.nan)
    fitted=np.full((2,count,M,M),np.nan+0j)
    would=np.full_like(fitted,np.nan+0j)
    gain=np.full((2,count,record['nominal'].shape[1]),np.nan)
    nominal_gain=np.full_like(gain,np.nan)
    grouped_prompts=np.full((count,M,len(prns)*4),np.nan+0j)
    noise_power=np.full(count,np.nan)
    original=record['nominal'][ANTENNAS]
    adjusted=amplitude[:,None]*original
    aided=np.array([aided_template(template) for template in templates])
    for index in range(count):
        start=index*WIDTH
        if start<STARTUP:continue
        stop=min(start+WIDTH,record['samples'].shape[1])
        samples=np.arange(start,stop)
        block=record['samples'][ANTENNAS,start:stop]/(
            calibration[:,None]*np.exp(1j*calibration_rate[:,None]*samples[None,:]/FS))
        noise=float(np.mean(abs(block)**2)/2046)
        local=np.stack([local_signal(record['codes'][star],template,samples/FS)
                        for star,template in zip(prns,aided)])
        chunks=len(samples)//2046
        prompts=np.einsum('mkn,skn->msk',block.reshape(M,chunks,2046),
                          local.conj().reshape(len(prns),chunks,2046))/2046
        grouped=group_five_ms(prompts)
        grouped_prompts[index]=grouped
        noise_power[index]=noise
        for mode,desired in enumerate((original,adjusted)):
            repeated=np.repeat(desired[:,prns],grouped.shape[1]//len(prns),axis=1)
            p,score,active,retained=trigger_fit(grouped,repeated,desired,noise)
            raw[mode,index]=score
            fitted[mode,index]=p
            would[mode,index]=active
            gain[mode,index]=retained
            nominal_gain[mode,index]=np.einsum(
                'mi,mn,ni->i',original.conj(),active,original).real/M
        if index%50==0:print('  amplitude fit block',index,'/',count,flush=True)
    return dict(block_start_s=np.arange(count)*.02,raw=raw,
                fitted_projector=fitted,would_projector=would,would_gain=gain,
                nominal_gain=nominal_gain,grouped_prompts=grouped_prompts,
                noise_power=noise_power,relative_amplitude=amplitude)


def audit_saved_raw(fit,nominal,prns):
    """Recompute every score from saved prompts with a complex two-column LS.

    The fitted projector supplies the nuisance direction. This audit does not
    call the receiver's residual_trigger or use source truth.
    """
    maximum=0.
    for mode,model in enumerate((nominal,fit['relative_amplitude'][:,None]*nominal)):
        desired=np.repeat(model[:,prns],4,axis=1)
        for index in range(5,len(fit['block_start_s'])):
            z=fit['grouped_prompts'][index]
            noise=fit['noise_power'][index]
            _,vectors=np.linalg.eigh(fit['fitted_projector'][mode,index])
            nuisance=vectors[:,0]
            dot=np.sum(desired.conj()*z,axis=0)
            cross=np.sum(desired.conj()*nuisance[:,None],axis=0)
            nuisance_dot=nuisance.conj()@z
            ridge=.01*M
            amplitude=(dot-cross*nuisance_dot)/(
                np.sum(abs(desired)**2,axis=0)+ridge-abs(cross)**2)
            nuisance_amplitude=nuisance_dot-cross.conj()*amplitude
            single=z-desired*(dot/M)
            residual=z-desired*amplitude-nuisance[:,None]*nuisance_amplitude
            reduction=np.sum(abs(single)**2)-np.sum(abs(residual)**2)
            score=float(reduction/(noise*(z.shape[1]+M-1)))
            maximum=max(maximum,abs(score-fit['raw'][mode,index]))
    assert maximum<1e-9
    return maximum


def preflight():
    stage35,stage37,stage38,sources=preflight_stage39()
    stage39=json.loads((STAGE39/'summary.json').read_text(encoding='utf-8'))
    assert stage39['source_snapshot_sha256']==digest(STAGE39/'source_snapshot.zip')
    verify_source_snapshot(STAGE39/'source_snapshot.zip',stage39['source_sha256'])
    for key,name in (('plan_sha256','plan.json'),
                     ('manifest_sha256','manifest.json'),
                     ('figure_sha256','overview.png')):
        assert stage39[key]==digest(STAGE39/name)
    for name,sha in stage39['source_sha256'].items():
        if name.startswith('core/'):
            assert sources[name]==sha
    h0_manifest=json.loads((H0/'manifest.json').read_text(encoding='utf-8'))
    h0_plan=json.loads((H0/'plan.json').read_text(encoding='utf-8'))
    generator_sha=h0_manifest['artifacts_sha256']['satellite_count_generator.py']
    assert generator_sha==h0_plan['generator_sha256']==digest(H0_GENERATOR)
    for seed in H0_SEEDS:
        folder=H0/'inputs'/f'{seed}_normal_24'
        for name in ('samples.npy','observations.npz','metadata.json'):
            key=f'inputs\\{seed}_normal_24\\{name}'
            assert h0_manifest['sha256'][key]==digest(folder/name)
        identity=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        assert identity['samples_sha256']==digest(folder/'samples.npy')
        assert identity['observations_sha256']==digest(folder/'observations.npz')
        assert identity['seed']==seed and identity['kind']=='normal'
        assert not identity['attacked'] and identity['case']['cn0']==45.
        assert identity['case']['stars']==24
    for name,(folder,_) in SAVED.items():
        summary=(stage39['cases']['weak' if name=='weak_minus5'
                                  else 'genuine_absent']
                 if name.endswith('minus5') else None)
        if summary is None:
            summary=(stage35['cases']['weak'] if name=='weak_0' else
                     stage35['cases']['no_attack_control'] if name=='control_0'
                     else stage38)
        for filename,sha in summary['input_sha256'].items():
            assert digest(folder/'input'/filename)==sha
        assert summary['fit_sha256']==digest(folder/'fit.npz')
        trace_sha=(summary['output']['aided_five_ms']['output_sha256']['trace.npz']
                   if name in ('weak_0','control_0') else
                   summary['receiver']['output_sha256']['trace.npz'])
        assert trace_sha==digest(folder/'aided_five_ms'/'trace.npz')
    return sources,h0_manifest,stage35,stage38,stage39


def source_folder(name):
    if name.startswith('h0_'):
        return H0/'inputs'/f'{name[3:]}_normal_24'
    return SAVED[name][0]/'input'


def load_record(name):
    folder=source_folder(name)
    with np.load(folder/'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(folder/'samples.npy',mmap_mode='r')
    identity=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    return record,identity


def saved_startup(name):
    folder=SAVED[name][0]
    with np.load(folder/'fit.npz') as saved:
        original=dict(saved)
    return (original['startup_prns'],original['startup_templates'],
            original['calibration'],original['calibration_rate'],original)


def check_original_fit(name,fit,original):
    """Match the existing aided-5 ms cache before interpreting a new model."""
    for key,old in (('raw','raw'),('fitted_projector','fitted_projector'),
                    ('would_projector','would_projector'),
                    ('would_gain','would_gain')):
        np.testing.assert_allclose(fit[key][0,5:],original[old][3,5:],
                                   rtol=0,atol=1e-9,err_msg=f'{name}.{key}')
    folder=SAVED[name][0]
    with np.load(folder/'aided_five_ms'/'trace.npz') as saved:
        np.testing.assert_allclose(fit['raw'][0,5:],
            saved['trace_raw_trigger'][5:],rtol=0,atol=1e-10)
        np.testing.assert_allclose(fit['would_projector'][0,5:],
            saved['trace_projectors'][5:],rtol=0,atol=1e-9)
        np.testing.assert_array_equal(np.where(fit['raw'][0,5:]>=3.,2,3),
                                      saved['trace_dimensions'][5:])


def model_weights(projectors,model,triggered):
    """Hypothetical weights for one steering model, matching both beam branches."""
    projected=np.einsum('bmn,ns->bms',projectors,model)
    denominator=np.sum(model.conj()[None,:,:]*projected,axis=1).real
    return np.where(triggered[:,None,None],
        projected/(denominator[:,None,:]+.01),model[None,:,:]/M)


def truth_scores(record,identity,fit,calibration,calibration_rate,folder):
    """Physics is used only after both fits and old-cache checks are done."""
    starts=fit['block_start_s']
    measured=starts>=.1
    nominal=record['nominal'][ANTENNAS]
    physical,fake=generator_truth(identity,record['nominal'])
    true_retention=np.full((2,len(starts),nominal.shape[1]),np.nan)
    fake_retention=np.full((2,len(starts)),np.nan)
    for index in np.flatnonzero(measured):
        time=starts[index]+.01
        gain=physical[ANTENNAS]/(
            calibration*np.exp(1j*calibration_rate*time))
        genuine=gain[:,None]*nominal
        for mode in range(2):
            projector=fit['would_projector'][mode,index]
            true_retention[mode,index]=projected_power_retention(projector,genuine)
            if identity['attacked'] and identity['case']['attack_start']<=time:
                fake_retention[mode,index]=projected_power_retention(
                    projector,gain*fake[ANTENNAS])
    phase={}
    complex_response=np.full((2,len(starts),nominal.shape[1]),np.nan+0j)
    boundary_jump=np.full((2,len(starts)-1,nominal.shape[1]),np.nan)
    midpoint_change=np.full_like(boundary_jump,np.nan)
    stable_boundary=np.zeros_like(boundary_jump,dtype=bool)
    amplitude=fit['relative_amplitude']
    for mode,label in enumerate(('original','amplitude')):
        projectors=fit['would_projector'][mode,measured]
        model=nominal if mode==0 else amplitude[:,None]*nominal
        triggered=fit['raw'][mode,measured]>=3.
        weights=model_weights(projectors,model,triggered)
        midpoint=starts[measured]+.01
        boundary=starts[measured][1:]
        response,_=effective_response(weights,nominal,physical[ANTENNAS],
                                       calibration,calibration_rate,midpoint)
        jump,change=phase_changes(weights,nominal,physical[ANTENNAS],
                                  calibration,calibration_rate,midpoint,boundary)
        complex_response[mode,measured]=response
        boundary_jump[mode,np.flatnonzero(measured)[:-1]]=jump
        midpoint_change[mode,np.flatnonzero(measured)[:-1]]=change
        stable=(fit['nominal_gain'][mode,measured][:-1]>=.05)&(
                fit['nominal_gain'][mode,measured][1:]>=.05)
        stable_boundary[mode,np.flatnonzero(measured)[:-1]]=stable
        magnitude=abs(jump)
        phase[label]=dict(all_prn_max_rad=float(magnitude.max()),
            all_prn_p99_rad=float(np.percentile(magnitude,99)),
            stable_count=int(stable.sum()),
            stable_max_rad=None if not stable.any() else float(magnitude[stable].max()),
            stable_p99_rad=None if not stable.any() else float(
                np.percentile(magnitude[stable],99)))
    np.savez_compressed(folder/'truth.npz',block_start_s=starts,
        true_power_retention=true_retention,
        fake_power_retention=fake_retention,
        complex_true_response=complex_response,
        same_boundary_phase_jump_rad=boundary_jump,
        midpoint_phase_change_rad=midpoint_change,
        stable_nominal_gain_boundary=stable_boundary)
    if identity['attacked']:
        post=starts>=identity['case']['attack_start']
        fake_summary={label:dict(median=float(np.nanmedian(fake_retention[index,post])),
                                 worst=float(np.nanmax(fake_retention[index,post])))
                      for index,label in enumerate(('original','amplitude'))}
    else:fake_summary=None
    return dict(result_sha256=digest(folder/'truth.npz'),
        true_direction_meaning=('counterfactual' if not identity['case']['genuine']
                                else 'present'),
        fake_direction_meaning=('present' if identity['attacked'] else 'absent'),
        fake_retention=fake_summary,phase_jump=phase,
        true_loss_median_by_prn={label:np.nanmedian(
            1.-true_retention[index,measured],axis=0).tolist()
            for index,label in enumerate(('original','amplitude'))})


def fit_case(name):
    record,identity=load_record(name)
    nominal=record['nominal'][ANTENNAS]
    if name.startswith('h0_'):
        seed=identity['seed']
        positions=np.array(list(mate70_phase_centers().values()))
        expected=steering(sky(seed,24),positions,0.)
        np.testing.assert_array_equal(record['nominal'],expected)
        prns,templates,calibration,rate=extract_startup(record)
        original=None
    else:
        prns,templates,calibration,rate,original=saved_startup(name)
    prompts=startup_prompts(record,prns,templates,calibration,rate)
    check_startup_prompts(record,prns,templates,calibration,rate,prompts)
    amplitude,covariance=estimate_amplitude(prompts,nominal,prns)
    folder=OUTPUT/name
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'startup.npz',prompts=prompts,prns=prns,
        templates=templates,calibration=calibration,calibration_rate=rate,
        covariance=covariance,relative_amplitude=amplitude)
    fitted=fit_blocks(record,prns,templates,calibration,rate,amplitude)
    np.savez_compressed(folder/'fit.npz',**fitted)
    raw_audit_error=audit_saved_raw(fitted,nominal,prns)
    if original is not None:check_original_fit(name,fitted,original)
    truth=truth_scores(record,identity,fitted,calibration,rate,folder)
    starts=fitted['block_start_s']
    masks=(dict(clean=starts>=.1,post=starts>=.3,steady=starts>=.5)
           if name.startswith('h0_') else
           dict(clean=(starts>=.1)&(starts<.3),
                post=starts>=.3,steady=starts>=.5))
    phases={}
    for label,mask in masks.items():
        phases[label]={mode:dict(blocks=int(mask.sum()),
            triggers=int((fitted['raw'][index,mask]>=3.).sum()),
            raw_min=float(np.min(fitted['raw'][index,mask])),
            raw_median=float(np.median(fitted['raw'][index,mask])),
            raw_max=float(np.max(fitted['raw'][index,mask])))
            for index,mode in enumerate(('original','amplitude'))}
    result=dict(input_sha256={filename:digest(source_folder(name)/filename)
                for filename in ('samples.npy','observations.npz','metadata.json')},
        startup_sha256=digest(folder/'startup.npz'),fit_sha256=digest(folder/'fit.npz'),
        relative_amplitude=amplitude.tolist(),
        independent_raw_max_abs_error=raw_audit_error,phases=phases,truth=truth)
    if name.startswith('h0_'):
        physical,_=generator_truth(identity,record['nominal'])
        magnitude=abs(physical[ANTENNAS])
        expected=magnitude/np.sqrt(np.mean(magnitude**2))
        result['generator_magnitude_audit']=dict(
            source='archived generator gain draws, not full RF regeneration',
            relative_magnitude=expected.tolist(),
            estimation_relative_error=(amplitude/expected-1.).tolist())
    del record,original,fitted,prompts
    gc.collect()
    return result


def plot_results(results):
    figure,axes=plt.subplots(4,2,figsize=(12,10),constrained_layout=True)
    for index,name in enumerate(results):
        row=index//2;column=index%2
        with np.load(OUTPUT/name/'fit.npz') as saved:
            time=saved['block_start_s']
            raw=saved['raw']
        axis=axes[row,column]
        axis.plot(time,raw[0],linewidth=.8,label='original steering')
        axis.plot(time,raw[1],linewidth=.8,label='startup amplitude steering')
        axis.axhline(3.,color='gray',linestyle=':',linewidth=.8)
        axis.set(title=name,xlabel='Block start (s)',ylabel='Offline raw fit score')
        axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    sources,h0_manifest,stage35,stage38,stage39=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    names=tuple(SAVED)+tuple(f'h0_{seed}' for seed in H0_SEEDS)
    plan=dict(scope=__doc__,inputs=names,startup_s=.1,block_s=.02,
        prompt_ms=5,threshold=3.,amplitude_definition=(
            'equal-PRN mean of calibrated y_m conj(y_p); ratios C12/C02, C21/C01; RMS one'),
        source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan),indent=2),
                                    encoding='utf-8')
    manifest=dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        stage35_summary_sha256=digest(STAGE35/'summary.json'),
        stage38_summary_sha256=digest(STAGE38/'summary.json'),
        stage39_summary_sha256=digest(STAGE39/'summary.json'),
        h0_manifest_sha256=digest(H0/'manifest.json'),
        h0_generator_sha256=digest(H0_GENERATOR))
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),indent=2),
                                        encoding='utf-8')
    results={}
    for name in names:
        results[name]=fit_case(name)
        print('startup amplitude offline diagnosis',name,'saved',flush=True)
    plot_results(results)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'),
        h0_generator_sha256=h0_manifest['artifacts_sha256']['satellite_count_generator.py'],
        cases=results)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
