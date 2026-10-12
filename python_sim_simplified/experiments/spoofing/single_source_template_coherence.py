"""Offline 2x2 startup-template and prompt-coherence beam-fit diagnosis.

This reads saved IQ and repeats only the beam-fit path. It does not navigate,
re-run the receiver frontend, or provide truth to any fitted projector.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.spoofing.array_acquisition import acquire_array
from core.presets import default_config,mate70_phase_centers
from core.provenance import source_hashes
from core.spoofing.recovery_geometry import C,FS,WAVELENGTH,sky,steering
from core.serialization import serializable
from core.spoofing.single_source import calibrate,residual_trigger
from core.spoofing.single_source_scene import local_signal
from core.tracking import track
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_long_coherent import (
    OUTPUT as STAGE21,source_snapshot,
)
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_spoof_power import (
    OUTPUT as STAGE26,preflight_inputs as preflight_stage26,
)
from experiments.spoofing.single_source_temporal_trigger import OUTPUT as STAGE25
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,digest,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'template_coherence'
SCENES=('weak','genuine_absent','shared_control')
VARIANTS=('original_1ms','original_5ms','aided_1ms','aided_5ms')
WIDTH=round(.02*FS)
STARTUP=round(.1*FS)
M=len(ANTENNAS)


def aided_template(template):
    """Change startup code-rate extrapolation using its carrier NCO only."""
    epoch,phase,_,frequency=template
    return epoch,phase,1.023e6*(1.+frequency/1575.42e6),frequency


def group_five_ms(prompts):
    """Four absolute 5 ms complex sums per PRN, preserving any bit flips."""
    antennas,stars,milliseconds=prompts.shape
    return (prompts.reshape(antennas,stars,milliseconds//5,5)
            .sum(axis=-1)/np.sqrt(5)).reshape(antennas,-1)


def trigger_fit(prompts,repeated_desired,nominal,noise):
    """Return fitted and actually triggered P, using the original score gate."""
    fitted,raw=residual_trigger(prompts,repeated_desired,noise)
    would=fitted if raw>=3. else np.eye(len(nominal))
    gain=np.einsum('mi,mn,ni->i',nominal.conj(),would,nominal).real/len(nominal)
    return fitted,raw,would,gain


def projected_power_retention(projector,directions):
    """Truth-only directional power ratio; never an input to trigger_fit."""
    projected=projector@directions
    return (np.sum(abs(projected)**2,axis=0)/
            np.sum(abs(directions)**2,axis=0)).real


def wrapped_code_error(local_phase,true_phase):
    """Signed code-template error in chips, wrapped to one 1023-chip period."""
    return (local_phase-true_phase+511.5)%1023.-511.5


def truth_code_phases(case,seed,time):
    """Generate physical code arguments solely for the after-fit audit."""
    directions=sky(seed,case['stars'])
    radius=21e6+np.arange(case['stars'])*17000.
    doppler=case['doppler_scale']*np.linspace(-1500.,2000.,case['stars'])
    drift=case['doppler_rate']*np.linspace(-1.,1.,case['stars'])
    distance=radius-WAVELENGTH*(doppler*time+.5*drift*time*time)
    east=case['velocity_east']*time
    false_east=east+case['fake_east']+case['fake_rate']*max(
        time-case['attack_start'],0.)
    true_range=np.sqrt(distance**2-2*distance*directions[:,0]*east+east**2)+75000.
    fake_range=np.sqrt(distance**2-2*distance*directions[:,0]*false_east+
                       false_east**2)+75000.
    return (time-true_range/C)*1.023e6,(time-fake_range/C)*1.023e6


def truth_directions(case,seed,nominal,calibration,calibration_rate,time):
    """Return calibrated physical true/fake vectors for post-fit loss only."""
    directions=sky(seed,case['stars'])
    tangent=np.cross(directions[0],[0,0,1.])
    tangent/=np.linalg.norm(tangent)
    source=directions[0]*np.cos(np.radians(case['angle']))+tangent*np.sin(
        np.radians(case['angle']))
    antenna_positions=np.array(list(mate70_phase_centers().values()))
    fake=steering(source[None],antenna_positions,0.)[ANTENNAS,0]
    rng=np.random.default_rng(seed)
    channel=np.exp(1j*np.radians(case['phase_deg']*rng.normal(
        size=case['antennas'])))
    channel*=10**(case['amplitude_db']*rng.normal(
        size=case['antennas'])/20)
    receiver=channel[ANTENNAS]/(
        calibration*np.exp(1j*calibration_rate*time))
    return receiver[:,None]*nominal,receiver*fake


def preflight_inputs():
    stage21,_,_=preflight_stage26()
    stage26=json.loads((STAGE26 / 'summary.json').read_text(encoding='utf-8'))
    assert stage26['manifest_sha256']==digest(STAGE26 / 'manifest.json')
    assert stage26['source_snapshot_sha256']==digest(STAGE26 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE26 / 'source_snapshot.zip',stage26['source_sha256'])
    current=source_hashes()
    assert {name:value for name,value in current.items() if name.startswith('core/')}=={
        name:value for name,value in stage26['source_sha256'].items()
        if name.startswith('core/')}
    for scene in ('weak','genuine_absent'):
        case=stage26['cases'][scene]
        for name,expected in case['input_sha256'].items():
            assert digest(STAGE26 / scene / 'input' / name)==expected
        for mode,outputs in case['output_sha256'].items():
            for name,expected in outputs.items():
                assert digest(STAGE26 / scene / mode / name)==expected
    for name,expected in stage21['cases']['no_attack_control']['input_sha256'].items():
        assert digest(STAGE21 / 'no_attack_control' / 'input' / name)==expected
    return stage21,stage26


def source_files(scene):
    if scene=='shared_control':
        folder=STAGE21 / 'no_attack_control'
        tracking=folder / 'tracking.npz'
    else:
        folder=STAGE26 / scene
        tracking=folder / 'off' / 'tracking.npz'
    return folder / 'input',tracking,folder / ('trace.npz' if scene=='shared_control' else 'off/trace.npz')


def extract_startup(record):
    """Use only the first 100 ms to acquire templates and phase calibration."""
    samples=record['samples'][ANTENNAS,:STARTUP]
    nominal=record['nominal'][ANTENNAS]
    cfg=default_config()
    cfg.acquisition.doppler_bins_hz=np.arange(-4000.,4001.,250.)
    cfg.acquisition.noncoherent_ms=100
    cfg.tracking.cn0_window_epochs=100
    cfg.tracking.max_epochs=round(record['samples'].shape[1]/FS*1000)
    cfg.tracking.code_replica_bandwidth_hz=.9e6
    cfg.tracking.refine_initial_phase=True
    cfg.tracking.carrier_discriminator='costas'
    prns=[];templates=[]
    for star,code in enumerate(record['codes']):
        acquisition=acquire_array(samples,FS,star+1,1.023e6,
                                  cfg.acquisition)
        if acquisition['success']:
            history=track(samples[0],FS,star+1,acquisition,
                          cfg.tracking,1575.42e6,1.023e6)
            if history['success']:
                prns.append(star)
                templates.append((history['epoch_start_samples'][-1]/FS,
                                  history['code_nco_phase_chips'][-1],
                                  history['code_nco_rate_hz'][-1],
                                  history['carrier_nco_frequency_hz'][-1]))
    t0=np.arange(STARTUP)/FS
    local=np.stack([local_signal(record['codes'][star],template,t0)
                    for star,template in zip(prns,templates)])
    calibration,rate=calibrate(samples,local,nominal[:,prns])
    return np.array(prns),np.array(templates),calibration,rate


def startup_templates(record,tracking):
    """Repeat frontend startup and check the saved receiver observations."""
    prns,templates,calibration,rate=extract_startup(record)
    np.testing.assert_array_equal(prns,tracking['startup_prns'])
    np.testing.assert_allclose(calibration,tracking['calibration'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(rate,tracking['calibration_rate'],rtol=0,atol=1e-12)
    return prns,templates,calibration,rate


def fit_saved_iq(record,tracking,reference_variant=0):
    """Fit every measurable block from IQ without any truth or future track."""
    prns,templates,calibration,calibration_rate=startup_templates(record,tracking)
    nominal=record['nominal'][ANTENNAS]
    count=(record['samples'].shape[1]+WIDTH-1)//WIDTH
    raw=np.full((4,count),np.nan)
    fitted=np.full((4,count,M,M),np.nan+0j)
    would=np.full_like(fitted,np.nan+0j)
    gain=np.full((4,count,nominal.shape[1]),np.nan)
    noise_power=np.full(count,np.nan)
    original=templates
    aided=np.array([aided_template(template) for template in templates])
    for block_index in range(count):
        start=block_index*WIDTH
        if start<STARTUP:continue
        end=min(start+WIDTH,record['samples'].shape[1])
        samples=np.arange(start,end)
        block=record['samples'][ANTENNAS,start:end]/(
            calibration[:,None]*np.exp(1j*calibration_rate[:,None]*samples[None,:]/FS))
        noise=float(np.mean(abs(block)**2)/2046)
        noise_power[block_index]=noise
        time=samples/FS
        for rate_index,rate_templates in enumerate((original,aided)):
            local=np.stack([local_signal(record['codes'][star],template,time)
                            for star,template in zip(prns,rate_templates)])
            chunks=len(samples)//2046
            prompts=np.einsum('mkn,skn->msk',block.reshape(M,chunks,2046),
                              local.conj().reshape(len(prns),chunks,2046))/2046
            for size_index,group in enumerate((prompts.reshape(M,-1),
                                               group_five_ms(prompts))):
                variant=2*rate_index+size_index
                repeated=np.repeat(nominal[:,prns],group.shape[1]//len(prns),axis=1)
                p,score,active,retained=trigger_fit(group,repeated,nominal,noise)
                raw[variant,block_index]=score
                fitted[variant,block_index]=p
                would[variant,block_index]=active
                gain[variant,block_index]=retained
        if block_index%50==0:print('  fitted block',block_index,'/',count,flush=True)
    indices=(tracking['time']/.02).astype(int)
    measured=indices>=STARTUP//WIDTH
    np.testing.assert_allclose(raw[reference_variant,indices[measured]],
                               tracking['ratio'][measured],rtol=0,atol=1e-10)
    return dict(block_start_s=np.arange(count)*.02,raw=raw,
                fitted_projector=fitted,would_projector=would,would_gain=gain,
                noise_power_per_prompt=noise_power,
                startup_prns=prns,startup_templates=templates,
                aided_code_rate_hz=aided[:,2],
                original_code_rate_hz=original[:,2],
                calibration=calibration,calibration_rate=calibration_rate)


def truth_audit(record,case,seed,fit,attacked):
    """Compute template error and source-direction loss after all fits exist."""
    count=len(fit['block_start_s'])
    stars=record['nominal'].shape[1]
    true_error=np.full((2,count,stars),np.nan)
    fake_error=np.full_like(true_error,np.nan)
    true_fitted=np.full((4,count,stars),np.nan)
    true_would=np.full_like(true_fitted,np.nan)
    fake_fitted=np.full((4,count),np.nan)
    fake_would=np.full_like(fake_fitted,np.nan)
    prns=fit['startup_prns']
    templates=fit['startup_templates']
    aided=np.array([aided_template(template) for template in templates])
    for index,start in enumerate(fit['block_start_s']):
        if index<STARTUP//WIDTH:continue
        time=start+.01
        genuine_phase,forged_phase=truth_code_phases(case,seed,time)
        for rate_index,rate_templates in enumerate((templates,aided)):
            local_phase=rate_templates[:,1]+rate_templates[:,2]*(time-rate_templates[:,0])
            true_error[rate_index,index,prns]=wrapped_code_error(
                local_phase,genuine_phase[prns])
            fake_error[rate_index,index,prns]=wrapped_code_error(
                local_phase,forged_phase[prns])
        genuine,fake=truth_directions(case,seed,record['nominal'][ANTENNAS],
            fit['calibration'],fit['calibration_rate'],time)
        for variant in range(4):
            true_fitted[variant,index]=projected_power_retention(
                fit['fitted_projector'][variant,index],genuine)
            true_would[variant,index]=projected_power_retention(
                fit['would_projector'][variant,index],genuine)
            if attacked and case['attack_start']<=time<case['attack_end']:
                fake_fitted[variant,index]=projected_power_retention(
                    fit['fitted_projector'][variant,index],fake)
                fake_would[variant,index]=projected_power_retention(
                    fit['would_projector'][variant,index],fake)
    return dict(true_template_error_chips=true_error,
                fake_template_error_chips=fake_error,
                fitted_true_power_retention=true_fitted,
                would_true_power_retention=true_would,
                fitted_fake_power_retention=fake_fitted,
                would_fake_power_retention=fake_would)


def phase_masks(starts,attack_start):
    measured=starts>=.1
    return dict(clean=measured&(starts<attack_start),
                post=starts>=attack_start,
                steady=starts>=.5)


def phase_statistics(fit,audit,attack_start,attacked):
    """Keep clean/post/steady and each of the four statistics separate."""
    phases={}
    for label,mask in phase_masks(fit['block_start_s'],attack_start).items():
        variants={}
        for index,name in enumerate(VARIANTS):
            scores=fit['raw'][index,mask]
            true_would=audit['would_true_power_retention'][index,mask]
            stats=dict(measured_blocks=int(len(scores)),
                       trigger_blocks=int((scores>=3.).sum()),
                       raw_median=float(np.median(scores)),
                       raw_p95=float(np.percentile(scores,95)),
                       raw_max=float(np.max(scores)),
                       nominal_gain_median=float(np.median(fit['would_gain'][index,mask])),
                       counterfactual_true_power_retention_median=float(
                           np.median(true_would)),
                       counterfactual_prn1_retention_median=float(
                           np.median(true_would[:,0])),
                       counterfactual_prn14_retention_median=float(
                           np.median(true_would[:,13])))
            rate_index=index//2
            error=abs(audit['true_template_error_chips'][rate_index,mask])
            stats['genuine_template_abs_error_median_chips']=float(np.nanmedian(error))
            stats['genuine_template_abs_error_max_chips']=float(np.nanmax(error))
            if attacked and label!='clean':
                stats['fitted_fake_power_retention_median']=float(np.median(
                    audit['fitted_fake_power_retention'][index,mask]))
                stats['would_fake_power_retention_median']=float(np.median(
                    audit['would_fake_power_retention'][index,mask]))
                fake_error=abs(audit['fake_template_error_chips'][rate_index,mask])
                stats['fake_template_abs_error_median_chips']=float(np.nanmedian(
                    fake_error))
            variants[name]=stats
        phases[label]=variants
    return phases


def plot_cases(plot_data):
    colors=('#343a40','#9b5de5','#ef476f','#118ab2')
    figure,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for row,scene in enumerate(SCENES):
        view=plot_data[scene]
        time=view['time']
        for index,name in enumerate(VARIANTS):
            axes[row,0].plot(time,view['raw'][index],color=colors[index],
                             linewidth=.75,label=name)
            axes[row,1].plot(time,view['true_retention'][index],
                             color=colors[index],linewidth=.75,label=name)
            if scene!='shared_control':
                axes[row,1].scatter(time,view['fake_retention'][index],
                                    color=colors[index],s=3,alpha=.3)
        axes[row,0].axhline(3.,color='black',linestyle='--',linewidth=.8)
        axes[row,0].set(title=scene,ylabel='Raw fitted trigger (threshold 3)')
        axes[row,1].set(ylabel='Would-project true direction retention; fake dots')
        for axis in axes[row]:
            axis.set(xlabel='Block start time (s)')
            axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    figure.savefig(OUTPUT / 'overview.png',dpi=150)
    plt.close(figure)


def main():
    stage21,stage26=preflight_inputs()
    sources=source_hashes()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    inputs={}
    for scene in SCENES:
        if scene=='shared_control':
            inputs[scene]=stage21['cases']['no_attack_control']['input_sha256']
        else:
            inputs[scene]=stage26['cases'][scene]['input_sha256']
    plan=dict(scope=__doc__,scenes=SCENES,variants=VARIANTS,
              input_sha256=inputs,source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
              stage26_summary_sha256=digest(STAGE26 / 'summary.json'),
              startup_duration_s=.1,block_duration_s=.02,
              group_duration_s=.005,threshold=3.,
              five_ms_definition='complex sum/sqrt(5) of unique same-PRN 1 ms prompts; no bit wiping',
              aided_rate_definition='1.023e6*(1+startup_carrier_hz/1575.42e6)',
              noise_definition='raw calibrated block mean power/2046 for both prompt sizes',
              no_receiver_navigation=True)
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),input_sha256=inputs)),
        indent=2,allow_nan=False),encoding='utf-8')
    cases={};plot_data={}
    for scene in SCENES:
        input_folder,tracking_path,trace_path=source_files(scene)
        with np.load(input_folder / 'observations.npz') as saved:
            record=dict(saved)
        record['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        with np.load(tracking_path) as saved:tracking=dict(saved)
        with np.load(trace_path) as saved:
            baseline_dimensions=saved['trace_dimensions'].copy()
            baseline_projectors=saved['trace_projectors'].copy()
        metadata=json.loads((input_folder / 'metadata.json').read_text(encoding='utf-8'))
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene,'startup and four-factor block fits start',flush=True)
        fit=fit_saved_iq(record,tracking)
        raw_path=(STAGE25 / 'shared_control' / 'tracking.npz' if scene=='shared_control'
                  else STAGE26 / scene / 'on' / 'tracking.npz')
        with np.load(raw_path) as saved:
            np.testing.assert_allclose(fit['raw'][0,5:],
                                       saved['beam_trigger_raw'][5:],
                                       rtol=0,atol=1e-10)
        np.testing.assert_array_equal(fit['would_projector'][0,5:],
                                      baseline_projectors[5:])
        np.testing.assert_array_equal((fit['raw'][0,5:]>=3.).astype(int),
                                      (baseline_dimensions[5:]==2).astype(int))
        np.savez_compressed(folder / 'fit.npz',**fit)
        print(scene,'fit saved; truth-only audit start',flush=True)
        audit=truth_audit(record,metadata['case'],metadata['seed'],fit,
                          metadata['attacked'])
        np.savez_compressed(folder / 'truth_audit.npz',**audit)
        cases[scene]=dict(
            source_input_sha256=inputs[scene],
            source_tracking_sha256=digest(tracking_path),
            source_block_raw_sha256=digest(raw_path),
            source_trace_sha256=digest(trace_path),
            output_sha256={name:digest(folder / name)
                           for name in ('fit.npz','truth_audit.npz')},
            startup_prns=fit['startup_prns'],
            startup_original_code_rate_hz=fit['original_code_rate_hz'],
            startup_aided_code_rate_hz=fit['aided_code_rate_hz'],
            startup_rate_difference_hz=(fit['aided_code_rate_hz']-
                                        fit['original_code_rate_hz']),
            phase_statistics=phase_statistics(fit,audit,
                                              metadata['case']['attack_start'],
                                              metadata['attacked']))
        true_curve=np.full_like(fit['raw'],np.nan)
        true_curve[:,5:]=np.median(
            audit['would_true_power_retention'][:,5:,:],axis=2)
        plot_data[scene]=dict(time=fit['block_start_s'],raw=fit['raw'],
            true_retention=true_curve,
            fake_retention=audit['would_fake_power_retention'])
        del record,tracking,fit,audit
        gc.collect()
        print(scene,'diagnostic saved',flush=True)
    plot_cases(plot_data)
    assert source_hashes()==sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=cases)),
        indent=2,allow_nan=False),encoding='utf-8')
    return cases


if __name__=='__main__':
    main()
