"""Offline aided beam fits for fixed 1/5/10/20 ms prompt groups.

This reads saved IQ from the minus-five-dB pair and the shared control. It
repeats 100 ms startup acquisition/tracking and the 20 ms block fit, without
the full receiver frontend, navigation, bit wiping, or truth in the fit.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from core.spoofing.single_source_scene import local_signal
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_long_coherent import OUTPUT as STAGE21,source_snapshot
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_spoof_power_minus_five import (
    OUTPUT as STAGE29,preflight as preflight_stage29,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,digest,
)
from experiments.spoofing.single_source_template_coherence import (
    OUTPUT as STAGE27,STARTUP,WIDTH,aided_template,phase_masks,
    projected_power_retention,startup_templates,trigger_fit,
    truth_code_phases,truth_directions,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'beam_coherence_lengths'
SCENES=('weak','genuine_absent','shared_control')
LENGTHS=(1,5,10,20)


def group_prompts(prompts,length):
    """Complex sum/sqrt(L) on aligned, same-PRN 1 ms prompts."""
    antennas,stars,milliseconds=prompts.shape
    return (prompts.reshape(antennas,stars,milliseconds//length,length)
            .sum(axis=-1)/np.sqrt(length)).reshape(antennas,-1)


def fit_prompt_lengths(prompts,desired,nominal,noise):
    """Fit all fixed lengths using each length's actual number of columns."""
    fitted=[];raw=[];would=[];gain=[]
    for length in LENGTHS:
        grouped=group_prompts(prompts,length)
        repeated=np.repeat(desired,20//length,axis=1)
        p,score,active,retained=trigger_fit(grouped,repeated,nominal,noise)
        fitted.append(p);raw.append(score)
        would.append(active);gain.append(retained)
    return np.array(fitted),np.array(raw),np.array(would),np.array(gain)


def preflight():
    """Bind all used IQ, tracking, and reference-fit files to frozen sources."""
    _,_,current=preflight_stage29()
    stage29=json.loads((STAGE29 / 'summary.json').read_text(encoding='utf-8'))
    stage27=json.loads((STAGE27 / 'summary.json').read_text(encoding='utf-8'))
    assert stage29['source_snapshot_sha256']==digest(STAGE29 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE29 / 'source_snapshot.zip',stage29['source_sha256'])
    assert stage29['plan_sha256']==digest(STAGE29 / 'plan.json')
    assert stage29['manifest_sha256']==digest(STAGE29 / 'manifest.json')
    assert stage29['figure_sha256']==digest(STAGE29 / 'overview.png')
    assert {key:value for key,value in current.items() if key.startswith('core/')}=={
        key:value for key,value in stage29['source_sha256'].items()
        if key.startswith('core/')}
    for scene in SCENES:
        if scene=='shared_control':
            assert stage27['cases'][scene]['output_sha256']['fit.npz']==digest(
                STAGE27 / scene / 'fit.npz')
        else:
            case=stage29['cases'][scene]
            for name,sha in case['input_sha256'].items():
                assert digest(STAGE29 / scene / 'input' / name)==sha
            for mode,outputs in case['output_sha256'].items():
                for name,sha in outputs.items():
                    assert digest(STAGE29 / scene / mode / name)==sha
            assert case['fit_sha256']==digest(STAGE29 / scene / 'fit.npz')
    return stage29,stage27,current


def source_files(scene):
    if scene=='shared_control':
        folder=STAGE21 / 'no_attack_control'
        return folder / 'input',folder / 'tracking.npz',STAGE27 / scene / 'fit.npz'
    folder=STAGE29 / scene
    return (folder / 'input',folder / 'default' / 'tracking.npz',
            folder / 'fit.npz')


def fit_saved_iq_lengths(record,tracking):
    """Repeat only the aided beam's independent 20 ms block fit."""
    prns,templates,calibration,calibration_rate=startup_templates(record,tracking)
    aided=np.array([aided_template(template) for template in templates])
    nominal=record['nominal'][ANTENNAS]
    antennas=len(ANTENNAS)
    count=(record['samples'].shape[1]+WIDTH-1)//WIDTH
    raw=np.full((len(LENGTHS),count),np.nan)
    fitted=np.full((len(LENGTHS),count,antennas,antennas),np.nan+0j)
    would=np.full_like(fitted,np.nan+0j)
    gain=np.full((len(LENGTHS),count,nominal.shape[1]),np.nan)
    noise_power=np.full(count,np.nan)
    for block_index in range(count):
        start=block_index*WIDTH
        if start<STARTUP:continue
        end=min(start+WIDTH,record['samples'].shape[1])
        samples=np.arange(start,end)
        block=record['samples'][ANTENNAS,start:end]/(
            calibration[:,None]*np.exp(1j*calibration_rate[:,None]*
                                         samples[None,:]/FS))
        noise=float(np.mean(abs(block)**2)/2046)
        noise_power[block_index]=noise
        time=samples/FS
        local=np.stack([local_signal(record['codes'][star],template,time)
                        for star,template in zip(prns,aided)])
        chunks=len(samples)//2046
        prompts=np.einsum('mkn,skn->msk',
                          block.reshape(antennas,chunks,2046),
                          local.conj().reshape(len(prns),chunks,2046))/2046
        p,score,active,retained=fit_prompt_lengths(
            prompts,nominal[:,prns],nominal,noise)
        fitted[:,block_index]=p;raw[:,block_index]=score
        would[:,block_index]=active;gain[:,block_index]=retained
        if block_index%50==0:
            print('  fitted block',block_index,'/',count,flush=True)
    return dict(block_start_s=np.arange(count)*.02,length_ms=np.array(LENGTHS),
                raw=raw,fitted_projector=fitted,would_projector=would,
                would_gain=gain,noise_power_per_prompt=noise_power,
                startup_prns=prns,startup_templates=templates,
                aided_code_rate_hz=aided[:,2],calibration=calibration,
                calibration_rate=calibration_rate)


def audit_saved_lengths(fit,reference):
    """The L=1/5 paths reproduce the frozen aided fits on this same IQ."""
    for index,old in ((0,2),(1,3)):
        np.testing.assert_allclose(fit['raw'][index,5:],
                                   reference['raw'][old,5:],rtol=0,atol=1e-10)
        for key in ('fitted_projector','would_projector'):
            np.testing.assert_allclose(fit[key][index,5:],
                                       reference[key][old,5:],
                                       rtol=0,atol=1e-9)
        np.testing.assert_allclose(fit['would_gain'][index,5:],
                                   reference['would_gain'][old,5:],
                                   rtol=0,atol=1e-9)
    for key in ('noise_power_per_prompt','startup_prns',
                'aided_code_rate_hz','calibration','calibration_rate'):
        np.testing.assert_allclose(fit[key],reference[key],rtol=0,atol=1e-10)


def truth_audit(record,metadata,fit):
    """Apply known directions and bit-delay geometry only after fitting."""
    count=len(fit['block_start_s'])
    stars=record['nominal'].shape[1]
    true_fitted=np.full((len(LENGTHS),count,stars),np.nan)
    true_would=np.full_like(true_fitted,np.nan)
    fake_fitted=np.full((len(LENGTHS),count),np.nan)
    fake_would=np.full_like(fake_fitted,np.nan)
    true_delay=np.full((count,stars),np.nan)
    fake_delay=np.full_like(true_delay,np.nan)
    case=metadata['case']
    for index,start in enumerate(fit['block_start_s']):
        if index<STARTUP//WIDTH:continue
        time=start+.01
        genuine_phase,forged_phase=truth_code_phases(
            case,metadata['seed'],time)
        true_delay[index]=(time-genuine_phase/1.023e6)%.02*1000.
        if metadata['attacked'] and case['attack_start']<=time<case['attack_end']:
            fake_delay[index]=(time-forged_phase/1.023e6)%.02*1000.
        genuine,fake=truth_directions(case,metadata['seed'],
            record['nominal'][ANTENNAS],fit['calibration'],
            fit['calibration_rate'],time)
        for length_index in range(len(LENGTHS)):
            true_fitted[length_index,index]=projected_power_retention(
                fit['fitted_projector'][length_index,index],genuine)
            true_would[length_index,index]=projected_power_retention(
                fit['would_projector'][length_index,index],genuine)
            if metadata['attacked'] and case['attack_start']<=time<case['attack_end']:
                fake_fitted[length_index,index]=projected_power_retention(
                    fit['fitted_projector'][length_index,index],fake)
                fake_would[length_index,index]=projected_power_retention(
                    fit['would_projector'][length_index,index],fake)
    return dict(fitted_true_power_retention=true_fitted,
                would_true_power_retention=true_would,
                fitted_fake_power_retention=fake_fitted,
                would_fake_power_retention=fake_would,
                genuine_propagation_delay_mod20_ms=true_delay,
                fake_propagation_delay_mod20_ms=fake_delay)


def summarize_fit(fit,audit,attack_start,attacked):
    phases={}
    for label,mask in phase_masks(fit['block_start_s'],attack_start).items():
        variants={}
        for index,length in enumerate(LENGTHS):
            scores=fit['raw'][index,mask]
            counts=(fit['would_gain'][index,mask]>=.05).sum(axis=1)
            true_retention=audit['would_true_power_retention'][index,mask]
            stats=dict(blocks=int(mask.sum()),triggered=int((scores>=3.).sum()),
                       raw_min=float(np.min(scores)),
                       raw_median=float(np.median(scores)),
                       raw_p95=float(np.percentile(scores,95)),
                       raw_max=float(np.max(scores)),
                       nominal_gain_ge05_min=int(np.min(counts)),
                       nominal_gain_ge05_median=float(np.median(counts)),
                       nominal_gain_ge05_max=int(np.max(counts)),
                       true_direction_retention_median=float(np.median(
                           true_retention)))
            if attacked and label!='clean':
                fake=audit['would_fake_power_retention'][index,mask]
                stats.update(fake_direction_retention_min=float(np.min(fake)),
                             fake_direction_retention_median=float(np.median(fake)),
                             fake_direction_retention_max=float(np.max(fake)))
            variants[f'{length}ms']=stats
        phases[label]=variants
    return phases


def plot_cases(plot_data):
    figure,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    colors=('#5a6770','#9163a6','#df6a3c','#18929a')
    for row,scene in enumerate(SCENES):
        view=plot_data[scene]
        time=view['time']
        for index,length in enumerate(LENGTHS):
            axes[row,0].plot(time,view['raw'][index],color=colors[index],
                             linewidth=.7,label=f'{length} ms')
            axes[row,1].plot(time,view['true'][index],color=colors[index],
                             linewidth=.7,label=f'{length} ms genuine')
            if scene!='shared_control':
                axes[row,1].scatter(time,view['fake'][index],
                                    color=colors[index],s=3,alpha=.3)
        axes[row,0].axhline(3.,color='black',linestyle='--',linewidth=.8)
        title=(scene+' (genuine direction counterfactual)'
               if scene=='genuine_absent' else scene)
        axes[row,0].set(title=title,ylabel='Raw fitted trigger')
        axes[row,1].set(ylabel='Direction power retained (fake: dots)')
        for axis in axes[row]:
            axis.set(xlabel='Block start time (s)')
            axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    figure.savefig(OUTPUT / 'overview.png',dpi=150)
    plt.close(figure)


def main():
    stage29,stage27,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    inputs={}
    for scene in SCENES:
        inputs[scene]=(stage27['cases'][scene]['source_input_sha256']
                       if scene=='shared_control' else
                       stage29['cases'][scene]['input_sha256'])
    plan=dict(scope=__doc__,scenes=SCENES,length_ms=LENGTHS,
              startup_duration_s=.1,block_duration_s=.02,threshold=3.,
              group_definition='same-PRN aligned complex sum/sqrt(L), no bit wiping',
              rate_definition='1.023e6*(1+startup_carrier_hz/1575.42e6)',
              noise_definition='calibrated block mean power/2046',
              bit_delay_audit='propagation delay modulo 20 ms locates possible bit boundaries; adjacent bits may have the same sign',
              same_numeric_threshold_not_same_false_alarm_rate=True,
              no_full_frontend_or_navigation=True,input_sha256=inputs,
              stage29_summary_sha256=digest(STAGE29 / 'summary.json'),
              stage27_summary_sha256=digest(STAGE27 / 'summary.json'),
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),input_sha256=inputs)),
        indent=2,allow_nan=False),encoding='utf-8')
    cases={};plot_data={}
    for scene in SCENES:
        input_folder,tracking_path,reference_path=source_files(scene)
        with np.load(input_folder / 'observations.npz') as saved:
            record=dict(saved)
        record['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        with np.load(tracking_path) as saved:
            tracking=dict(saved)
        with np.load(reference_path) as saved:
            reference=dict(saved)
        metadata=json.loads((input_folder / 'metadata.json').read_text(
            encoding='utf-8'))
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene,'aided 1/5/10/20 ms block fits start',flush=True)
        fit=fit_saved_iq_lengths(record,tracking)
        audit_saved_lengths(fit,reference)
        np.savez_compressed(folder / 'fit.npz',**fit)
        print(scene,'fits saved; truth-only audit start',flush=True)
        audit=truth_audit(record,metadata,fit)
        np.savez_compressed(folder / 'truth_audit.npz',**audit)
        cases[scene]=dict(source_input_sha256=inputs[scene],
                          source_tracking_sha256=digest(tracking_path),
                          reference_fit_sha256=digest(reference_path),
                          output_sha256={name:digest(folder / name)
                                         for name in ('fit.npz','truth_audit.npz')},
                          startup_prns=fit['startup_prns'],
                          startup_aided_code_rate_hz=fit['aided_code_rate_hz'],
                          phase_statistics=summarize_fit(fit,audit,
                              metadata['case']['attack_start'],metadata['attacked']))
        plot_data[scene]=dict(time=fit['block_start_s'].copy(),
            raw=fit['raw'].copy(),
            true=np.median(audit['would_true_power_retention'],axis=2),
            fake=audit['would_fake_power_retention'].copy())
        del record,tracking,reference,fit,audit
        gc.collect()
        print(scene,'audit saved',flush=True)
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
