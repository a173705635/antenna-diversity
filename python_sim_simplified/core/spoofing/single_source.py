"""Continuous sampled RF -> acquisition -> tracking -> aided PVT research.

No attack schedule or true position is supplied. Nominal attitude, ephemeris,
code-period aiding and synchronized samples remain explicit assumptions.
"""
from dataclasses import replace
import numpy as np
from ..acquisition import acquire
from .array_acquisition import acquire_array
from ..tracking import track
from ..presets import default_config
from ..positioning import code_pseudorange, carrier_range, solve_position
from .single_source_scene import local_signal
from .recovery_geometry import FS, WAVELENGTH, position_solution
from ..code_replica import _sample_code
from .array_processing import prompt_projector,startup_relative_amplitude
from ..gps_simulation import receiver_reference
from .mechanism_phase import (phase_prediction, phase_prediction_leave_one_out,
                              phase_prediction_long_anchor)
from .mechanism_smoothing import smooth_history


def coherent_cn0(prompt,width):
    """Estimate (E[I虏]-E[Q虏])/(2 E[Q虏] T); phase alignment is required.

    Nonpositive coherent power is an unobservable signal estimate, represented
    as NaN. This is not an acquisition metric or an independent phase test.
    """
    count=np.minimum(np.arange(len(prompt))+1,width)
    i=np.convolve(prompt.real**2,np.ones(width),'full')[:len(prompt)]/count
    q=np.convolve(prompt.imag**2,np.ones(width),'full')[:len(prompt)]/count
    # Fitted initial phase leaves no single-sample noise degree of freedom.
    measured=(count>=2)&(i>q)
    result=np.full(len(prompt),np.nan)
    result[measured]=10*np.log10((i[measured]-q[measured])/(2*q[measured]*.001))
    return result


def _coherent_loss_quality(prefix,lock,pll):
    """Apply the receive quality gate to the latest completed 100 ms prompt."""
    recent=prefix[-100:]
    i=np.convolve(recent.real**2,np.ones(100),'valid')[0]/100
    q=np.convolve(recent.imag**2,np.ones(100),'valid')[0]/100
    cn0=(10*np.log10((i-q)/(2*q*.001)) if i>q else np.nan)
    snr=10**(cn0/10)*.001
    expected=snr/(1+snr)
    return lock>=.8*expected and cn0>=25 and abs(pll)<np.pi/2


def receive_measured(signal,prn,acquisition_cfg,tracking_cfg,fast_truncation=False):
    window=round(FS*.001*acquisition_cfg.noncoherent_ms*acquisition_cfg.coherent_ms)
    cfg=replace(tracking_cfg,acquisition_ms=acquisition_cfg.noncoherent_ms*acquisition_cfg.coherent_ms)
    cursor=0;segments=[]
    while cursor+window+4092<=len(signal):
        acq=acquire(signal[cursor:cursor+window],FS,prn,1.023e6,acquisition_cfg)
        if not acq['success']:
            cursor+=window
            continue
        if fast_truncation:
            h=track(signal[cursor:],FS,prn,acq,cfg,1575.42e6,1.023e6,
                    loss_confirm_epochs=5,loss_settle_epochs=100,
                    loss_quality=_coherent_loss_quality)
        else:
            h=track(signal[cursor:],FS,prn,acq,cfg,1575.42e6,1.023e6)
        if not h['success']:break
        h['cn0_db_hz']=coherent_cn0(h['prompt'],100)
        snr=10**(h['cn0_db_hz']/10)*.001
        expected=snr/(1+snr)
        good=(h['phase_lock_metric']>=.8*expected)&(h['cn0_db_hz']>=25)&(abs(h['pll_error_rad'])<np.pi/2)
        bad=0;end=None
        for k in range(100,len(good)):
            bad=0 if good[k] else bad+1
            if bad>=5:
                end=k+1
                break
        if fast_truncation and h['status']=='tracking_quality_lost':
            assert end==len(h['prompt'])
        if end is not None:
            h={key:value[:end] if isinstance(value,np.ndarray) else value for key,value in h.items()}
        h['epoch_start_samples']+=cursor
        segments.append(dict(history=h,tracking_id=f'tracking-{len(segments)}'))
        cursor=int(h['epoch_start_samples'][-1])+2046
        if end is None:break
    return dict(segments=segments)


def spatial_match(block,local,desired):
    prompts=(block*local.conj()).reshape(len(block),-1,2046).mean(axis=2)
    covariance=prompts@prompts.conj().T
    return float(np.vdot(desired,covariance@desired).real/(len(desired)*np.trace(covariance).real))


def calibrate(block, local, desired):
    m,n=block.shape;chunks=n//2046
    z=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(len(local),chunks,2046))/2046
    cross=np.sum(z*z[0].conj()[None,:,:]*(desired[0]/desired)[:,:,None],axis=1)
    time=(np.arange(chunks)+.5)*.001
    angles=np.unwrap(np.angle(cross),axis=1).T
    rate,phase=np.polyfit(time,angles,1)
    residual=angles-(time[:,None]*rate+phase)
    error=np.sqrt(np.sum(residual**2,axis=0)/(chunks-2)/np.sum((time-time.mean())**2))
    rate=np.where(abs(rate)>3*error,rate,0.)
    phase=angles.mean(axis=0)-time.mean()*rate
    return np.exp(1j*phase),rate



def residual_trigger(prompts,desired,noise=1.):
    """Score residual reduction per added complex parameter and noise power.

    Noise is measured raw sample power / coherent sample count, so unresolved
    signals inflate this estimate. Threshold 3 is a research setting, not a Pfa.
    """
    fitted=desired*np.sum(desired.conj()*prompts,axis=0)/len(desired)
    projector,_=prompt_projector(prompts,desired,6)
    projected=projector@desired
    amplitude=np.sum(projected.conj()*prompts,axis=0)/(np.sum(desired.conj()*projected,axis=0).real+.01*len(desired))
    residual=projector@(prompts-desired*amplitude)
    reduction=np.sum(abs(prompts-fitted)**2)-np.sum(abs(residual)**2)
    return projector,float(reduction/(noise*(prompts.shape[1]+len(desired)-1)))


def source_evidence(block,local,desired):
    """At least two residual dimensions and a nonzero desired vector required.

    On-direction energy is compared with mean orthogonal energy. White noise
    gives a ratio near one; threshold three is an uncalibrated research choice.
    """
    prompts=(block*local.conj()).reshape(len(block),-1,2046).mean(axis=2)
    covariance=prompts@prompts.conj().T
    on=np.vdot(desired,covariance@desired).real/np.vdot(desired,desired).real
    off=(np.trace(covariance).real-on)/(len(block)-1)
    return float(on/off)


def split_source_evidence(block,local,steering,bases,begin,end,width,return_energy=False):
    """Combine prompt energies from the actual spatial basis of each block."""
    on_energy=0.;off_energy=0.
    cursor=begin
    while cursor<end:
        segment_end=min(end,(cursor//width+1)*width)
        offset=cursor-begin
        length=segment_end-cursor
        basis=bases[cursor//width]
        desired=basis.conj().T@steering
        dimension=basis.shape[1]
        if dimension<2 or np.vdot(desired,desired).real/len(steering)<.05:
            return (np.nan,np.nan,np.nan) if return_energy else np.nan
        projected=basis.conj().T@block[:,offset:offset+length]
        prompts=(projected*local[None,offset:offset+length].conj()).reshape(dimension,-1,2046).mean(axis=2)
        segment_on=np.sum(abs(desired.conj()@prompts)**2)/np.vdot(desired,desired).real
        on_energy+=segment_on
        off_energy+=(np.sum(abs(prompts)**2)-segment_on)/(dimension-1)
        cursor=segment_end
    ratio=float(on_energy/off_energy)
    return (ratio,float(on_energy),float(off_energy)) if return_energy else ratio


def temporal_beam_trigger(raw, history, beam_projection_threshold=3.):
    """Use measured current/past five block scores; keep the current fit."""
    history.append(raw)
    del history[:-5]
    mean=float(np.mean(history))
    effective=max(raw,mean)
    return mean,effective,raw<beam_projection_threshold<=effective


def beam(block, a, local, startup_cov, prns, trigger_history=None,
         coherent_ms=1, beam_projection_threshold=3.):
    m,n=block.shape;chunks=n//2046
    prompts=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(len(local),chunks,2046))/2046
    if coherent_ms in (5,10):
        prompts=(prompts.reshape(m,len(local),chunks//coherent_ms,coherent_ms)
                 .sum(axis=-1)/np.sqrt(coherent_ms))
        chunks//=coherent_ms
    else:assert coherent_ms==1
    noise=float(np.mean(abs(block)**2)/2046)
    p,raw=residual_trigger(prompts.reshape(m,-1),np.repeat(a[:,prns],chunks,axis=1),noise)
    mean=raw
    effective=raw
    if trigger_history is not None:
        mean,effective,_=temporal_beam_trigger(
            raw,trigger_history,beam_projection_threshold)
    if effective < beam_projection_threshold:
        output=(a/len(a),np.ones(a.shape[1]),raw)
    else:
        projected=p@a
        response=np.sum(a.conj()*projected,axis=0).real
        output=(projected/(response+.01)[None,:],response/len(a),raw)
    return output if trigger_history is None else (*output,mean,effective)


def beam_history_weight(weights, steering, ratio, history, blocks,
                        beam_projection_threshold=3.):
    """Replace a triggered block's null using recent antenna-space nulls.

    History includes the current measured block. Its phase-independent outer
    products are averaged before extracting the dominant null direction.
    """
    if ratio < beam_projection_threshold:
        history.clear()
        return weights, np.ones(steering.shape[1]), 0, np.nan
    left, values, _ = np.linalg.svd(weights, full_matrices=True)
    assert values[-1] <= 1e-10 * values[0] < values[-2]
    null = left[:, -1]
    history.append(np.outer(null, null.conj()))
    del history[:-blocks]
    eigenvalues, vectors = np.linalg.eigh(np.mean(history, axis=0))
    direction = vectors[:, -1]
    projector = np.eye(len(steering), dtype=complex) - np.outer(direction, direction.conj())
    projected = projector @ steering
    response = np.sum(steering.conj() * projected, axis=0).real
    return projected / (response + .01)[None, :], response / len(steering), len(history), float(eigenvalues[-1] - eigenvalues[-2])


LOOP_HISTORY_FIELDS = (
    'epoch_start_samples', 'prompt', 'pll_error_rad', 'fll_error_hz',
    'carrier_nco_phase_rad', 'carrier_nco_frequency_hz', 'carrier_frequency_hz',
    'code_nco_phase_chips', 'code_nco_rate_hz', 'dll_error', 'cn0_db_hz',
    'phase_lock_metric',
)


def frontend(record, subset, split_support=False, receiver_pll=None, beam_history_blocks=1,
             evidence_trace=False, temporal_beam_trigger_enabled=False,
             beam_template_carrier_aiding=False, beam_coherent_ms=1,
             stream_memmap_path=None, beam_startup_amplitude=False, loop_history=None,
             fast_truncation=False, beam_projection_threshold=3.):
    assert not evidence_trace or split_support
    assert not (temporal_beam_trigger_enabled and beam_history_blocks != 1)
    x=record['samples'][subset]; a=record['nominal'][subset]
    if beam_startup_amplitude:assert len(a)==3
    codes=record['codes']; times=record['times']; stars=len(codes); count=len(times)
    cfg=default_config(); cfg.acquisition.doppler_bins_hz=np.arange(-4000.,4001.,250.)
    cfg.acquisition.noncoherent_ms=100
    cfg.tracking.cn0_window_epochs=100
    cfg.tracking.max_epochs=round(x.shape[1]/FS*1000)
    cfg.tracking.code_replica_bandwidth_hz=.9e6;cfg.tracking.refine_initial_phase=True
    cfg.tracking.carrier_discriminator='costas'
    receive_tracking=cfg.tracking if receiver_pll is None else replace(
        cfg.tracking,pll_kp=receiver_pll[0],pll_ki_hz=receiver_pll[1])
    startup=round(.1*FS); width=round(.02*FS)
    templates=[]; prns=[]
    for i in range(stars):
        acq=acquire_array(x[:,:startup],FS,i+1,1.023e6,cfg.acquisition)
        if acq['success']:
            h=track(x[0,:startup],FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
            if h['success']:
                prns.append(i)
                templates.append((h['epoch_start_samples'][-1]/FS,h['code_nco_phase_chips'][-1],
                                  h['code_nco_rate_hz'][-1],h['carrier_nco_frequency_hz'][-1]))
    # At least four actually acquired/tracked PRNs are required for startup.
    ranges=np.full((count,stars),np.nan); lock=ranges.copy(); cn0=ranges.copy(); carrier=ranges.copy();support=ranges.copy();arc=np.full((count,stars),-1,int)
    if split_support:
        alternative_support=support.copy()
        split_on_energy=support.copy();split_off_energy=support.copy()
    if evidence_trace:
        trace_k=[];trace_star=[];trace_arc=[];trace_start=[];trace_available=[];trace_prompts=[]
    if len(prns)<4:
        result=dict(time=times,available_time=times+.02,satellites=record['satellites'],
                    ranges=ranges,lock=lock,cn0=cn0,carrier=carrier,gain=np.zeros_like(ranges),
                    startup_prns=np.array(prns),calibration=np.full(len(a),np.nan),ratio=np.zeros(count),support=support,arc=arc)
        if split_support:
            result.update(support_split=alternative_support,
                          split_on_energy=split_on_energy,split_off_energy=split_off_energy)
        if evidence_trace:
            result.update(trace_window_k=np.empty(0,int),trace_star=np.empty(0,int),
                          trace_arc=np.empty(0,int),trace_start_sample=np.empty(0,int),
                          trace_available_time=np.empty(0),
                          trace_prompts=np.empty((0,20,len(a)),complex),
                          trace_projectors=np.empty((0,len(a),len(a)),complex),
                          trace_dimensions=np.empty(0,int),trace_nominal=a)
            if beam_template_carrier_aiding or beam_coherent_ms!=1:
                result['trace_raw_trigger']=np.full((x.shape[1]+width-1)//width,np.nan)
        if temporal_beam_trigger_enabled:
            blocks=(x.shape[1]+width-1)//width
            result.update(beam_trigger_raw=np.full(blocks,np.nan),
                          beam_trigger_history_mean=np.full(blocks,np.nan),
                          beam_trigger_effective=np.full(blocks,np.nan),
                          beam_trigger_history_only=np.zeros(blocks,bool))
        return result
    t0=np.arange(startup)/FS
    local0=np.stack([local_signal(codes[i],template,t0) for i,template in zip(prns,templates)])
    gain,rate=calibrate(x[:,:startup],local0,a[:,prns])
    beam_templates=templates
    if beam_template_carrier_aiding:
        beam_templates=[(epoch,phase,1.023e6*(1.+frequency/1575.42e6),frequency)
                        for epoch,phase,_,frequency in templates]
    beam_options={} if beam_coherent_ms==1 else {'coherent_ms':beam_coherent_ms}
    if beam_projection_threshold!=3.:
        beam_options['beam_projection_threshold']=beam_projection_threshold
    calibrated=x/(gain[:,None]*np.exp(1j*rate[:,None]*np.arange(x.shape[1])[None,:]/FS))
    startup_cov=calibrated[:,:startup]@calibrated[:,:startup].conj().T/startup
    model=a
    if beam_startup_amplitude:
        prompts=np.einsum('mkn,skn->msk',
            calibrated[:,:startup].reshape(len(a),startup//2046,2046),
            local0.conj().reshape(len(prns),startup//2046,2046))/2046
        relative_amplitude,_=startup_relative_amplitude(prompts,a,np.array(prns))
        model=relative_amplitude[:,None]*a
    if stream_memmap_path is None:
        stream=np.empty((stars,x.shape[1]),complex)
    else:
        stream=np.memmap(stream_memmap_path,dtype=np.complex128,mode='w+',
                         shape=(stars,x.shape[1]),order='C')
    gains=[];ratios=[];bases=[]
    if temporal_beam_trigger_enabled:
        trigger_history=[];trigger_raw=[];trigger_mean=[]
        trigger_effective=[];trigger_history_only=[]
    if beam_history_blocks != 1:
        assert beam_history_blocks == 5
        assert np.all(rate == 0.), "Beam history study assumes static calibration"
        null_history=[];history_weights=[];history_counts=[];history_gaps=[]
    for start in range(0,x.shape[1],width):
        block=calibrated[:,start:start+width]
        t=np.arange(start,start+len(block[0]))/FS
        local=np.stack([local_signal(codes[i],template,t) for i,template in zip(prns,beam_templates)])
        # Templates only exist for acquired PRNs; beam steering still covers all.
        if start < startup:
            w=model/len(a); retained=np.ones(stars); ratio=0.
            if temporal_beam_trigger_enabled:
                trigger_raw.append(np.nan);trigger_mean.append(np.nan)
                trigger_effective.append(np.nan);trigger_history_only.append(False)
        else:
            if temporal_beam_trigger_enabled:
                w,retained,raw,mean,ratio=beam(
                    block,model,local,startup_cov,prns,trigger_history=trigger_history,
                    **beam_options)
                trigger_raw.append(raw);trigger_mean.append(mean)
                trigger_effective.append(ratio)
                trigger_history_only.append(
                    raw<beam_projection_threshold<=ratio)
            else:
                w,retained,ratio=beam(block,model,local,startup_cov,prns,
                                      **beam_options)
        if beam_history_blocks != 1:
            w,retained,history_count,eigengap=beam_history_weight(
                w,model,ratio,null_history,beam_history_blocks,
                beam_projection_threshold)
            history_weights.append(w);history_counts.append(history_count);history_gaps.append(eigengap)
        if ratio>=beam_projection_threshold:
            _,vectors=np.linalg.eigh(w@w.conj().T);basis=vectors[:,1:]
        else:basis=np.eye(len(a))
        bases.append(basis)
        stream[:,start:start+block.shape[1]]=w.conj().T@block
        gains.append(retained)
        ratios.append(raw if temporal_beam_trigger_enabled and start >= startup else ratio)
    if stream_memmap_path is not None:
        stream.flush()
    for i,signal in enumerate(stream):
        if fast_truncation:
            reception=receive_measured(signal,i+1,cfg.acquisition,receive_tracking,
                                       fast_truncation=True)
        else:
            reception=receive_measured(signal,i+1,cfg.acquisition,receive_tracking)
        for identity,segment in enumerate(reception['segments']):
                h=segment['history'];starts=h['epoch_start_samples']/FS
                if loop_history is not None:
                    loop_history.append(dict(prn=i+1,arc=identity,
                        **{name:np.array(h[name],copy=True) for name in LOOP_HISTORY_FIELDS}))
                selected=(times>=starts[0]+.02)&(times<starts[-1]+.001)
                rr=code_pseudorange(h,times,FS,.001,record['coarse'][i])
                ranges[selected,i]=rr[selected]
                carrier[selected,i]=carrier_range(h,times,FS,.001,WAVELENGTH)[selected]
                idx=np.searchsorted(starts+.001,times[selected],side='right')-1
                lock[selected,i]=h['phase_lock_metric'][idx];cn0[selected,i]=h['cn0_db_hz'][idx];arc[selected,i]=identity
                for k in np.flatnonzero(selected):
                    t=times[k]
                    end=round(t*FS);begin=end-width;sample_time=np.arange(begin,end)/FS
                    hi=np.searchsorted(starts,sample_time,side='right')-1
                    dt=sample_time-starts[hi]
                    code_phase=h['code_nco_phase_chips'][hi]+h['code_nco_rate_hz'][hi]*dt
                    phase=h['carrier_nco_phase_rad'][hi]+2*np.pi*h['carrier_nco_frequency_hz'][hi]*dt
                    local=_sample_code(codes[i],code_phase)*np.exp(1j*phase)
                    if evidence_trace:
                        prompts=(calibrated[:,begin:end]*local.conj()).reshape(len(a),20,2046).mean(axis=2).T
                        trace_k.append(k);trace_star.append(i);trace_arc.append(identity)
                        trace_start.append(begin);trace_available.append(times[k]+.02)
                        trace_prompts.append(prompts)
                    basis=bases[end//width-1];desired=basis.conj().T@model[:,i]
                    # A residual 1-D array or nulled genuine steering provides no directional evidence.
                    if basis.shape[1]>=2 and np.vdot(desired,desired).real/len(a)>=.05:
                        support[k,i]=source_evidence(basis.conj().T@calibrated[:,begin:end],local,desired)
                    if split_support:
                        (alternative_support[k,i],split_on_energy[k,i],
                         split_off_energy[k,i])=split_source_evidence(
                            calibrated[:,begin:end],local,model[:,i],bases,begin,end,width,
                            return_energy=True)
    indices=(times/.02).astype(int)
    result=dict(time=times,available_time=times+.02,satellites=record['satellites'],
                ranges=ranges,lock=lock,cn0=cn0,carrier=carrier,gain=np.array(gains)[indices],
                startup_prns=np.array(prns),calibration=gain,calibration_rate=rate,ratio=np.array(ratios)[indices],support=support,arc=arc)
    if beam_startup_amplitude:
        result['startup_relative_amplitude']=relative_amplitude
    if split_support:
        result.update(support_split=alternative_support,
                      split_on_energy=split_on_energy,split_off_energy=split_off_energy)
    if beam_history_blocks != 1:
        result.update(beam_history_weights=np.array(history_weights),
                      beam_history_count=np.array(history_counts),
                      beam_history_eigengap=np.array(history_gaps))
    if temporal_beam_trigger_enabled:
        result.update(beam_trigger_raw=np.array(trigger_raw),
                      beam_trigger_history_mean=np.array(trigger_mean),
                      beam_trigger_effective=np.array(trigger_effective),
                      beam_trigger_history_only=np.array(trigger_history_only))
    if evidence_trace:
        result.update(trace_window_k=np.array(trace_k,int),trace_star=np.array(trace_star,int),
                      trace_arc=np.array(trace_arc,int),trace_start_sample=np.array(trace_start,int),
                      trace_available_time=np.array(trace_available),
                      trace_prompts=np.array(trace_prompts).reshape(-1,20,len(a)),
                      trace_projectors=np.array([basis@basis.conj().T for basis in bases]),
                      trace_dimensions=np.array([basis.shape[1] for basis in bases]),
                      trace_nominal=model)
        if beam_template_carrier_aiding or beam_coherent_ms!=1:
            block_raw=np.array(ratios,dtype=float)
            block_raw[:startup//width]=np.nan
            result['trace_raw_trigger']=block_raw
    del signal,stream
    return result


def navigate(z, diagnostics=False, history_gap_s=.025, code_blend=.1,
             phase_leave_one_out=False, phase_long_anchor=False,
             phase_state_continuity=False, phase_state_eligible=None):
    out=np.full((len(z['time']),3),np.nan); candidates=out.copy(); initialized=None; stable=[];last=None;isolated=False;recovery_count=0
    if phase_state_continuity:
        phase_state=np.full_like(out,np.nan)
        unpublished_run=0
    finite=np.isfinite(z['ranges'])
    lock=finite&(z['lock']>=.5)
    cn0=lock&(z['cn0']>=25)
    gain=cn0&(z['gain']>=.05)
    valid=gain&(z['support']>=3.)
    history=smooth_history(z,valid,history_gap_s=history_gap_s,
                           diagnostics=diagnostics and history_gap_s>.025)
    ranges,ages=history[:2]
    valid &= np.isfinite(ranges)
    if diagnostics:
        n=len(out)
        diagnostic=dict(raw_count=finite.sum(axis=1),lock_count=lock.sum(axis=1),
                        cn0_count=cn0.sum(axis=1),gain_count=gain.sum(axis=1),
                        support_count=(gain&(z['support']>=3.)).sum(axis=1),
                        selected_count=valid.sum(axis=1),
                        unanchored_selected_common_arc_count=np.zeros(n,int),
                        anchored_selected_common_arc_count=np.zeros(n,int),
                        raw_failed=(~finite).sum(axis=1),
                        lock_failed=(finite&~(z['lock']>=.5)).sum(axis=1),
                        cn0_failed=(lock&~(z['cn0']>=25)).sum(axis=1),
                        gain_failed=(cn0&~(z['gain']>=.05)).sum(axis=1),
                        support_failed=(gain&~(z['support']>=3.)).sum(axis=1),
                        smoothing_failed=(gain&(z['support']>=3.)&~np.isfinite(ranges)).sum(axis=1),
                        mean_smoothing_age=np.array([ages[k,valid[k]].mean() if valid[k].any() else np.nan for k in range(n)]),
                        condition=np.full(n,np.nan),code_separation_m=np.full(n,np.nan),
                        phase_ok=np.zeros(n,bool),isolated=np.zeros(n,bool),
                        recovery_count=np.zeros(n,int),recovery_attempt_count=np.zeros(n,int),
                        recovery_confirmed=np.zeros(n,bool),
                        reason=np.full(n,'',dtype='<U32'))
        if history_gap_s>.025:diagnostic.update(history[2])
        if phase_leave_one_out or phase_long_anchor:
            diagnostic.update(phase_lag_epochs=np.zeros(n,int),
                              phase_excluded_prn=np.zeros(n,int),
                              phase_full_common_count=np.zeros((n,2),int),
                              phase_full_residual_m=np.full((n,2,z['carrier'].shape[1]),np.nan),
                              phase_candidate_common_count=np.zeros(n,int),
                              phase_candidate_residual_m=np.full_like(z['carrier'],np.nan))
        if phase_long_anchor:
            diagnostic.update(phase_short_prediction_m=np.full((n,3),np.nan),
                              phase_long_anchor_index=np.full(n,-1,int),
                              phase_long_anchor_position_m=np.full((n,3),np.nan),
                              phase_long_elapsed_s=np.full(n,np.nan),
                              phase_long_excluded_prn=np.zeros(n,int),
                              phase_long_reason=np.full(n,'',dtype='<U24'),
                              phase_long_path=np.full(n,'short',dtype='<U8'),
                              phase_long_full_common_count=np.zeros(n,int),
                              phase_long_full_residual_m=np.full_like(z['carrier'],np.nan),
                              phase_long_candidate_common_count=np.zeros(n,int),
                              phase_long_candidate_residual_m=np.full_like(z['carrier'],np.nan),
                              phase_long_candidate_prediction_m=np.full((n,3),np.nan),
                              phase_long_jump_m=np.full(n,np.nan),
                              phase_long_horizontal_jump_m=np.full(n,np.nan))
        if phase_state_continuity:
            diagnostic.update(phase_state_position_m=phase_state,
                              phase_state_source=np.full(n,'none',dtype='<U12'),
                              phase_state_short_anchor_index=np.full(n,-1,int),
                              phase_state_short_anchor_published=np.zeros(n,bool),
                              phase_state_unpublished_run_epochs=np.zeros(n,int))
            if phase_state_eligible is not None:
                diagnostic.update(phase_state_eligible=np.array(phase_state_eligible,bool),
                                  phase_state_blocked_internal_anchor_index=
                                  np.full((n,2),-1,int))
    anchor=-1000
    for k in range(len(out)):
        separation=np.inf
        phase_ok=False
        short_anchor_index=-1
        p,condition,residual=position_solution(z['satellites'][k],ranges[k],valid[k])
        if diagnostics:
            diagnostic['condition'][k]=condition
            for j in range(k-1,max(-1,k-3),-1):
                if z['time'][k]-z['time'][j]>.025:continue
                common=valid[k]&valid[j]&np.isfinite(z['carrier'][k])&np.isfinite(z['carrier'][j])&(z['arc'][k]==z['arc'][j])
                diagnostic['unanchored_selected_common_arc_count'][k]=max(diagnostic['unanchored_selected_common_arc_count'][k],int(common.sum()))
                if np.isfinite(out[j]).all():
                    diagnostic['anchored_selected_common_arc_count'][k]=max(diagnostic['anchored_selected_common_arc_count'][k],int(common.sum()))
        if valid[k].sum()>=4 and condition<=1000:
            reference,rotation=receiver_reference()
            sigma=10**((45-z['cn0'][k,valid[k]])/20)
            fit=solve_position(z['satellites'][k,valid[k]],ranges[k,valid[k]],sigma_m=sigma)
            p=(fit['position_m']-reference)@rotation.T
            fresh=p.copy()
            if valid[k].sum()>=6:
                indices=np.flatnonzero(valid[k]);separation=0.
                for drop in indices:
                    retained=indices[indices!=drop]
                    alternate=solve_position(z['satellites'][k,retained],ranges[k,retained],sigma_m=10**((45-z['cn0'][k,retained])/20))
                    delta=(alternate['position_m']-fit['position_m'])@rotation.T
                    separation=max(separation,float(np.linalg.norm(delta[:2])))
            positions=phase_state if phase_state_continuity else out
            if phase_state_continuity and phase_state_eligible is not None:
                if not phase_state_eligible[k]:
                    positions=out
                for lag in (1,2):
                    j=k-lag
                    if (j>=0 and np.isfinite(phase_state[j]).all()
                        and not np.isfinite(out[j]).all()
                        and not np.all(phase_state_eligible[j:k+1])):
                        if diagnostics:
                            diagnostic['phase_state_blocked_internal_anchor_index'][k,lag-1]=j
                        if positions is phase_state:
                            positions=phase_state.copy()
                        if positions is not out:
                            positions[j]=out[j]
            if phase_leave_one_out or phase_long_anchor:
                prediction,phase_details=phase_prediction_leave_one_out(z,valid,positions,k)
                if phase_state_continuity and prediction is not None and phase_details['lag_epochs']:
                    short_anchor_index=k-int(phase_details['lag_epochs'])
                if diagnostics:
                    for key,value in phase_details.items():
                        diagnostic[f'phase_{key}'][k]=value
                if phase_long_anchor:
                    short_prediction=prediction
                    long_prediction,long_details=phase_prediction_long_anchor(
                        z,valid,out,k,phase_details['excluded_prn'])
                    if diagnostics:
                        if short_prediction is not None:
                            diagnostic['phase_short_prediction_m'][k]=short_prediction
                        for key,value in long_details.items():
                            diagnostic[f'phase_long_{key}'][k]=value
                        j=long_details['anchor_index']
                        if j>=0:
                            diagnostic['phase_long_anchor_position_m'][k]=out[j]
                    if long_prediction is not None:
                        prediction=long_prediction
                        short_anchor_index=-1
                        if diagnostics:
                            diagnostic['phase_long_path'][k]='long'
                            if short_prediction is not None:
                                diagnostic['phase_long_jump_m'][k]=float(
                                    np.linalg.norm(long_prediction-short_prediction))
                                diagnostic['phase_long_horizontal_jump_m'][k]=float(
                                    np.linalg.norm((long_prediction-short_prediction)[:2]))
            else:
                if phase_state_continuity and diagnostics:
                    prediction,short_anchor_index=phase_prediction(
                        z,valid,positions,k,return_anchor=True)
                else:
                    prediction=phase_prediction(z,valid,positions,k)
            phase_ok=prediction is not None
            if phase_ok:p=code_blend*p+(1-code_blend)*prediction if separation<=20. else prediction
            if diagnostics and phase_state_continuity and short_anchor_index>=0:
                diagnostic['phase_state_short_anchor_index'][k]=short_anchor_index
                diagnostic['phase_state_short_anchor_published'][k]=np.isfinite(
                    out[short_anchor_index]).all()
        if diagnostics:
            diagnostic['code_separation_m'][k]=separation
            diagnostic['phase_ok'][k]=phase_ok
        if separation>20. and not phase_ok:p=np.full(3,np.nan)
        if separation<=20.:anchor=k
        last=p.copy() if np.all(np.isfinite(p)) else None
        candidates[k]=p
        if initialized is None:
            if np.all(np.isfinite(p)):
                stable.append(p)
                if len(stable)>=3:
                    window=np.array(stable[-3:])
                    if np.max(np.linalg.norm(window[:,:2]-window[:,:2].mean(axis=0),axis=1))<=20:
                        initialized=k
            else:stable=[]
        if initialized is not None:
            isolated |= not np.all(np.isfinite(p))
            if isolated:
                confirmed=(valid[k]&(z['support'][k]>=3.)).sum()>=6 and np.all(np.isfinite(p)) and separation<=20.
                if diagnostics:diagnostic['recovery_confirmed'][k]=confirmed
                recovery_count=recovery_count+1 if confirmed else 0
                if diagnostics:diagnostic['recovery_attempt_count'][k]=recovery_count
                if recovery_count>=10:
                    isolated=False;p=fresh.copy();last=p.copy();recovery_count=0
            if not isolated and (separation<=20. or phase_ok and k-anchor<=20):out[k]=p
        if phase_state_continuity:
            if np.isfinite(out[k]).all():
                phase_state[k]=out[k]
                unpublished_run=0
                source='published'
            elif (initialized is not None and not isolated and phase_ok
                  and np.isfinite(p).all() and separation>20. and k-anchor>20
                  and (phase_state_eligible is None or phase_state_eligible[k])):
                phase_state[k]=p
                unpublished_run+=1
                source='output_gate'
            else:
                unpublished_run=0
                source='none'
            if diagnostics:
                diagnostic['phase_state_source'][k]=source
                diagnostic['phase_state_unpublished_run_epochs'][k]=unpublished_run
        if diagnostics:
            diagnostic['isolated'][k]=isolated
            diagnostic['recovery_count'][k]=recovery_count
            if initialized is None:reason='initializing'
            elif np.isfinite(out[k]).all():reason='published'
            elif valid[k].sum()<4:reason='fewer_than_four'
            elif condition>1000:reason='geometry_condition'
            elif valid[k].sum()<6 and not phase_ok:reason='fewer_than_six_for_code'
            elif separation>20. and not phase_ok:reason='code_or_phase'
            elif isolated:reason='isolated_recovery'
            else:reason='output_gate'
            diagnostic['reason'][k]=reason
    result=dict(position=out,candidate_position=candidates,selected=valid,
                initialization_time=None if initialized is None else float(z['available_time'][initialized]))
    if diagnostics:result['diagnostics']=diagnostic
    return result
