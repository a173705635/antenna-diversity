"""Offline L1 capture/track/reacquire schedule driven by observed quality.

Acquisition consumes complete windows before tracking starts. No outage truth
or scheduled return time is accepted. Sample times exclude compute latency.
"""
from dataclasses import replace

from acquisition import acquire
from tracking import track


def receive(x, fs_hz, prn, code_rate_hz, carrier_hz, acquisition_cfg, tracking_cfg,
            *, loss_confirm_epochs=5, loss_settle_epochs=100):
    window=round(fs_hz*.001*acquisition_cfg.coherent_ms*acquisition_cfg.noncoherent_ms)
    epoch_samples=round(fs_hz*.001*tracking_cfg.coherent_ms)
    cfg=replace(tracking_cfg, acquisition_ms=acquisition_cfg.coherent_ms*acquisition_cfg.noncoherent_ms)
    cursor=0;events=[];segments=[]
    while cursor+window+2*epoch_samples <= len(x):
        acq=acquire(x[cursor:cursor+window],fs_hz,prn,code_rate_hz,acquisition_cfg)
        events.append(dict(kind='acquisition',start_sample=cursor,
                           evidence_end_sample=cursor+window,success=acq['success'],metric=acq['metric']))
        if not acq['success']:
            cursor+=window
            continue
        h=track(x[cursor:],fs_hz,prn,acq,cfg,carrier_hz,code_rate_hz,
                loss_confirm_epochs=loss_confirm_epochs,loss_settle_epochs=loss_settle_epochs)
        if not h['success']:
            events.append(dict(kind='tracking_failure',start_sample=cursor,status=h['status']))
            break
        h['epoch_start_samples']+=cursor
        end=int(h['epoch_start_samples'][-1])+epoch_samples
        identity=f'tracking-{len(segments)}'
        segments.append(dict(tracking_id=identity,history=h))
        events.append(dict(kind='tracking_end',tracking_id=identity,evidence_end_sample=end,status=h['status']))
        cursor=end
        if h['status']!='tracking_quality_lost':
            break
    return dict(events=events,segments=segments,processed_until_sample=cursor,
                scope='offline L1 schedule; quality recovery does not authenticate a new phase reference')
