"""GPS L1 C/A LNAV synchronization, parity and broadcast fields."""
import numpy as np


def decode_lnav_word(word):
    """IS-GPS-200 user parity algorithm, 2 previous + 30 current bits.

    Protocol parity masks cross-checked against RTKLIB src/rtkcmn.c decode_word.
    The input includes the preceding word's D29*/D30* bits.
    """
    masks = (0xBB1F3480,0x5D8F9A40,0xAEC7CD00,0x5763E680,0x6BB1F340,0x8B7A89C0)
    corrected = word ^ (0x3FFFFFC0 if word & 0x40000000 else 0)
    parity = 0
    for mask in masks:
        parity = (parity << 1) | ((corrected & mask).bit_count() & 1)
    return (corrected >> 6) & 0xFFFFFF if parity == (word & 63) else None


def navigation_check(h):
    """Offline 20-ms bit synchronization, word parity and TLM/HOW evidence.

    No truth bits supplied, no NCO modification. A long consecutive parity
    chain is evidence of demodulation, not authentication or BER truth.
    """
    z = h['prompt'][500:]  # discard startup, same for every PRN/mode
    if len(z) < 700: return {'status':'too_short'}
    scores=[]
    for offset in range(20):
        n=(len(z)-offset)//20
        scores.append(float(np.mean(abs(z[offset:offset+n*20].real.reshape(n,20).sum(axis=1)))))
    offset=int(np.argmax(scores)); n=(len(z)-offset)//20
    sums=z[offset:offset+n*20].real.reshape(n,20).sum(axis=1)
    bits=(sums<0).astype(int).tolist()
    candidates=[]
    for polarity in (0,1):
        b=[v^polarity for v in bits]
        for start in range(2,32):
            words=[]; longest=run=0
            for i in range(start,len(b)-29,30):
                word=int(''.join(map(str,b[i-2:i+30])),2)
                payload=decode_lnav_word(word)
                run=run+1 if payload is not None else 0
                longest=max(longest,run)
                words.append({'bit_index':i,'payload':payload})
            valid=sum(w['payload'] is not None for w in words)
            candidates.append((longest,valid,polarity,start,words))
    longest,valid,polarity,start,words=max(candidates,key=lambda c:(c[0],c[1]))
    frames=[]
    for j,w in enumerate(words[:-1]):
        d=w['payload']; how=words[j+1]['payload']
        if d is not None and d>>16==0x8b and how is not None:
            frames.append({'bit_index':w['bit_index'],'next_subframe_tow_s':(how>>7)*6,
                'subframe_id':(how>>2)&7,
                'ten_words_parity_ok':len(words[j:j+10])==10 and all(v['payload'] is not None for v in words[j:j+10])})
    return {'status':'evaluated','bit_offset_epochs_after_500':offset,
        'bit_sync_scores':scores,'bit_count':n,'bits':bits,'polarity':polarity,
        'word_start_bit':start,'words':words,'valid_words':valid,'tested_words':len(words),
        'longest_consecutive_valid_words':longest,'preambles':frames,
        'half_cycle_ambiguity_resolved_for_nco':False,
        'demodulation_evidence':longest>=4 and len(set(bits))==2 and any(
            1<=f['subframe_id']<=5 and 0<=f['next_subframe_tow_s']<604800 for f in frames)}


def decode_ephemeris(subframes):
    """Decode parity-validated 24-bit payload words from GPS subframes 1/2/3.

    Bit positions exclude the six parity bits of each word, as in the IS-GPS-200
    fields and RTKLIB's packed-payload convention. Week remains modulo 1024;
    no operating-system date is silently used to resolve historical recordings.
    """
    if not {1,2,3}.issubset(subframes):
        raise ValueError('subframes 1, 2 and 3 required')
    streams={}
    for sf in (1,2,3):
        words=subframes[sf]
        if len(words)!=10 or any(not 0<=w<2**24 for w in words):
            raise ValueError('ten validated payload words required')
        if words[0]>>16!=0x8b or ((words[1]>>2)&7)!=sf:
            raise ValueError('subframe header mismatch')
        streams[sf]=''.join(f'{w:024b}' for w in words)
    def field(sf,offset,n,signed=False):
        v=int(streams[sf][offset:offset+n],2)
        return v-(1<<n) if signed and v&(1<<(n-1)) else v
    eph={'week_mod1024':field(1,48,10),'health':field(1,64,6),'ura_index':field(1,60,4),
         'iodc':(field(1,70,2)<<8)|field(1,168,8),'iode':field(2,48,8),
         'toc':field(1,176,16)*16.,'toe':field(2,216,16)*16.,
         'tgd':field(1,160,8,True)*2**-31,
         'af2':field(1,192,8,True)*2**-55,'af1':field(1,200,16,True)*2**-43,
         'af0':field(1,216,22,True)*2**-31}
    if field(3,216,8)!=eph['iode'] or (eph['iodc']&255)!=eph['iode']:
        raise ValueError('ephemeris issue identifiers are inconsistent')
    for name,sf,offset,n,power,angular,signed in (
        ('crs',2,56,16,-5,False,True),('delta_n',2,72,16,-43,True,True),
        ('m0',2,88,32,-31,True,True),('cuc',2,120,16,-29,False,True),
        ('e',2,136,32,-33,False,False),('cus',2,168,16,-29,False,True),
        ('sqrt_a',2,184,32,-19,False,False),('cic',3,48,16,-29,False,True),
        ('omega0',3,64,32,-31,True,True),('cis',3,96,16,-29,False,True),
        ('i0',3,112,32,-31,True,True),('crc',3,144,16,-5,False,True),
        ('omega',3,160,32,-31,True,True),('omega_dot',3,192,24,-43,True,True),
        ('idot',3,224,14,-43,True,True)):
        eph[name]=field(sf,offset,n,signed)*2.**power*(np.pi if angular else 1.)
    eph['tgd_unavailable']=field(1,160,8,True)==-128
    if eph['tgd_unavailable']: eph['tgd']=0.
    if not (0<=eph['e']<1 and 2e7<eph['sqrt_a']**2<3e7):
        raise ValueError('invalid decoded orbit')
    return eph


def decode_lnav(history, fs_hz):
    """Offline GPS LNAV decoding and verified TLM-to-sample time anchors.

    Complete subframes require all ten words to pass parity. Timing requires
    at least two TLM/HOW anchors agreeing in code-epoch count and subframe ID.
    Ephemeris can be assembled out of arrival order only with matching IODs.
    """
    diagnostics=navigation_check(history)
    out={'status':'no_verified_time','diagnostics':diagnostics,'subframes':[],
         'time_anchors':[],'ephemeris':None,'half_cycle_resolved':False}
    if diagnostics.get('status')!='evaluated' or not diagnostics['demodulation_evidence']:
        return out
    starts=np.asarray(history['epoch_start_samples'])
    if len(starts)!=len(history['prompt']) or not np.allclose(np.diff(starts)/fs_hz,.001,atol=1e-10,rtol=0):
        out['status']='discontinuous_tracking';return out
    words=diagnostics['words']; anchors=[]; complete={}
    for j,w in enumerate(words[:-1]):
        d=w['payload'];how=words[j+1]['payload']
        if d is None or d>>16!=0x8b or how is None:continue
        sf=(how>>2)&7;tow_next=(how>>7)*6
        if not 1<=sf<=5 or not 0<=tow_next<604800:continue
        k=500+diagnostics['bit_offset_epochs_after_500']+w['bit_index']*20
        anchor={'epoch_index':k,'tow_s':(tow_next-6)%604800,'subframe_id':sf,
                'sample_time_s':float(starts[k]/fs_hz)}
        anchors.append(anchor)
        frame=words[j:j+10]
        if len(frame)==10 and all(v['payload'] is not None for v in frame):
            payloads=[v['payload'] for v in frame]
            out['subframes'].append({**anchor,'payload_words':payloads})
            if sf in (1,2,3):complete[sf]=payloads
    for a,b in zip(anchors,anchors[1:]):
        gap=(b['tow_s']-a['tow_s'])%604800
        if (gap>0 and gap%6==0 and gap<=30 and
            b['epoch_index']-a['epoch_index']==round(gap*1000) and
            (b['subframe_id']-a['subframe_id'])%5==int(gap/6)%5):
            if a not in out['time_anchors']:out['time_anchors'].append(a)
            if b not in out['time_anchors']:out['time_anchors'].append(b)
    if not out['time_anchors']:return out
    out['status']='time_only'
    if {1,2,3}.issubset(complete):
        try:out['ephemeris']=decode_ephemeris(complete)
        except ValueError as e:out['ephemeris_error']=str(e)
        else:out['status']='ephemeris_decoded'
    return out
