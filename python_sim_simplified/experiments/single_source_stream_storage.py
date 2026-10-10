"""Replay frozen 8 s IQ with a disk-backed beam stream; audit exact output equivalence."""
import json

import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.single_source import frontend
from experiments.paths import RESULTS
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import build_coherent_support
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_ramp_takeover import OUTPUT as STAGE33
from experiments.single_source_spoof_power import run_navigation
from experiments.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,RECEIVER_PLL,digest,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'stream_storage'
SCENE='weak'
MODE='aided_five_ms'
FILES=('tracking.npz','trace.npz','history.npz','groups.npz',
       'support.npz','navigation.npz')


def assert_saved_arrays(actual,path):
    """Compare each scientific field, including NaN placement and dtype."""
    with np.load(path) as saved:
        assert set(actual)==set(saved.files)
        for name,value in actual.items():
            assert np.asarray(value).dtype==saved[name].dtype
            np.testing.assert_array_equal(value,saved[name],err_msg=f'{path}/{name}')


def preflight():
    old=json.loads((STAGE33/'summary.json').read_text(encoding='utf-8'))
    current=source_hashes()
    assert old['source_snapshot_sha256']==digest(STAGE33/'source_snapshot.zip')
    verify_source_snapshot(STAGE33/'source_snapshot.zip',old['source_sha256'])
    for key,filename in (('plan_sha256','plan.json'),
                         ('manifest_sha256','manifest.json'),
                         ('figure_sha256','overview.png')):
        assert old[key]==digest(STAGE33/filename)
    for name,sha in old['source_sha256'].items():
        if name.startswith('core/') and name!='core/single_source.py':
            assert current[name]==sha
    folder=STAGE33/SCENE
    for filename,sha in old['cases'][SCENE]['input_sha256'].items():
        assert digest(folder/'input'/filename)==sha
    for filename,sha in old['cases'][SCENE]['output_sha256'][MODE].items():
        assert digest(folder/MODE/filename)==sha
    return old,current


def main():
    old,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    plan=dict(scene=SCENE,source=STAGE33.name,receiver_pll=RECEIVER_PLL,
              processed_antennas=ANTENNAS,beam_template_carrier_aiding=True,
              beam_coherent_ms=5,temporal_beam_trigger_enabled=False,
              stream_dtype='complex128',stream_order='C',
              source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan),indent=2),
                                    encoding='utf-8')
    manifest=dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                  plan_sha256=digest(OUTPUT/'plan.json'),
                  stage33_summary_sha256=digest(STAGE33/'summary.json'),
                  input_sha256=old['cases'][SCENE]['input_sha256'],
                  cached_output_sha256=old['cases'][SCENE]['output_sha256'][MODE])
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),indent=2),
                                        encoding='utf-8')
    input_folder=STAGE33/SCENE/'input'
    with np.load(input_folder/'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(input_folder/'samples.npy',mmap_mode='r')
    scratch=OUTPUT/'beam_stream.dat'
    observations=frontend(record,ANTENNAS,split_support=True,
        receiver_pll=RECEIVER_PLL,evidence_trace=True,
        temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=True,beam_coherent_ms=5,
        stream_memmap_path=scratch)
    scratch_bytes=scratch.stat().st_size
    assert scratch_bytes==len(record['codes'])*record['samples'].shape[1]*np.dtype(
        np.complex128).itemsize
    trace={key:value for key,value in observations.items()
           if key.startswith('trace_')}
    tracking={key:value for key,value in observations.items()
              if not key.startswith('trace_')}
    np.savez_compressed(OUTPUT/'tracking.npz',**tracking)
    np.savez_compressed(OUTPUT/'trace.npz',**trace)
    scratch.unlink()
    assert not scratch.exists()
    print('frontend saved and beam stream scratch removed',flush=True)
    reference=STAGE33/SCENE/MODE
    assert_saved_arrays(tracking,reference/'tracking.npz')
    assert_saved_arrays(trace,reference/'trace.npz')
    history=analyze_trace(observations)
    np.savez_compressed(OUTPUT/'history.npz',**history)
    assert_saved_arrays(history,reference/'history.npz')
    groups=analyze_groups(trace)
    np.savez_compressed(OUTPUT/'groups.npz',**groups)
    assert_saved_arrays(groups,reference/'groups.npz')
    coherent=build_coherent_support(tracking,groups)
    support={f'coherent_{name}':value for name,value in coherent.items()}
    np.savez_compressed(OUTPUT/'support.npz',**support)
    assert_saved_arrays(support,reference/'support.npz')
    metadata=json.loads((input_folder/'metadata.json').read_text(encoding='utf-8'))
    run_navigation(tracking,coherent['support'],metadata['case'],OUTPUT)
    with np.load(OUTPUT/'navigation.npz') as saved:
        navigation=dict(saved)
    assert_saved_arrays(navigation,reference/'navigation.npz')
    assert source_hashes()==sources
    result=dict(scope=__doc__,source_sha256=sources,
                source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                plan_sha256=digest(OUTPUT/'plan.json'),
                manifest_sha256=digest(OUTPUT/'manifest.json'),
                scratch_bytes=scratch_bytes,scratch_deleted=True,
                input_sha256=manifest['input_sha256'],
                output_sha256={name:digest(OUTPUT/name) for name in FILES},
                reference_output_sha256=manifest['cached_output_sha256'],
                exact_fields={name:len(data) for name,data in (
                    ('tracking.npz',tracking),('trace.npz',trace),
                    ('history.npz',history),('groups.npz',groups),
                    ('support.npz',support),('navigation.npz',navigation))})
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(result),indent=2),
                                       encoding='utf-8')
    return result


if __name__=='__main__':
    main()
