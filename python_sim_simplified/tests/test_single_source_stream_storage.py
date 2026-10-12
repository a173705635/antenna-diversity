"""Saved receiver arrays must match the frozen in-memory path exactly."""
import numpy as np
import pytest

from experiments.spoofing.single_source_stream_storage import assert_saved_arrays


def test_saved_array_audit_rejects_changed_receiver_field(tmp_path):
    path=tmp_path/'tracking.npz'
    reference=dict(time=np.array([.25,.5]),
                   support=np.array([[np.nan,4.],[5.,np.nan]]))
    np.savez_compressed(path,**reference)
    assert_saved_arrays(reference,path)
    changed={**reference,'support':np.array([[np.nan,4.],[6.,np.nan]])}
    with pytest.raises(AssertionError):
        assert_saved_arrays(changed,path)
    changed_dtype={**reference,'time':reference['time'].astype(np.float32)}
    with pytest.raises(AssertionError):
        assert_saved_arrays(changed_dtype,path)
