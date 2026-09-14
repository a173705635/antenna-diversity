import numpy as np
from spatial_reference_direction import estimate_with_reference


def test_trusted_reference_removes_unknown_common_channel_bias():
    p=np.array([[0.,0,0],[.04,0,0],[0,-.04,0],[.04,-.04,0]])
    truth=np.array([.3,.4,np.sqrt(.75)])
    reference=np.array([-.2,.5,np.sqrt(.71)])
    x=np.broadcast_to((-p@np.array([truth,reference]).T)[None],(10,4,2)).copy()
    x+=np.array([0,.13,-.07,.18])[None,:,None]
    result=estimate_with_reference(x,p,.190293672798,reference_index=1,
                                   target_indices=[0],reference_direction=reference,harmonic=1)
    assert result['status']=='single_upper_candidate'
    assert any(np.linalg.norm(np.array(c['unit_vector'])-truth)<1e-4 for c in result['candidates'])


def test_reference_cannot_also_be_target():
    p=np.array([[0.,0,0],[.04,0,0],[0,-.04,0]])
    result=estimate_with_reference(np.zeros((10,3,3)),p,.19,reference_index=2,
                                   target_indices=[0,1,2],reference_direction=[0,0,1])
    assert result['status']=='reference_in_target'
