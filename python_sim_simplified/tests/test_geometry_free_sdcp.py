import numpy as np
import pytest
from sdcp import geometry_free_sdcp


def test_geometry_cancels_but_static_bias_is_indistinguishable():
    t = np.arange(4.)
    g = np.array([0., .01, -.03, .07])
    a = np.zeros((4, 2))
    b = np.column_stack([g, -g])
    assert np.allclose(geometry_free_sdcp(t, a, b, t, a, b), 0.)
    # Either a hardware bias or static multipath can produce this same observation.
    contaminated = geometry_free_sdcp(t, a, b + .008, t, a, b)
    assert np.allclose(contaminated, .008)


def test_common_frequency_error_is_invisible_and_cycles_remain():
    t = np.arange(3.)
    a = np.zeros((3, 1))
    assert np.allclose(geometry_free_sdcp(t, a, a+.02, t, a, a+.02), 0.)
    l1, l5 = 299792458/1575.42e6, 299792458/1176.45e6
    result = geometry_free_sdcp(t, a, a+l1, t, a, a+l5)
    assert np.allclose(result, l1-l5)
    assert not np.allclose(result, 0.)
