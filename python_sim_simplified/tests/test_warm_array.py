import numpy as np
from research_warm_array import generate, SUBSETS
from warm_array import process


def test_paired_subsets_share_exact_samples_and_no_truth_interface():
    record, truth = generate(111003, 'attack')
    assert 'truth' not in record
    np.testing.assert_array_equal(record['samples'][:, SUBSETS['three']][:, :2],
                                  record['samples'][:, SUBSETS['two_short']])
    before = process(record, SUBSETS['three'])
    changed = dict(record, samples=record['samples'].copy())
    changed['samples'][3:] *= 30j
    after = process(changed, SUBSETS['three'])
    np.testing.assert_array_equal(before['position'][:3], after['position'][:3])


def test_attack_and_genuine_disappearance_start_only_after_initialization():
    normal, _ = generate(111005, 'normal')
    absent, _ = generate(111005, 'absent')
    np.testing.assert_array_equal(normal['samples'][:3], absent['samples'][:3])
    assert not np.array_equal(normal['samples'][3:], absent['samples'][3:])


def test_estimated_null_preserves_nominal_response_and_excludes_degenerate_star():
    from warm_array import weights
    rng = np.random.default_rng(111006)
    v = np.ones(4, complex)/2
    a = np.column_stack((np.ones(4), [1, -1, 1j, -1j]))
    x = 100*v[:, None]*np.exp(1j*np.arange(2046))[None, :]
    x += (rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))/np.sqrt(2)
    w, admitted = weights(x, a, np.eye(4))
    assert not admitted[0] and admitted[1]
    assert abs(np.vdot(w[:, 1], v)) < .005
    assert abs(np.vdot(w[:, 1], a[:, 1])-1) < .01
