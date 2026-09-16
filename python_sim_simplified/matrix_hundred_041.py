from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Full projective matrix fit models off-diagonal coupling that diagonal gains cannot.',
 change='Replace diagonal homogeneous basis with all m squared matrix entries.',
 input_model='24 authentic clean sky responses with unknown complex per-PRN amplitude; 16 heldout directions; 4 fixed antennas.')
def experiment(seed=41060):
    d=scene(seed)
    diag,_=homogeneous_fit(d['a'],d['clean'],diagonal=True)
    full,s=homogeneous_fit(d['a'],d['clean'])
    return dict(diagonal_heldout_error=projective_error(diag@d['test_a'],d['test_y']),
                full_heldout_error=projective_error(full@d['test_a'],d['test_y']), singular_values=s.tolist())
