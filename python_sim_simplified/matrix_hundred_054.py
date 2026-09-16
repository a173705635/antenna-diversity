from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Unknown calibration-reference group delay is indistinguishable from a common hardware delay even with full-rank pilots.',
 change='Introduce unknown frequency-dependent pilot reference transfer; construct two physical decompositions with identical commanded/measured data.',
 input_model='13 tones, full matrix constant; reference cable delay 120 ns in world A versus common hardware delay 120 ns in world B; no monitor.')
def experiment(seed=41060):
    d=scene(seed);f=np.linspace(-1e6,1e6,13);r=np.exp(-2j*np.pi*f*120e-9)
    world_a=r[:,None,None]*d['h'];world_b=np.array([d['h']*ri for ri in r])
    return dict(indistinguishable_measurement_norm=float(np.linalg.norm(world_a-world_b)),
      ignored_reference_matrix_error=relative_error(world_a,np.broadcast_to(d['h'],world_a.shape)),
      ignored_reference_delay_ns=120,common_range_equivalent_m=299792458*120e-9,
      spatial_projective_error=projective_error(world_a[0]@d['a'],d['h']@d['a']),
      positioning_caveat='A common delay can be absorbed in receiver clock; 35.98 m range-equivalent is not horizontal position error.',identified=False)
