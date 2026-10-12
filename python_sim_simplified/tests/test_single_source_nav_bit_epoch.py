"""The navigation-bit epoch changes only the modeled data-bit signs."""
from copy import deepcopy

import numpy as np
import pytest

from core.spoofing import single_source_scene as scene
from core.spoofing.recovery_geometry import C,FS,WAVELENGTH,sky,steering
from core.presets import mate70_phase_centers


def small_case(**changes):
    values=dict(duration=.06,stars=3,antennas=3,cn0=35.,
                attack_start=.03,angle=5.,spoof_db=-5.,
                velocity_east=2.,phase_deg=0.,amplitude_db=0.)
    return scene.Case(**(values | changes))


def test_epoch_zero_is_default_and_bits_off_ignore_epoch():
    default=scene.generate(small_case(),19,True)[0]
    explicit=scene.generate(small_case(nav_bit_epoch_s=0.),19,True)[0]
    np.testing.assert_array_equal(default,explicit)
    old_off=scene.generate(small_case(data_bits=False),19,True)[0]
    shifted_off=scene.generate(small_case(data_bits=False,
                                          nav_bit_epoch_s=.005),19,True)[0]
    np.testing.assert_array_equal(old_off,shifted_off)


@pytest.mark.parametrize('genuine,attacked,fake_rate,fake_east,bit_epoch',(
    (True,True,0.,100.,.005),(False,True,0.,100.,.005),
    (True,False,0.,100.,.005),(True,True,3.,0.,0.),
    (False,True,3.,0.,0.)))
def test_three_star_three_antenna_bit_epoch_matches_independent_components(
        monkeypatch,genuine,attacked,fake_rate,fake_east,bit_epoch):
    case=small_case(nav_bit_epoch_s=bit_epoch,genuine=genuine,
                    fake_rate=fake_rate,fake_east=fake_east)
    seed=31
    original=np.random.default_rng
    generators=[]

    def tracked(value):
        generator=original(value)
        generators.append(generator)
        return generator

    monkeypatch.setattr(scene.np.random,'default_rng',tracked)
    shifted=scene.generate(case,seed,attacked)[0]
    shifted_state=deepcopy(generators[0].bit_generator.state)
    generators.clear()
    plain=scene.generate(small_case(data_bits=False,genuine=genuine,
                                    fake_rate=fake_rate,fake_east=fake_east),
                         seed,attacked)[0]
    plain_state=deepcopy(generators[0].bit_generator.state)
    assert shifted_state==plain_state

    rng=original(seed)
    rng.normal(size=case.antennas)
    rng.normal(size=case.antennas)
    count=round(FS*case.duration)
    real_noise=rng.normal(size=(case.antennas,count))
    imaginary_noise=rng.normal(size=(case.antennas,count))
    time=np.arange(count)/FS
    directions=sky(seed,case.stars)
    antennas=np.array(list(mate70_phase_centers().values()))[:case.antennas]
    nominal=steering(directions,antennas,0.)
    tangent=np.cross(directions[0],[0,0,1.])
    tangent/=np.linalg.norm(tangent)
    source=directions[0]*np.cos(np.radians(case.angle))+tangent*np.sin(
        np.radians(case.angle))
    forged_vector=steering(source[None],antennas,0.)[:,0]
    attack=(time>=case.attack_start)&attacked
    expected=np.sqrt(FS/10**(case.cn0/10))*(
        real_noise+1j*imaginary_noise)/np.sqrt(2)
    for star in range(case.stars):
        bits=rng.choice([-1.,1.],size=round(case.duration/.02)+10)
        code=scene.bandlimited_code_replica(scene.ca_code(star+1),
                                             1.023e6,.9e6)
        distance=21e6+star*17000-WAVELENGTH*np.linspace(
            -1500.,2000.,case.stars)[star]*time
        east=case.velocity_east*time
        genuine=np.sqrt(distance**2-2*distance*directions[star,0]*east+
                        east**2)+75000
        false_east=east+case.fake_east+case.fake_rate*np.maximum(
            time-case.attack_start,0.)
        fake=np.sqrt(distance**2-2*distance*directions[star,0]*false_east+
                     false_east**2)+75000
        for ranges,vector,scale in ((genuine,nominal[:,star],
                                    np.where(attack&~np.array(case.genuine),0.,1.)),
                                    (fake,forged_vector,
                                     attack*np.sqrt(10**(case.spoof_db/10)))):
            waveform=scene._sample_code(code,(time-ranges/C)*1.023e6)*np.exp(
                -2j*np.pi*ranges/WAVELENGTH)
            bit_index=(np.floor((time-ranges/C-case.nav_bit_epoch_s)/.02)
                       .astype(int)+5)%len(bits)
            expected+=vector[:,None]*(waveform*scale*bits[bit_index])
    np.testing.assert_allclose(shifted,expected,rtol=0,atol=2e-10)
    assert np.any(shifted!=plain)
