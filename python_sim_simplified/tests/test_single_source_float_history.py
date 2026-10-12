"""Square-root causal arc-bias compression versus full-prefix batch fitting."""

import numpy as np

from experiments.spoofing.single_source_float_window_study import solve_history, solve_window


def fixture(epochs=4):
    geometry = np.array([[1., 0., 0.], [-1., 0., 0.],
                         [0., 1., 0.], [0., -1., 0.],
                         [0., 0., 1.], [0., 0., -1.]])
    design = np.broadcast_to(geometry, (epochs, 6, 3)).copy()
    position = np.column_stack((2. * np.arange(epochs),
                                -.5 * np.arange(epochs),
                                .25 * np.arange(epochs)))
    clock = 75_000. + 3. * np.arange(epochs)
    code = np.einsum("tsd,td->ts", design, position) + clock[:, None]
    carrier = code + 1000. * np.arange(6)[None, :]
    valid = np.ones((epochs, 6), bool)
    arc = np.zeros((epochs, 6), int)
    return design, code, carrier, valid, arc, position


def test_history_matches_every_full_prefix_batch_position_and_formal_covariance():
    design, code, carrier, valid, arc, _ = fixture()
    carrier[2:, 0] += 13.
    arc[2:, 0] = 1

    history = solve_history(design, code, carrier, valid, arc)

    for target in range(len(design)):
        batch = solve_window(design, code, carrier, valid, arc, target, width=100)
        assert history["observable"][target] == batch["observable"]
        np.testing.assert_allclose(history["position_m"][target], batch["position_m"], atol=1e-6)
        np.testing.assert_allclose(history["formal_sigma_xyz_m"][target],
                                   batch["formal_sigma_xyz_m"], atol=1e-8)


def test_carrier_arc_origin_shift_does_not_move_position_or_formal_sigma():
    design, code, carrier, valid, arc, _ = fixture()
    original = solve_history(design, code, carrier, valid, arc)
    carrier += 1e6 + np.arange(6)[None, :] * 7e5
    shifted = solve_history(design, code, carrier, valid, arc)

    np.testing.assert_allclose(shifted["position_m"], original["position_m"], atol=1e-6)
    np.testing.assert_allclose(shifted["formal_sigma_xyz_m"],
                               original["formal_sigma_xyz_m"], atol=1e-8)


def test_future_and_missing_current_rows_do_not_create_position():
    design, code, carrier, valid, arc, _ = fixture()
    prefix = solve_history(design[:2], code[:2], carrier[:2], valid[:2], arc[:2])
    code[2:] += 100000.
    carrier[2:] -= 50000.
    full = solve_history(design, code, carrier, valid, arc)
    np.testing.assert_array_equal(full["position_m"][:2], prefix["position_m"])

    valid[-1] = False
    missing = solve_history(design, code, carrier, valid, arc)
    assert not missing["observable"][-1]
    assert np.isnan(missing["position_m"][-1]).all()


def test_noisy_changing_constellation_matches_batch_after_empty_epoch():
    rng = np.random.default_rng(2026092507)
    epochs, satellites = 8, 9
    design = rng.normal(size=(epochs, satellites, 3))
    design /= np.linalg.norm(design, axis=2, keepdims=True)
    position = rng.normal(size=(epochs, 3)) * 4
    clock = 75000. + rng.normal(size=epochs) * 2
    geometric = np.einsum("tsd,td->ts", design, position) + clock[:, None]
    code = geometric + rng.normal(size=(epochs, satellites)) * 30
    carrier = geometric + 2000 * np.arange(satellites)[None, :]
    carrier += rng.normal(size=(epochs, satellites)) * .03
    valid = rng.random((epochs, satellites)) < .72
    valid[:2] = False
    valid[5] = False
    valid[6] = True
    arc = np.zeros((epochs, satellites), int)
    arc[4:, 2] = 1
    carrier[4:, 2] += 120.

    history = solve_history(design, code, carrier, valid, arc)

    assert not history["observable"][0]
    assert not history["observable"][1]
    assert not history["observable"][5]
    assert history["observable"][6]
    for target in range(epochs):
        batch = solve_window(design, code, carrier, valid, arc, target, width=100)
        np.testing.assert_allclose(history["position_m"][target],
                                   batch["position_m"], atol=1e-4, equal_nan=True)
        np.testing.assert_allclose(history["formal_sigma_xyz_m"][target],
                                   batch["formal_sigma_xyz_m"], atol=1e-6, equal_nan=True)
