"""Geometry and positioning for continuous array recovery."""
import numpy as np
from ..positioning import solve_position
from ..gps_simulation import receiver_reference

C = 299792458.
FS = 2.046e6
WAVELENGTH = C / 1575.42e6


def sky(seed, count, clustered=False):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2*np.pi, 24)[:count]
    el = np.arcsin(rng.uniform(np.sin(np.radians(15)), .97, 24)[:count])
    if clustered:
        az = .5 + .1 * az
        el = .5 + .1 * el
    return np.column_stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)))


def yaw_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def steering(directions, positions, yaw):
    return np.exp(2j*np.pi/WAVELENGTH * (positions @ yaw_matrix(yaw).T @ directions.T))


def position_solution(satellites, ranges, valid):
    """Explicit observability outcomes are research results, not default fixes."""
    if valid.sum() < 4:
        return np.full(3, np.nan), np.nan, np.nan
    reference, rotation = receiver_reference()
    # Rank/geometry eligibility is part of the estimator's measurement admission.
    direction = satellites[valid]-reference
    design = np.column_stack((-direction/np.linalg.norm(direction, axis=1)[:, None], np.ones(valid.sum())))
    condition = np.linalg.cond(design)
    if condition > 1000.:
        return np.full(3, np.nan), condition, np.nan
    result = solve_position(satellites[valid], ranges[valid])
    return (result['position_m']-reference) @ rotation.T, condition, float(np.sqrt(np.mean(result['residual_m']**2)))
