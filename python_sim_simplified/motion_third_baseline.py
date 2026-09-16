"""Phone differential carrier/gyro observation-domain research, no navigation."""
import numpy as np
from scipy.spatial.transform import Rotation


def estimate(o):
    t = o['time']
    rate = o['gyro'].copy()
    rotations = np.repeat(np.eye(3)[None], len(t), axis=0)
    # ROTATION
    baselines = o['baselines'].copy()
    x = np.einsum('tij,bj->tbi', rotations, baselines) * o['wave_number']
    y = o['phase'].copy()
    # PHASE
    # CENTER
    # NUISANCE
    a = x.reshape(-1, 3)
    b = y.reshape(-1, y.shape[-1])
    u = np.linalg.lstsq(a[:, :2], b, rcond=None)[0]
    u = np.vstack((u, np.zeros((1, b.shape[1]))))
    residual = b - a @ u
    spread = float(np.sqrt(np.mean((u-u.mean(axis=1, keepdims=True))**2)))
    rms = float(np.sqrt(np.mean(residual**2)))
    singular = np.linalg.svd(a, compute_uv=False)
    status = 'common' if spread < .12 else 'diverse'
    # STATUS
    mean = u.mean(axis=1)
    direction = mean / (np.linalg.norm(mean)+1e-30)
    # DIRECTION
    return {'status': status, 'spread': spread, 'rms': rms,
            'singular_min': float(singular[-1]), 'direction': direction.tolist(),
            'direction_available': True, 'direction_radius_deg': 0.,
            'absolute_position_available': False}
