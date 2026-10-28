"""Observation/snapshot utilities that leave the dynamics integration state intact."""
import hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

PHYSICS = ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat', 'qacc_warmstart', 'time')


def snapshot(runtime):
    return {k: np.array(getattr(runtime.d, k), copy=True) for k in PHYSICS}


def physical_hash(runtime):
    h = hashlib.sha256()
    for name in PHYSICS:
        h.update(np.asarray(getattr(runtime.d, name)).tobytes())
    return h.hexdigest()


def robot_state(runtime):
    for k in PHYSICS:
        if k == 'time':
            runtime.shadow.time = runtime.d.time
        else:
            getattr(runtime.shadow, k)[:] = getattr(runtime.d, k)
    mujoco.mj_forward(runtime.m, runtime.shadow)
    out = np.zeros((2, 31), np.float64)
    for arm, side in enumerate(('right', 'left')):
        if side not in runtime.e._native_ctrl_ids:
            continue
        sid = runtime.e._tcp_sites[side]
        q = runtime.shadow.qpos[runtime.e._full_hand_qpos[side]]
        out[arm, :3] = runtime.shadow.site_xpos[sid]
        out[arm, 3:7] = Rotation.from_matrix(runtime.shadow.site_xmat[sid].reshape(3, 3)).as_quat(scalar_first=True)
        out[arm, 7:7 + len(q)] = q
    return out


def state_error(actual, expected, mask):
    actual, expected = np.array(actual, copy=True), np.asarray(expected)
    for arm in range(2):
        if mask[arm, 0] and np.dot(actual[arm, 3:7], expected[arm, 3:7]) < 0:
            actual[arm, 3:7] *= -1
    return float(np.max(np.abs(actual - expected)[mask]))
