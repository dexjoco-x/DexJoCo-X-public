from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def joint_qpos_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    """Return qpos ids for scalar joints in ``joint_names``."""

    qpos_ids: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"Unknown joint name: {name!r}")
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            raise ValueError(
                f"Position IK only supports scalar hinge/slide joints, got {name!r}."
            )
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(qpos_ids, dtype=int)


def joint_position_limits(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar joint lower/upper limits, using infinities when unlimited."""

    lower = np.full(len(joint_names), -np.inf, dtype=np.float64)
    upper = np.full(len(joint_names), np.inf, dtype=np.float64)
    for idx, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"Unknown joint name: {name!r}")
        if bool(model.jnt_limited[joint_id]):
            lower[idx], upper[idx] = model.jnt_range[joint_id]
    return lower, upper


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    return quat / norm


def _rotation_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    return R.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()


def _damped_pinv_task_delta(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    damping = max(float(damping), 0.0)
    lhs = jacobian @ jacobian.T
    lhs += np.eye(lhs.shape[0], dtype=np.float64) * (damping * damping)
    pinv = jacobian.T @ np.linalg.solve(lhs, np.eye(lhs.shape[0], dtype=np.float64))
    return pinv @ error, pinv


def solve_position_ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    dof_ids: np.ndarray,
    target_pos: np.ndarray | None = None,
    target_quat_wxyz: np.ndarray | None = None,
    q_reference: np.ndarray | None = None,
    qpos_ids: np.ndarray | None = None,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    pos_gain: float = 1.0,
    ori_gain: float = 1.0,
    damping: float = 1e-3,
    nullspace_gain: float = 0.02,
    max_delta: float = 0.05,
) -> np.ndarray:
    """Compute one joint-position IK target for position-controlled arms.

    The returned vector is intended for MuJoCo ``position`` actuators.  It does
    not mutate ``data.ctrl`` so callers can clamp, filter, or combine it with a
    gripper command before stepping the simulation.
    """

    dof_ids = np.asarray(dof_ids, dtype=int).reshape(-1)
    if qpos_ids is None:
        qpos_ids = dof_ids
    else:
        qpos_ids = np.asarray(qpos_ids, dtype=int).reshape(-1)
    if qpos_ids.shape != dof_ids.shape:
        raise ValueError("qpos_ids and dof_ids must have the same shape.")

    mujoco.mj_forward(model, data)

    jac_pos = np.zeros((3, model.nv), dtype=np.float64)
    jac_rot = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, int(site_id))

    rows: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    if target_pos is not None:
        pos_error = np.asarray(target_pos, dtype=np.float64).reshape(3) - data.site_xpos[site_id]
        rows.append(float(pos_gain) * jac_pos[:, dof_ids])
        errors.append(float(pos_gain) * pos_error)
    if target_quat_wxyz is not None:
        target_rot = _rotation_from_quat_wxyz(target_quat_wxyz)
        current_rot = data.site_xmat[site_id].reshape(3, 3)
        rot_error = R.from_matrix(target_rot @ current_rot.T).as_rotvec()
        rows.append(float(ori_gain) * jac_rot[:, dof_ids])
        errors.append(float(ori_gain) * rot_error)

    q = data.qpos[qpos_ids].copy()
    if rows:
        jacobian = np.concatenate(rows, axis=0)
        error = np.concatenate(errors, axis=0)
        dq, task_pinv = _damped_pinv_task_delta(jacobian, error, damping)
        if q_reference is not None and float(nullspace_gain) > 0.0:
            q_ref = np.asarray(q_reference, dtype=np.float64).reshape(q.shape)
            projector = np.eye(len(dof_ids), dtype=np.float64) - task_pinv @ jacobian
            dq += projector @ (float(nullspace_gain) * (q_ref - q))
    elif q_reference is not None:
        q_ref = np.asarray(q_reference, dtype=np.float64).reshape(q.shape)
        dq = float(nullspace_gain) * (q_ref - q)
    else:
        raise ValueError("Expected at least one target pose component or q_reference.")

    max_delta = float(max_delta)
    if max_delta > 0.0:
        dq = np.clip(dq, -max_delta, max_delta)
    q_target = q + dq

    if joint_lower is not None or joint_upper is not None:
        lower = (
            np.asarray(joint_lower, dtype=np.float64).reshape(q.shape)
            if joint_lower is not None
            else np.full(q.shape, -np.inf, dtype=np.float64)
        )
        upper = (
            np.asarray(joint_upper, dtype=np.float64).reshape(q.shape)
            if joint_upper is not None
            else np.full(q.shape, np.inf, dtype=np.float64)
        )
        q_target = np.clip(q_target, lower, upper)
    return q_target


def apply_position_ik_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    dof_ids: np.ndarray,
    ctrl_ids: np.ndarray,
    target_pos: np.ndarray | None = None,
    target_quat_wxyz: np.ndarray | None = None,
    q_reference: np.ndarray | None = None,
    qpos_ids: np.ndarray | None = None,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    pos_gain: float = 1.0,
    ori_gain: float = 1.0,
    damping: float = 1e-3,
    nullspace_gain: float = 0.02,
    max_delta: float = 0.05,
) -> np.ndarray:
    """Write one position-IK target into ``data.ctrl`` and return it."""

    q_target = solve_position_ik_step(
        model=model,
        data=data,
        site_id=site_id,
        dof_ids=dof_ids,
        target_pos=target_pos,
        target_quat_wxyz=target_quat_wxyz,
        q_reference=q_reference,
        qpos_ids=qpos_ids,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        pos_gain=pos_gain,
        ori_gain=ori_gain,
        damping=damping,
        nullspace_gain=nullspace_gain,
        max_delta=max_delta,
    )
    data.ctrl[np.asarray(ctrl_ids, dtype=int).reshape(-1)] = q_target
    return q_target
