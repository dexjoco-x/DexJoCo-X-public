from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .ik_position import (
    apply_position_ik_control,
    joint_position_limits,
    joint_qpos_ids,
)


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.asarray((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))


def _quat_wxyz_from_orientation(ori: np.ndarray | None) -> np.ndarray | None:
    if ori is None:
        return None
    ori = np.asarray(ori, dtype=np.float64)
    if ori.shape == (3, 3):
        return _quat_wxyz_from_matrix(ori)
    return ori.reshape(4)


def apply_arm_pose_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    profile,
    site_id: int,
    dof_ids: np.ndarray,
    ctrl_ids: np.ndarray,
    pos: np.ndarray | None = None,
    ori: np.ndarray | None = None,
    joint: np.ndarray | None = None,
    pos_gains: tuple[float, float, float] | np.ndarray = (200.0, 200.0, 200.0),
    ori_gains: tuple[float, float, float] | np.ndarray = (200.0, 200.0, 200.0),
    damping_ratio: float = 1.0,
    nullspace_stiffness: float = 0.5,
    max_pos_acceleration: float | None = None,
    max_ori_acceleration: float | None = None,
    gravity_comp: bool = True,
    position_ik_pos_gain: float = 1.0,
    position_ik_ori_gain: float = 1.0,
    position_ik_damping: float = 1e-3,
    position_ik_nullspace_gain: float = 0.02,
    position_ik_max_delta: float = 0.03,
) -> np.ndarray:
    """Apply an end-effector pose command for torque or position-controlled arms."""

    control_mode = str(getattr(profile, "control_mode", "torque")).lower()
    if control_mode == "position":
        qpos_ids = joint_qpos_ids(model, profile.joint_names)
        lower, upper = joint_position_limits(model, profile.joint_names)
        return apply_position_ik_control(
            model=model,
            data=data,
            site_id=site_id,
            dof_ids=dof_ids,
            ctrl_ids=ctrl_ids,
            target_pos=pos,
            target_quat_wxyz=_quat_wxyz_from_orientation(ori),
            q_reference=joint,
            qpos_ids=qpos_ids,
            joint_lower=lower,
            joint_upper=upper,
            pos_gain=position_ik_pos_gain,
            ori_gain=position_ik_ori_gain,
            damping=position_ik_damping,
            nullspace_gain=position_ik_nullspace_gain,
            max_delta=position_ik_max_delta,
        )

    if control_mode != "torque":
        raise ValueError(f"Unsupported arm control mode: {control_mode!r}")

    from .opspace import opspace

    command = opspace(
        model=model,
        data=data,
        site_id=site_id,
        dof_ids=dof_ids,
        pos=pos,
        ori=ori,
        joint=joint,
        pos_gains=pos_gains,
        ori_gains=ori_gains,
        damping_ratio=damping_ratio,
        nullspace_stiffness=nullspace_stiffness,
        max_pos_acceleration=max_pos_acceleration,
        max_ori_acceleration=max_ori_acceleration,
        gravity_comp=gravity_comp,
    )
    data.ctrl[np.asarray(ctrl_ids, dtype=int).reshape(-1)] = command
    return command
