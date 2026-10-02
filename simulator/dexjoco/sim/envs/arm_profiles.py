from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
UR10E_HOME = np.asarray(
    (0.399258, 4.855564, -2.217920, -2.552952, -1.170867, -2.937512),
    dtype=np.float64,
)
IIWA14_HOME = np.asarray(
    (1.55623, 1.31128, -0.83377, -1.92607, -0.57963, 1.07115, -0.65251),
    dtype=np.float64,
)
RIZON4S_HOME = np.asarray((0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0), dtype=np.float64)
OPENARM_HOME = np.asarray((0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0), dtype=np.float64)

PANDA_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
PANDA_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
IIWA14_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
IIWA14_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
RIZON4S_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
RIZON4S_ACTUATOR_NAMES = RIZON4S_JOINT_NAMES
UR10E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR10E_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
OPENARM_RIGHT_JOINT_NAMES = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
OPENARM_LEFT_JOINT_NAMES = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
OPENARM_RIGHT_ACTUATOR_NAMES = tuple(f"right_joint{i}_ctrl" for i in range(1, 8))
OPENARM_LEFT_ACTUATOR_NAMES = tuple(f"left_joint{i}_ctrl" for i in range(1, 8))


@dataclass(frozen=True)
class ArmProfile:
    name: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    home_qpos: np.ndarray
    tau_limit: np.ndarray
    control_mode: str = "torque"


PANDA_PROFILE = ArmProfile(
    name="panda",
    joint_names=PANDA_JOINT_NAMES,
    actuator_names=PANDA_ACTUATOR_NAMES,
    home_qpos=PANDA_HOME,
    tau_limit=np.full(7, np.inf, dtype=np.float64),
)
UR10E_PROFILE = ArmProfile(
    name="ur10e",
    joint_names=UR10E_JOINT_NAMES,
    actuator_names=UR10E_ACTUATOR_NAMES,
    home_qpos=UR10E_HOME,
    tau_limit=np.asarray((330.0, 330.0, 150.0, 56.0, 56.0, 56.0), dtype=np.float64),
)
IIWA14_PROFILE = ArmProfile(
    name="iiwa14",
    joint_names=IIWA14_JOINT_NAMES,
    actuator_names=IIWA14_ACTUATOR_NAMES,
    home_qpos=IIWA14_HOME,
    tau_limit=np.asarray((320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0), dtype=np.float64),
)
RIZON4S_PROFILE = ArmProfile(
    name="rizon4s",
    joint_names=RIZON4S_JOINT_NAMES,
    actuator_names=RIZON4S_ACTUATOR_NAMES,
    home_qpos=RIZON4S_HOME,
    tau_limit=np.asarray((246.0, 246.0, 128.0, 128.0, 78.0, 78.0, 78.0), dtype=np.float64),
)
OPENARM_RIGHT_PROFILE = ArmProfile(
    name="openarm_right",
    joint_names=OPENARM_RIGHT_JOINT_NAMES,
    actuator_names=OPENARM_RIGHT_ACTUATOR_NAMES,
    home_qpos=OPENARM_HOME,
    tau_limit=np.full(7, np.inf, dtype=np.float64),
    control_mode="position",
)
OPENARM_LEFT_PROFILE = ArmProfile(
    name="openarm_left",
    joint_names=OPENARM_LEFT_JOINT_NAMES,
    actuator_names=OPENARM_LEFT_ACTUATOR_NAMES,
    home_qpos=OPENARM_HOME,
    tau_limit=np.full(7, np.inf, dtype=np.float64),
    control_mode="position",
)


def _has_all(model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> bool:
    return all(mujoco.mj_name2id(model, obj_type, name) >= 0 for name in names)


def _joint_dof_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Unknown joint name: {name!r}")
    return int(model.jnt_dofadr[joint_id])


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise ValueError(f"Unknown actuator name: {name!r}")
    return int(actuator_id)


def detect_arm_profile(model: mujoco.MjModel) -> ArmProfile:
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, OPENARM_RIGHT_JOINT_NAMES):
        return OPENARM_RIGHT_PROFILE
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, OPENARM_LEFT_JOINT_NAMES):
        return OPENARM_LEFT_PROFILE
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, UR10E_JOINT_NAMES):
        return UR10E_PROFILE
    if _has_all(model, mujoco.mjtObj.mjOBJ_MESH, ("kuka", "link_2_orange")):
        return IIWA14_PROFILE
    if _has_all(model, mujoco.mjtObj.mjOBJ_MESH, ("link0", "link7")) and _has_all(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ("base", "link7"),
    ):
        return RIZON4S_PROFILE
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, PANDA_JOINT_NAMES):
        return PANDA_PROFILE
    raise RuntimeError(
        "Could not detect supported arm profile. Expected Panda, UR10e, iiwa14, "
        "Rizon4S, or OpenArm joints."
    )


def configure_arm(model: mujoco.MjModel) -> tuple[ArmProfile, np.ndarray, np.ndarray]:
    profile = detect_arm_profile(model)
    dof_ids = np.asarray(
        [_joint_dof_id(model, name) for name in profile.joint_names],
        dtype=int,
    )
    ctrl_ids = np.asarray(
        [_actuator_id(model, name) for name in profile.actuator_names],
        dtype=int,
    )
    return profile, dof_ids, ctrl_ids


def _suffix_names(names: tuple[str, ...], side: str) -> tuple[str, ...]:
    return tuple(f"{name}_{side}" for name in names)


def _bimanual_profile(profile: ArmProfile, side: str) -> ArmProfile:
    return ArmProfile(
        name=f"{profile.name}_{side}",
        joint_names=_suffix_names(profile.joint_names, side),
        actuator_names=_suffix_names(profile.actuator_names, side),
        home_qpos=profile.home_qpos,
        tau_limit=profile.tau_limit,
        control_mode=profile.control_mode,
    )


def detect_bimanual_arm_profile(model: mujoco.MjModel, side: str) -> ArmProfile:
    side = str(side).lower()
    if side not in {"right", "left"}:
        raise ValueError(f"Expected side to be 'right' or 'left', got {side!r}.")
    openarm_profile = OPENARM_RIGHT_PROFILE if side == "right" else OPENARM_LEFT_PROFILE
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, openarm_profile.joint_names):
        return openarm_profile
    ur10e_profile = _bimanual_profile(UR10E_PROFILE, side)
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, ur10e_profile.joint_names):
        return ur10e_profile
    rizon4s_profile = _bimanual_profile(RIZON4S_PROFILE, side)
    if (
        _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, rizon4s_profile.joint_names)
        and _has_all(model, mujoco.mjtObj.mjOBJ_BODY, (f"base_{side}", f"link7_{side}"))
        and _has_all(model, mujoco.mjtObj.mjOBJ_MESH, (f"link0_{side}", f"link7_{side}"))
    ):
        return rizon4s_profile
    iiwa14_profile = ArmProfile(
        name=f"iiwa14_{side}",
        joint_names=tuple(f"joint{i}_{side}" for i in range(1, 8)),
        actuator_names=tuple(f"actuator{i}_{side}" for i in range(1, 8)),
        home_qpos=IIWA14_HOME,
        tau_limit=IIWA14_PROFILE.tau_limit,
    )
    if _has_all(model, mujoco.mjtObj.mjOBJ_JOINT, iiwa14_profile.joint_names):
        return iiwa14_profile
    raise RuntimeError(
        f"Could not detect supported {side} bimanual arm profile. "
        "Expected OpenArm, suffixed UR10e, Rizon4S, or joint1..joint7 names."
    )


def configure_bimanual_arm(
    model: mujoco.MjModel,
    side: str,
) -> tuple[ArmProfile, np.ndarray, np.ndarray]:
    profile = detect_bimanual_arm_profile(model, side)
    dof_ids = np.asarray(
        [_joint_dof_id(model, name) for name in profile.joint_names],
        dtype=int,
    )
    ctrl_ids = np.asarray(
        [_actuator_id(model, name) for name in profile.actuator_names],
        dtype=int,
    )
    return profile, dof_ids, ctrl_ids
