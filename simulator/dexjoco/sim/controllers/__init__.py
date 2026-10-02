from .arm_pose_control import apply_arm_pose_control
from .ik_position import (
    apply_position_ik_control,
    joint_position_limits,
    joint_qpos_ids,
    solve_position_ik_step,
)

try:
    from .opspace import opspace
except ModuleNotFoundError as exc:
    if exc.name != "dm_robotics":
        raise
    _OPSPACE_IMPORT_ERROR = exc

    def opspace(*args, **kwargs):
        raise ModuleNotFoundError(
            "opspace requires the dm-robotics-transformations package."
        ) from _OPSPACE_IMPORT_ERROR

__all__ = [
    "apply_arm_pose_control",
    "apply_position_ik_control",
    "joint_position_limits",
    "joint_qpos_ids",
    "opspace",
    "solve_position_ik_step",
]
