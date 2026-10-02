"""Inspire hammer-nail environment with native per-step TCP delta actions.

This is intentionally a separate environment from
``PandaHammerNailInspireGymEnv``.  The historical environment consumes an
absolute 7-D TCP target.  This variant consumes a 6-D delta pose and resolves
it against the *current simulated TCP* on every call to :meth:`step`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from .panda_hammer_nail_inspire_env import PandaHammerNailInspireGymEnv


ACTION_SEMANTICS = "tcp_delta_pose_current_actual"
ROTATION_SEMANTICS = "world_left_multiply_rotvec"
NATIVE_ARM_POS_GAIN_BASE = 3600.0
NATIVE_ARM_ORI_GAIN_BASE = 3600.0
NATIVE_ARM_DAMPING_RATIO = 1.0
NATIVE_IMPACT_VELOCITY_TIMEBASE = "physics_dt"


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(
        np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    ).as_quat()
    return np.asarray(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def apply_delta_pose_to_actual_tcp(
    actual_tcp_matrix: np.ndarray,
    delta_pose: np.ndarray,
) -> np.ndarray:
    """Apply ``[delta_xyz, delta_rotvec]`` using OPFA/IsaacLab semantics.

    Translation is expressed in the world/base frame.  Rotation is a world
    frame left multiplication:

    ``target_p = actual_p + delta_p``
    ``target_R = Exp(delta_rotvec) @ actual_R``
    """

    actual = np.asarray(actual_tcp_matrix, dtype=np.float64).reshape(4, 4)
    delta = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    target = actual.copy()
    target[:3, 3] = actual[:3, 3] + delta[:3]
    target[:3, :3] = (
        Rotation.from_rotvec(delta[3:6]).as_matrix() @ actual[:3, :3]
    )
    return target


class PandaHammerNailInspireNativeDeltaGymEnv(PandaHammerNailInspireGymEnv):
    """Hammer-nail task whose arm action is a native current-TCP delta."""

    action_semantics = ACTION_SEMANTICS
    rotation_semantics = ROTATION_SEMANTICS
    arm_action_dim = 6

    def __init__(
        self,
        *args,
        delta_position_limit: float = 0.02,
        delta_rotation_limit: float = 0.03,
        **kwargs,
    ):
        hz = float(kwargs.get("hz", 20))
        if not np.isfinite(hz) or hz <= 0.0:
            raise ValueError("hz must be finite and positive.")
        # A native delta is defined over one collection interval.  The
        # historical environment simulated only 20 ms even when collection
        # ran at 20 Hz (50 ms), so most of each current-TCP-relative command
        # was still unfinished when the next frame arrived and was discarded.
        kwargs.setdefault("control_dt", 1.0 / hz)
        kwargs.setdefault("arm_pos_gain_base", NATIVE_ARM_POS_GAIN_BASE)
        kwargs.setdefault("arm_ori_gain_base", NATIVE_ARM_ORI_GAIN_BASE)
        kwargs.setdefault("arm_damping_ratio", NATIVE_ARM_DAMPING_RATIO)
        kwargs.setdefault(
            "impact_velocity_timebase",
            NATIVE_IMPACT_VELOCITY_TIMEBASE,
        )
        super().__init__(*args, **kwargs)
        self._delta_position_limit = float(delta_position_limit)
        self._delta_rotation_limit = float(delta_rotation_limit)

        hand_joint_ids = [
            int(self._model.joint(name).id) for name in self._allegro_joint_names
        ]
        hand_lower = self._model.jnt_range[hand_joint_ids, 0].astype(np.float32)
        hand_upper = self._model.jnt_range[hand_joint_ids, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    np.full(3, -self._delta_position_limit, dtype=np.float32),
                    np.full(3, -self._delta_rotation_limit, dtype=np.float32),
                    hand_lower,
                ]
            ),
            high=np.concatenate(
                [
                    np.full(3, self._delta_position_limit, dtype=np.float32),
                    np.full(3, self._delta_rotation_limit, dtype=np.float32),
                    hand_upper,
                ]
            ),
            dtype=np.float32,
        )

    def _actual_tcp_matrix(self) -> np.ndarray:
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = np.asarray(
            self._data.site_xpos[self._site_id], dtype=np.float64
        )
        target[:3, :3] = np.asarray(
            self._data.site_xmat[self._site_id], dtype=np.float64
        ).reshape(3, 3)
        return target

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = self.arm_action_dim + self._n_hand
        if action.shape != (expected,):
            raise ValueError(
                "Expected native-delta Inspire action "
                f"[delta_pose_6, hand_{self._n_hand}] with shape ({expected},), "
                f"got {action.shape}."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Native delta action contains NaN or Inf.")

        requested_delta = action[:6].copy()
        applied_delta = requested_delta.copy()
        applied_delta[:3] = np.clip(
            applied_delta[:3],
            -self._delta_position_limit,
            self._delta_position_limit,
        )
        applied_delta[3:6] = np.clip(
            applied_delta[3:6],
            -self._delta_rotation_limit,
            self._delta_rotation_limit,
        )

        actual_before = self._actual_tcp_matrix()
        target = apply_delta_pose_to_actual_tcp(actual_before, applied_delta)
        absolute_action = np.concatenate(
            [
                target[:3, 3],
                _quat_wxyz_from_matrix(target[:3, :3]),
                action[6:],
            ]
        )

        obs, rew, done, truncated, info = super().step(absolute_action)
        actual_after = self._actual_tcp_matrix()
        actual_motion = np.concatenate(
            [
                actual_after[:3, 3] - actual_before[:3, 3],
                Rotation.from_matrix(
                    actual_after[:3, :3] @ actual_before[:3, :3].T
                ).as_rotvec(),
            ]
        )
        requested_translation_norm = float(np.linalg.norm(applied_delta[:3]))
        requested_rotation_norm = float(np.linalg.norm(applied_delta[3:6]))
        actual_translation_norm = float(np.linalg.norm(actual_motion[:3]))
        actual_rotation_norm = float(np.linalg.norm(actual_motion[3:6]))
        info = dict(info)
        info.update(
            {
                "action_semantics": self.action_semantics,
                "rotation_semantics": self.rotation_semantics,
                "requested_delta_pose": requested_delta,
                "applied_delta_pose": applied_delta,
                "actual_tcp_pose7_before": np.concatenate(
                    [
                        actual_before[:3, 3],
                        _quat_wxyz_from_matrix(actual_before[:3, :3]),
                    ]
                ),
                "relative_target_pose7": np.concatenate(
                    [
                        target[:3, 3],
                        _quat_wxyz_from_matrix(target[:3, :3]),
                    ]
                ),
                "actual_delta_pose6": actual_motion,
                "native_delta_translation_tracking_ratio": (
                    actual_translation_norm / requested_translation_norm
                    if requested_translation_norm > 1e-9
                    else np.nan
                ),
                "native_delta_rotation_tracking_ratio": (
                    actual_rotation_norm / requested_rotation_norm
                    if requested_rotation_norm > 1e-9
                    else np.nan
                ),
            }
        )
        return obs, rew, done, truncated, info
