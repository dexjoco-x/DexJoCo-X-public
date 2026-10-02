"""Shared current-TCP-relative action interface for pick-bucket tasks.

The historical pick-bucket environments consume an absolute 7-D TCP target.
This mixin is used only by the dedicated ``*_native_delta`` task variants and
keeps the historical environments and their datasets unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from .panda_hammer_nail_inspire_native_delta_env import (
    ACTION_SEMANTICS,
    NATIVE_ARM_DAMPING_RATIO,
    NATIVE_ARM_ORI_GAIN_BASE,
    NATIVE_ARM_POS_GAIN_BASE,
    ROTATION_SEMANTICS,
    _quat_wxyz_from_matrix,
    apply_delta_pose_to_actual_tcp,
)


class PickBucketNativeDeltaMixin:
    """Convert one 6-D world-frame delta into a fresh absolute TCP target."""

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

        # One native delta is defined over one policy/collection interval.
        # The official OPFA Isaac environments use dt=0.01 and decimation=5,
        # hence a 0.05 s (20 Hz) action interval.
        kwargs.setdefault("control_dt", 1.0 / hz)
        kwargs.setdefault("arm_pos_gain_base", NATIVE_ARM_POS_GAIN_BASE)
        kwargs.setdefault("arm_ori_gain_base", NATIVE_ARM_ORI_GAIN_BASE)
        kwargs.setdefault("arm_damping_ratio", NATIVE_ARM_DAMPING_RATIO)
        super().__init__(*args, **kwargs)

        self._delta_position_limit = float(delta_position_limit)
        self._delta_rotation_limit = float(delta_rotation_limit)
        if (
            not np.isfinite(self._delta_position_limit)
            or self._delta_position_limit <= 0.0
        ):
            raise ValueError("delta_position_limit must be finite and positive.")
        if (
            not np.isfinite(self._delta_rotation_limit)
            or self._delta_rotation_limit <= 0.0
        ):
            raise ValueError("delta_rotation_limit must be finite and positive.")

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
        actual = np.eye(4, dtype=np.float64)
        actual[:3, 3] = np.asarray(
            self._data.site_xpos[self._site_id],
            dtype=np.float64,
        )
        actual[:3, :3] = np.asarray(
            self._data.site_xmat[self._site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        return actual

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = self.arm_action_dim + self._n_hand
        if action.shape != (expected,):
            raise ValueError(
                "Expected native-delta pick-bucket action "
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

        observation, reward, done, truncated, info = super().step(absolute_action)
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
        return observation, reward, done, truncated, info
