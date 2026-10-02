from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InspireHandProfile:
    """DexJoCo's first Inspire right-hand control profile."""

    hand_type: str
    urdf_path: Path
    active_joint_names: tuple[str, ...]
    full_joint_names: tuple[str, ...]
    mimic_rules: dict[str, tuple[str, float, float]]
    actuator_names: tuple[str, ...]
    home_qpos: np.ndarray
    full_home_qpos: np.ndarray

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.active_joint_names


_INSPIRE_ROOT = (Path(__file__).resolve().parents[4] / "teleoperation" / "inspire") / "assets" / "right"

# Inspire URDF has 12 revolute joints, but only these 6 are active.
# The remaining joints are mimic joints and should not be driven from UDP.
INSPIRE_RIGHT_ACTIVE_JOINT_NAMES = (
    "inspire_thumb_proximal_yaw_joint",
    "inspire_thumb_proximal_pitch_joint",
    "inspire_index_proximal_joint",
    "inspire_middle_proximal_joint",
    "inspire_ring_proximal_joint",
    "inspire_pinky_proximal_joint",
)

INSPIRE_RIGHT_FULL_JOINT_NAMES = (
    "inspire_thumb_proximal_yaw_joint",
    "inspire_thumb_proximal_pitch_joint",
    "inspire_thumb_intermediate_joint",
    "inspire_thumb_distal_joint",
    "inspire_index_proximal_joint",
    "inspire_index_intermediate_joint",
    "inspire_middle_proximal_joint",
    "inspire_middle_intermediate_joint",
    "inspire_ring_proximal_joint",
    "inspire_ring_intermediate_joint",
    "inspire_pinky_proximal_joint",
    "inspire_pinky_intermediate_joint",
)

INSPIRE_RIGHT_MIMIC_RULES = {
    "inspire_thumb_intermediate_joint": ("inspire_thumb_proximal_pitch_joint", 1.334, 0.0),
    "inspire_thumb_distal_joint": ("inspire_thumb_proximal_pitch_joint", 0.667, 0.0),
    "inspire_index_intermediate_joint": ("inspire_index_proximal_joint", 1.06399, -0.04545),
    "inspire_middle_intermediate_joint": ("inspire_middle_proximal_joint", 1.06399, -0.04545),
    "inspire_ring_intermediate_joint": ("inspire_ring_proximal_joint", 1.06399, -0.04545),
    "inspire_pinky_intermediate_joint": ("inspire_pinky_proximal_joint", 1.06399, -0.04545),
}

INSPIRE_LEFT_ACTIVE_JOINT_NAMES = INSPIRE_RIGHT_ACTIVE_JOINT_NAMES
INSPIRE_LEFT_FULL_JOINT_NAMES = INSPIRE_RIGHT_FULL_JOINT_NAMES
INSPIRE_LEFT_MIMIC_RULES = INSPIRE_RIGHT_MIMIC_RULES

INSPIRE_RIGHT_ACTUATOR_NAMES = tuple(f"{name}_ctrl" for name in INSPIRE_RIGHT_ACTIVE_JOINT_NAMES)
INSPIRE_LEFT_ACTUATOR_NAMES = tuple(f"{name}_ctrl" for name in INSPIRE_LEFT_ACTIVE_JOINT_NAMES)
INSPIRE_RIGHT_HOME = np.zeros(len(INSPIRE_RIGHT_ACTIVE_JOINT_NAMES), dtype=np.float32)
INSPIRE_RIGHT_FULL_HOME = np.zeros(len(INSPIRE_RIGHT_FULL_JOINT_NAMES), dtype=np.float32)
INSPIRE_LEFT_HOME = np.zeros(len(INSPIRE_LEFT_ACTIVE_JOINT_NAMES), dtype=np.float32)
INSPIRE_LEFT_FULL_HOME = np.zeros(len(INSPIRE_LEFT_FULL_JOINT_NAMES), dtype=np.float32)

INSPIRE_RIGHT_PROFILE = InspireHandProfile(
    hand_type="inspire_right",
    urdf_path=_INSPIRE_ROOT / "main.urdf",
    active_joint_names=INSPIRE_RIGHT_ACTIVE_JOINT_NAMES,
    full_joint_names=INSPIRE_RIGHT_FULL_JOINT_NAMES,
    mimic_rules=INSPIRE_RIGHT_MIMIC_RULES,
    actuator_names=INSPIRE_RIGHT_ACTUATOR_NAMES,
    home_qpos=INSPIRE_RIGHT_HOME,
    full_home_qpos=INSPIRE_RIGHT_FULL_HOME,
)

INSPIRE_LEFT_PROFILE = InspireHandProfile(
    hand_type="inspire_left",
    urdf_path=_INSPIRE_ROOT / "main.urdf",
    active_joint_names=INSPIRE_LEFT_ACTIVE_JOINT_NAMES,
    full_joint_names=INSPIRE_LEFT_FULL_JOINT_NAMES,
    mimic_rules=INSPIRE_LEFT_MIMIC_RULES,
    actuator_names=INSPIRE_LEFT_ACTUATOR_NAMES,
    home_qpos=INSPIRE_LEFT_HOME,
    full_home_qpos=INSPIRE_LEFT_FULL_HOME,
)


def get_inspire_right_profile() -> InspireHandProfile:
    return INSPIRE_RIGHT_PROFILE


def get_inspire_left_profile() -> InspireHandProfile:
    return INSPIRE_LEFT_PROFILE
