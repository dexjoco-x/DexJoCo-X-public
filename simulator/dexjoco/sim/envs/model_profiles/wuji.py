from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WujiHandProfile:
    """DexJoCo's first Wuji right-hand control profile."""

    hand_type: str
    urdf_path: Path
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    home_qpos: np.ndarray


_WUJI_ASSET_ROOT = (Path(__file__).resolve().parents[4] / "teleoperation" / "wuji") / "assets"
_WUJI_RIGHT_ROOT = _WUJI_ASSET_ROOT / "right"
_WUJI_LEFT_ROOT = _WUJI_ASSET_ROOT / "left"

# Wuji right hand from UniDex has 20 actuated revolute joints:
# F1 = thumb, F2 = index, F3 = middle, F4 = ring, F5 = pinky.
WUJI_RIGHT_JOINT_NAMES = (
    "wuji_F1J1",
    "wuji_F1J2",
    "wuji_F1J3",
    "wuji_F1J4",
    "wuji_F2J1",
    "wuji_F2J2",
    "wuji_F2J3",
    "wuji_F2J4",
    "wuji_F3J1",
    "wuji_F3J2",
    "wuji_F3J3",
    "wuji_F3J4",
    "wuji_F4J1",
    "wuji_F4J2",
    "wuji_F4J3",
    "wuji_F4J4",
    "wuji_F5J1",
    "wuji_F5J2",
    "wuji_F5J3",
    "wuji_F5J4",
)

WUJI_RIGHT_ACTUATOR_NAMES = tuple(f"{name}_ctrl" for name in WUJI_RIGHT_JOINT_NAMES)
WUJI_NATURAL_OPEN_SPREAD = np.asarray(
    (0.18, 0.0, 0.14, 0.20),
    dtype=np.float32,
)
WUJI_RIGHT_HOME = np.zeros(len(WUJI_RIGHT_JOINT_NAMES), dtype=np.float32)
WUJI_RIGHT_HOME[[5, 9, 13, 17]] = WUJI_NATURAL_OPEN_SPREAD

WUJI_LEFT_JOINT_NAMES = tuple(WUJI_RIGHT_JOINT_NAMES)
WUJI_LEFT_ACTUATOR_NAMES = tuple(f"{name}_ctrl" for name in WUJI_LEFT_JOINT_NAMES)
WUJI_LEFT_HOME = WUJI_RIGHT_HOME.copy()

WUJI_RIGHT_PROFILE = WujiHandProfile(
    hand_type="wuji_right",
    urdf_path=_WUJI_RIGHT_ROOT / "main.urdf",
    joint_names=WUJI_RIGHT_JOINT_NAMES,
    actuator_names=WUJI_RIGHT_ACTUATOR_NAMES,
    home_qpos=WUJI_RIGHT_HOME,
)

WUJI_LEFT_PROFILE = WujiHandProfile(
    hand_type="wuji_left",
    urdf_path=_WUJI_LEFT_ROOT / "main.urdf",
    joint_names=WUJI_LEFT_JOINT_NAMES,
    actuator_names=WUJI_LEFT_ACTUATOR_NAMES,
    home_qpos=WUJI_LEFT_HOME,
)


def get_wuji_right_profile() -> WujiHandProfile:
    return WUJI_RIGHT_PROFILE


def get_wuji_left_profile() -> WujiHandProfile:
    return WUJI_LEFT_PROFILE
