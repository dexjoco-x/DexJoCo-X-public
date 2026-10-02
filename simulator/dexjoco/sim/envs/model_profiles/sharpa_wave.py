from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SharpaWaveHandProfile:
    """Sharpa Wave 右手的本地资产与关节顺序定义。"""

    hand_type: str
    xml_path: Path
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    home_qpos: np.ndarray


_SHARPA_ROOT = (Path(__file__).resolve().parents[4] / "teleoperation" / "sharpa_wave") / "mujoco_menagerie" / "sharpa_wave"

# 这里固定 DexJoCo 内部统一采用的 Sharpa Wave 右手 22 维关节顺序。
# 后续 viewer、UDP bridge、任务环境、数据采集都应复用这套顺序。
SHARPA_WAVE_RIGHT_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

# menagerie 这版 Sharpa 的 actuator name 与 joint name 一一对应，可以直接共用。
SHARPA_WAVE_RIGHT_ACTUATOR_NAMES = SHARPA_WAVE_RIGHT_JOINT_NAMES

# 第一版默认 home pose 先使用全零位。
# 这和 menagerie 场景的默认展示一致，后面如果需要更适合抓取的初始姿态，再单独调表。
SHARPA_WAVE_RIGHT_HOME = np.zeros(len(SHARPA_WAVE_RIGHT_JOINT_NAMES), dtype=np.float32)

SHARPA_WAVE_LEFT_JOINT_NAMES = tuple(
    name.replace("right_", "left_") for name in SHARPA_WAVE_RIGHT_JOINT_NAMES
)

SHARPA_WAVE_LEFT_ACTUATOR_NAMES = tuple(
    name.replace("right_", "left_") for name in SHARPA_WAVE_RIGHT_ACTUATOR_NAMES
)

SHARPA_WAVE_LEFT_HOME = np.zeros(len(SHARPA_WAVE_LEFT_JOINT_NAMES), dtype=np.float32)

SHARPA_WAVE_RIGHT_PROFILE = SharpaWaveHandProfile(
    hand_type="sharpa_wave_right",
    xml_path=_SHARPA_ROOT / "right_hand.xml",
    joint_names=SHARPA_WAVE_RIGHT_JOINT_NAMES,
    actuator_names=SHARPA_WAVE_RIGHT_ACTUATOR_NAMES,
    home_qpos=SHARPA_WAVE_RIGHT_HOME,
)


def get_sharpa_wave_right_profile() -> SharpaWaveHandProfile:
    return SHARPA_WAVE_RIGHT_PROFILE
