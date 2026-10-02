"""Control order of the supplied L20 V10.1 right hand (not L25 hardware)."""
from pathlib import Path
import numpy as np

HAND_TYPE = "linkerhand_l20_right"
JOINT_NAMES = tuple("l20_" + name for name in (
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
))
ACTUATOR_NAMES = tuple(name + "_ctrl" for name in JOINT_NAMES)
HOME = np.zeros(len(JOINT_NAMES), dtype=np.float32)


def environment_kwargs(task):
    if task not in ("hammer_nail", "bucket_pick"):
        raise ValueError(task)
    return dict(
        xml_path=Path(__file__).parent / "xmls" / f"arena_arm_hand_{task}_linkerhand_l20.xml",
        hand_joint_names=JOINT_NAMES, hand_actuator_names=ACTUATOR_NAMES,
        hand_home=HOME.copy(), hand_type=HAND_TYPE, hand_command_mode="actuator",
    )
