"""Twelve independent XHand controls, in source URDF order."""
from pathlib import Path
import numpy as np

HAND_TYPE='xhand_right'
JOINT_NAMES=tuple('xhand_right_hand_'+name for name in (
    'thumb_bend_joint','thumb_rota_joint1','thumb_rota_joint2',
    'index_bend_joint','index_joint1','index_joint2',
    'mid_joint1','mid_joint2','ring_joint1','ring_joint2','pinky_joint1','pinky_joint2',
))
ACTUATOR_NAMES=tuple(name+'_ctrl' for name in JOINT_NAMES)
HOME=np.zeros(len(JOINT_NAMES),dtype=np.float32)
TIP_BODY_NAMES=tuple('xhand_right_hand_'+part for part in (
    'thumb_rota_tip','index_rota_tip','mid_tip','ring_tip','pinky_tip'))


def environment_kwargs(task):
    if task not in ('hammer_nail','bucket_pick'):raise ValueError(task)
    return dict(xml_path=Path(__file__).parent/'xmls'/f'arena_arm_hand_{task}_xhand.xml',
        hand_joint_names=JOINT_NAMES,hand_actuator_names=ACTUATOR_NAMES,hand_home=HOME.copy(),
        hand_type=HAND_TYPE,hand_command_mode='actuator')
