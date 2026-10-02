"""20 Hz XHand policy tasks; hardware teleoperation is not calibrated."""
from .config import TaskConfigBase
from .native_delta_teleop import SingleArmNativeDeltaPolicyWrapper
from .obs_adapters import DexjocoObsAdapter
from ..sim.envs.panda_xhand_env import PandaHammerNailXHandEnv,PandaPickBucketXHandEnv


class HammerNailXHandConfig(TaskConfigBase):
    action_semantics='tcp_delta_pose_current_actual'
    rotation_semantics='world_left_multiply_rotvec'
    action_layout=('arm_delta_pose_6','xhand_joint_target_12')
    arm_action_dim=6
    hand_action_dim=12
    control_hz=20
    proprio_keys=['tcp_pose','gripper_pose','hammer_ori_pose','nail_ori_pose','table_delta_height']
    env_class=PandaHammerNailXHandEnv

    def get_environment(self,policy_mode=True,render_mode='rgb_array',randomize=False,hz=20,**kwargs):
        if not policy_mode:raise NotImplementedError('XHand hardware teleoperation has not been calibrated.')
        if float(hz)!=self.control_hz:raise ValueError('XHand native-delta tasks require 20 Hz.')
        env=self.env_class(render_mode=render_mode,randomize=randomize,hz=hz,**kwargs)
        return DexjocoObsAdapter(SingleArmNativeDeltaPolicyWrapper(env),proprio_keys=self.proprio_keys)

    def process_demos(self,demo):return demo


class PickBucketXHandConfig(HammerNailXHandConfig):
    proprio_keys=['tcp_pose','gripper_pose','bucket_ori_pose','boxed_food_ori_pose','table_delta_height']
    env_class=PandaPickBucketXHandEnv
