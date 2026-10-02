from typing import Literal
from ...sim.envs.panda_hammer_nail_inspire_native_delta_env import PandaHammerNailInspireNativeDeltaGymEnv
from ..config import TaskConfigBase
from ..native_delta_teleop import NATIVE_DELTA_CLIPPING_MODE, NATIVE_DELTA_INPUT_MODE, SingleArmNativeDeltaPolicyWrapper
from ..obs_adapters import DexjocoObsAdapter

class TaskConfig(TaskConfigBase):
    """Separate task ID for native current-TCP delta collection/evaluation."""
    action_semantics = 'tcp_delta_pose_current_actual'
    rotation_semantics = 'world_left_multiply_rotvec'
    action_layout = ('arm_delta_pose_6', 'inspire_joint_target_6')
    arm_action_dim = 6
    hand_action_dim = 6
    delta_input_mode = NATIVE_DELTA_INPUT_MODE
    delta_clipping_mode = NATIVE_DELTA_CLIPPING_MODE
    accumulate_clipped_residual = False
    proprio_keys = ['tcp_pose', 'gripper_pose', 'hammer_ori_pose', 'nail_ori_pose', 'table_delta_height']

    def get_environment(self, policy_mode=True, render_mode: Literal['rgb_array', 'human']='human', randomize=False, hz: int=30, **kwargs):
        if not policy_mode:
            raise ValueError('Live glove capture is not part of the public release')
        env = PandaHammerNailInspireNativeDeltaGymEnv(render_mode=render_mode, randomize=randomize, hz=hz, delta_position_limit=0.02, delta_rotation_limit=0.03, **kwargs)
        env = SingleArmNativeDeltaPolicyWrapper(env)
        return DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
