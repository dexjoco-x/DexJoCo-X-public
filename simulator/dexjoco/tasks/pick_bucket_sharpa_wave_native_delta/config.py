from typing import Literal
from ...sim.envs.hand_specs import get_pick_bucket_hand_spec
from ...sim.envs.panda_pick_bucket_sharpa_wave_native_delta_env import PandaPickBucketSharpaWaveNativeDeltaGymEnv
from ..config import TaskConfigBase
from ..native_delta_teleop import NATIVE_DELTA_CLIPPING_MODE, NATIVE_DELTA_INPUT_MODE, SingleArmNativeDeltaPolicyWrapper
from ..obs_adapters import DexjocoObsAdapter
_HAND_SPEC = get_pick_bucket_hand_spec('sharpa_wave_right')

class TaskConfig(TaskConfigBase):
    """Matched Sharpa Wave collector for the three-hand transfer track."""
    action_semantics = 'tcp_delta_pose_current_actual'
    rotation_semantics = 'world_left_multiply_rotvec'
    action_layout = ('arm_delta_pose_6', 'sharpa_wave_joint_target_22')
    arm_action_dim = 6
    hand_action_dim = len(_HAND_SPEC.joint_names)
    control_hz = 20
    delta_input_mode = NATIVE_DELTA_INPUT_MODE
    delta_clipping_mode = NATIVE_DELTA_CLIPPING_MODE
    accumulate_clipped_residual = False
    proprio_keys = ['tcp_pose', 'gripper_pose', 'bucket_ori_pose', 'boxed_food_ori_pose', 'table_delta_height']

    def get_environment(self, policy_mode=True, render_mode: Literal['rgb_array', 'human']='human', randomize=False, hz: int=20, **kwargs):
        if not policy_mode:
            raise ValueError('Live glove capture is not part of the public release')
        if float(hz) != float(self.control_hz):
            raise ValueError(f'pick_bucket_sharpa_wave_native_delta is a fixed 20 Hz benchmark task; got hz={hz}.')
        env = PandaPickBucketSharpaWaveNativeDeltaGymEnv(render_mode=render_mode, randomize=randomize, hz=hz, delta_position_limit=0.02, delta_rotation_limit=0.03, **kwargs)
        env = SingleArmNativeDeltaPolicyWrapper(env)
        return DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)

    def process_demos(self, demo):
        return demo
