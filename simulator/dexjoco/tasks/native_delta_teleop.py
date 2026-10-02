"""Identity wrapper for frozen single-arm donor replay; no live capture."""
import gymnasium as gym
import numpy as np
NATIVE_DELTA_INPUT_MODE = "vive_consecutive_pose_increment"
NATIVE_DELTA_CLIPPING_MODE = "per_axis_saturation_no_residual"

class SingleArmNativeDeltaPolicyWrapper(gym.ActionWrapper):
    """Identity policy interface with an explicit native-delta shape check."""
    action_semantics = 'tcp_delta_pose_current_actual'
    rotation_semantics = 'world_left_multiply_rotvec'
    arm_action_dim = 6

    def __init__(self, env):
        super().__init__(env)
        raw_env = getattr(env, 'unwrapped', env)
        self.hand_dim = int(getattr(raw_env, '_n_hand', 6))
        self.policy_action_dim = self.arm_action_dim + self.hand_dim
        self.action_space = gym.spaces.Box(low=np.asarray(env.action_space.low, dtype=np.float32).copy(), high=np.asarray(env.action_space.high, dtype=np.float32).copy(), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.policy_action_dim,):
            raise ValueError(f'Expected native-delta policy action ({self.policy_action_dim},), got {action.shape}.')
        return action
