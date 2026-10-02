from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space

# 它的作用是：把原始环境 observation 转换成 DexJoCo/OpenPI 更容易使用的格式
# 尤其是把多个低维状态字段拼接成一个扁平的一维向量。

# ObservationWrapper 的作用是：只修改环境返回的 observation，不改变 action、reward、done 等逻辑。
class DexjocoObsAdapter(gym.ObservationWrapper):
    """Adapt raw task observations into the flattened Dexjoco collection format."""

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        ordered_spaces = OrderedDict(
            (key, self.env.observation_space["state"][key]) for key in self.proprio_keys
        )
        self.proprio_space = gym.spaces.Dict(ordered_spaces)

        # 定义新的观测空间
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space.get("images", {})),
            }
        )

    def observation(self, obs):
        proprio_values = [np.asarray(obs["state"][key]).ravel() for key in self.proprio_keys]
        if proprio_values:
            flat_state = np.concatenate(proprio_values, axis=0).astype(np.float64, copy=False)
        else:
            flat_state = np.zeros((0,), dtype=np.float64)
        return {"state": flat_state, **obs.get("images", {})}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

# 把环境原始 observation 转成 DexJoCo/OpenPI policy 需要的 observation 格式。
'''
1. 从 obs["state"] 里挑选 proprio_keys 指定的字段
2. 将每个字段转成一维数组
3. 按固定顺序拼接成一个 flat_state
4. 把 flat_state 放到返回结果的 "state"
5. 把 obs["images"] 里的图像展开到顶层
6. 更新 observation_space，让 Gym 知道新 observation 的结构
'''
'''
原始 obs：{"state": {"tcp_pose": ..., "gripper_pose": ...}, "images": {...}}

适配后：{"state": 一维状态向量, "camera_xxx": 图像, ...}

'''

'''
举个例子
假设原始环境返回：

obs = {
    "state": {
        "tcp_pose": np.array([1, 2, 3]),
        "gripper_pose": np.array([0.5]),
    },
    "images": {
        "front": np.zeros((224, 224, 3), dtype=np.uint8),
    }
}

会得到：
{
    "state": np.array([1.0, 2.0, 3.0, 0.5]),
    "front": np.zeros((224, 224, 3), dtype=np.uint8),
}
'''
