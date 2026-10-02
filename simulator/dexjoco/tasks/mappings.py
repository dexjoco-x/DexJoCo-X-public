"""Exactly seven hands by six release tasks; no historical task registration."""
from dexjoco.benchmark6 import HANDS, TARGET_TASKS
from .config import TaskConfigBase

class BenchmarkTaskConfig(TaskConfigBase):
    task = None
    hand = None
    action_semantics = "tcp_delta_pose_current_actual"
    rotation_semantics = "world_left_multiply_rotvec"
    control_hz = 20

    def get_environment(self, policy_mode=True, render_mode="rgb_array", randomize=True,
                        hz=20, seed=0, randomize_dynamics=False, **kwargs):
        if not policy_mode:
            raise ValueError("Live glove capture is not part of the public release")
        if hz != self.control_hz or randomize_dynamics or kwargs:
            raise ValueError("Use the frozen 20-Hz benchmark settings; only the scene seed varies")
        from dexjoco.benchmark6.environments import make_env
        return make_env(self.task, self.hand, seed=int(seed), pixels=render_mode != "none")

    def process_demos(self, demo):
        return demo

CONFIG_MAPPING = {
    f"{task}/{hand}": type(
        f"{task}_{hand}_Config", (BenchmarkTaskConfig,), {"task": task, "hand": hand, "__module__": __name__}
    )
    for task in TARGET_TASKS for hand in HANDS
}
