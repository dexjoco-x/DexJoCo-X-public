from abc import ABC, abstractmethod

class TaskConfigBase(ABC):
    """Base config for task environments."""

    proprio_keys = None

    @abstractmethod
    # 创建仿真环境
    def get_environment(self, policy_mode=False, render_mode="human", randomize=False, **kwargs):
        """Create the task environment.

        When ``policy_mode`` is true, implementations are expected to attach the
        policy action wrapper instead of the teleoperation wrapper.
        """
        pass

    @abstractmethod
    # process_demos(...)：处理 demonstration 数据，通常用于训练/转换数据，而不是加载 checkpoint
    def process_demos(self, demo):
        pass
