from .hand_specs import get_hammer_nail_hand_spec
from .panda_hammer_nail_inspire_base_env import PandaHammerNailGymEnv


class PandaHammerNailInspireGymEnv(PandaHammerNailGymEnv):
    """Hammer-nail task variant with a right Inspire hand on the Panda wrist."""

    def __init__(self, *args, **kwargs):
        hand_spec = get_hammer_nail_hand_spec("inspire_right")
        super().__init__(
            *args,
            xml_path=hand_spec.xml_path,
            hand_joint_names=hand_spec.joint_names,
            hand_actuator_names=hand_spec.actuator_names,
            hand_home=hand_spec.home_qpos,
            hand_command_mode="actuator",
            **kwargs,
        )
