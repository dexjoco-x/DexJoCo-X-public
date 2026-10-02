from .hand_specs import get_pick_bucket_hand_spec
from .panda_pick_bucket_inspire_base_env import PandaPickBucketGymEnv


class PandaPickBucketInspireGymEnv(PandaPickBucketGymEnv):
    """Pick-bucket task variant with a right Inspire hand on the Panda wrist."""

    def __init__(self, *args, **kwargs):
        hand_spec = get_pick_bucket_hand_spec("inspire_right")
        super().__init__(
            *args,
            xml_path=hand_spec.xml_path,
            hand_joint_names=hand_spec.joint_names,
            hand_actuator_names=hand_spec.actuator_names,
            hand_home=hand_spec.home_qpos,
            hand_type=hand_spec.hand_type,
            hand_command_mode="actuator",
            **kwargs,
        )
