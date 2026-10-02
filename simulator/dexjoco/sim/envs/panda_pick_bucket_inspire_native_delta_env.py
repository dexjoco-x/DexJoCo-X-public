"""Inspire pick-bucket environment with native current-TCP delta actions."""

from .panda_pick_bucket_inspire_env import PandaPickBucketInspireGymEnv
from .panda_pick_bucket_native_delta_env import PickBucketNativeDeltaMixin


class PandaPickBucketInspireNativeDeltaGymEnv(
    PickBucketNativeDeltaMixin,
    PandaPickBucketInspireGymEnv,
):
    """Dedicated Inspire native-delta variant; historical task is unchanged."""

