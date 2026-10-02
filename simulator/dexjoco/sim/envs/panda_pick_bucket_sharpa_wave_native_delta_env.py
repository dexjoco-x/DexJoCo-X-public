"""Sharpa Wave pick-bucket environment with current-TCP delta actions."""

from .panda_pick_bucket_native_delta_env import PickBucketNativeDeltaMixin
from .panda_pick_bucket_sharpa_wave_env import PandaPickBucketSharpaWaveGymEnv


class PandaPickBucketSharpaWaveNativeDeltaGymEnv(
    PickBucketNativeDeltaMixin,
    PandaPickBucketSharpaWaveGymEnv,
):
    """Dedicated Sharpa Wave variant; the historical absolute task is unchanged."""

