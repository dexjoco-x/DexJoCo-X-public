"""Panda + XHand variants with unchanged DexJoCo task objects and criteria."""
from .xhand_specs import environment_kwargs
from .panda_hammer_nail_inspire_base_env import PandaHammerNailGymEnv
from .panda_pick_bucket_inspire_base_env import PandaPickBucketGymEnv
from .panda_pick_bucket_native_delta_env import PickBucketNativeDeltaMixin


class PandaHammerNailXHandEnv(PickBucketNativeDeltaMixin,PandaHammerNailGymEnv):
    def __init__(self,**kwargs):
        kwargs.setdefault('hz',20)
        kwargs.setdefault('impact_velocity_timebase','physics_dt')
        super().__init__(**environment_kwargs('hammer_nail'),**kwargs)


class PandaPickBucketXHandEnv(PickBucketNativeDeltaMixin,PandaPickBucketGymEnv):
    def __init__(self,**kwargs):
        kwargs.setdefault('hz',20)
        super().__init__(**environment_kwargs('bucket_pick'),**kwargs)
