from __future__ import annotations

from importlib import import_module


_ENV_EXPORTS = {'PandaBimanualMicrowaveCookGymEnv': 'panda_bimanual_microwave_cook_env', 'PandaBimanualPhotographGymEnv': 'panda_bimanual_photograph_env', 'PandaBimanualUnlockIpadGymEnv': 'panda_bimanual_unlock_ipad_env', 'PandaHammerNailInspireGymEnv': 'panda_hammer_nail_inspire_env', 'PandaHammerNailInspireNativeDeltaGymEnv': 'panda_hammer_nail_inspire_native_delta_env', 'PandaPickBucketInspireGymEnv': 'panda_pick_bucket_inspire_env', 'PandaPickBucketLeapGymEnv': 'panda_pick_bucket_leap_env', 'PandaPickBucketSharpaWaveGymEnv': 'panda_pick_bucket_sharpa_wave_env', 'PandaPickBucketSharpaWaveNativeDeltaGymEnv': 'panda_pick_bucket_sharpa_wave_native_delta_env', 'PandaBimanualHanoiGymEnv': 'panda_bimanual_hanoi_env'}

__all__ = tuple(_ENV_EXPORTS)


def __getattr__(name: str):
    if name not in _ENV_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_ENV_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
