"""Frozen donor-demo interfaces, separate from the 42 release environments."""
from .hammer_nail_inspire_native_delta.config import TaskConfig as HammerNailInspireNativeDeltaConfig
from .pick_bucket_inspire_native_delta.config import TaskConfig as PickBucketInspireNativeDeltaConfig
from .pick_bucket_leap_native_delta.config import TaskConfig as PickBucketLeapNativeDeltaConfig
from .pick_bucket_sharpa_wave_native_delta.config import TaskConfig as PickBucketSharpaWaveNativeDeltaConfig
from .linkerhand_l20 import HammerNailLinkerL20Config, PickBucketLinkerL20Config
from .xhand import HammerNailXHandConfig, PickBucketXHandConfig

SOURCE_CONFIG_MAPPING = {
    'hammer_nail_xhand_native_delta': HammerNailXHandConfig,
    'pick_bucket_xhand_native_delta': PickBucketXHandConfig,
    'hammer_nail_linkerhand_l20_native_delta': HammerNailLinkerL20Config,
    'pick_bucket_linkerhand_l20_native_delta': PickBucketLinkerL20Config,
    'hammer_nail_inspire_native_delta': HammerNailInspireNativeDeltaConfig,
    'pick_bucket_inspire_native_delta': PickBucketInspireNativeDeltaConfig,
    'pick_bucket_leap_native_delta': PickBucketLeapNativeDeltaConfig,
    'pick_bucket_sharpa_wave_native_delta': PickBucketSharpaWaveNativeDeltaConfig,
}
