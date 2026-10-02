"""Trusted task registrations; no model-generated imports or permissive fallback."""
from copy import deepcopy

import numpy as np

from .bucket_chain import BUCKET_CHAIN, make_bucket_phase_manifests
from .collection_contracts import CollectionTaskSpec
from .hammer_chain import HAMMER_CHAIN, make_hammer_phase_manifests
from .hammer_observation import StationaryHammerObserver, stationary_token
from .hammer_perception import HammerNailSceneEstimate


HAMMER_TASK = "hammer_nail_inspire_native_delta"
BUCKET_TASK = "pick_bucket_inspire_native_delta"


def hammer_task_spec():
    return CollectionTaskSpec(HAMMER_TASK, HAMMER_CHAIN,
        make_hammer_phase_manifests(task_id=HAMMER_TASK, expected_action_dim=12),
        "根据当前静止仿真画面提出受限锤钉阶段计划，或重新观察、停止。计划经本地验证后仅允许执行第一阶段；其余阶段强制禁止。本次不是完整任务采集。",
        "第一阶段运动后，检查仍不确定。根据当前阶段图像和检查结果选择重新取证或停止；禁止生成动作计划，也不要把机械臂运动完成当作抓牢。")


def bucket_task_spec():
    return CollectionTaskSpec(BUCKET_TASK, BUCKET_CHAIN,
        make_bucket_phase_manifests(task_id=BUCKET_TASK, expected_action_dim=12),
        "根据当前静止仿真画面提出受限提桶阶段计划，或重新观察、停止。计划经本地验证后仅允许执行第一阶段；其余阶段强制禁止。本次不是完整任务采集。",
        "提桶第一阶段运动后，检查仍不确定。根据当前图像和检查结果选择重新取证或停止；禁止生成动作计划，不要把运动完成当作桶已被稳定提起。")


class HammerCollectionAdapter:
    def __init__(self):
        self.spec = hammer_task_spec()

    def make_reset_observer(self, **kwargs):
        return StationaryHammerObserver(**kwargs)

    def export_latest_scene(self, observer):
        if observer.latest_scene_estimate is None:
            raise ValueError("no current accepted hammer estimate")
        source = deepcopy(observer.latest_scene_estimate)
        for name in ("hammer_pose7", "nail_pose7"):
            source[name] = np.asarray(source[name], dtype=float)
        return HammerNailSceneEstimate(**source)

    def source_objects(self):
        return (type(self), hammer_task_spec, StationaryHammerObserver, HammerNailSceneEstimate)

    def stationary_token(self, env):
        return stationary_token(env)


class BucketCollectionAdapter:
    """Contract registered for offline tests; live sensing deliberately unavailable.

    Existing bucket prompted-depth perception is front-only calibrated. A real
    reset receipt/exporter and post-stage segmentation association must be
    validated before enabling this adapter in the simulation collection CLI.
    """
    def __init__(self):
        self.spec = bucket_task_spec()

    def make_reset_observer(self, **kwargs):
        raise NotImplementedError("bucket live observation adapter is not validated; existing geometry supports front only")

    def export_latest_scene(self, observer):
        raise NotImplementedError("bucket live pose provenance exporter is not yet validated")

    def source_objects(self):
        return (type(self), bucket_task_spec)

    def stationary_token(self, env):
        # Equality-only MuJoCo guard; this does not enable bucket live sensing.
        return stationary_token(env)
