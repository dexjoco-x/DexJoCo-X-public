"""Conservative RGB-D lift/co-motion evidence, not force-closure certification.

This data-only checker does not receive an environment, object reset pose,
simulator segmentation, or task-verifier result. Loss of visibility is UNKNOWN,
not a fabricated grasp failure. Large inconsistency is not labelled proven slip.
"""
from dataclasses import asdict, dataclass, field
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .stage_checks import CheckStatus, StageCheckResult, StageSnapshot


@dataclass(frozen=True)
class ObjectSample:
    frame_id: str
    camera: str
    control_step: int
    tcp_pose7: tuple[float, ...]
    centroid_world: tuple[float, ...]
    robust_size_m: float
    median_rgb: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ObjectWindow:
    baseline: ObjectSample | None
    samples: tuple[ObjectSample, ...]
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class VisualStageSnapshot(StageSnapshot):
    object_window: ObjectWindow | None = None
    artifact_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualLiftCoMotionCheck:
    minimum_lift_m: float = .025
    minimum_tcp_motion_m: float = .015
    minimum_object_motion_m: float = .010
    maximum_relative_drift_m: float = .025
    inconsistent_relative_drift_m: float = .060
    maximum_tcp_distance_m: float = .30
    maximum_window_rotation_rad: float = .35
    maximum_size_ratio: float = 2.5
    maximum_color_change: float = .20
    check_id: str = field(default="visual_lift_comotion", init=False)
    scope: str = field(default="object_effect", init=False)

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in {"check_id", "scope"} and (not math.isfinite(value) or value <= 0):
                raise ValueError("visual effect thresholds must be finite and positive")
        if self.inconsistent_relative_drift_m <= self.maximum_relative_drift_m or self.maximum_size_ratio <= 1:
            raise ValueError("invalid visual effect ambiguity band")

    def configuration(self):
        return {**asdict(self), "claim": "visible_part_lifted_and_co_moving_during_sampled_window",
                "not_certified": ["force_closure", "continuous_grasp", "confirmed_slip", "task_success"]}

    def evaluate(self, snapshot):
        def result(status, reason, measurements=()):
            return StageCheckResult(self.check_id, status, self.scope, reason, tuple(measurements))
        window = getattr(snapshot, "object_window", None)
        if window is None or window.unavailable_reason:
            return result(CheckStatus.UNKNOWN, "visual evidence unavailable: " +
                          (window.unavailable_reason if window else "no RGB-D window"))
        if window.baseline is None or len(window.samples) != 3:
            return result(CheckStatus.UNKNOWN, "reset baseline and three temporal samples are required")
        baseline, samples = window.baseline, window.samples
        all_samples = (baseline, *samples)
        if (baseline.control_step != 0 or len({s.frame_id for s in all_samples}) != 4
                or len({s.camera for s in all_samples}) != 1
                or not 0 < samples[0].control_step < samples[1].control_step < samples[2].control_step
                or samples[-1].control_step != snapshot.control_step
                or samples[-1].frame_id != snapshot.observation_id):
            raise ValueError("visual window is stale, mixed-camera, or from a different control time")
        for sample in all_samples:
            if (len(sample.tcp_pose7) != 7 or len(sample.centroid_world) != 3
                    or not np.all(np.isfinite((*sample.tcp_pose7, *sample.centroid_world, sample.robust_size_m)))
                    or sample.robust_size_m <= 0):
                raise ValueError("malformed visual object sample")
        if tuple(samples[-1].tcp_pose7) != snapshot.proprioception[:7]:
            raise ValueError("terminal image is not paired with the current TCP observation")
        sizes = np.array([s.robust_size_m for s in all_samples])
        if sizes.max() / sizes.min() > self.maximum_size_ratio:
            return result(CheckStatus.UNKNOWN, "visible-part size changed too much for temporal association")
        if any(s.median_rgb is None for s in all_samples):
            return result(CheckStatus.UNKNOWN, "appearance evidence required for temporal object association is missing")
        colors = np.array([s.median_rgb for s in all_samples])
        if colors.shape != (4, 3) or not np.all(np.isfinite(colors)) or np.any((colors < 0) | (colors > 1)):
            raise ValueError("appearance must be a normalized finite RGB three-vector")
        if np.max(np.linalg.norm(colors - colors[0], axis=1)) > self.maximum_color_change:
            return result(CheckStatus.UNKNOWN, "appearance changed too much to associate the same visible object part")
        tcp = np.array([s.tcp_pose7 for s in samples])
        objects = np.array([s.centroid_world for s in samples])
        rotations = Rotation.from_quat(tcp[:, 3:], scalar_first=True)
        angular_change = float(np.max((rotations[0].inv() * rotations).magnitude()))
        if angular_change > self.maximum_window_rotation_rad:
            return result(CheckStatus.UNKNOWN, "rotation exceeds the visible-centroid association envelope")
        relative = rotations.inv().apply(objects - tcp[:, :3])
        drift = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
        tcp_motion = float(np.linalg.norm(tcp[-1, :3] - tcp[0, :3]))
        object_motion = float(np.linalg.norm(objects[-1] - objects[0]))
        lift = float(objects[-1, 2] - baseline.centroid_world[2])
        distance = float(np.max(np.linalg.norm(relative, axis=1)))
        measurements = (("lift_from_reset_m", lift), ("tcp_motion_m", tcp_motion),
                        ("object_motion_m", object_motion), ("relative_drift_m", drift),
                        ("maximum_tcp_distance_m", distance), ("window_rotation_rad", angular_change))
        if drift >= self.inconsistent_relative_drift_m or distance > self.maximum_tcp_distance_m:
            return result(CheckStatus.FAIL, "observed object/hand motion is inconsistent; this does not prove slip", measurements)
        if tcp_motion < self.minimum_tcp_motion_m:
            return result(CheckStatus.UNKNOWN, "hand motion is too small to test co-motion", measurements)
        if lift < self.minimum_lift_m or object_motion < self.minimum_object_motion_m:
            return result(CheckStatus.UNKNOWN, "visible lift or object motion is insufficient to certify co-motion", measurements)
        if drift > self.maximum_relative_drift_m:
            return result(CheckStatus.UNKNOWN, "relative drift lies in the visual ambiguity band", measurements)
        return result(CheckStatus.PASS, "visible part lifted and co-moved in the sampled window; grasp security unverified", measurements)
