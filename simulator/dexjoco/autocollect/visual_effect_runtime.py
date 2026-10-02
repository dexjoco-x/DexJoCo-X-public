"""Real camera acquisition and SAM/depth association for stage effect gates.

Only designated frames of existing actions are recorded. Alternate views have
their own temporal history; a refresh renders a NEW endpoint without actuation.
No post-grasp reset-orientation/CAD pose estimator is used here.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from scipy import ndimage

from dexjoco.data.depth_capture import _env_viewer_and_camera_ids, collect_depth_frames
from .core import ExecutionBudgetExceeded, _json_safe
from .hammer_perception import _camera_projection_inputs
from .hammer_recovery import reset_fingerprint
from .perception import _masked_depth_to_world_points
from .recovery import PerceptionEvidenceError
from .stage_checks import StageGate, TcpEndpointCheck
from .visual_effects import ObjectSample, ObjectWindow, VisualLiftCoMotionCheck, VisualStageSnapshot


@dataclass(frozen=True)
class CameraFrame:
    frame_id: str
    camera: str
    control_step: int
    tcp_pose7: tuple[float, ...]
    rgb: np.ndarray
    projection: dict
    search_center_world: tuple[float, ...] | None = None
    table_height_m: float | None = None


class StageBaselineRejected(RuntimeError):
    """Typed pre-motion stop: the selected views lack valid temporal baselines."""

    def __init__(self, report):
        super().__init__("selected stage views do not all have accepted pre-motion baselines")
        self.report = report


def capture_rgbd(context, camera):
    """Atomic, aligned RGB/depth plus allowed camera calibration; no stepping."""
    before = reset_fingerprint(context.env)  # Equality-only audit, not input to a checker.
    before_time = getattr(getattr(context.env.unwrapped, "data", None), "time", None)
    viewer, ids = _env_viewer_and_camera_ids(context.env.unwrapped, [camera])
    if viewer is None or ids is None:
        raise RuntimeError(f"post-stage camera unavailable: {camera}")
    rgb = np.asarray(viewer.render(render_mode="rgb_array", camera_id=int(ids[0]))).copy()
    depths = collect_depth_frames(context.env, [camera])
    if camera not in depths:
        raise RuntimeError(f"post-stage depth unavailable: {camera}")
    depth = depths[camera]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("post-stage RGB/depth are malformed or not aligned")
    projection = _camera_projection_inputs(context.env, camera_key=camera, depth=depth)
    # A fixed broad sensor workspace, not a crop derived from simulator objects.
    projection["workspace_lower"] = np.array([-.65, -.75, .80])
    projection["workspace_upper"] = np.array([.50, .75, 1.85])
    if (reset_fingerprint(context.env) != before
            or getattr(getattr(context.env.unwrapped, "data", None), "time", None) != before_time):
        raise RuntimeError("physics changed during camera-only stage observation")
    return CameraFrame(uuid.uuid4().hex, camera, context.total_steps,
                       tuple(np.asarray(context.observation["state"], dtype=float)[:7]), rgb, projection)


def prepare_search_image(frame, *, radius_m=.24):
    """Depth/proprioception search region, NOT a semantic or success mask.

Reset center comes from accepted initial perception; later centers are TCP
positions. Actual object centroids are always measured from a new SAM mask.
"""
    if frame.search_center_world is None:
        return frame.rgb, (0, 0, frame.rgb.shape[1], frame.rgb.shape[0]), np.ones(frame.rgb.shape[:2], bool)
    if frame.table_height_m is None or not np.isfinite(frame.table_height_m):
        raise ValueError("search crop requires a depth-estimated table height")
    depth = frame.projection["depth"]
    height, width = depth.shape
    row, column = np.indices(depth.shape)
    focal = .5 * height / np.tan(.5 * np.deg2rad(frame.projection["camera_fovy_deg"]))
    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, 0)
    camera = np.stack(((column - (width - 1)*.5)*safe_depth/focal,
                       -(row - (height - 1)*.5)*safe_depth/focal, -safe_depth), axis=-1)
    world = camera @ frame.projection["camera_rotation_world"].T + frame.projection["camera_position_world"]
    region = (valid & (np.linalg.norm(world - np.asarray(frame.search_center_world), axis=-1) <= radius_m)
              & (world[..., 2] > frame.table_height_m + .008))
    y, x = np.nonzero(region)
    if len(x) < 200:
        raise PerceptionEvidenceError("insufficient depth support in the declared visual search region")
    x0, x1 = max(0, int(x.min())-12), min(width, int(x.max())+13)
    y0, y1 = max(0, int(y.min())-12), min(height, int(y.max())+13)
    masked_rgb = np.where(region[..., None], frame.rgb, np.uint8(127))
    return masked_rgb[y0:y1, x0:x1], (x0, y0, x1, y1), region


def object_sample_from_mask(frame, mask, *, minimum_points=200, minimum_valid_fraction=.80, audit=None):
    audit = audit if audit is not None else {}
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != frame.rgb.shape[:2]:
        raise ValueError("object mask must be an aligned boolean image")
    count = int(mask.sum())
    audit.update(mask_pixels=count, mask_fraction=float(count / mask.size))
    if count < minimum_points or count / mask.size > .20:
        raise PerceptionEvidenceError("visible-part mask is too small or covers too much of the image")
    labels, _ = ndimage.label(mask)
    counts = np.bincount(labels.reshape(-1))[1:]
    audit["largest_component_fraction"] = float(counts.max() / count) if len(counts) else 0.
    if not len(counts) or counts.max() / count < .85:
        raise PerceptionEvidenceError("visible-part mask is fragmented")
    mask = labels == (int(np.argmax(counts)) + 1)
    points = _masked_depth_to_world_points(mask=mask, **frame.projection)
    audit.update(valid_world_points=len(points), valid_depth_fraction=float(len(points) / int(mask.sum())))
    if len(points) < minimum_points or len(points) / int(mask.sum()) < minimum_valid_fraction:
        raise PerceptionEvidenceError("insufficient aligned foreground depth inside fixed sensor workspace")
    lower, upper = np.quantile(points, [.05, .95], axis=0)
    size = float(np.linalg.norm(upper - lower))
    audit["robust_size_m"] = size
    if not .01 <= size <= .30:
        raise PerceptionEvidenceError("visible-part depth extent is outside the declared association envelope")
    return ObjectSample(frame.frame_id, frame.camera, frame.control_step, frame.tcp_pose7,
                        tuple(np.median(points, axis=0)), size,
                        tuple(np.median(frame.rgb[mask], axis=0) / 255.))


class VisualEffectRuntime:
    """Fresh instance for each candidate/context, with bounded capture schedule."""
    cameras = ("ego_left", "ego_right")
    available_cameras = ("ego_left", "ego_right", "fixed_front", "top")
    query = "wooden stick"

    def __init__(self, *, context, stage_ends, provider, artifact_root,
                 initial_search_position=None, table_height_m=None, cameras=None):
        self.cameras = self.validate_cameras(cameras)
        if getattr(provider, "privileged_calibration_only", True):
            raise ValueError("visual effects require a nonprivileged RGB mask provider")
        if context.total_steps != 0:
            raise ValueError("visual evidence must be bound before any candidate motion")
        self.context, self.provider = context, provider
        self.initial_search_position = initial_search_position
        self.table_height_m = table_height_m
        if initial_search_position is not None:
            position = np.asarray(initial_search_position)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("initial search position must be a perceived finite three-vector")
            if table_height_m is None or not np.isfinite(table_height_m):
                raise ValueError("table height must be explicitly estimated from depth")
        self.root = Path(artifact_root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.windows = {}
        previous = 0
        for stage, end in stage_ends.items():
            if type(end) is not int or end - previous < 9:
                raise ValueError("visual checkpoints require at least nine control steps per stage")
            self.windows[stage] = (end - 8, end - 4, end)
            previous = end
        self.scheduled_steps = {0, *(step for window in self.windows.values() for step in window)}
        self.frames = {}
        self.samples = {}
        self.frame_artifacts = {}
        self.all_records = []

    @classmethod
    def validate_cameras(cls, cameras=None):
        selected = cls.cameras if cameras is None else tuple(cameras)
        if (len(selected) != 2 or len(set(selected)) != 2
                or any(camera not in cls.available_cameras for camera in selected)):
            raise ValueError("stage views require two distinct stationary cameras from "
                             "ego_left, ego_right, fixed_front, top")
        return selected

    @classmethod
    def configuration(cls, cameras=None):
        return {"backend": "sam_depth_temporal_stage_v1", "query": cls.query,
                "cameras": list(cls.validate_cameras(cameras)), "window_offsets_steps": [-8, -4, 0],
                "minimum_mask_points": 200, "minimum_depth_fraction": .80,
                "maximum_mask_fraction": .20, "minimum_component_fraction": .85,
                "allowed_visible_part_size_m": [.01, .30],
                "search_radius_m": .24, "table_exclusion_height_m": .008,
                "search_center": "perceived_reset_position_at_step_zero_else_TCP",
                "outside_search_region_RGB": 127, "minimum_mask_search_overlap": .90,
                "maximum_reset_association_distance_m": .10,
                "workspace_lower": [-.65, -.75, .80], "workspace_upper": [.50, .75, 1.85],
                "robot_probe_actions_added": 0, "object_pose_or_segmentation_oracle_used": False,
                "effect_check": VisualLiftCoMotionCheck().configuration()}

    def _budget(self):
        import time
        if self.context.deadline_monotonic is not None and time.monotonic() >= self.context.deadline_monotonic:
            raise ExecutionBudgetExceeded("visual stage acquisition exhausted the shared deadline")

    def capture_due(self, context):
        if context is not self.context:
            raise ValueError("visual runtime cannot be reused with another context")
        if context.total_steps not in self.scheduled_steps:
            return
        for camera in self.cameras:
            self._budget()
            key = (camera, context.total_steps)
            if key in self.frames:
                raise ValueError("scheduled visual frame was captured twice")
            self.frames[key] = self._capture(camera)
            self._budget()

    def _capture(self, camera):
        captured = capture_rgbd(self.context, camera)
        if self.initial_search_position is not None:
            center = self.initial_search_position if captured.control_step == 0 else captured.tcp_pose7[:3]
            captured = replace(captured, search_center_world=tuple(center), table_height_m=self.table_height_m)
        # Preserve even unused alternate views. These are sensor archives, not
        # new SAM/LLM observations and never evidence of success by themselves.
        from .hammer_observation import png_bytes
        directory = self.root / "captured_frames" / captured.frame_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "rgb.png").write_bytes(png_bytes(captured.rgb))
        np.savez_compressed(directory / "rgbd.npz", rgb=captured.rgb, **captured.projection)
        metadata = {"frame_id": captured.frame_id, "camera": captured.camera,
                    "control_step": captured.control_step, "tcp_pose7": captured.tcp_pose7,
                    "search_center_world": captured.search_center_world, "table_height_m": captured.table_height_m,
                    "rgb_sha256": hashlib.sha256(captured.rgb.tobytes()).hexdigest(),
                    "role": "sensor_archive_not_an_additional_planner_observation"}
        (directory / "frame.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
        return captured

    def require_baselines(self):
        """Opt-in preflight of both already-captured step-zero frames; no API."""
        if self.context.total_steps != 0 or any((camera, 0) not in self.frames for camera in self.cameras):
            raise ValueError("baseline preflight requires both captured views before motion")
        from .hammer_observation import stationary_token
        before = stationary_token(self.context.env)
        rows = []
        try:
            for camera in self.cameras:
                frame = self.frames[(camera, 0)]
                sample, error = self._sample(frame)
                rows.append({"camera": camera, "frame_id": frame.frame_id,
                             "status": "accepted" if sample is not None else "unknown",
                             "reason": error, "artifact_dir": self.frame_artifacts[frame.frame_id]})
        finally:
            if stationary_token(self.context.env) != before or self.context.total_steps != 0:
                raise RuntimeError("scene changed during pre-motion baseline checking")
        report = {"status": "pass" if all(row["status"] == "accepted" for row in rows) else "unknown",
                  "control_steps": 0, "api_calls_added": 0, "views": rows,
                  "configuration": self.configuration(self.cameras),
                  "report_path": str(self.root / "baseline_preflight.json")}
        (self.root / "baseline_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report["status"] != "pass":
            raise StageBaselineRejected(report)
        return report

    def _sample(self, frame):
        if frame.frame_id in self.samples:
            return self.samples[frame.frame_id]
        self._budget()
        directory = self.root / frame.frame_id
        directory.mkdir(exist_ok=False)
        np.savez_compressed(directory / "rgbd.npz", rgb=frame.rgb, **frame.projection)
        record = {"frame_id": frame.frame_id, "camera": frame.camera, "control_step": frame.control_step,
                  "tcp_pose7": list(frame.tcp_pose7), "query": self.query,
                  "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
                  "depth_sha256": hashlib.sha256(frame.projection["depth"].tobytes()).hexdigest(),
                  "uses_simulator_object_pose": False, "uses_segmentation_labels": False,
                  "quality": {}}
        record.update(search_center_world=frame.search_center_world, table_height_m=frame.table_height_m,
                      search_center_source="initial_perception" if frame.control_step == 0 else "proprioception")
        previous_directory = self.provider.diagnostic_dir
        self.provider.diagnostic_dir = directory
        try:
            clear = getattr(self.provider, "clear_prediction_evidence", None)
            if callable(clear):
                clear()  # A pre-inference failure must not retain another frame's diagnostics.
            image, crop, region = prepare_search_image(frame)
            record["model_input_crop_xyxy"] = crop
            masks = self.provider.predict(image, {"held_part": self.query})
            self._budget()
            if "held_part" not in masks.masks:
                raise ValueError("provider omitted required visible-part mask")
            small_mask = masks.masks["held_part"]
            if small_mask.dtype != np.bool_ or small_mask.shape != image.shape[:2]:
                raise ValueError("SAM mask does not align with the cropped model input")
            full_mask = np.zeros(frame.rgb.shape[:2], bool)
            x0, y0, x1, y1 = crop
            full_mask[y0:y1, x0:x1] = small_mask
            if not full_mask.any() or np.count_nonzero(full_mask & region) / full_mask.sum() < .90:
                raise PerceptionEvidenceError("predicted object mask lies outside its declared search region")
            sample = object_sample_from_mask(frame, full_mask, audit=record["quality"])
            if frame.control_step == 0 and frame.search_center_world is not None:
                association_distance = float(np.linalg.norm(np.asarray(sample.centroid_world) - frame.search_center_world))
                record["quality"]["reset_association_distance_m"] = association_distance
                if association_distance > .10:
                    raise PerceptionEvidenceError("reset mask does not agree with the initially perceived target location")
            record.update(status="accepted", centroid_world=sample.centroid_world,
                          robust_size_m=sample.robust_size_m, median_rgb=sample.median_rgb)
            self.samples[frame.frame_id] = (sample, None)
        except PerceptionEvidenceError as exc:
            record.update(status="unknown", reason=str(exc))
            self.samples[frame.frame_id] = (None, str(exc))
        except Exception as exc:
            record.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.provider.diagnostic_dir = previous_directory
            record["model_diagnostics"] = self.provider.last_diagnostics
            (directory / "evidence.json").write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")
            self.frame_artifacts[frame.frame_id] = str(directory)
            self.all_records.append(record)
        return self.samples[frame.frame_id]

    def _observe(self, snapshot, camera_index, *, fresh_endpoint):
        if snapshot.task_id != self.context.task_id or snapshot.control_step != self.context.total_steps:
            raise ValueError("cannot attach historical images to a different live stage")
        camera = self.cameras[camera_index]
        if self.windows[snapshot.stage_id][-1] != snapshot.control_step:
            raise ValueError("stage endpoint does not match the declared visual schedule")
        if fresh_endpoint:
            self._budget()
            self.frames[(camera, snapshot.control_step)] = self._capture(camera)
        frames = [self.frames[(camera, step)] for step in (0, *self.windows[snapshot.stage_id])]
        # These are stationary calibrated cameras. A changed camera invalidates
        # visible-centroid temporal comparisons even if world projection exists.
        for frame in frames[1:]:
            for key in ("camera_position_world", "camera_rotation_world"):
                if not np.array_equal(frame.projection[key], frames[0].projection[key]):
                    raise ValueError("camera changed within a temporal evidence window")
        samples, errors = [], []
        for frame in frames:
            sample, error = self._sample(frame)
            samples.append(sample)
            if error:
                errors.append(f"step {frame.control_step}: {error}")
        window = ObjectWindow(samples[0], tuple(sample for sample in samples[1:] if sample is not None),
                              "; ".join(errors) if errors else None)
        endpoint = frames[-1]
        refs = tuple(self.frame_artifacts[frame.frame_id] for frame in frames)
        (self.root / "manifest.json").write_text(json.dumps(_json_safe({
            "configuration": self.configuration(self.cameras), "observed_stage": snapshot.stage_id,
            "records": self.all_records}), indent=2), encoding="utf-8")
        return VisualStageSnapshot(snapshot.task_id, snapshot.stage_id, endpoint.frame_id, snapshot.control_step,
                                   snapshot.proprioception, snapshot.reference_tcp_pose7, window, refs)

    def observe(self, snapshot):
        return self._observe(snapshot, 0, fresh_endpoint=False)

    def refresh(self, snapshot):
        # Two fixed views => exactly one alternate-view request is possible.
        if not isinstance(snapshot, VisualStageSnapshot):
            raise ValueError("refresh requires a visual snapshot from this runtime")
        return self._observe(snapshot, 1, fresh_endpoint=True)

    def planning_image(self, snapshot):
        """Current stage RGB only; no reset pose estimator or simulator labels."""
        from .agent import PlanningImage
        from .hammer_observation import png_bytes
        from .planning_snapshot import _image
        if (not isinstance(snapshot, VisualStageSnapshot) or snapshot.task_id != self.context.task_id
                or snapshot.control_step != self.context.total_steps
                or self.windows[snapshot.stage_id][-1] != snapshot.control_step):
            raise ValueError("stage planning image must match the current live endpoint")
        matching = [frame for frame in self.frames.values() if frame.frame_id == snapshot.observation_id
                    and frame.control_step == snapshot.control_step]
        if len(matching) != 1:
            raise ValueError("stage planning image is stale, reused or unavailable")
        frame = matching[0]
        records = [record for record in self.all_records if record["frame_id"] == frame.frame_id]
        if len(records) != 1:
            raise ValueError("stage planning image requires a unique perception audit record")
        diagnostic = records[0].get("model_diagnostics", {})
        if diagnostic.get("artifact_dir"):
            return _image(diagnostic, root=self.root / frame.frame_id,
                          frame_id=frame.frame_id, camera=frame.camera)
        return PlanningImage(frame.frame_id, frame.camera, png_bytes(frame.rgb))

    def make_gate(self):
        return StageGate({stage: (TcpEndpointCheck(), VisualLiftCoMotionCheck()) for stage in self.windows},
                         observe=self.observe, refresh=self.refresh, maximum_reobservations=1)
