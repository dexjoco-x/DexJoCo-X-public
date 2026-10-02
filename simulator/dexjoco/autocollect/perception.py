"""Perception backends for no-VLA pick-bucket automatic collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.data.depth_capture import (
    collect_depth_frames,
    collect_segmentation_frames,
)

from .core import _json_safe
from .recovery import PerceptionEvidenceError
from .retarget import BUCKET_SLICE, FOOD_SLICE


@dataclass(frozen=True)
class SceneEstimate:
    """Object poses produced by one perception invocation."""

    bucket_pose7: np.ndarray
    food_pose7: np.ndarray
    confidence: float
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, pose in (
            ("bucket_pose7", self.bucket_pose7),
            ("food_pose7", self.food_pose7),
        ):
            array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if array.shape != (7,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite 7-vector")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_pose7": _json_safe(self.bucket_pose7),
            "food_pose7": _json_safe(self.food_pose7),
            "confidence": self.confidence,
            "backend": self.backend,
            "diagnostics": _json_safe(self.diagnostics),
        }


class PickBucketPerception(Protocol):
    name: str

    def estimate(self, env: Any, observation: dict[str, Any]) -> SceneEstimate:
        """Estimate the initial bucket and payload poses."""


@dataclass(frozen=True)
class PromptedMaskSet:
    """Named instance masks returned by a detector-plus-segmenter backend."""

    masks: Mapping[str, np.ndarray]
    scores: Mapping[str, float]
    backend: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("mask backend name must not be empty")
        if not self.masks:
            raise ValueError("a prompted mask result must contain at least one mask")
        shapes: set[tuple[int, int]] = set()
        for label, value in self.masks.items():
            if not str(label).strip():
                raise ValueError("mask labels must not be empty")
            mask = np.asarray(value)
            if mask.ndim != 2:
                raise ValueError(f"mask {label!r} must be HxW, got {mask.shape}")
            if mask.dtype != np.bool_:
                raise TypeError(f"mask {label!r} must have boolean dtype")
            shapes.add(mask.shape)
            if label not in self.scores:
                raise ValueError(f"mask {label!r} has no confidence score")
            score = float(self.scores[label])
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"mask {label!r} score must be within [0, 1]")
        if len(shapes) != 1:
            raise ValueError("all prompted masks must share one image shape")

    @property
    def image_shape(self) -> tuple[int, int]:
        first = next(iter(self.masks.values()))
        return tuple(np.asarray(first).shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "image_shape": list(self.image_shape),
            "labels": sorted(self.masks),
            "pixel_counts": {
                label: int(np.count_nonzero(mask))
                for label, mask in self.masks.items()
            },
            "scores": {label: float(score) for label, score in self.scores.items()},
            "diagnostics": _json_safe(self.diagnostics),
        }


class PromptedMaskProvider(Protocol):
    """Pluggable 2-D instance segmentation boundary.

    A production implementation should normally combine an object detector
    (for example GroundingDINO) with box-prompted SAM.  SAM alone segments a
    prompted region but does not assign the semantic labels used here.
    """

    name: str

    def predict(
        self,
        image_rgb: np.ndarray,
        object_queries: Mapping[str, str],
    ) -> PromptedMaskSet:
        """Return one named mask for every requested object query."""


class CallablePromptedMaskProvider:
    """Adapter that turns an external detector/SAM callable into a provider."""

    def __init__(
        self,
        *,
        name: str,
        predict_fn: Callable[
            [np.ndarray, Mapping[str, str]],
            PromptedMaskSet,
        ],
    ) -> None:
        self.name = str(name).strip()
        if not self.name:
            raise ValueError("mask provider name must not be empty")
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable")
        self._predict_fn = predict_fn

    def predict(
        self,
        image_rgb: np.ndarray,
        object_queries: Mapping[str, str],
    ) -> PromptedMaskSet:
        result = self._predict_fn(image_rgb, object_queries)
        if not isinstance(result, PromptedMaskSet):
            raise TypeError("mask provider must return PromptedMaskSet")
        return result


def perception_configuration(perception: Any) -> dict[str, Any]:
    """Return auditable non-weight configuration for a perception backend."""

    configuration: dict[str, Any] = {
        "name": str(perception.name),
        "requires_rgb": bool(getattr(perception, "requires_rgb", False)),
    }
    for attribute in (
        "camera_key",
        "minimum_bucket_points",
        "minimum_food_points",
        "minimum_table_points",
        "table_clearance_m",
        "minimum_mask_score",
        "minimum_handle_points",
        "minimum_head_points",
        "minimum_nail_points",
        "maximum_part_overlap",
        "crop_to_workspace",
        "privileged_calibration_only",
    ):
        if hasattr(perception, attribute):
            configuration[attribute] = getattr(perception, attribute)
    provider = getattr(perception, "mask_provider", None)
    if provider is not None:
        configuration["mask_provider"] = str(provider.name)
        provider_configuration = getattr(provider, "configuration", None)
        if callable(provider_configuration):
            configuration["mask_provider_configuration"] = provider_configuration()
    queries = getattr(perception, "object_queries", None)
    if queries is not None:
        configuration["object_queries"] = dict(queries)
    return _json_safe(configuration)


def perception_source_objects(perception: Any) -> tuple[Any, ...]:
    """Return implementation types that must enter the policy source hash."""

    objects: list[Any] = [type(perception)]
    for attribute in ("mask_provider", "detector", "segmenter"):
        component = getattr(perception, attribute, None)
        if component is not None and type(component) not in objects:
            objects.append(type(component))
    provider = getattr(perception, "mask_provider", None)
    predict_fn = getattr(provider, "_predict_fn", None)
    if callable(predict_fn) and predict_fn not in objects:
        # A CallablePromptedMaskProvider is only a thin wrapper.  Hash the
        # injected detector/SAM implementation as well, otherwise two visual
        # policies with different inference code would share one identity.
        objects.append(predict_fn)
    return tuple(objects)


class OracleStatePerception:
    """State-vector baseline retained only for controlled comparisons."""

    name = "oracle_state"
    requires_rgb = False

    def estimate(self, env: Any, observation: dict[str, Any]) -> SceneEstimate:
        del env
        state = np.asarray(observation["state"], dtype=np.float64).reshape(-1)
        return SceneEstimate(
            bucket_pose7=state[BUCKET_SLICE].copy(),
            food_pose7=state[FOOD_SLICE].copy(),
            confidence=1.0,
            backend=self.name,
            diagnostics={"uses_simulator_object_pose": True},
        )


def _descendant_geom_ids(model: mujoco.MjModel, root_body_id: int) -> np.ndarray:
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id and body_id != root_body_id:
            body_id = int(model.body_parentid[body_id])
        if body_id == root_body_id:
            geom_ids.append(geom_id)
    if not geom_ids:
        raise ValueError(f"body id {root_body_id} has no renderable geoms")
    return np.asarray(geom_ids, dtype=np.int32)


def _segmented_depth_to_world_points(
    *,
    depth: np.ndarray,
    segmentation: np.ndarray,
    geom_ids: np.ndarray,
    camera_fovy_deg: float,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    workspace_lower: np.ndarray,
    workspace_upper: np.ndarray,
) -> np.ndarray:
    """Project selected MuJoCo geom pixels into world XYZ."""

    depth = np.asarray(depth, dtype=np.float32)
    segmentation = np.asarray(segmentation)
    if segmentation.shape != (*depth.shape, 2):
        raise ValueError(
            f"segmentation shape {segmentation.shape} does not match depth {depth.shape}"
        )
    mask = (
        (segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM))
        & np.isin(segmentation[..., 1], geom_ids)
        & np.isfinite(depth)
        & (depth > 0.0)
    )
    return _masked_depth_to_world_points(
        depth=depth,
        mask=mask,
        camera_fovy_deg=camera_fovy_deg,
        camera_position_world=camera_position_world,
        camera_rotation_world=camera_rotation_world,
        workspace_lower=workspace_lower,
        workspace_upper=workspace_upper,
    )


def _masked_depth_to_world_points(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    camera_fovy_deg: float,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    workspace_lower: np.ndarray,
    workspace_upper: np.ndarray,
) -> np.ndarray:
    """Project a sensor-space mask without assigning any semantic label."""

    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth.shape}")
    if mask.shape != depth.shape:
        raise ValueError(f"mask shape {mask.shape} does not match depth {depth.shape}")
    mask = mask & np.isfinite(depth) & (depth > 0.0)
    row, column = np.nonzero(mask)
    if len(row) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    height, width = depth.shape
    z = depth[row, column].astype(np.float64)
    focal = 0.5 * height / np.tan(0.5 * np.deg2rad(camera_fovy_deg))
    x = (column.astype(np.float64) - (width - 1) * 0.5) * z / focal
    y = (row.astype(np.float64) - (height - 1) * 0.5) * z / focal
    # OpenCV pixels use +Y down and +Z forward. MuJoCo camera coordinates use
    # +Y up and +Z backward.
    camera_xyz = np.column_stack([x, -y, -z])
    camera_position_world = np.asarray(camera_position_world, dtype=np.float64)
    camera_rotation_world = np.asarray(
        camera_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    world_xyz = camera_xyz @ camera_rotation_world.T + camera_position_world
    keep = np.all(world_xyz >= workspace_lower, axis=1) & np.all(
        world_xyz <= workspace_upper, axis=1
    )
    return world_xyz[keep]


def _depth_to_world_points(
    *,
    depth: np.ndarray,
    camera_fovy_deg: float,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    workspace_lower: np.ndarray,
    workspace_upper: np.ndarray,
) -> np.ndarray:
    """Project every valid depth pixel inside a calibrated world workspace."""

    depth = np.asarray(depth, dtype=np.float32)
    return _masked_depth_to_world_points(
        depth=depth,
        mask=np.isfinite(depth) & (depth > 0.0),
        camera_fovy_deg=camera_fovy_deg,
        camera_position_world=camera_position_world,
        camera_rotation_world=camera_rotation_world,
        workspace_lower=workspace_lower,
        workspace_upper=workspace_upper,
    )


def _crop_world_points(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64).reshape(3)
    upper = np.asarray(upper, dtype=np.float64).reshape(3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if np.any(lower >= upper):
        raise ValueError("crop lower bounds must be smaller than upper bounds")
    keep = np.all(points >= lower, axis=1) & np.all(points <= upper, axis=1)
    return points[keep]


def _estimate_table_height(
    points: np.ndarray,
    *,
    minimum_points: int = 10_000,
    bin_width_m: float = 0.0005,
) -> tuple[float, dict[str, Any]]:
    """Find the dominant horizontal table surface from unlabelled depth.

    The XY and Z limits are task calibration, analogous to a real workcell's
    table region.  The returned height comes only from measured depth points.
    """

    if minimum_points <= 0:
        raise ValueError("minimum_points must be positive")
    if bin_width_m <= 0.0:
        raise ValueError("bin_width_m must be positive")
    table_region = _crop_world_points(
        points,
        lower=np.asarray([-0.38, -0.70, 0.88]),
        upper=np.asarray([0.38, 0.70, 0.98]),
    )
    if len(table_region) < minimum_points:
        raise PerceptionEvidenceError(
            f"table region has only {len(table_region)} depth points; "
            f"need {minimum_points}"
        )
    edges = np.arange(0.88, 0.98 + bin_width_m, bin_width_m)
    counts, edges = np.histogram(table_region[:, 2], bins=edges)
    peak_index = int(np.argmax(counts))
    peak_lower = float(edges[peak_index])
    peak_upper = float(edges[peak_index + 1])
    peak_points = table_region[
        (table_region[:, 2] >= peak_lower)
        & (table_region[:, 2] < peak_upper)
    ]
    if len(peak_points) == 0:
        raise PerceptionEvidenceError("dominant table-depth bin is unexpectedly empty")
    height = float(np.median(peak_points[:, 2]))
    return height, {
        "region_point_count": len(table_region),
        "peak_point_count": len(peak_points),
        "histogram_bin_width_m": float(bin_width_m),
        "peak_interval_m": [peak_lower, peak_upper],
    }


def _minimum_area_xy_box(
    points: np.ndarray,
    *,
    nominal_axis_angle_rad: float,
    search_half_range_deg: float = 15.0,
    trim_quantile: float = 0.005,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a robust 2-D box while restricting yaw to the task's reset range."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
        raise ValueError("at least 32 finite XYZ points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("point cloud contains NaN or Inf")
    if not 0.0 <= trim_quantile < 0.5:
        raise ValueError("trim_quantile must be within [0, 0.5)")

    angles = nominal_axis_angle_rad + np.deg2rad(
        np.linspace(-search_half_range_deg, search_half_range_deg, 301)
    )
    xy = points[:, :2]
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for angle in angles:
        first_axis = np.asarray([np.cos(angle), np.sin(angle)])
        second_axis = np.asarray([-np.sin(angle), np.cos(angle)])
        coordinates = np.column_stack([xy @ first_axis, xy @ second_axis])
        lower, upper = np.quantile(
            coordinates,
            [trim_quantile, 1.0 - trim_quantile],
            axis=0,
        )
        area = float(np.prod(upper - lower))
        if best is None or area < best[0]:
            best = (area, float(angle), lower, upper)
    assert best is not None
    _area, angle, lower, upper = best
    first_axis = np.asarray([np.cos(angle), np.sin(angle)])
    second_axis = np.asarray([-np.sin(angle), np.cos(angle)])
    midpoint = 0.5 * (lower + upper)
    center_xy = midpoint[0] * first_axis + midpoint[1] * second_axis
    return angle, center_xy, upper - lower


def _estimate_bucket_pose(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    angle, center_xy, extents_xy = _minimum_area_xy_box(
        points,
        nominal_axis_angle_rad=np.pi,
    )
    lower_z, upper_z = np.quantile(points[:, 2], [0.005, 0.995])
    geometric_center = np.asarray(
        [center_xy[0], center_xy[1], 0.5 * (lower_z + upper_z)]
    )
    local_x = np.asarray([np.cos(angle), np.sin(angle), 0.0])
    local_y = np.asarray([0.0, 0.0, 1.0])
    local_z = np.cross(local_x, local_y)
    rotation = np.column_stack([local_x, local_y, local_z])
    # A fixed camera/mesh calibration converts the robust visible-surface box
    # centre into the bucket body origin. It was fitted once across multiple
    # reset seeds; no per-episode simulator pose enters this computation.
    geometric_center_in_body = np.asarray([-0.01133, 0.01026, -0.00430])
    body_origin = geometric_center - rotation @ geometric_center_in_body
    quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    return np.concatenate([body_origin, quaternion]), {
        "xy_extents_m": extents_xy,
        "z_extent_m": float(upper_z - lower_z),
        "estimated_axis_angle_deg": float(np.degrees(angle)),
        "body_center_calibration_m": geometric_center_in_body,
    }


def _estimate_food_pose(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    angle, center_xy, extents_xy = _minimum_area_xy_box(
        points,
        nominal_axis_angle_rad=0.0,
        trim_quantile=0.01,
    )
    lower_z, upper_z = np.quantile(points[:, 2], [0.01, 0.99])
    geometric_center = np.asarray(
        [center_xy[0], center_xy[1], 0.5 * (lower_z + upper_z)]
    )
    rotation = Rotation.from_euler("z", angle).as_matrix()
    # Union of the two collision boxes is centred slightly away from the body
    # origin. This is a fixed CAD offset, not a per-scene simulator pose.
    geometric_center_in_body = np.asarray([-0.00075, 0.0, 0.0014])
    body_origin = geometric_center - rotation @ geometric_center_in_body
    quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    return np.concatenate([body_origin, quaternion]), {
        "xy_extents_m": extents_xy,
        "z_extent_m": float(upper_z - lower_z),
        "estimated_axis_angle_deg": float(np.degrees(angle)),
    }


class DepthGeometryBucketPerception:
    """Estimate the scene from ordinary depth and calibrated workcell ROIs.

    This backend does not render or consume MuJoCo object IDs.  It first finds
    the table as the dominant horizontal depth surface, removes that surface,
    then separates the bucket and payload using their disjoint initial
    workspaces.  Those workspace bounds are an explicit task prior and must be
    recalibrated if the physical workcell or reset distribution changes.
    """

    name = "depth_geometry_roi_obb_v1"
    requires_rgb = False

    def __init__(
        self,
        *,
        camera_key: str = "front",
        minimum_bucket_points: int = 1000,
        minimum_food_points: int = 200,
        minimum_table_points: int = 10_000,
        table_clearance_m: float = 0.001,
    ) -> None:
        self.camera_key = camera_key
        self.minimum_bucket_points = int(minimum_bucket_points)
        self.minimum_food_points = int(minimum_food_points)
        self.minimum_table_points = int(minimum_table_points)
        self.table_clearance_m = float(table_clearance_m)
        if self.camera_key != "front":
            raise ValueError(
                "the calibrated geometry estimator currently supports front only"
            )
        if min(
            self.minimum_bucket_points,
            self.minimum_food_points,
            self.minimum_table_points,
        ) <= 0:
            raise ValueError("minimum point counts must be positive")
        if self.table_clearance_m <= 0.0:
            raise ValueError("table_clearance_m must be positive")

    def estimate(self, env: Any, observation: dict[str, Any]) -> SceneEstimate:
        del observation
        raw = env.unwrapped
        depths = collect_depth_frames(env, [self.camera_key])
        if self.camera_key not in depths:
            raise RuntimeError(f"depth is unavailable for {self.camera_key!r}")
        camera_id = int(getattr(raw, "_front_camera_id"))
        camera_position = np.asarray(raw.data.cam_xpos[camera_id]).copy()
        camera_rotation = np.asarray(raw.data.cam_xmat[camera_id]).reshape(3, 3)
        fovy = float(raw.model.cam_fovy[camera_id])
        points = _depth_to_world_points(
            depth=depths[self.camera_key],
            camera_fovy_deg=fovy,
            camera_position_world=camera_position,
            camera_rotation_world=camera_rotation,
            workspace_lower=np.asarray([-0.50, -0.75, 0.86]),
            workspace_upper=np.asarray([0.45, 0.75, 1.30]),
        )
        table_height, table_diagnostics = _estimate_table_height(
            points,
            minimum_points=self.minimum_table_points,
        )
        object_floor = table_height + self.table_clearance_m
        bucket_bounds = (
            np.asarray([-0.43, -0.43, object_floor]),
            np.asarray([0.08, -0.04, table_height + 0.29]),
        )
        food_bounds = (
            np.asarray([-0.43, 0.08, object_floor]),
            np.asarray([-0.23, 0.28, table_height + 0.14]),
        )
        bucket_points = _crop_world_points(points, *bucket_bounds)
        food_points = _crop_world_points(points, *food_bounds)
        if len(bucket_points) < self.minimum_bucket_points:
            raise RuntimeError(
                f"bucket workspace has only {len(bucket_points)} valid depth "
                f"points; need {self.minimum_bucket_points}"
            )
        if len(food_points) < self.minimum_food_points:
            raise RuntimeError(
                f"payload workspace has only {len(food_points)} valid depth "
                f"points; need {self.minimum_food_points}"
            )

        bucket_pose, bucket_diagnostics = _estimate_bucket_pose(bucket_points)
        food_pose, food_diagnostics = _estimate_food_pose(food_points)
        bucket_point_score = min(1.0, len(bucket_points) / 10_000.0)
        food_point_score = min(1.0, len(food_points) / 1_000.0)
        confidence = float(min(bucket_point_score, food_point_score))
        return SceneEstimate(
            bucket_pose7=bucket_pose,
            food_pose7=food_pose,
            confidence=confidence,
            backend=self.name,
            diagnostics={
                "camera_key": self.camera_key,
                "sensor_inputs": ["depth", "camera_calibration"],
                "projected_point_count": len(points),
                "estimated_table_height_m": table_height,
                "table_clearance_m": self.table_clearance_m,
                "table": table_diagnostics,
                "bucket_point_count": len(bucket_points),
                "food_point_count": len(food_points),
                "bucket_workspace_bounds_m": bucket_bounds,
                "food_workspace_bounds_m": food_bounds,
                "bucket": bucket_diagnostics,
                "food": food_diagnostics,
                "task_prior": "disjoint_initial_workspace_rois",
                "uses_simulator_object_pose": False,
                "uses_segmentation_labels": False,
                "uses_rgb": False,
            },
        )


class DepthSegmentationBucketPerception:
    """Estimate bucket and payload pose from one aligned depth/segment frame."""

    name = "depth_segmentation_obb_v2"
    requires_rgb = False

    def __init__(
        self,
        *,
        camera_key: str = "front",
        minimum_bucket_points: int = 1000,
        minimum_food_points: int = 200,
    ) -> None:
        self.camera_key = camera_key
        self.minimum_bucket_points = int(minimum_bucket_points)
        self.minimum_food_points = int(minimum_food_points)
        if self.camera_key != "front":
            raise ValueError(
                "the calibrated v2 estimator currently supports front only"
            )
        if self.minimum_bucket_points <= 0 or self.minimum_food_points <= 0:
            raise ValueError("minimum point counts must be positive")

    def estimate(self, env: Any, observation: dict[str, Any]) -> SceneEstimate:
        del observation
        raw = env.unwrapped
        depths = collect_depth_frames(env, [self.camera_key])
        segmentations = collect_segmentation_frames(env, [self.camera_key])
        if self.camera_key not in depths or self.camera_key not in segmentations:
            raise RuntimeError(
                f"depth or segmentation is unavailable for {self.camera_key!r}"
            )
        camera_id = int(getattr(raw, "_front_camera_id"))
        camera_position = np.asarray(raw.data.cam_xpos[camera_id]).copy()
        camera_rotation = np.asarray(raw.data.cam_xmat[camera_id]).reshape(3, 3)
        fovy = float(raw.model.cam_fovy[camera_id])
        workspace_lower = np.asarray([-0.58, -0.52, 0.87])
        workspace_upper = np.asarray([0.18, 0.42, 1.38])

        bucket_body_id = int(raw.model.body("bucket").id)
        food_body_id = int(raw.model.body("boxed_food_0").id)
        bucket_points = _segmented_depth_to_world_points(
            depth=depths[self.camera_key],
            segmentation=segmentations[self.camera_key],
            geom_ids=_descendant_geom_ids(raw.model, bucket_body_id),
            camera_fovy_deg=fovy,
            camera_position_world=camera_position,
            camera_rotation_world=camera_rotation,
            workspace_lower=workspace_lower,
            workspace_upper=workspace_upper,
        )
        food_points = _segmented_depth_to_world_points(
            depth=depths[self.camera_key],
            segmentation=segmentations[self.camera_key],
            geom_ids=_descendant_geom_ids(raw.model, food_body_id),
            camera_fovy_deg=fovy,
            camera_position_world=camera_position,
            camera_rotation_world=camera_rotation,
            workspace_lower=workspace_lower,
            workspace_upper=workspace_upper,
        )
        if len(bucket_points) < self.minimum_bucket_points:
            raise RuntimeError(
                f"bucket has only {len(bucket_points)} valid depth points; "
                f"need {self.minimum_bucket_points}"
            )
        if len(food_points) < self.minimum_food_points:
            raise RuntimeError(
                f"payload has only {len(food_points)} valid depth points; "
                f"need {self.minimum_food_points}"
            )

        bucket_pose, bucket_diagnostics = _estimate_bucket_pose(bucket_points)
        food_pose, food_diagnostics = _estimate_food_pose(food_points)
        bucket_point_score = min(1.0, len(bucket_points) / 10_000.0)
        food_point_score = min(1.0, len(food_points) / 1_000.0)
        confidence = float(min(bucket_point_score, food_point_score))
        return SceneEstimate(
            bucket_pose7=bucket_pose,
            food_pose7=food_pose,
            confidence=confidence,
            backend=self.name,
            diagnostics={
                "camera_key": self.camera_key,
                "bucket_point_count": len(bucket_points),
                "food_point_count": len(food_points),
                "bucket": bucket_diagnostics,
                "food": food_diagnostics,
                "uses_simulator_object_pose": False,
                "uses_segmentation_labels": True,
            },
        )


class PromptedMaskDepthBucketPerception:
    """Fuse detector/SAM instance masks with aligned depth for 6-D poses.

    The injected provider owns RGB recognition and 2-D segmentation.  This
    class owns geometric projection, minimum-evidence checks, and object-pose
    fitting.  It never reads MuJoCo object IDs or object poses.
    """

    requires_rgb = True

    def __init__(
        self,
        *,
        mask_provider: PromptedMaskProvider,
        camera_key: str = "front",
        object_queries: Mapping[str, str] | None = None,
        minimum_bucket_points: int = 1000,
        minimum_food_points: int = 200,
        minimum_mask_score: float = 0.5,
    ) -> None:
        self.mask_provider = mask_provider
        self.camera_key = str(camera_key)
        self.object_queries = dict(
            object_queries
            or {
                "bucket": "bucket with a handle",
                "payload": "small boxed food package",
            }
        )
        self.minimum_bucket_points = int(minimum_bucket_points)
        self.minimum_food_points = int(minimum_food_points)
        self.minimum_mask_score = float(minimum_mask_score)
        self.name = f"prompted_mask_depth_obb_v1:{mask_provider.name}"
        if self.camera_key != "front":
            raise ValueError(
                "the calibrated prompted-mask estimator currently supports front only"
            )
        if set(self.object_queries) != {"bucket", "payload"}:
            raise ValueError(
                "object_queries must contain exactly 'bucket' and 'payload'"
            )
        if any(not query.strip() for query in self.object_queries.values()):
            raise ValueError("object queries must not be empty")
        if self.minimum_bucket_points <= 0 or self.minimum_food_points <= 0:
            raise ValueError("minimum point counts must be positive")
        if (
            not np.isfinite(self.minimum_mask_score)
            or not 0.0 <= self.minimum_mask_score <= 1.0
        ):
            raise ValueError("minimum_mask_score must be within [0, 1]")

    def estimate(self, env: Any, observation: dict[str, Any]) -> SceneEstimate:
        if self.camera_key not in observation:
            raise RuntimeError(
                f"RGB observation is unavailable for {self.camera_key!r}"
            )
        image_rgb = np.asarray(observation[self.camera_key])
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"RGB image must have shape HxWx3, got {image_rgb.shape}"
            )
        if image_rgb.dtype != np.uint8:
            raise ValueError("RGB image supplied to the mask provider must be uint8")
        mask_set = self.mask_provider.predict(image_rgb, self.object_queries)
        missing = sorted(set(self.object_queries) - set(mask_set.masks))
        unexpected = sorted(set(mask_set.masks) - set(self.object_queries))
        if missing or unexpected:
            raise ValueError(
                "mask provider labels do not match requested objects; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if mask_set.image_shape != tuple(image_rgb.shape[:2]):
            raise ValueError(
                f"mask image shape {mask_set.image_shape} does not match RGB "
                f"shape {image_rgb.shape[:2]}"
            )
        low_scores = {
            label: float(mask_set.scores[label])
            for label in self.object_queries
            if float(mask_set.scores[label]) < self.minimum_mask_score
        }
        if low_scores:
            raise RuntimeError(
                f"mask confidence is below {self.minimum_mask_score}: {low_scores}"
            )

        depths = collect_depth_frames(env, [self.camera_key])
        if self.camera_key not in depths:
            raise RuntimeError(f"depth is unavailable for {self.camera_key!r}")
        depth = depths[self.camera_key]
        if depth.shape != mask_set.image_shape:
            raise ValueError(
                f"depth shape {depth.shape} does not match masks {mask_set.image_shape}"
            )
        raw = env.unwrapped
        camera_id = int(getattr(raw, "_front_camera_id"))
        camera_position = np.asarray(raw.data.cam_xpos[camera_id]).copy()
        camera_rotation = np.asarray(raw.data.cam_xmat[camera_id]).reshape(3, 3)
        fovy = float(raw.model.cam_fovy[camera_id])
        projection = {
            "depth": depth,
            "camera_fovy_deg": fovy,
            "camera_position_world": camera_position,
            "camera_rotation_world": camera_rotation,
            "workspace_lower": np.asarray([-0.58, -0.52, 0.87]),
            "workspace_upper": np.asarray([0.18, 0.42, 1.38]),
        }
        bucket_points = _masked_depth_to_world_points(
            mask=np.asarray(mask_set.masks["bucket"], dtype=bool),
            **projection,
        )
        food_points = _masked_depth_to_world_points(
            mask=np.asarray(mask_set.masks["payload"], dtype=bool),
            **projection,
        )
        if len(bucket_points) < self.minimum_bucket_points:
            raise RuntimeError(
                f"bucket mask has only {len(bucket_points)} valid depth points; "
                f"need {self.minimum_bucket_points}"
            )
        if len(food_points) < self.minimum_food_points:
            raise RuntimeError(
                f"payload mask has only {len(food_points)} valid depth points; "
                f"need {self.minimum_food_points}"
            )

        bucket_pose, bucket_diagnostics = _estimate_bucket_pose(bucket_points)
        food_pose, food_diagnostics = _estimate_food_pose(food_points)
        point_score = min(
            1.0,
            len(bucket_points) / 10_000.0,
            len(food_points) / 1_000.0,
        )
        confidence = float(
            min(
                point_score,
                *(float(mask_set.scores[label]) for label in self.object_queries),
            )
        )
        overlap_pixels = int(
            np.count_nonzero(
                np.asarray(mask_set.masks["bucket"], dtype=bool)
                & np.asarray(mask_set.masks["payload"], dtype=bool)
            )
        )
        return SceneEstimate(
            bucket_pose7=bucket_pose,
            food_pose7=food_pose,
            confidence=confidence,
            backend=self.name,
            diagnostics={
                "camera_key": self.camera_key,
                "sensor_inputs": ["rgb", "depth", "camera_calibration"],
                "mask_provider": mask_set.to_dict(),
                "object_queries": self.object_queries,
                "minimum_mask_score": self.minimum_mask_score,
                "bucket_point_count": len(bucket_points),
                "food_point_count": len(food_points),
                "overlap_pixel_count": overlap_pixels,
                "bucket": bucket_diagnostics,
                "food": food_diagnostics,
                "uses_simulator_object_pose": False,
                "uses_simulator_segmentation_labels": False,
                "sam_role": "box_or_point_prompted_pixel_segmentation_only",
                "semantic_recognition_role": "injected_detector_or_mask_provider",
            },
        )
