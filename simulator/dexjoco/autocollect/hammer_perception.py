"""RGB-D perception contracts for random-reset hammer-nail scenes.

The production boundary accepts semantically named masks from an external
detector plus segmenter (normally a text-prompted detector followed by SAM).
MuJoCo segmentation is kept as an explicitly privileged calibration backend;
neither backend reads the simulator's hammer or nail pose as a fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.data.depth_capture import (
    _env_viewer_and_camera_ids,
    collect_depth_frames,
    collect_segmentation_frames,
)

from .core import _json_safe
from .recovery import PerceptionEvidenceError
from .perception import (
    PromptedMaskProvider,
    PromptedMaskSet,
    _depth_to_world_points,
    _estimate_table_height,
    _masked_depth_to_world_points,
    _segmented_depth_to_world_points,
)


# These are fixed CAD/workcell calibrations, not per-scene simulator values.
# The hammer body origin is 8 cm above the table surface at reset, and the nail
# body origin is 16 cm above it.  The remaining constants come from the CAD
# geometry and a small radial visibility correction, never a per-scene pose.
HAMMER_ORIGIN_ABOVE_TABLE_M = 0.08
NAIL_ORIGIN_ABOVE_TABLE_M = 0.16
HAMMER_HANDLE_HALF_WIDTH_M = 0.0254
HAMMER_HEAD_CENTER_ALONG_HANDLE_M = 0.1575
NAIL_VISIBLE_CENTROID_RADIAL_BIAS_M = 0.006


@dataclass(frozen=True)
class HammerNailSceneEstimate:
    """Perceived reset poses for the two independently randomized objects."""

    hammer_pose7: np.ndarray
    nail_pose7: np.ndarray
    confidence: float
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, pose in (
            ("hammer_pose7", self.hammer_pose7),
            ("nail_pose7", self.nail_pose7),
        ):
            array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if array.shape != (7,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite 7-vector")
            quaternion_norm = float(np.linalg.norm(array[3:7]))
            if not np.isclose(quaternion_norm, 1.0, atol=1e-5):
                raise ValueError(f"{name} quaternion must have unit norm")
        if not str(self.backend).strip():
            raise ValueError("perception backend must not be empty")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hammer_pose7": _json_safe(self.hammer_pose7),
            "nail_pose7": _json_safe(self.nail_pose7),
            "confidence": float(self.confidence),
            "backend": self.backend,
            "diagnostics": _json_safe(self.diagnostics),
        }


class HammerNailPerception(Protocol):
    name: str
    requires_rgb: bool
    privileged_calibration_only: bool

    def estimate(
        self,
        env: Any,
        observation: Mapping[str, Any],
    ) -> HammerNailSceneEstimate:
        """Estimate initial hammer and nail poses without an oracle fallback."""


def _validate_points(name: str, points: np.ndarray, minimum: int = 32) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} points must have shape Nx3")
    if len(array) < minimum:
        raise ValueError(f"{name} needs at least {minimum} finite points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} points contain NaN or Inf")
    return array


def _estimate_hammer_pose_from_parts(
    handle_points: np.ndarray,
    head_points: np.ndarray,
    *,
    table_height_m: float,
    camera_position_world: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the reset hammer frame from separate handle/head point sets.

    The vector from the visible handle centroid to the metal head resolves the
    180-degree ambiguity that a whole-object PCA or bounding box would retain.
    Reset constrains the hammer's local X axis to point vertically downward;
    only its world yaw is randomized.
    """

    handle = _validate_points("hammer handle", handle_points)
    head = _validate_points("hammer head", head_points)
    table_height_m = float(table_height_m)
    if not np.isfinite(table_height_m):
        raise ValueError("table_height_m must be finite")
    camera_position = np.asarray(
        camera_position_world, dtype=np.float64
    ).reshape(-1)
    if camera_position.shape != (3,) or not np.all(np.isfinite(camera_position)):
        raise ValueError("camera_position_world must be a finite 3-vector")

    handle_centroid = np.mean(handle, axis=0)
    head_centroid = np.mean(head, axis=0)
    local_z_xy = head_centroid[:2] - handle_centroid[:2]
    separation_m = float(np.linalg.norm(local_z_xy))
    if separation_m < 0.04:
        raise PerceptionEvidenceError(
            "hammer head and handle centroids are too close to resolve direction"
        )
    local_z_xy /= separation_m
    local_x = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    local_z = np.asarray([local_z_xy[0], local_z_xy[1], 0.0])
    local_y = np.cross(local_z, local_x)
    rotation = np.column_stack([local_x, local_y, local_z])

    # Recover the absolute body coordinate along local Y from the visible
    # camera-facing handle surface.  Along local Z, use the two robust bounds
    # of the metal head, whose CAD centre is 0.1575 m from the body origin.
    # This remains stable when the camera is randomized, unlike a fixed
    # front-camera visible-centroid offset.
    handle_coordinates = handle @ rotation
    head_coordinates = head @ rotation
    camera_direction = camera_position - handle_centroid
    camera_side_y = 1.0 if float(camera_direction @ local_y) >= 0.0 else -1.0
    surface_quantile = 0.99 if camera_side_y > 0.0 else 0.01
    body_coordinate_y = float(
        np.quantile(handle_coordinates[:, 1], surface_quantile)
        - camera_side_y * HAMMER_HANDLE_HALF_WIDTH_M
    )
    head_lower_z, head_upper_z = np.quantile(
        head_coordinates[:, 2], [0.01, 0.99]
    )
    body_coordinate_z = float(
        0.5 * (head_lower_z + head_upper_z)
        - HAMMER_HEAD_CENTER_ALONG_HANDLE_M
    )
    origin = local_y * body_coordinate_y + local_z * body_coordinate_z
    origin[2] = table_height_m + HAMMER_ORIGIN_ABOVE_TABLE_M
    quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    pose = np.concatenate([origin, quaternion])
    return pose, {
        "handle_centroid_world_m": handle_centroid,
        "head_centroid_world_m": head_centroid,
        "head_handle_xy_separation_m": separation_m,
        "handle_axis_yaw_deg": float(
            np.degrees(np.arctan2(local_z[1], local_z[0]))
        ),
        "handle_half_width_cad_m": HAMMER_HANDLE_HALF_WIDTH_M,
        "head_center_along_handle_cad_m": (
            HAMMER_HEAD_CENTER_ALONG_HANDLE_M
        ),
        "camera_side_local_y": camera_side_y,
        "body_coordinate_y_m": body_coordinate_y,
        "body_coordinate_z_m": body_coordinate_z,
        "origin_height_above_table_m": HAMMER_ORIGIN_ABOVE_TABLE_M,
    }


def _estimate_nail_pose(
    nail_points: np.ndarray,
    *,
    table_height_m: float,
    camera_position_world: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the vertical nail pose using its mask and workcell geometry."""

    nail = _validate_points("nail", nail_points)
    table_height_m = float(table_height_m)
    if not np.isfinite(table_height_m):
        raise ValueError("table_height_m must be finite")
    camera_position = np.asarray(
        camera_position_world, dtype=np.float64
    ).reshape(-1)
    if camera_position.shape != (3,) or not np.all(np.isfinite(camera_position)):
        raise ValueError("camera_position_world must be a finite 3-vector")
    visible_centroid = np.mean(nail, axis=0)
    radial_direction_xy = camera_position[:2] - visible_centroid[:2]
    radial_norm = float(np.linalg.norm(radial_direction_xy))
    if radial_norm < 1e-6:
        raise PerceptionEvidenceError("camera XY position is too close to nail centre")
    radial_direction_xy /= radial_norm
    origin = np.asarray(
        [
            visible_centroid[0]
            - NAIL_VISIBLE_CENTROID_RADIAL_BIAS_M * radial_direction_xy[0],
            visible_centroid[1]
            - NAIL_VISIBLE_CENTROID_RADIAL_BIAS_M * radial_direction_xy[1],
            table_height_m + NAIL_ORIGIN_ABOVE_TABLE_M,
        ],
        dtype=np.float64,
    )
    pose = np.concatenate([origin, np.asarray([1.0, 0.0, 0.0, 0.0])])
    return pose, {
        "visible_centroid_world_m": visible_centroid,
        "camera_radial_direction_xy": radial_direction_xy,
        "visible_centroid_radial_bias_calibration_m": (
            NAIL_VISIBLE_CENTROID_RADIAL_BIAS_M
        ),
        "origin_height_above_table_m": NAIL_ORIGIN_ABOVE_TABLE_M,
        "orientation_prior": "vertical_nail_fixed_by_fixture",
    }


def _camera_projection_inputs(
    env: Any,
    *,
    camera_key: str,
    depth: np.ndarray,
) -> dict[str, Any]:
    raw = env.unwrapped
    if camera_key in {"front", "random_camera"}:
        camera_id = int(raw._front_camera_id)
    elif camera_key in {"ego_left", "ego_right", "wrist"}:
        camera_id = int(raw.camera_id[{"ego_left": 1, "ego_right": 2, "wrist": 3}[camera_key]])
    elif camera_key in {"fixed_front", "top"}:
        _, ids = _env_viewer_and_camera_ids(raw, [camera_key])
        if ids is None:
            raise ValueError(f"calibrated hammer camera unavailable: {camera_key!r}")
        camera_id = int(ids[0])
    else:
        raise ValueError(f"unsupported calibrated hammer camera {camera_key!r}")
    camera_body_id = int(raw.model.cam_bodyid[camera_id])
    if camera_body_id == 0:
        # A world-attached camera's model pose is already in world coordinates.
        # Body-attached cameras below require the forward-kinematic data pose.
        camera_position = np.asarray(raw.model.cam_pos[camera_id]).copy()
        camera_rotation = Rotation.from_quat(
            np.asarray(raw.model.cam_quat[camera_id]),
            scalar_first=True,
        ).as_matrix()
    else:
        camera_position = np.asarray(raw.data.cam_xpos[camera_id]).copy()
        camera_rotation = np.asarray(
            raw.data.cam_xmat[camera_id]
        ).reshape(3, 3)
    return {
        "depth": np.asarray(depth, dtype=np.float32),
        "camera_fovy_deg": float(raw.model.cam_fovy[camera_id]),
        "camera_position_world": camera_position,
        "camera_rotation_world": camera_rotation,
        "workspace_lower": np.asarray([-0.60, -0.70, 0.86]),
        "workspace_upper": np.asarray([0.30, 0.50, 1.40]),
    }


def _table_height_from_depth(
    projection: Mapping[str, Any],
    *,
    minimum_table_points: int,
) -> tuple[float, dict[str, Any]]:
    points = _depth_to_world_points(**projection)
    height, diagnostics = _estimate_table_height(
        points,
        minimum_points=minimum_table_points,
    )
    return height, {
        **diagnostics,
        "projected_scene_point_count": len(points),
    }


class DepthSegmentationHammerNailPerception:
    """Privileged MuJoCo-label baseline used only to calibrate RGB-D geometry."""

    name = "hammer_nail_depth_segmentation_calibration_v1"
    requires_rgb = False
    privileged_calibration_only = True

    def __init__(
        self,
        *,
        camera_key: str = "front",
        minimum_handle_points: int = 250,
        minimum_head_points: int = 750,
        minimum_nail_points: int = 200,
        minimum_table_points: int = 10_000,
    ) -> None:
        self.camera_key = str(camera_key)
        self.minimum_handle_points = int(minimum_handle_points)
        self.minimum_head_points = int(minimum_head_points)
        self.minimum_nail_points = int(minimum_nail_points)
        self.minimum_table_points = int(minimum_table_points)
        if self.camera_key not in {"front", "random_camera"}:
            raise ValueError("camera_key must be 'front' or 'random_camera'")
        if min(
            self.minimum_handle_points,
            self.minimum_head_points,
            self.minimum_nail_points,
            self.minimum_table_points,
        ) <= 0:
            raise ValueError("minimum point counts must be positive")

    def estimate(
        self,
        env: Any,
        observation: Mapping[str, Any],
    ) -> HammerNailSceneEstimate:
        del observation
        depths = collect_depth_frames(env, [self.camera_key])
        segmentations = collect_segmentation_frames(env, [self.camera_key])
        if self.camera_key not in depths or self.camera_key not in segmentations:
            raise RuntimeError(
                f"aligned depth/segmentation unavailable for {self.camera_key!r}"
            )
        raw = env.unwrapped
        projection = _camera_projection_inputs(
            env,
            camera_key=self.camera_key,
            depth=depths[self.camera_key],
        )
        segmentation = segmentations[self.camera_key]

        def project_geoms(names: tuple[str, ...]) -> np.ndarray:
            geom_ids = np.asarray(
                [int(raw.model.geom(name).id) for name in names],
                dtype=np.int32,
            )
            return _segmented_depth_to_world_points(
                segmentation=segmentation,
                geom_ids=geom_ids,
                **projection,
            )

        handle_points = project_geoms(("handle",))
        head_points = project_geoms(("head", "neck", "face", "claw"))
        nail_points = project_geoms(("nail_head", "nail_shaft"))
        requirements = {
            "hammer_handle": (len(handle_points), self.minimum_handle_points),
            "hammer_head": (len(head_points), self.minimum_head_points),
            "nail": (len(nail_points), self.minimum_nail_points),
        }
        insufficient = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in requirements.items()
            if actual < required
        }
        if insufficient:
            raise RuntimeError(f"insufficient segmented depth evidence: {insufficient}")

        table_height, table_diagnostics = _table_height_from_depth(
            projection,
            minimum_table_points=self.minimum_table_points,
        )
        hammer_pose, hammer_diagnostics = _estimate_hammer_pose_from_parts(
            handle_points,
            head_points,
            table_height_m=table_height,
            camera_position_world=projection["camera_position_world"],
        )
        nail_pose, nail_diagnostics = _estimate_nail_pose(
            nail_points,
            table_height_m=table_height,
            camera_position_world=projection["camera_position_world"],
        )
        confidence = float(
            min(
                1.0,
                len(handle_points) / 1000.0,
                len(head_points) / 2500.0,
                len(nail_points) / 350.0,
            )
        )
        return HammerNailSceneEstimate(
            hammer_pose7=hammer_pose,
            nail_pose7=nail_pose,
            confidence=confidence,
            backend=self.name,
            diagnostics={
                "camera_key": self.camera_key,
                "sensor_inputs": [
                    "depth",
                    "camera_calibration",
                    "mujoco_geom_segmentation",
                ],
                "point_counts": {
                    "hammer_handle": len(handle_points),
                    "hammer_head": len(head_points),
                    "nail": len(nail_points),
                },
                "estimated_table_height_m": table_height,
                "table": table_diagnostics,
                "hammer": hammer_diagnostics,
                "nail": nail_diagnostics,
                "uses_simulator_object_pose": False,
                "uses_simulator_segmentation_labels": True,
                "privileged_calibration_only": True,
                "eligible_as_nonprivileged_collection": False,
            },
        )


class PromptedMaskDepthHammerNailPerception:
    """Fuse detector/SAM masks with aligned depth into hammer/nail poses.

    The provider must distinguish the wooden handle, metal head, and nail.
    Keeping the two hammer parts separate resolves the direction of the hammer
    axis in this centroid-based estimator (whole-object CAD fitting is a
    different possible estimator).
    """

    requires_rgb = True
    privileged_calibration_only = False

    def __init__(
        self,
        *,
        mask_provider: PromptedMaskProvider,
        camera_key: str = "front",
        object_queries: Mapping[str, str] | None = None,
        minimum_handle_points: int = 250,
        minimum_head_points: int = 750,
        minimum_nail_points: int = 200,
        minimum_table_points: int = 10_000,
        minimum_mask_score: float = 0.5,
        maximum_part_overlap: float = 0.25,
        crop_to_workspace: bool = False,
    ) -> None:
        self.mask_provider = mask_provider
        self.camera_key = str(camera_key)
        self.object_queries = dict(
            object_queries
            or {
                "hammer_handle": "wooden hammer handle",
                "hammer_head": "metal hammer head",
                "nail": "upright metal nail in the wooden board",
            }
        )
        self.minimum_handle_points = int(minimum_handle_points)
        self.minimum_head_points = int(minimum_head_points)
        self.minimum_nail_points = int(minimum_nail_points)
        self.minimum_table_points = int(minimum_table_points)
        self.minimum_mask_score = float(minimum_mask_score)
        self.maximum_part_overlap = float(maximum_part_overlap)
        self.crop_to_workspace = bool(crop_to_workspace)
        self.last_model_input_crop: list[int] | None = None
        if not np.isfinite(self.maximum_part_overlap) or not 0 <= self.maximum_part_overlap < 1:
            raise ValueError("maximum_part_overlap must be in [0, 1)")
        self.name = f"hammer_nail_prompted_mask_depth_v1:{mask_provider.name}"
        expected_labels = {"hammer_handle", "hammer_head", "nail"}
        if set(self.object_queries) != expected_labels:
            raise ValueError(
                "object_queries must contain hammer_handle, hammer_head, and nail"
            )
        if self.camera_key not in {"front", "random_camera", "ego_left", "ego_right", "wrist"}:
            raise ValueError("unsupported calibrated hammer camera")
        if any(not str(query).strip() for query in self.object_queries.values()):
            raise ValueError("object queries must not be empty")
        if min(
            self.minimum_handle_points,
            self.minimum_head_points,
            self.minimum_nail_points,
            self.minimum_table_points,
        ) <= 0:
            raise ValueError("minimum point counts must be positive")
        if (
            not np.isfinite(self.minimum_mask_score)
            or not 0.0 <= self.minimum_mask_score <= 1.0
        ):
            raise ValueError("minimum_mask_score must be within [0, 1]")

    def estimate(
        self,
        env: Any,
        observation: Mapping[str, Any],
    ) -> HammerNailSceneEstimate:
        self.last_model_input_crop = None
        clear_evidence = getattr(self.mask_provider, "clear_prediction_evidence", None)
        if callable(clear_evidence):
            clear_evidence()
        if self.camera_key not in observation:
            raise RuntimeError(f"RGB observation unavailable for {self.camera_key!r}")
        image_rgb = np.asarray(observation[self.camera_key])
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"RGB image must have shape HxWx3, got {image_rgb.shape}")
        if image_rgb.dtype != np.uint8:
            raise ValueError("RGB image supplied to the mask provider must be uint8")

        depths = collect_depth_frames(env, [self.camera_key])
        if self.camera_key not in depths:
            raise RuntimeError(f"depth unavailable for {self.camera_key!r}")
        depth = depths[self.camera_key]
        if tuple(depth.shape) != tuple(image_rgb.shape[:2]):
            raise ValueError("depth shape does not match RGB")
        projection = _camera_projection_inputs(env, camera_key=self.camera_key, depth=depth)
        table_height, table_diagnostics = _table_height_from_depth(
            projection, minimum_table_points=self.minimum_table_points,
        )
        if self.crop_to_workspace:
            crop = _workspace_image_crop(projection, table_height_m=table_height)
            self.last_model_input_crop = list(crop)
            x0, y0, x1, y1 = crop
            cropped = self.mask_provider.predict(image_rgb[y0:y1, x0:x1].copy(), self.object_queries)
            if cropped.image_shape != (y1 - y0, x1 - x0):
                raise ValueError("provider mask shape does not match cropped RGB")
            masks = {}
            for label, mask in cropped.masks.items():
                restored = np.zeros(image_rgb.shape[:2], dtype=bool)
                restored[y0:y1, x0:x1] = mask
                masks[label] = restored
            mask_set = PromptedMaskSet(
                masks=masks, scores=cropped.scores, backend=cropped.backend,
                diagnostics={**cropped.diagnostics, "input_crop_xyxy": list(crop)},
            )
        else:
            mask_set = self.mask_provider.predict(image_rgb, self.object_queries)
        self._validate_mask_set(mask_set, image_rgb.shape[:2])
        low_scores = {
            label: float(mask_set.scores[label])
            for label in self.object_queries
            if float(mask_set.scores[label]) < self.minimum_mask_score
        }
        if low_scores:
            raise PerceptionEvidenceError(
                f"mask confidence is below {self.minimum_mask_score}: {low_scores}"
            )

        if tuple(depth.shape) != mask_set.image_shape:
            raise ValueError(
                f"depth shape {depth.shape} does not match masks {mask_set.image_shape}"
            )
        points = {
            label: _masked_depth_to_world_points(
                mask=np.asarray(mask_set.masks[label], dtype=bool),
                **projection,
            )
            for label in self.object_queries
        }
        requirements = {
            "hammer_handle": self.minimum_handle_points,
            "hammer_head": self.minimum_head_points,
            "nail": self.minimum_nail_points,
        }
        insufficient = {
            label: {"actual": len(points[label]), "required": required}
            for label, required in requirements.items()
            if len(points[label]) < required
        }
        if insufficient:
            raise PerceptionEvidenceError(f"insufficient masked depth evidence: {insufficient}")

        hammer_pose, hammer_diagnostics = _estimate_hammer_pose_from_parts(
            points["hammer_handle"],
            points["hammer_head"],
            table_height_m=table_height,
            camera_position_world=projection["camera_position_world"],
        )
        nail_pose, nail_diagnostics = _estimate_nail_pose(
            points["nail"],
            table_height_m=table_height,
            camera_position_world=projection["camera_position_world"],
        )
        point_score = min(
            1.0,
            len(points["hammer_handle"]) / 1000.0,
            len(points["hammer_head"]) / 2500.0,
            len(points["nail"]) / 350.0,
        )
        confidence = float(
            min(
                point_score,
                *(float(mask_set.scores[label]) for label in self.object_queries),
            )
        )
        overlap = np.zeros(mask_set.image_shape, dtype=np.uint8)
        for mask in mask_set.masks.values():
            overlap += np.asarray(mask, dtype=np.uint8)
        return HammerNailSceneEstimate(
            hammer_pose7=hammer_pose,
            nail_pose7=nail_pose,
            confidence=confidence,
            backend=self.name,
            diagnostics={
                "camera_key": self.camera_key,
                "sensor_inputs": ["rgb", "depth", "camera_calibration"],
                "mask_provider": mask_set.to_dict(),
                "point_counts": {
                    label: len(value) for label, value in points.items()
                },
                "overlap_pixel_count": int(np.count_nonzero(overlap > 1)),
                "estimated_table_height_m": table_height,
                "table": table_diagnostics,
                "hammer": hammer_diagnostics,
                "nail": nail_diagnostics,
                "uses_simulator_object_pose": False,
                "uses_simulator_segmentation_labels": False,
                "privileged_calibration_only": False,
                "eligible_as_nonprivileged_collection": True,
            },
        )

    def _validate_mask_set(
        self,
        mask_set: PromptedMaskSet,
        image_shape: tuple[int, int],
    ) -> None:
        expected = set(self.object_queries)
        actual = set(mask_set.masks)
        if actual != expected:
            raise ValueError(
                "mask labels do not match requested objects; "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        if mask_set.image_shape != tuple(image_shape):
            raise ValueError(
                f"mask image shape {mask_set.image_shape} does not match RGB "
                f"shape {tuple(image_shape)}"
            )
        labels = list(self.object_queries)
        for i, first in enumerate(labels):
            for second in labels[i + 1:]:
                a = np.asarray(mask_set.masks[first], dtype=bool)
                b = np.asarray(mask_set.masks[second], dtype=bool)
                overlap = np.count_nonzero(a & b) / max(1, min(np.count_nonzero(a), np.count_nonzero(b)))
                if overlap > self.maximum_part_overlap:
                    raise PerceptionEvidenceError(
                        f"ambiguous part masks: {first} and {second} overlap={overlap:.3f} "
                        f"exceeds {self.maximum_part_overlap}"
                    )


def _workspace_image_crop(
    projection: Mapping[str, Any], *, table_height_m: float, padding_pixels: int = 12,
) -> tuple[int, int, int, int]:
    """Crop measured above-table surfaces, without object labels or object poses.

    Height interval is a task/workcell prior: objects initially rest within
    24 cm above the table. No scene-specific image coordinates are configured.
    """
    depth = np.asarray(projection["depth"])
    height, width = depth.shape
    row, column = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[row, column].astype(np.float64)
    focal = .5 * height / np.tan(.5 * np.deg2rad(projection["camera_fovy_deg"]))
    camera_xyz = np.column_stack([
        (column - (width - 1) * .5) * z / focal,
        -(row - (height - 1) * .5) * z / focal, -z,
    ])
    world = camera_xyz @ np.asarray(projection["camera_rotation_world"]).T + projection["camera_position_world"]
    keep = (
        np.all(world >= projection["workspace_lower"], axis=1)
        & np.all(world <= projection["workspace_upper"], axis=1)
        & (world[:, 2] > table_height_m + .008)
        & (world[:, 2] < table_height_m + .24)
    )
    row, column = row[keep], column[keep]
    if len(row) < 100:
        raise PerceptionEvidenceError("insufficient above-table depth evidence for model crop")
    return (
        max(0, int(column.min()) - padding_pixels),
        max(0, int(row.min()) - padding_pixels),
        min(width, int(column.max()) + padding_pixels + 1),
        min(height, int(row.max()) + padding_pixels + 1),
    )
