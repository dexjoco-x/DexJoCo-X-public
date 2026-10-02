"""Baseline-only geometric correspondence, separate from semantic identity.

Saved masks and measured RGB-D only. No model, renderer, object-state lookup,
mask winner, fused action pose, actuation permission, or training eligibility.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy import ndimage

from .part_refinement import assess_refinement
from .perception import _masked_depth_to_world_points
from .visual_effect_runtime import prepare_search_image


@dataclass(frozen=True)
class CrossViewPolicy:
    depth_tolerance_m: float = .01
    mask_boundary_tolerance_px: int = 2
    minimum_observable_pixels: int = 50
    minimum_observable_fraction: float = .10
    minimum_mask_agreement_fraction: float = .90
    maximum_free_space_conflict_fraction: float = .05
    maximum_candidates_per_view: int = 30

    def __post_init__(self):
        if not np.isfinite(self.depth_tolerance_m) or not 0 < self.depth_tolerance_m <= .05:
            raise ValueError("invalid metric depth tolerance")
        for key, lower, upper in (("mask_boundary_tolerance_px", 0, 3),
                                  ("minimum_observable_pixels", 1, 10000),
                                  ("maximum_candidates_per_view", 1, 30)):
            value = getattr(self, key)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid bounded cross-view policy")
        for key in ("minimum_observable_fraction", "minimum_mask_agreement_fraction",
                    "maximum_free_space_conflict_fraction"):
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("invalid cross-view fraction")


def validate_projection(frame):
    p = frame.projection
    if frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3:
        raise ValueError("cross-view RGB must be HxWx3 uint8")
    depth = np.asarray(p["depth"])
    if depth.shape != frame.rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth must be an aligned floating-point metric image")
    rotation, position = np.asarray(p["camera_rotation_world"]), np.asarray(p["camera_position_world"])
    lower, upper = np.asarray(p["workspace_lower"]), np.asarray(p["workspace_upper"])
    fovy = float(p["camera_fovy_deg"])
    if (rotation.shape != (3, 3) or position.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
            or not all(np.isfinite(a).all() for a in (rotation, position, lower, upper))
            or not np.all(lower < upper) or not np.isfinite(fovy) or not 0 < fovy < 179
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
        raise ValueError("invalid calibrated camera projection")


def frame_binding(frame, *, scene_id, snapshot_id):
    validate_projection(frame)
    if (not all(isinstance(v, str) and v.strip() for v in (scene_id, snapshot_id, frame.frame_id, frame.camera))
            or type(frame.control_step) is not int or frame.control_step != 0):
        raise ValueError("cross-view evidence requires named stationary baseline bindings")
    calibration = {key: np.asarray(value).tolist() for key, value in frame.projection.items() if key != "depth"}
    calibration.update(search_center_world=frame.search_center_world, table_height_m=frame.table_height_m,
                       tcp_pose7=frame.tcp_pose7)
    return {"scene_id": scene_id, "snapshot_id": snapshot_id, "frame_id": frame.frame_id,
            "camera": frame.camera, "control_step": 0, "image_shape": list(frame.rgb.shape),
            "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(frame.projection["depth"].tobytes()).hexdigest(),
            "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()}


@dataclass(frozen=True)
class PreparedView:
    frame: object
    target: object
    binding: dict
    assessments: tuple
    rows: tuple
    masks: tuple
    mask_hashes: tuple


def prepare_view(*, frame, scene_id, snapshot_id, target, reference, batches, diagnostics=None):
    """Up to two retained batches (old boxes + point refinements), <=30 masks.

Each batch is {'candidates': ..., optionally 'prompt_plan': ...}. Re-assess
the actual pixels rather than trusting a previously saved compatible flag.
The caller must establish a common stationary capture group; equal step
numbers alone do not establish that two frames came from the same reset.
"""
    binding = frame_binding(frame, scene_id=scene_id, snapshot_id=snapshot_id)
    if not 1 <= len(batches) <= 2 or sum(len(batch["candidates"]) for batch in batches) > 30:
        raise ValueError("only two bounded saved candidate batches per view")
    image, crop, _ = prepare_search_image(frame)
    assessments, rows, masks = [], [], []
    for batch in batches:
        assessed = assess_refinement(frame=frame, scene_id=scene_id, target=target, reference=reference,
            candidates=batch["candidates"], prompt_plan=batch.get("prompt_plan"), diagnostics=diagnostics)
        assessments.append(assessed)
        for candidate, row in zip(batch["candidates"], assessed["candidates"]):
            full = np.zeros(frame.rgb.shape[:2], bool)
            x0, y0, x1, y1 = crop
            full[y0:y1, x0:x1] = candidate.mask
            full.setflags(write=False)
            masks.append(full)
            rows.append(row)
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("candidate identities must be unique across batches")
    return PreparedView(frame, target, binding, tuple(assessments), tuple(rows), tuple(masks),
                        tuple(hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks))


def _project_into(points, frame):
    """Nearest-pixel projection with a source z-buffer (no duplicate votes)."""
    p = frame.projection
    camera = (points - p["camera_position_world"]) @ p["camera_rotation_world"]
    z = -camera[:, 2]
    height, width = frame.rgb.shape[:2]
    focal = .5 * height / np.tan(.5 * np.deg2rad(float(p["camera_fovy_deg"])))
    forward = np.isfinite(camera).all(axis=1) & (z > 1e-8)
    camera, z = camera[forward], z[forward]
    u, v = focal * camera[:, 0] / z + (width-1)*.5, -focal * camera[:, 1] / z + (height-1)*.5
    inside = (u >= 0) & (u <= width-1) & (v >= 0) & (v <= height-1)
    x, y, z = np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int), z[inside]
    flat = y * width + x
    order = np.lexsort((z, flat))
    flat, z = flat[order], z[order]
    keep = np.r_[True, flat[1:] != flat[:-1]] if len(flat) else np.zeros(0, bool)
    flat, z = flat[keep], z[keep]
    return flat // width, flat % width, z, {"source_world_points": len(points),
        "behind_camera_points": int(len(points) - forward.sum()), "out_of_view_points": int(len(inside) - inside.sum()),
        "projected_source_points": int(inside.sum()), "unique_projected_pixels": len(flat)}


def _direction(projected, target_frame, target_mask, policy):
    y, x, projected_depth, counts = projected
    measured_depth = target_frame.projection["depth"][y, x]
    valid = np.isfinite(measured_depth) & (measured_depth > 0)
    delta = projected_depth - measured_depth
    occluded = valid & (delta > policy.depth_tolerance_m)
    free = valid & (delta < -policy.depth_tolerance_m)
    surface = valid & (np.abs(delta) <= policy.depth_tolerance_m)
    observable = surface | free
    # A zero-width band must NOT call scipy dilation(iterations=0), which
    # iterates until convergence rather than leaving the input untouched.
    expanded = (ndimage.binary_dilation(target_mask, structure=np.ones((3, 3), bool),
                iterations=policy.mask_boundary_tolerance_px) if policy.mask_boundary_tolerance_px else target_mask)
    matches = surface & expanded[y, x]
    n, total = int(observable.sum()), len(y)
    result = {**counts, "invalid_target_depth_pixels": int((~valid).sum()), "occluded_pixels": int(occluded.sum()),
              "surface_consistent_pixels": int(surface.sum()), "free_space_conflict_pixels": int(free.sum()),
              "observable_pixels": n, "matching_mask_pixels": int(matches.sum()),
              "observable_fraction": n / total if total else 0.,
              "mask_agreement_fraction": float(matches.sum() / n) if n else None,
              "free_space_conflict_fraction": float(free.sum() / n) if n else None}
    if n < policy.minimum_observable_pixels or result["observable_fraction"] < policy.minimum_observable_fraction:
        result.update(status="unknown", reason="insufficient mutually observable support; occlusion is not agreement")
    elif result["free_space_conflict_fraction"] > policy.maximum_free_space_conflict_fraction:
        result.update(status="conflict", reason="source surface projects into observed free space")
    elif result["mask_agreement_fraction"] < policy.minimum_mask_agreement_fraction:
        result.update(status="conflict", reason="visible source surface does not agree with the other candidate mask")
    else:
        result.update(status="supported", reason="observable depth and projected mask support agree")
    return result


def associate_views(left, right, *, policy=CrossViewPolicy()):
    if (left.target != right.target or left.binding["scene_id"] != right.binding["scene_id"]
            or left.binding["snapshot_id"] != right.binding["snapshot_id"]
            or left.binding["frame_id"] == right.binding["frame_id"]
            or left.binding["camera"] == right.binding["camera"]):
        raise ValueError("association requires the same target and stationary capture group, with distinct views")
    for view in (left, right):
        if view.binding != frame_binding(view.frame, scene_id=view.target.scene_id, snapshot_id=view.binding["snapshot_id"]):
            raise ValueError("RGB/depth/calibration changed after binding")
        if (not 1 <= len(view.rows) <= policy.maximum_candidates_per_view
                or tuple(hashlib.sha256(mask.tobytes()).hexdigest() for mask in view.masks) != view.mask_hashes):
            raise ValueError("candidate masks changed or exceeded the declared budget")
    a, b = left.frame.projection, right.frame.projection
    if (np.allclose(a["camera_position_world"], b["camera_position_world"], atol=1e-6, rtol=0)
            and np.allclose(a["camera_rotation_world"], b["camera_rotation_world"], atol=1e-6, rtol=0)):
        raise ValueError("duplicate camera geometry is not independent cross-view evidence")
    usable = lambda row: (row["evidence"]["segmentation"]["status"] == "usable"
                          and row["evidence"]["spatial"]["usable_for_association"])
    # Deliberately keep color-inconsistent but measurable candidates. This
    # diagnoses geometry; it does NOT remove the legacy color acceptance gate.
    prepared = []
    for source, target in ((left, right), (right, left)):
        projections = {}
        for i, row in enumerate(source.rows):
            if usable(row):
                points = _masked_depth_to_world_points(mask=source.masks[i], **source.frame.projection)
                projections[i] = _project_into(points, target.frame)
        prepared.append(projections)
    pairs, linked = [], set()
    for i, first in enumerate(left.rows):
        for j, second in enumerate(right.rows):
            pair = {"left_candidate_id": first["candidate_id"], "right_candidate_id": second["candidate_id"]}
            if i not in prepared[0] or j not in prepared[1]:
                pair.update(status="not_evaluated", reason="segmentation/spatial quality insufficient; identity not inferred")
            else:
                forward = _direction(prepared[0][i], right.frame, right.masks[j], policy)
                backward = _direction(prepared[1][j], left.frame, left.masks[i], policy)
                states = (forward["status"], backward["status"])
                state = "conflict" if "conflict" in states else "supported" if states == ("supported", "supported") else "unknown"
                pair.update(status=state, left_to_right=forward, right_to_left=backward)
                if state == "supported":
                    linked.add((left.mask_hashes[i], right.mask_hashes[j]))
            pairs.append(pair)
    counts = {state: sum(pair["status"] == state for pair in pairs)
              for state in ("supported", "conflict", "unknown", "not_evaluated")}
    return {"schema_version": "cross_view_geometry_v1", "target": asdict(left.target),
            "left_binding": left.binding, "right_binding": right.binding, "policy": asdict(policy),
            "status": "spatial_links_present_identity_unresolved" if linked else "no_supported_spatial_link",
            "pair_counts": counts, "distinct_supported_mask_pairs": len(linked), "pairs": pairs,
            "selected_candidate_ids": None, "semantic_identity_verified": False, "actuation_allowed": False,
            "training_eligible": False, "legacy_acceptance_unchanged": True,
            "claim": "bidirectional visible-surface correspondence only; not semantic identity, grasp, or a fused pose",
            "independence_limit": "geometry does not use RGB prototypes, but masks/prompts share reference evidence"}
