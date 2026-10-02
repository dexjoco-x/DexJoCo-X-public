"""Sensor-only adapter for the bounded hammer grasp probe.

The reset pose estimator explicitly retains the existing hammer CAD/upright
reset prior. Post-action checks use measured visible points, never that reset
pose estimator. No task oracle or target-scene object state is consumed.
"""
from dataclasses import replace
import hashlib

import numpy as np
from scipy import ndimage

from .hammer_perception import _estimate_hammer_pose_from_parts, _table_height_from_depth, _workspace_image_crop
from .perception import _masked_depth_to_world_points
from .recovery import PerceptionEvidenceError
from .sam_candidates import MaskCandidate
from .visual_effect_runtime import object_sample_from_mask, prepare_search_image


def candidate_for_mask(frame, mask, box, label, provider):
    diagnostic = provider.last_diagnostics
    index = list(diagnostic["queries"]).index(label)
    scores = np.asarray(diagnostic["sam_scores"][index], dtype=float)
    variant = int(np.argmax(scores))
    selected = np.asarray(diagnostic["selected_boxes"][label])
    detected = min(diagnostic["detections"][label], key=lambda row:
                   float(np.linalg.norm(np.asarray(row["box_xyxy"]) - selected)))
    return MaskCandidate(f"{label}_{frame.frame_id[:12]}", 1, variant, tuple(box), float(detected["score"]), float(scores[variant]),
        mask.copy(), hashlib.sha256(frame.rgb.tobytes()).hexdigest())


def reset_hammer_evidence(frame, provider):
    """One disjoint named handle/head proposal; no nail needed for acquisition."""
    if frame.control_step != 0:
        raise ValueError("upright reset pose estimation cannot be used after motion")
    projection = dict(frame.projection)
    projection.update(workspace_lower=np.array([-.60, -.70, .86]), workspace_upper=np.array([.30, .50, 1.40]))
    table_height, table_diagnostic = _table_height_from_depth(projection, minimum_table_points=10_000)
    crop = _workspace_image_crop(projection, table_height_m=table_height)
    x0, y0, x1, y1 = crop
    labels = {"hammer_handle": "wooden stick", "hammer_head": "metal hammer head"}
    mask_set = provider.predict(frame.rgb[y0:y1, x0:x1].copy(), labels)
    if set(mask_set.masks) != set(labels) or mask_set.image_shape != (y1-y0, x1-x0):
        raise ValueError("reset part masks do not match source RGB crop")
    candidates = []
    for label in labels:
        small = mask_set.masks[label]
        if small.dtype != np.bool_ or not np.isfinite(mask_set.scores[label]):
            raise ValueError("malformed reset mask or score")
        mask = np.zeros(frame.rgb.shape[:2], bool)
        mask[y0:y1, x0:x1] = small
        box = np.asarray(provider.last_diagnostics["selected_boxes"][label]) + [x0, y0, x0, y0]
        candidates.append(candidate_for_mask(frame, mask, box, label, provider))
    return reset_pose_from_candidates(frame, candidates, model_crop=crop)


def reset_pose_from_candidates(frame, candidates, *, model_crop=None):
    """Shared geometric checks for detector or explicit point-guided proposals."""
    if frame.control_step != 0 or len(candidates) != 2:
        raise ValueError("reset pose needs ordered handle/head candidates before any action")
    projection = dict(frame.projection)
    projection.update(workspace_lower=np.array([-.60, -.70, .86]), workspace_upper=np.array([.30, .50, 1.40]))
    table_height, table_diagnostic = _table_height_from_depth(projection, minimum_table_points=10_000)
    restored, points, quality = {}, {}, {}
    for label, candidate in zip(("hammer_handle", "hammer_head"), candidates):
        if candidate.input_rgb_sha256 != hashlib.sha256(frame.rgb.tobytes()).hexdigest():
            raise ValueError("part mask is bound to a different RGB frame")
        mask = candidate.mask
        restored[label] = mask
        quality[label] = {}
        sample = object_sample_from_mask(frame, mask, audit=quality[label])
        components, _ = ndimage.label(mask)
        sizes = np.bincount(components.ravel())[1:]
        main = components == (int(np.argmax(sizes)) + 1)
        points[label] = _masked_depth_to_world_points(mask=main, **projection)
        quality[label]["pose_points_scope"] = "largest connected component; original mask preserved"
        minimum = 250 if label == "hammer_handle" else 750
        if len(points[label]) < minimum:
            raise PerceptionEvidenceError(f"{label}: {len(points[label])} valid points, minimum {minimum}")
        if label == "hammer_handle":
            handle_sample = sample
    overlap = int(np.count_nonzero(restored["hammer_handle"] & restored["hammer_head"])) / min(
        int(restored["hammer_handle"].sum()), int(restored["hammer_head"].sum()))
    if overlap > .25:
        raise PerceptionEvidenceError("handle/head masks overlap beyond the existing .25 limit")
    pose, diagnostic = _estimate_hammer_pose_from_parts(points["hammer_handle"], points["hammer_head"],
        table_height_m=table_height, camera_position_world=projection["camera_position_world"])
    return {"pose7": pose, "table_height_m": table_height, "candidates": tuple(candidates),
            "handle_sample": handle_sample, "diagnostics": {"table": table_diagnostic, "hammer": diagnostic,
                "quality": quality, "part_overlap": overlap, "model_crop_xyxy": model_crop,
                "geometry_workspace_lower": projection["workspace_lower"], "geometry_workspace_upper": projection["workspace_upper"],
                "uses_object_truth": False, "pose_scope": "existing upright reset CAD calibration, not general 6D estimation"}}


def visible_handle_evidence(frame, provider, *, table_height, search_center):
    """Fresh mask and visible centroid, not a post-grasp CAD/reset-pose estimate."""
    search = replace(frame, search_center_world=tuple(search_center), table_height_m=float(table_height))
    image, crop, region = prepare_search_image(search)
    x0, y0, x1, y1 = crop
    mask_set = provider.predict(image, {"held_part": "wooden stick"})
    if set(mask_set.masks) != {"held_part"} or mask_set.image_shape != image.shape[:2]:
        raise ValueError("held-part mask does not match RGB crop")
    small = mask_set.masks["held_part"]
    if small.dtype != np.bool_:
        raise ValueError("held-part mask must be boolean")
    mask = np.zeros(frame.rgb.shape[:2], bool)
    mask[y0:y1, x0:x1] = small
    if not mask.any() or np.count_nonzero(mask & region) / mask.sum() < .90:
        raise PerceptionEvidenceError("held-part mask outside perceived/proprioceptive search region")
    quality = {}
    sample = object_sample_from_mask(frame, mask, audit=quality)
    box = np.asarray(provider.last_diagnostics["selected_boxes"]["held_part"]) + [x0, y0, x0, y0]
    candidate = candidate_for_mask(frame, mask, box, "held_part", provider)
    return {"candidate": candidate, "sample": sample,
            "diagnostics": {"quality": quality, "model_crop_xyxy": crop, "search_center_world": search_center,
                "search_center_source": "reset_perception" if frame.control_step == 0 else "TCP_proprioception",
                "uses_object_truth": False, "reset_pose_estimator_used": False}}
