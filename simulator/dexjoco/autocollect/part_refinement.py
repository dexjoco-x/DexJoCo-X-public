"""Offline evidence-sourced SAM prompts and boundary uncertainty diagnostics.

No semantic identity claim, mask voting, selected action mask, or robot access.
Point colors are proposal cues; they cannot validate their own semantic identity.
"""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from scipy import ndimage

from .sam_candidates import recorded_boxes
from .visual_candidates import AppearancePolicy, appearance_check, assess_part_candidates, resolve_candidates
from .visual_effect_runtime import prepare_search_image


@dataclass(frozen=True)
class PointPromptPolicy:
    minimum_seed_pixels: int = 25
    minimum_interior_distance_px: float = 2.
    minimum_color_margin: float = .05
    minimum_dominant_fraction: float = .85
    maximum_point_prompts: int = 5

    def __post_init__(self):
        if type(self.minimum_seed_pixels) is not int or self.minimum_seed_pixels < 1:
            raise ValueError("invalid point support")
        if type(self.maximum_point_prompts) is not int or not 1 <= self.maximum_point_prompts <= 5:
            raise ValueError("point inference is bounded to 1..5 prompts per image")
        if not np.isfinite(self.minimum_interior_distance_px) or self.minimum_interior_distance_px < 1:
            raise ValueError("point must be inside a supported region")
        for value in (self.minimum_color_margin, self.minimum_dominant_fraction):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("invalid point color/component policy")


@dataclass(frozen=True)
class BoundaryPolicy:
    boundary_band_px: int = 2
    minimum_common_pixels: int = 200
    minimum_common_component_fraction: float = .85
    maximum_centroid_spread_m: float = .005
    maximum_size_spread_m: float = .01

    def __post_init__(self):
        if (type(self.boundary_band_px) is not int or not 1 <= self.boundary_band_px <= 3
                or type(self.minimum_common_pixels) is not int or self.minimum_common_pixels < 1):
            raise ValueError("invalid boundary support policy")
        for value in (self.minimum_common_component_fraction, self.maximum_centroid_spread_m,
                      self.maximum_size_spread_m):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("invalid boundary geometry policy")


def _interior_point(support, policy):
    labels, _ = ndimage.label(support)
    counts = np.bincount(labels.ravel())[1:]
    total = int(support.sum())
    info = {"support_pixels": total}
    if not len(counts) or int(counts.max()) < policy.minimum_seed_pixels:
        return None, {**info, "reason": "insufficient connected color support"}
    fraction = float(counts.max() / total)
    info["largest_component_fraction"] = fraction
    if fraction < policy.minimum_dominant_fraction:
        return None, {**info, "reason": "multiple competing color regions"}
    component = labels == int(np.argmax(counts)) + 1
    # Explicit outside padding also handles components touching the image edge.
    distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distance), support.shape)
    info["interior_distance_px"] = float(distance[y, x])
    if distance[y, x] < policy.minimum_interior_distance_px:
        return None, {**info, "reason": "support has no sufficiently interior pixel"}
    return [int(x), int(y)], info


def _positive_hypotheses(support, policy):
    """Branch over every supported component, not a largest-component winner.

Separate regions may be texture fragments OR different objects. Neither case
is resolved here. Small/thin regions cannot provide an interior seed and remain
listed in the plan. The caller refuses the whole plan if its budget overflows.
"""
    labels, _ = ndimage.label(support)
    counts = np.bincount(labels.ravel())[1:]
    proposed, rejected = [], []
    for label, count in enumerate(counts, 1):
        info = {"component_id": label, "support_pixels": int(count),
                "fraction_of_positive_support": float(count / support.sum())}
        if count < policy.minimum_seed_pixels:
            rejected.append({**info, "reason": "insufficient connected color support"})
            continue
        point, interior = _interior_point(labels == label, policy)
        info.update(interior)
        if point is None:
            rejected.append(info)
        else:
            proposed.append((point, info))
    return proposed, rejected


def build_part_point_plan(*, frame, scene_id, target, reference, diagnostics,
                          appearance_policy=AppearancePolicy(), policy=PointPromptPolicy()):
    """One positive + one excluded-part point per hypothesis, <=5 in total.

All coordinates derive from RGB/reference prototypes inside the unchanged depth
search region. No hand-picked pixels, selected SAM mask, or simulator labels.
"""
    if target != reference.target or target.extent != "part" or scene_id != target.scene_id:
        raise ValueError("point prompts require matching scene/object/part/query reference")
    if frame.control_step != 0 or reference.control_step != 0 or frame.frame_id == reference.frame_id:
        raise ValueError("point prompts require an independent baseline frame")
    if frame.search_center_world is None or frame.table_height_m is None:
        raise ValueError("point prompts require perceived position and depth table estimate")
    image, crop, region = prepare_search_image(frame)
    boxes = recorded_boxes(image, diagnostics, label="held_part")
    # Reuse validation of finite/distinguishable reference prototypes.
    check = appearance_check(image, np.ones(image.shape[:2], bool), reference, appearance_policy)
    separation = check.get("prototype_separation")
    x0, y0, x1, y1 = crop
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    plan = {"target": asdict(target), "frame_id": frame.frame_id, "control_step": frame.control_step,
            "input_rgb_sha256": digest, "raw_rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(frame.projection["depth"].tobytes()).hexdigest(),
            "crop_xyxy": list(crop), "reference": asdict(reference), "point_policy": asdict(policy),
            "appearance_policy": asdict(appearance_policy), "prompts": [], "skipped_boxes": [],
            "source": "recorded detector boxes + independent reference RGB prototypes + measured depth search region",
            "semantic_identity_verified": False, "actuation_allowed": False}
    if separation is None:
        plan.update(status="unknown", reason="reference appearance cannot distinguish the requested parts")
        return plan
    pixels = image.astype(float) / 255.
    positive_distance = np.linalg.norm(pixels - np.asarray(reference.part_rgb), axis=-1)
    negative_distance = np.min(np.linalg.norm(
        pixels[:, :, None, :] - np.asarray(reference.excluded_rgb)[None, None, :, :], axis=-1), axis=-1)
    positive = ((positive_distance <= appearance_policy.maximum_color_distance)
                & (negative_distance - positive_distance >= policy.minimum_color_margin))
    negative = ((negative_distance <= appearance_policy.maximum_color_distance)
                & (positive_distance - negative_distance >= policy.minimum_color_margin))
    ys, xs = np.indices(image.shape[:2])
    for rank, box in enumerate(boxes, 1):
        left, top, right, bottom = box["box_xyxy"]
        allowed = ((xs >= left) & (xs <= right) & (ys >= top) & (ys <= bottom) & region[y0:y1, x0:x1])
        positives, rejected = _positive_hypotheses(positive & allowed, policy)
        neg, neg_info = _interior_point(negative & allowed, policy)
        record = {"box_rank": rank, "box_xyxy": list(box["box_xyxy"]), "detector_score": box["score"],
                  "eligible_positive_components": len(positives), "rejected_positive_components": rejected,
                  "negative_evidence": neg_info}
        if not positives or neg is None:
            plan["skipped_boxes"].append(record)
            continue
        for pos, pos_info in positives:
            plan["prompts"].append({**record, "positive_evidence": pos_info,
                "prompt_id": f"box_{rank}_part_component_{pos_info['component_id']}",
                "points_xy": [pos, neg], "labels": [1, 0],
                "points_full_image_xy": [[x + x0, y + y0] for x, y in (pos, neg)]})
    if len(plan["prompts"]) > policy.maximum_point_prompts:
        plan["unexecuted_prompts"] = plan["prompts"]
        plan.update(prompts=[], status="unknown", reason="point hypothesis budget exceeded; no subset selected")
        return plan
    plan["status"] = "proposed" if plan["prompts"] else "unknown"
    return plan


def _boundary_pair(first, second, policy):
    """Changes must stay in BOTH masks' thin contour bands, not internal holes."""
    structure = np.ones((3, 3), bool)
    for a, b in ((first, second), (second, first)):
        inner = ndimage.binary_erosion(a, structure=structure, iterations=policy.boundary_band_px)
        outer = ndimage.binary_dilation(a, structure=structure, iterations=policy.boundary_band_px)
        if np.any(inner & ~b) or np.any(b & ~outer):
            return False
    return True


def boundary_diagnosis(rows, masks, *, policy=BoundaryPolicy()):
    """Characterize proposals, never replace the strict selector or pick a mask.

Complete-link morphology avoids transitive merging; every family member must
pass unchanged gates and bounded 3-D spread. A failed nearby variant cannot be
discarded just to turn the family's result into PASS.
"""
    if (not rows or len(rows) > 15 or len(rows) != len(masks)
            or len({row["candidate_id"] for row in rows}) != len(rows)
            or any(type(row["compatible"]) is not bool for row in rows)
            or any(mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape != masks[0].shape for mask in masks)):
        raise ValueError("boundary diagnosis requires 1..15 unique aligned proposals")
    order = sorted(range(len(rows)), key=lambda i: (hashlib.sha256(masks[i].tobytes()).hexdigest(), rows[i]["candidate_id"]))
    groups = []
    for i in order:
        for group in groups:
            if all(_boundary_pair(masks[i], masks[j], policy) for j in group):
                group.append(i)
                break
        else:
            groups.append([i])
    families = []
    for group in groups:
        common = np.logical_and.reduce([masks[i] for i in group])
        union = np.logical_or.reduce([masks[i] for i in group])
        labels, _ = ndimage.label(common)
        counts = np.bincount(labels.ravel())[1:]
        fraction = float(counts.max() / common.sum()) if len(counts) else 0.
        compatible = [rows[i]["compatible"] for i in group]
        family = {"candidate_ids": sorted(rows[i]["candidate_id"] for i in group),
                  "common_pixels": int(common.sum()), "union_pixels": int(union.sum()),
                  "boundary_uncertain_pixels": int(np.count_nonzero(union & ~common)),
                  "common_component_fraction": fraction}
        if not any(compatible):
            family["status"] = "rejected"
        elif not all(compatible):
            family["status"] = "boundary_gate_disagreement"
        elif common.sum() < policy.minimum_common_pixels or fraction < policy.minimum_common_component_fraction:
            family["status"] = "insufficient_common_support"
        else:
            centroids = np.asarray([rows[i]["quality"].get("centroid_world", [np.nan] * 3) for i in group])
            sizes = np.asarray([rows[i]["quality"].get("robust_size_m", np.nan) for i in group])
            if centroids.shape != (len(group), 3) or not np.isfinite(centroids).all() or not np.isfinite(sizes).all():
                family["status"] = "geometry_unavailable"
            else:
                spread = float(np.max(np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)))
                family.update(centroid_lower_m=centroids.min(axis=0).tolist(), centroid_upper_m=centroids.max(axis=0).tolist(),
                              centroid_spread_m=spread, size_range_m=[float(sizes.min()), float(sizes.max())])
                family["status"] = ("bounded_boundary_family" if spread <= policy.maximum_centroid_spread_m
                                    and np.ptp(sizes) <= policy.maximum_size_spread_m else "geometry_disagreement")
        families.append(family)
    usable = [family for family in families if family["status"] == "bounded_boundary_family"]
    uncertain = [family for family in families if family["status"] not in ("bounded_boundary_family", "rejected")]
    if uncertain:
        status = "unresolved_boundary_or_geometry"
    elif len(usable) > 1:
        status = "identity_or_extent_ambiguity"
    elif len(usable) == 1:
        status = "boundary_uncertainty_only" if usable[0]["boundary_uncertain_pixels"] else "single_mask_proposal"
    else:
        status = "no_compatible_proposal"
    return {"status": status, "families": families, "policy": asdict(policy),
            "selected_candidate_id": None, "semantic_identity_verified": False, "actuation_allowed": False,
            "claim": "observed proposal variation only; ranges are not calibrated confidence intervals or identity proof"}


def assess_refinement(*, frame, scene_id, target, reference, candidates,
                      appearance_policy=AppearancePolicy(), boundary_policy=BoundaryPolicy(),
                      prompt_plan=None, diagnostics=None):
    result = assess_part_candidates(frame=frame, scene_id=scene_id, target=target, reference=reference,
                                    candidates=candidates, policy=appearance_policy)
    eligible = [i for i, candidate in enumerate(candidates) if candidate.sam_score >= .65]
    if prompt_plan is not None:
        expected = build_part_point_plan(frame=frame, scene_id=scene_id, target=target, reference=reference,
                                        diagnostics=diagnostics, appearance_policy=appearance_policy)
        if expected != prompt_plan:
            raise ValueError("assessment point plan differs from bound evidence")
        constraints = {f"{p['prompt_id']}_mask_{v}": p for p in expected["prompts"] for v in range(3)}
        if set(constraints) != {c.candidate_id for c in candidates}:
            raise ValueError("assessment must retain every prompted SAM variant")
        for row, candidate in zip(result["candidates"], candidates):
            prompt = constraints[candidate.candidate_id]
            satisfied = [bool(candidate.mask[y, x]) == bool(label)
                         for (x, y), label in zip(prompt["points_xy"], prompt["labels"])]
            row["point_adherence"] = {"positive_included": satisfied[0], "negative_excluded": satisfied[1]}
            if not all(satisfied):
                row["reasons"].append("SAM mask contradicts its sourced positive/negative prompt")
                row["compatible"] = False
                row["evidence"]["segmentation"]["status"] = "unknown"
                row["evidence"]["segmentation"]["reasons"].append(row["reasons"][-1])
        if eligible:
            result["selection"] = resolve_candidates([result["candidates"][i] for i in eligible],
                [candidates[i].mask for i in eligible], equivalent_mask_iou=appearance_policy.equivalent_mask_iou)
    result["boundary_diagnosis"] = (boundary_diagnosis([result["candidates"][i] for i in eligible],
        [candidates[i].mask for i in eligible], policy=boundary_policy) if eligible else
        {"status": "no_compatible_proposal", "families": [], "selected_candidate_id": None, "actuation_allowed": False})
    return result
