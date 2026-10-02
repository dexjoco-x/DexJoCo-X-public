"""Offline, baseline-only part proposal filtering; never an actuation permit.

Task-independent identity scope and reference appearance are explicit. RGB
prototype consistency is NOT semantic identity, a pose estimate or grasp proof.
"""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .recovery import PerceptionEvidenceError
from .visual_effect_runtime import object_sample_from_mask, prepare_search_image


@dataclass(frozen=True)
class VisualTarget:
    scene_id: str
    object_id: str
    extent: str
    part_id: str | None
    query: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in (self.scene_id, self.object_id, self.query)):
            raise ValueError("target requires explicit scene/object/query identity")
        if self.extent not in ("part", "whole"):
            raise ValueError("target extent must explicitly be part or whole")
        if ((self.extent == "part" and (not isinstance(self.part_id, str) or not self.part_id.strip()))
                or (self.extent == "whole" and self.part_id is not None)):
            raise ValueError("part identity and whole-object identity are not interchangeable")


@dataclass(frozen=True)
class AppearancePolicy:
    # New, predeclared appearance-only gates. Existing geometric gates unchanged.
    minimum_reference_pixels: int = 200
    maximum_color_distance: float = .20
    minimum_prototype_separation: float = .10
    minimum_part_support: float = .80
    maximum_excluded_support: float = .05
    equivalent_mask_iou: float = .95

    def __post_init__(self):
        if type(self.minimum_reference_pixels) is not int or self.minimum_reference_pixels <= 0:
            raise ValueError("invalid reference support")
        for key, value in asdict(self).items():
            if key != "minimum_reference_pixels" and (not np.isfinite(value) or not 0 < value <= 1):
                raise ValueError("appearance policy values must lie in (0, 1]")


@dataclass(frozen=True)
class PartAppearanceReference:
    target: VisualTarget
    frame_id: str
    rgb_sha256: str
    part_rgb: tuple[float, ...]
    excluded_rgb: tuple[tuple[float, ...], ...]
    excluded_parts: tuple[str, ...]
    control_step: int = 0


def make_part_reference(*, target, frame_id, rgb, part_mask, excluded_masks, expected_rgb_sha256,
                        policy=AppearancePolicy()):
    if target.extent != "part" or not frame_id or not excluded_masks or target.part_id in excluded_masks:
        raise ValueError("part selection needs a named independent reference and excluded sibling parts")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("reference RGB must be HxWx3 uint8")
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    if digest != expected_rgb_sha256:
        raise ValueError("reference pixels do not match the recorded image")
    masks = [part_mask, *excluded_masks.values()]
    for mask in masks:
        if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or mask.sum() < policy.minimum_reference_pixels:
            raise ValueError("reference masks must be aligned with sufficient visible pixels")
    # Incompatible reference assignments are not silently fixed by subtraction.
    for excluded in masks[1:]:
        if np.count_nonzero(part_mask & excluded) / min(part_mask.sum(), excluded.sum()) > .25:
            raise ValueError("reference part masks overlap ambiguously")
    prototypes = [tuple(np.median(rgb[mask], axis=0) / 255.) for mask in masks]
    return PartAppearanceReference(target, frame_id, digest, prototypes[0], tuple(prototypes[1:]),
                                   tuple(excluded_masks))


def appearance_check(rgb, mask, reference, policy):
    positive = np.asarray(reference.part_rgb)
    negative = np.asarray(reference.excluded_rgb)
    if (positive.shape != (3,) or negative.ndim != 2 or negative.shape[1] != 3 or not len(negative)
            or not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative))
            or np.any((positive < 0) | (positive > 1)) or np.any((negative < 0) | (negative > 1))):
        raise ValueError("malformed appearance reference")
    separation = float(np.min(np.linalg.norm(negative - positive, axis=1)))
    if separation < policy.minimum_prototype_separation:
        return {"status": "unknown", "reason": "reference parts are not distinguishable by this appearance test"}
    if not mask.any():
        return {"status": "unknown", "reason": "empty candidate"}
    pixels = rgb[mask].astype(float) / 255.
    part_distance = np.linalg.norm(pixels - positive, axis=1)
    excluded_distance = np.min(np.linalg.norm(pixels[:, None, :] - negative[None, :, :], axis=-1), axis=1)
    part_fraction = float(np.mean((part_distance <= policy.maximum_color_distance) & (part_distance < excluded_distance)))
    excluded_fraction = float(np.mean((excluded_distance <= policy.maximum_color_distance) & (excluded_distance < part_distance)))
    compatible = part_fraction >= policy.minimum_part_support and excluded_fraction <= policy.maximum_excluded_support
    return {"status": "consistent" if compatible else "unknown",
            "reason": "reference appearance consistent; semantic identity unverified" if compatible
                      else "insufficient intended-part support or excluded-part appearance included",
            "part_support_fraction": part_fraction, "excluded_support_fraction": excluded_fraction,
            "prototype_separation": separation}


def mask_iou(first, second):
    union = int(np.count_nonzero(first | second))
    return np.count_nonzero(first & second) / union if union else 1.


def resolve_candidates(rows, masks, *, equivalent_mask_iou=.95):
    """Complete-link groups prevent duplicate masks from manufacturing agreement.

Scores never choose between distinct compatible groups. Equivalent masks with
different gate results remain ambiguous, including threshold-boundary flips.
"""
    if len(rows) != len(masks) or not rows or len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("unique candidate records and masks are required")
    if (len(rows) > 15 or not np.isfinite(equivalent_mask_iou) or not 0 < equivalent_mask_iou <= 1
            or any(type(row["compatible"]) is not bool for row in rows)
            or any(mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape != masks[0].shape for mask in masks)):
        raise ValueError("malformed or unbounded mask grouping inputs")
    order = sorted(range(len(rows)), key=lambda index: (hashlib.sha256(masks[index].tobytes()).hexdigest(), rows[index]["candidate_id"]))
    groups = []
    for index in order:
        for group in groups:
            if all(mask_iou(masks[index], masks[other]) >= equivalent_mask_iou for other in group):
                group.append(index)
                break
        else:
            groups.append([index])
    valid, ambiguous, reports = [], [], []
    for group in groups:
        usable = [index for index in group if rows[index]["compatible"]]
        state = "compatible" if len(usable) == len(group) else "ambiguous" if usable else "rejected"
        reports.append({"candidate_ids": sorted(rows[index]["candidate_id"] for index in group), "status": state})
        if state == "compatible":
            valid.append(group)
        elif state == "ambiguous":
            ambiguous.append(group)
    if len(valid) == 1 and not ambiguous:
        # All members agree; deterministic representative, not a score contest.
        selected = min(rows[index]["candidate_id"] for index in valid[0])
        return {"status": "unique_compatible_proposal", "selected_candidate_id": selected, "groups": reports}
    return {"status": "unknown", "selected_candidate_id": None, "groups": reports,
            "reason": "distinct or boundary-ambiguous candidates" if valid or ambiguous else "no compatible candidates"}


def assess_part_candidates(*, frame, scene_id, target, reference, candidates, policy=AppearancePolicy(), sam_threshold=.65):
    if target != reference.target or target.extent != "part" or scene_id != target.scene_id:
        raise ValueError("whole/part, scene, object and query identity must match the reference contract")
    if frame.control_step != 0 or reference.control_step != 0 or frame.frame_id == reference.frame_id:
        raise ValueError("this selector is baseline-only and requires an independent reference frame")
    if frame.search_center_world is None or frame.table_height_m is None:
        raise ValueError("baseline association requires the previously perceived target and table height")
    if not 1 <= len(candidates) <= 15:
        raise ValueError("selection requires 1..15 predeclared proposals")
    if not np.isfinite(sam_threshold) or not 0 < sam_threshold <= 1:
        raise ValueError("invalid SAM quality threshold")
    image, crop, region = prepare_search_image(frame)
    input_hash = hashlib.sha256(image.tobytes()).hexdigest()
    rows, full_masks = [], []
    for candidate in candidates:
        if candidate.input_rgb_sha256 != input_hash:
            raise ValueError("candidate masks are not bound to these exact RGB pixels")
        if candidate.mask.dtype != np.bool_ or candidate.mask.shape != image.shape[:2]:
            raise ValueError("candidate masks must align with the exact model crop")
        if not np.isfinite(candidate.sam_score) or not np.isfinite(candidate.detector_score):
            raise ValueError("candidate scores must be finite")
        mask = np.zeros(frame.rgb.shape[:2], bool)
        x0, y0, x1, y1 = crop
        mask[y0:y1, x0:x1] = candidate.mask
        quality, reasons = {}, []
        segmentation_reasons, spatial_reasons = [], []
        sample = None
        appearance = appearance_check(frame.rgb, mask, reference, policy)
        if candidate.sam_score < sam_threshold:
            reasons.append("SAM quality below the unchanged threshold")
            segmentation_reasons.append(reasons[-1])
        if not mask.any() or np.count_nonzero(mask & region) / mask.sum() < .90:
            reasons.append("predicted mask lies outside the unchanged search region")
            segmentation_reasons.append(reasons[-1])
        try:
            sample = object_sample_from_mask(frame, mask, audit=quality)
            distance = float(np.linalg.norm(np.asarray(sample.centroid_world) - frame.search_center_world))
            quality["reset_association_distance_m"] = distance
            quality["centroid_world"] = list(sample.centroid_world)
            if distance > .10:
                reasons.append("reset association exceeds the unchanged 0.10 m limit")
                spatial_reasons.append(reasons[-1])
        except PerceptionEvidenceError as exc:
            reasons.append(str(exc))
            # The legacy measurement routine checks mask support/components
            # BEFORE depth and extent. Keep these evidence axes separate.
            if ("largest_component_fraction" not in quality
                    or quality["largest_component_fraction"] < .85):
                segmentation_reasons.append(str(exc))
            else:
                spatial_reasons.append(str(exc))
        if appearance["status"] != "consistent":
            reasons.append(appearance["reason"])
        rows.append({"candidate_id": candidate.candidate_id, "box_rank": candidate.box_rank,
                     "sam_variant": candidate.variant, "detector_score": candidate.detector_score,
                     "sam_score": candidate.sam_score, "box_xyxy": list(candidate.box_xyxy),
                     "compatible": not reasons, "reasons": reasons, "quality": quality, "appearance": appearance,
                     "evidence": {
                         "segmentation": {"status": "usable" if not segmentation_reasons else "unknown",
                                          "reasons": segmentation_reasons,
                                          "claim": "mask support/quality only, not correct part identity"},
                         "spatial": {"status": "measured" if sample is not None else "unknown",
                                     "usable_for_association": sample is not None and not spatial_reasons,
                                     "reasons": spatial_reasons if sample is not None or spatial_reasons else
                                                ["insufficient segmentation support for spatial measurement"],
                                     "centroid_world_m": list(sample.centroid_world) if sample is not None else None,
                                     "claim": "visible surface centroid, not complete object pose"},
                         "identity": {"status": "unknown", "appearance_status": appearance["status"],
                                      "reason": "reference appearance is not independent semantic or instance identity evidence"}}})
        full_masks.append(mask)
    # Low-SAM-confidence variants stay in the audit but cannot create or resolve
    # competing semantic hypotheses. All high-confidence variants are grouped,
    # including ones rejected by geometric/appearance checks.
    eligible = [i for i, candidate in enumerate(candidates) if candidate.sam_score >= sam_threshold]
    selection = (resolve_candidates([rows[i] for i in eligible], [full_masks[i] for i in eligible],
                                    equivalent_mask_iou=policy.equivalent_mask_iou) if eligible else
                 {"status": "unknown", "selected_candidate_id": None, "groups": [], "reason": "no reliable SAM proposals"})
    return {"target": asdict(target), "reference_frame_id": reference.frame_id, "frame_id": frame.frame_id,
            "camera": frame.camera, "control_step": frame.control_step, "appearance_policy": asdict(policy),
            "candidates": rows, "selection": selection, "semantic_identity_verified": False,
            "grasp_verified": False, "actuation_allowed": False, "training_eligible": False}
