"""Bounded local SAM proposals from already-recorded RGB detector boxes.

No detector rerun, network, environment or robot API. Preserve all SAM variants
instead of silently replacing a rejected mask with whichever variant passes.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .grounded_sam import _snapshot_fingerprint, _validate_box


@dataclass(frozen=True)
class MaskCandidate:
    candidate_id: str
    box_rank: int
    variant: int
    box_xyxy: tuple[float, ...]
    detector_score: float | None  # None for explicitly detector-free point proposals.
    sam_score: float
    mask: np.ndarray
    input_rgb_sha256: str


def recorded_boxes(image, diagnostics, *, label, maximum_boxes=5):
    if type(maximum_boxes) is not int or not 1 <= maximum_boxes <= 5:
        raise ValueError("candidate inference is bounded to 1..5 boxes")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("candidate RGB must be HxWx3 uint8")
    if (diagnostics.get("input_shape") != list(image.shape)
            or diagnostics.get("input_rgb_sha256") != hashlib.sha256(image.tobytes()).hexdigest()
            or diagnostics.get("uses_simulator_labels") is not False):
        raise ValueError("recorded boxes must be bound to these exact nonprivileged RGB pixels")
    detections = diagnostics.get("detections", {}).get(label)
    if not isinstance(detections, list) or not detections:
        raise ValueError("recorded label has no detections")
    validated = []
    for item in detections:
        score = float(item["score"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("detector score must be finite in [0, 1]")
        validated.append({"box_xyxy": _validate_box(item["box_xyxy"], image.shape[:2]), "score": score})
    return sorted(validated, key=lambda item: (-item["score"], item["box_xyxy"]))[:maximum_boxes]


class SamBoxCandidateGenerator:
    """Load only the existing local SAM checkpoint; at most 15 masks per image."""
    def __init__(self, *, segmenter_path, device="cuda:0"):
        import torch
        from transformers import SamModel, SamProcessor
        self.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        path = Path(segmenter_path).resolve()
        self.provenance = _snapshot_fingerprint(path)
        self.processor = SamProcessor.from_pretrained(path, local_files_only=True)
        self.model = SamModel.from_pretrained(path, local_files_only=True, use_safetensors=True).eval().to(device)
        self.calls = 0

    @classmethod
    def from_provider(cls, provider):
        """Share already-loaded local SAM weights; do not load a second copy."""
        instance = cls.__new__(cls)
        instance.device = provider.device
        instance.processor = provider.segmenter_processor
        instance.model = provider.segmenter
        instance.provenance = provider.provenance["segmenter"]
        instance.calls = 0
        return instance

    def generate(self, image, diagnostics, *, label, maximum_boxes=5):
        boxes = recorded_boxes(image, diagnostics, label=label, maximum_boxes=maximum_boxes)
        return self._generate(image, diagnostics, boxes)

    def generate_refined(self, *, frame, scene_id, target, reference, diagnostics, prompt_plan):
        from .part_refinement import build_part_point_plan
        from .visual_effect_runtime import prepare_search_image
        expected = build_part_point_plan(frame=frame, scene_id=scene_id, target=target,
                                         reference=reference, diagnostics=diagnostics)
        if prompt_plan != expected:
            raise ValueError("point plan differs from the current evidence-derived plan")
        if not expected["prompts"]:
            return ()
        image = prepare_search_image(frame)[0]
        boxes = [{"box_xyxy": row["box_xyxy"], "score": row["detector_score"]} for row in expected["prompts"]]
        return self._generate(image, diagnostics, boxes, prompts=expected["prompts"])

    def _generate(self, image, diagnostics, boxes, prompts=None):
        import torch
        from PIL import Image
        extra = {} if prompts is None else {"input_points": [[row["points_xy"] for row in prompts]],
                                            "input_labels": [[row["labels"] for row in prompts]]}
        with torch.inference_mode():
            inputs = self.processor(images=Image.fromarray(image), input_boxes=[[row["box_xyxy"] for row in boxes]],
                                    return_tensors="pt", **extra).to(self.device)
            self.calls += 1
            outputs = self.model(**inputs, multimask_output=True)
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
            scores = outputs.iou_scores[0].cpu().numpy()
        if tuple(masks.shape) != (len(boxes), 3, *image.shape[:2]) or scores.shape != (len(boxes), 3):
            raise ValueError("SAM returned an unexpected candidate layout")
        candidates = []
        for index, box in enumerate(boxes):
            for variant in range(3):
                score = float(scores[index, variant])
                if not np.isfinite(score):
                    raise ValueError("SAM returned a nonfinite quality score")
                rank = index + 1 if prompts is None else prompts[index]["box_rank"]
                prefix = f"box_{rank}" if prompts is None else prompts[index]["prompt_id"]
                candidates.append(MaskCandidate(f"{prefix}_mask_{variant}", rank, variant,
                    tuple(box["box_xyxy"]), box["score"], score, masks[index, variant].numpy().astype(bool),
                    diagnostics["input_rgb_sha256"]))
        return tuple(candidates)
