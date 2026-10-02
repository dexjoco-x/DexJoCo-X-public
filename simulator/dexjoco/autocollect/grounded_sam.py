"""Local Grounding DINO + box-prompted SAM, with RGB-only model inputs.

Optional dependencies are imported at construction, keeping the base robot
environment independent of transformers. Model revisions and weights are
fingerprinted before a rollout; inference never downloads code or weights.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from itertools import combinations, product
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from .perception import PromptedMaskSet
from .recovery import PerceptionEvidenceError


def _snapshot_fingerprint(path: Path) -> dict[str, Any]:
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix in {".json", ".txt", ".safetensors"}
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"no safetensors weights found in {path}")
    fingerprints = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        fingerprints[file.name] = digest.hexdigest()
    return {"snapshot": str(path), "revision": path.name, "files_sha256": fingerprints}


def _validate_box(box: Any, image_shape: tuple[int, int]) -> list[float]:
    array = np.asarray(box, dtype=float)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError("detected box must be four finite xyxy coordinates")
    height, width = image_shape
    array = np.clip(array, [0, 0, 0, 0], [width, height, width, height])
    if np.any(array[2:] - array[:2] < 2):
        raise ValueError("detected box is empty or smaller than two pixels")
    return array.tolist()


def _select_disjoint_boxes(
    detections: Mapping[str, list[dict[str, Any]]], *, maximum_overlap: float = .25,
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Choose jointly compatible named detections, without task coordinates.

    Applicable only to labels representing distinct visible parts, not nested
    whole/part labels. Scores rank proposals; they are not calibrated certainty.
    """
    if not detections or any(not candidates for candidates in detections.values()):
        raise PerceptionEvidenceError("every query needs a detection before joint selection")
    if not 0 <= maximum_overlap < 1 or top_k < 1:
        raise ValueError("invalid joint box selection limits")
    pools = [sorted(items, key=lambda item: item["score"], reverse=True)[:top_k] for items in detections.values()]
    if math.prod(len(pool) for pool in pools) > 10_000:
        raise ValueError("joint box search exceeds 10000 combinations")
    best, best_score = None, -np.inf
    for group in product(*pools):
        boxes = np.asarray([item["box_xyxy"] for item in group], dtype=float)
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 2:] <= boxes[:, :2]):
            continue
        compatible = True
        for a, b in combinations(boxes, 2):
            intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
            smaller_area = min(np.prod(a[2:] - a[:2]), np.prod(b[2:] - b[:2]))
            if intersection / smaller_area > maximum_overlap:
                compatible = False
                break
        score = sum(np.log(max(1e-12, item["score"])) for item in group)
        if compatible and score > best_score:
            best, best_score = group, score
    if best is None:
        raise PerceptionEvidenceError("no mutually compatible detection boxes for distinct parts")
    return dict(zip(detections, best))


class GroundedSamMaskProvider:
    """Detect named queries, optionally resolve conflicts, and segment locally."""

    name = "grounding_dino_tiny_sam_base_v1"
    privileged_calibration_only = False

    def __init__(
        self,
        *,
        detector_path: str | Path,
        segmenter_path: str | Path,
        device: str = "cuda:0",
        detection_threshold: float = 0.20,
        text_threshold: float = 0.20,
        sam_threshold: float = 0.65,
        diagnostic_dir: str | Path | None = None,
        disjoint_queries: bool = False,
    ) -> None:
        import torch
        import transformers
        from transformers import (
            GroundingDinoForObjectDetection, GroundingDinoProcessor,
            SamModel, SamProcessor,
        )

        self.device = str(device)
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else None
        self.disjoint_queries = bool(disjoint_queries)
        self.detection_threshold = float(detection_threshold)
        self.text_threshold = float(text_threshold)
        self.sam_threshold = float(sam_threshold)
        for threshold in (
            self.detection_threshold, self.text_threshold, self.sam_threshold,
        ):
            if not np.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError("model thresholds must lie in (0, 1]")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        detector_path = Path(detector_path).expanduser().resolve()
        segmenter_path = Path(segmenter_path).expanduser().resolve()
        self.provenance = {
            "detector": _snapshot_fingerprint(detector_path),
            "segmenter": _snapshot_fingerprint(segmenter_path),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        self.detector_processor = GroundingDinoProcessor.from_pretrained(
            detector_path, local_files_only=True,
        )
        self.detector = GroundingDinoForObjectDetection.from_pretrained(
            detector_path, local_files_only=True, use_safetensors=True,
            disable_custom_kernels=True,
        ).eval().to(self.device)
        self.segmenter_processor = SamProcessor.from_pretrained(
            segmenter_path, local_files_only=True,
        )
        self.segmenter = SamModel.from_pretrained(
            segmenter_path, local_files_only=True, use_safetensors=True,
        ).eval().to(self.device)
        self.last_diagnostics: dict[str, Any] = {}
        self.last_masks: dict[str, np.ndarray] = {}

    def clear_prediction_evidence(self) -> None:
        self.last_masks = {}
        self.last_diagnostics = {}

    def configuration(self) -> dict[str, Any]:
        return {
            "name": self.name, "device": self.device,
            "detection_threshold": self.detection_threshold,
            "text_threshold": self.text_threshold,
            "sam_threshold": self.sam_threshold,
            "detection_strategy": "joint_disjoint_top5" if self.disjoint_queries else "highest_score_per_named_query",
            "maximum_box_overlap": .25 if self.disjoint_queries else None,
            "segmentation_strategy": "box_prompt_best_sam_iou",
            "provenance": self.provenance,
        }

    def predict(
        self, image_rgb: np.ndarray, object_queries: Mapping[str, str],
    ) -> PromptedMaskSet:
        self.clear_prediction_evidence()
        try:
            result = self._predict(image_rgb, object_queries)
            self.last_diagnostics["status"] = "pass"
            return result
        except Exception as exc:
            self.last_diagnostics.update(
                status="error", error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            if self.diagnostic_dir is not None:
                self._save_diagnostics(image_rgb)

    def _save_diagnostics(self, image_rgb: np.ndarray) -> None:
        """Persist model evidence even when detection/segmentation rejects a scene."""
        from PIL import Image, ImageDraw

        directory = self.diagnostic_dir / f"prediction_{uuid.uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=False)
        self.last_diagnostics["artifact_dir"] = str(directory.resolve())
        image = np.asarray(image_rgb)
        if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
            Image.fromarray(image).save(directory / "rgb.png")
            canvas = image.copy()
            colors = [(255, 70, 70), (50, 220, 90), (50, 130, 255)]
            for i, (label, mask) in enumerate(self.last_masks.items()):
                Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"mask_{i}.png")
                canvas[mask] = .5 * canvas[mask] + .5 * np.asarray(colors[i % 3])
            overlay = Image.fromarray(canvas)
            draw = ImageDraw.Draw(overlay)
            for i, (label, candidates) in enumerate(self.last_diagnostics.get("detections", {}).items()):
                if candidates:
                    best = max(candidates, key=lambda item: item["score"])
                    box = self.last_diagnostics.get("selected_boxes", {}).get(label, best["box_xyxy"])
                    best = min(candidates, key=lambda item: np.linalg.norm(np.asarray(item["box_xyxy"]) - box))
                    draw.rectangle(box, outline=colors[i % 3], width=2)
                    draw.text((box[0], max(0, box[1] - 12)), f"{label}: {best['score']:.3f}", fill=colors[i % 3])
            overlay.save(directory / "overlay.png")
        self.last_diagnostics["mask_file_labels"] = {
            f"mask_{i}.png": label for i, label in enumerate(self.last_masks)
        }
        (directory / "diagnostics.json").write_text(
            json.dumps(self.last_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    def _predict(
        self, image_rgb: np.ndarray, object_queries: Mapping[str, str],
    ) -> PromptedMaskSet:
        import torch
        from PIL import Image

        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("model input must be an HxWx3 uint8 RGB image")
        if not object_queries or any(not str(q).strip() for q in object_queries.values()):
            raise ValueError("named detection queries must not be empty")
        self.last_masks = {}
        self.last_diagnostics = {
            "input_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "input_shape": list(image.shape), "queries": dict(object_queries),
            "detections": {}, "uses_simulator_labels": False,
        }
        pil_image = Image.fromarray(image)
        labels = list(object_queries)
        boxes = []
        detection_scores = []
        with torch.inference_mode():
            for label in labels:
                query = str(object_queries[label]).strip().rstrip(".") + "."
                inputs = self.detector_processor(
                    images=pil_image, text=query, return_tensors="pt",
                ).to(self.device)
                outputs = self.detector(**inputs)
                result = self.detector_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=self.detection_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[image.shape[:2]],
                )[0]
                candidates = [
                    {"box_xyxy": b.tolist(), "score": float(s)}
                    for b, s in zip(result["boxes"].cpu(), result["scores"].cpu())
                ]
                self.last_diagnostics["detections"][label] = candidates
                if not candidates:
                    continue
                best = max(candidates, key=lambda item: item["score"])
                boxes.append(_validate_box(best["box_xyxy"], image.shape[:2]))
                detection_scores.append(best["score"])

            missing = [label for label in labels if not self.last_diagnostics["detections"][label]]
            if missing:
                raise PerceptionEvidenceError(f"detector found no confident objects: {missing}")

            if self.disjoint_queries:
                selected = _select_disjoint_boxes(self.last_diagnostics["detections"])
                boxes = [_validate_box(selected[label]["box_xyxy"], image.shape[:2]) for label in labels]
                detection_scores = [selected[label]["score"] for label in labels]

            sam_inputs = self.segmenter_processor(
                images=pil_image, input_boxes=[boxes], return_tensors="pt",
            ).to(self.device)
            outputs = self.segmenter(**sam_inputs, multimask_output=True)
            masks = self.segmenter_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]
            scores = {}
            sam_scores = outputs.iou_scores[0].cpu().numpy()
            self.last_diagnostics["selected_boxes"] = dict(zip(labels, boxes))
            self.last_diagnostics["sam_scores"] = sam_scores.tolist()
            for i, label in enumerate(labels):
                best_index = int(np.argmax(sam_scores[i]))
                score = float(sam_scores[i, best_index])
                mask = masks[i, best_index].numpy().astype(bool)
                self.last_masks[label] = mask
                if not np.isfinite(score) or score < self.sam_threshold or not mask.any():
                    raise PerceptionEvidenceError(f"SAM has insufficient evidence for {label!r}: {score}")
                scores[label] = min(detection_scores[i], max(0., min(1., score)))

        return PromptedMaskSet(
            masks=dict(self.last_masks), scores=scores, backend=self.name,
            diagnostics=self.last_diagnostics,
        )
