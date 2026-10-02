"""Immutable, historical observation evidence for a read-only semantic reviewer.

This is not a PlanningRequest, action program, live receipt or identity permit.
Diagnostic sheets display all candidates, never a score-selected subset.
"""
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .agent import PlanningImage, _canonical_json, _sha256
from .cross_view_evidence import associate_views
from .program import _unique_object, _reject_json_constant
from .visual_effect_runtime import prepare_search_image


def _diagnostic_sheet(view, ordered, frame_alias):
    """Deterministic annotations of actual pixels; not generated sensor data."""
    frame = view.frame
    height, width = frame.rgb.shape[:2]
    if height > 1024 or width > 1024:
        raise ValueError("diagnostic layout only supports source images up to 1024 pixels per side")
    crop_rgb, crop, _ = prepare_search_image(frame)
    grid_x, tile_w, tile_h, columns = width + 32, 190, 205, 4
    canvas = Image.new("RGB", (grid_x + columns * tile_w + 16, max(height + 140, 65 + ((len(ordered)+3)//4)*tile_h)), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((16, 10), f"{frame_alias}: {frame.camera} | HISTORICAL DIAGNOSTIC, NOT A NEW OBSERVATION", fill="black", font=font)
    draw.text((16, 32), "LEFT: original RGB. RIGHT: all candidate masks in red (display overlay only).", fill="black", font=font)
    canvas.paste(Image.fromarray(frame.rgb), (16, 64))
    draw.text((16, height + 80), "No candidate is an identity/grasp/motion permission.", fill="black", font=font)
    draw.text((16, height + 98), "Grid order is by candidate ID, not quality or preference.", fill="black", font=font)
    locations = {}
    for ordinal, index in enumerate(ordered, 1):
        row, mask = view.rows[index], view.masks[index]
        alias = f"{frame_alias}C{ordinal:02d}"
        x, y = grid_x + ((ordinal-1) % columns)*tile_w, 64 + ((ordinal-1)//columns)*tile_h
        draw.rectangle((x, y, x+tile_w-4, y+tile_h-4), outline=(175, 175, 175))
        draw.text((x+5, y+4), alias, fill="black", font=font)
        for line, text in enumerate(textwrap.wrap(row["candidate_id"], width=27)[:2]):
            draw.text((x+5, y+17+line*11), text, fill="black", font=font)
        x0, y0, x1, y1 = crop
        small = mask[y0:y1, x0:x1]
        overlay = crop_rgb.copy()
        overlay[small] = np.rint(.55 * overlay[small] + .45 * np.array([255, 0, 0])).astype(np.uint8)
        picture = Image.fromarray(overlay)
        picture.thumbnail((tile_w-12, tile_h-55), Image.Resampling.NEAREST)
        # Original RGB is never resized; only clearly labeled mask thumbnails.
        canvas.paste(picture, (x+5, y+45))
        locations[alias] = {"tile_xyxy": [x, y, x+tile_w-4, y+tile_h-4],
                            "source_model_crop_xyxy": list(crop)}
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    return PlanningImage(frame.frame_id, frame.camera, stream.getvalue()), locations


@dataclass(frozen=True)
class ObservationEvidencePackage:
    """Canonical bytes prevent in-place mutation of reviewed observation data."""
    canonical_json: bytes
    images: tuple[PlanningImage, ...]

    def to_dict(self):
        return json.loads(self.canonical_json, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)

    def validate(self):
        if not isinstance(self.canonical_json, bytes) or len(self.canonical_json) > 48 * 1024:
            raise ValueError("evidence package exceeds the bounded context budget")
        body = self.to_dict()
        if (body.get("schema_version") != "observation_evidence_v1"
                or body.get("source_kind") != "historical_sensor_replay"
                or body.get("historical_snapshot") is not True
                or body.get("execution_capabilities") != []
                or body.get("semantic_identity_verified") is not False
                or body.get("actuation_allowed") is not False
                or body.get("training_eligible") is not False
                or body.get("allowed_advice") != ["request_fresh_observation", "stop"]
                or len(self.images) != 2 or len(body.get("views", [])) != 2):
            raise ValueError("invalid historical, non-actuating evidence package")
        identifier = body.pop("package_id", None)
        if identifier != _sha256(body):
            raise ValueError("evidence package digest mismatch")
        for view, image in zip(body["views"], self.images):
            if (view["attachment"] != image.to_dict() or view["binding"]["frame_id"] != image.frame_id
                    or view["binding"]["camera"] != image.camera or view["binding"]["control_step"] != 0):
                raise ValueError("diagnostic attachment and source frame differ")
        candidates = body["candidates"]
        if not 1 <= len(candidates) <= 60 or len({c["candidate_id"] for c in candidates}) != len(candidates):
            raise ValueError("candidate catalog must be complete, unique and bounded")
        catalog = [identifier for view in body["views"] for identifier in view["candidate_ids"]]
        if (len(catalog) != len(candidates) or set(catalog) != {c["candidate_id"] for c in candidates}
                or any(c["identity"] != "unknown" for c in candidates)):
            raise ValueError("catalog mismatch or identity verification is not authorized")
        return self


def build_observation_package(left, right):
    association = associate_views(left, right)  # Revalidate exact sensor bindings.
    views, images, records, aliases = [], [], [], []
    for number, view in enumerate((left, right), 1):
        frame_alias = f"V{number}"
        order = sorted(range(len(view.rows)), key=lambda i: view.rows[i]["candidate_id"])
        attachment, locations = _diagnostic_sheet(view, order, frame_alias)
        images.append(attachment)
        mapping = {}
        for ordinal, index in enumerate(order, 1):
            alias = f"{frame_alias}C{ordinal:02d}"
            row = view.rows[index]
            mapping[row["candidate_id"]] = alias
            evidence = row["evidence"]
            appearance = row["appearance"]
            records.append({"candidate_id": alias, "source_candidate_id": row["candidate_id"],
                "view_id": frame_alias, "mask_sha256": view.mask_hashes[index], **locations[alias],
                "segmentation": evidence["segmentation"]["status"],
                "spatial": evidence["spatial"]["status"],
                "spatial_association_usable": evidence["spatial"]["usable_for_association"],
                "centroid_world_m": evidence["spatial"]["centroid_world_m"],
                "identity": evidence["identity"]["status"], "appearance": appearance["status"],
                "part_color_support": appearance.get("part_support_fraction"),
                "excluded_color_support": appearance.get("excluded_support_fraction"),
                "legacy_compatible": row["compatible"], "point_adherence": row.get("point_adherence")})
        views.append({"view_id": frame_alias, "binding": view.binding, "attachment": attachment.to_dict(),
                      "image_role": "annotated_diagnostic_including_unmodified_raw_RGB_not_new_sensor_frame",
                      "raw_rgb_panel_xyxy": [16, 64, 16+view.frame.rgb.shape[1], 64+view.frame.rgb.shape[0]],
                      "candidate_ids": list(mapping.values())})
        aliases.append(mapping)
    pair_by_id = {(pair["left_candidate_id"], pair["right_candidate_id"]): pair for pair in association["pairs"]}
    codes = {"supported": "S", "conflict": "C", "unknown": "U", "not_evaluated": "N"}
    matrix = ["".join(codes[pair_by_id[(a, b)]["status"]] for b in aliases[1]) for a in aliases[0]]
    body = {"schema_version": "observation_evidence_v1", "source_kind": "historical_sensor_replay",
        "historical_snapshot": True, "target": asdict(left.target), "views": views, "candidates": records,
        "spatial_relationships": {"row_candidate_ids": list(aliases[0].values()),
            "column_candidate_ids": list(aliases[1].values()), "matrix_rows": matrix,
            "legend": {value: key for key, value in codes.items()}, "counts": association["pair_counts"],
            "distinct_supported_mask_pairs": association["distinct_supported_mask_pairs"],
            "claim": "bidirectional visible-surface correspondence, not semantic identity or candidate votes"},
        "missing_evidence": ["independent_part_semantics", "target_instance_identity", "live_current_observation"],
        "allowed_advice": ["request_fresh_observation", "stop"], "execution_capabilities": [],
        "semantic_identity_verified": False, "actuation_allowed": False, "training_eligible": False,
        "constraints": ["All candidates are hypotheses; no score-ranked subset was selected.",
            "Historical images do not describe a current live scene; no fresh observation was acquired.",
            "Color mismatch does not imply missing segmentation; geometry match does not verify identity.",
            "Masks and color cues share reference provenance; matches are not independent identity votes.",
            "Do not produce skills, commands, coordinates for action, or changes to local acceptance rules."]}
    body["package_id"] = _sha256(body)
    return ObservationEvidencePackage(_canonical_json(body), tuple(images)).validate()
