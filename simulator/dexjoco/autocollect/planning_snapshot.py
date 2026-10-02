"""Read-only, allowlisted export of existing hammer observations to a planner.

This first task adapter supports reset or blocked-stage historical snapshots.
It never exports full traces, recorded model plans, success labels or nail depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .agent import AutoCollectAgent, PlanningImage, PlanningRequest
from .core import SkillContext, SkillExecutor
from .hammer_chain import HAMMER_CHAIN, make_hammer_phase_manifests
from .program import ProgramLimits, _reject_json_constant, _unique_object


TASK = "hammer_nail_inspire_native_delta"
LIMITS = ProgramLimits(maximum_invocations=3, maximum_total_steps=1000,
                       maximum_wall_time_seconds=180)


def _read(path: Path, limit: int):
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("snapshot file exceeds size limit")
    return data


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)


def _under(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("snapshot image reference escapes the matching artifact directory")
    return path


def _image(diagnostics, *, root, frame_id, camera):
    from PIL import Image
    if diagnostics.get("uses_simulator_labels") is not False:
        raise ValueError("image provenance does not explicitly exclude simulator labels")
    path = _under(Path(diagnostics["artifact_dir"]) / "rgb.png", root)
    item = PlanningImage(frame_id=frame_id, camera=camera, png=_read(path, 1024 * 1024))
    with Image.open(io.BytesIO(item.png)) as picture:
        rgb = np.asarray(picture.convert("RGB"))
    if (list(rgb.shape) != diagnostics["input_shape"]
            or hashlib.sha256(rgb.tobytes()).hexdigest() != diagnostics["input_rgb_sha256"]):
        raise ValueError("snapshot PNG does not match the recorded perception input")
    return item


def _completed_before(event, later):
    """Sequence IDs mark span starts; also check completion time, not just IDs."""
    try:
        end = datetime.fromisoformat(event["finished_at_utc"])
        start = datetime.fromisoformat(later["started_at_utc"])
        return (event["attempt_id"] == later["attempt_id"]
                and type(event["sequence"]) is int and type(later["sequence"]) is int
                and event["sequence"] < later["sequence"]
                and end.tzinfo is not None and start.tzinfo is not None and end <= start)
    except (KeyError, TypeError, ValueError):
        return False


def perception_evidence_state(events, planning_event, source_scene, recorded_feedback):
    """Distinguish an earlier rejected view from the accepted current estimate.

    Supersession is narrowly scoped to which observation supplied reset poses.
    It NEVER clears a physical-effect/safety check or asserts visual correctness.
    Missing, unrelated or contradictory receipts remain unresolved.
    """
    views = [e for e in events if e.get("name") == "perception.hammer_nail_view"
             and _completed_before(e, planning_event)]
    latest = max(views, key=lambda e: (datetime.fromisoformat(e["finished_at_utc"]), e["sequence"]), default=None)
    accepted = (latest if latest is not None and latest.get("status") == "succeeded"
                and latest.get("start_step") == 0 and latest.get("end_step") == 0
                and latest.get("outputs", {}).get("scene_estimate") == source_scene else None)
    current = {
        "status": "accepted_estimate" if accepted else "temporal_provenance_unverified",
        "source_event_id": accepted["event_id"] if accepted else None,
        "camera": source_scene["diagnostics"]["camera_key"],
        "observation_index": accepted["inputs"].get("observation_index") if accepted else None,
        "finished_at_utc": accepted["finished_at_utc"] if accepted else None,
        "frame_id": "reset_detector_crop",
        "input_rgb_sha256": source_scene["diagnostics"]["mask_provider"]["diagnostics"]["input_rgb_sha256"],
        "claim": "source of current reset poses only; grasp and visual correctness remain unverified",
    }
    feedback = []
    for item in recorded_feedback:
        entry = item.get("feedback", {})
        if entry.get("stage") != "perception":
            continue  # No verifier memory or cross-run summaries exported.
        matches = [e for e in events if e.get("name") == "recovery.decide"
                   and e.get("status") == "succeeded" and _completed_before(e, planning_event)
                   and e.get("outputs", {}).get("feedback") == entry
                   and ("sequence" not in item or e["outputs"].get("sequence") == item["sequence"])]
        origin = matches[0] if len(matches) == 1 else None
        previous_views = [e for e in views if origin is not None and _completed_before(e, origin)]
        rejected = max(previous_views, key=lambda e: e["sequence"], default=None)
        superseded = (accepted is not None and origin is not None
                      and origin["outputs"].get("action") == "reobserve"
                      and _completed_before(origin, accepted)
                      and rejected is not None and rejected.get("status") == "failed")
        feedback.append({
            "stage": "perception", "code": str(entry.get("code", ""))[:128],
            "reason": str(entry.get("reason", ""))[:512],
            "scope": "earlier_perception_attempt", "source_event_id": origin["event_id"] if origin else None,
            "camera": rejected["inputs"].get("camera_key") if rejected else None,
            "observation_index": rejected["inputs"].get("observation_index") if rejected else None,
            "temporal_status": "superseded_for_reset_pose_source" if superseded else "unresolved_or_unlinked",
            "current_blocker": not superseded,
            "superseded_by_event_id": accepted["event_id"] if superseded else None,
            "physical_problem_resolved": False,
        })
    return {"schema_version": 1, "current_pose_observation": current,
            "unresolved_perception_feedback_count": sum(row["current_blocker"] for row in feedback),
            "physical_effect_status": "unverified"}, feedback


class _PlanningOnlySkill:
    def __init__(self, name):
        self.name = name

    def execute(self, context, parameters):
        raise RuntimeError("historical planning snapshots have no motion executor")


@dataclass(frozen=True)
class PlanningSnapshot:
    request: PlanningRequest
    provenance: dict

    def make_agent(self, planner):
        return make_planning_only_hammer_agent(planner)

    def context(self):
        return SkillContext(env=None, task_id=TASK, config=None, observation={})


def make_planning_only_hammer_agent(planner):
    executor = SkillExecutor(require_manifests=True)
    for manifest in make_hammer_phase_manifests(task_id=TASK, expected_action_dim=12):
        executor.register(_PlanningOnlySkill(manifest.name), manifest=manifest)
    return AutoCollectAgent(planner=planner, skill_executor=executor,
                            limits=LIMITS, chain_contract=HAMMER_CHAIN)


def validated_hammer_scene(source_scene):
    """Shared live/snapshot allowlist; never infer poses or weaken thresholds."""
    diagnostic = source_scene["diagnostics"]
    if (any(diagnostic.get(key) is not False for key in
            ("uses_simulator_object_pose", "uses_simulator_segmentation_labels", "privileged_calibration_only"))
            or diagnostic.get("eligible_as_nonprivileged_collection") is not True):
        raise ValueError("snapshot is privileged or has incomplete perception provenance")
    backend = "hammer_nail_prompted_mask_depth_v1:grounding_dino_tiny_sam_base_v1"
    if source_scene["backend"] != backend:
        raise ValueError("snapshot perception backend is not supported by this exporter")
    scene = {key: source_scene[key] for key in ("hammer_pose7", "nail_pose7", "confidence", "backend")}
    for field in ("hammer_pose7", "nail_pose7"):
        if not isinstance(scene[field], list) or any(type(v) not in (int, float) for v in scene[field]):
            raise ValueError("perceived poses must be JSON numeric arrays")
        pose = np.asarray(scene[field], dtype=float)
        if (pose.shape != (7,) or not np.isfinite(pose).all()
                or np.any(pose[:3] < [-.65, -.75, .8]) or np.any(pose[:3] > [.5, .75, 1.85])
                or abs(np.linalg.norm(pose[3:]) - 1) > 1e-5):
            raise ValueError("perceived pose is invalid or outside the declared workspace")
    if type(scene["confidence"]) not in (int, float) or not 0 <= scene["confidence"] <= 1:
        raise ValueError("invalid perception confidence")
    parameters = {"target_hammer_pose7": scene["hammer_pose7"],
                  "target_nail_pose7": scene["nail_pose7"], "target_pose_source": backend}
    manifests = make_hammer_phase_manifests(task_id=TASK, expected_action_dim=12)
    for manifest in manifests:
        manifest.validate_invocation(task_id=TASK, parameters=parameters)
    return scene, {m.name: dict(parameters) for m in manifests}


def load_hammer_snapshot(trace_path: Path, *, mode="reset") -> PlanningSnapshot:
    if mode not in ("reset", "blocked"):
        raise ValueError("snapshot mode must be reset or blocked")
    trace_path = Path(trace_path).resolve()
    if trace_path.parent.parent.name != "attempt_traces":
        raise ValueError("expected an attempt_traces/seed_*/candidate_*.jsonl snapshot")
    root = trace_path.parents[2]
    raw = _read(trace_path, 8 * 1024 * 1024)
    events = [_json(line) for line in raw.splitlines() if line.strip()]
    planning = [e for e in events if e.get("name") == "agent.plan"]
    if len(planning) != 1 or planning[0].get("start_step") != 0:
        raise ValueError("snapshot must contain exactly one reset planning request")
    event = planning[0]
    recorded = event["inputs"]["request"]
    if recorded["task_id"] != TASK:
        raise ValueError("the first snapshot adapter supports only the Inspire hammer task")
    manifests = make_hammer_phase_manifests(task_id=TASK, expected_action_dim=12)
    catalog = tuple(m.to_dict() for m in sorted(manifests, key=lambda m: m.name))
    if tuple(recorded["skill_catalog"]) != catalog:
        raise ValueError("snapshot skill catalog differs from current trusted manifests")
    source_scene = recorded["observation_summary"]["scene_estimate"]
    diagnostic = source_scene["diagnostics"]
    scene, locked = validated_hammer_scene(source_scene)
    if recorded["locked_parameters"] != locked:
        raise ValueError("snapshot locked poses differ from the perception result")
    provider = diagnostic["mask_provider"]
    reset_image = _image(provider["diagnostics"], root=root / "perception_audit" / trace_path.parent.name,
                         frame_id="reset_detector_crop", camera=diagnostic["camera_key"])
    images = [reset_image]
    scene["image_description"] = "reset camera RGB crop used by the detector; not a full-frame image"
    scene["confidence_note"] = "detector confidence is not calibrated grasp-success probability"
    evidence_state, feedback = perception_evidence_state(
        events, event, source_scene, recorded.get("prior_feedback", []))
    temporal_ready = (evidence_state["current_pose_observation"]["status"] == "accepted_estimate"
                      and evidence_state["unresolved_perception_feedback_count"] == 0)
    summary = {"snapshot_mode": mode, "historical_snapshot": True, "control_step": 0,
               "scene_estimate": scene, "required_skill_chain": HAMMER_CHAIN.to_dict(),
               "physical_effect_verified": False, "evidence_state": evidence_state,
               "planning_gate": {"program_allowed": temporal_ready,
                                 "reason": ("reset poses passed provenance, workspace and temporal checks; runtime effects unverified"
                                            if temporal_ready else "current perception chronology is unverified or has unresolved feedback")}}
    if mode == "blocked":
        gates = [e for e in events if e.get("name") == "stage.check_and_route"
                 and e.get("attempt_id") == event.get("attempt_id")
                 and e.get("sequence", -1) > event["sequence"]]
        if not gates or gates[0]["outputs"].get("action") not in ("skip", "reobserve", "abort"):
            raise ValueError("snapshot has no blocked stage gate")
        gate = gates[0]
        attempts = gate["outputs"]["observation_attempts"]
        if not attempts:
            raise ValueError("blocked gate has no observation evidence")
        latest = attempts[-1]
        frame_id = latest["observation_id"]
        if not isinstance(frame_id, str) or not frame_id.isalnum():
            raise ValueError("invalid blocked observation ID")
        stage_root = root / "stage_vision" / trace_path.parent.name / trace_path.stem
        evidence_path = _under(stage_root / frame_id / "evidence.json", stage_root)
        evidence = _json(_read(evidence_path, 256 * 1024))
        if (evidence.get("frame_id") != frame_id
                or evidence.get("control_step") != gate["end_step"]
                or evidence.get("uses_simulator_object_pose") is not False
                or evidence.get("uses_segmentation_labels") is not False):
            raise ValueError("blocked image has mismatched or privileged provenance")
        images.append(_image(evidence["model_diagnostics"], root=evidence_path.parent,
                             frame_id=frame_id, camera=evidence["camera"]))
        feedback.append({"stage": gate["outputs"]["stage_id"], "action": gate["outputs"]["action"],
                         "scope": "current_stage_evidence", "temporal_status": "current_unresolved",
                         "current_blocker": True, "source_event_id": gate.get("event_id"),
                         "observation_id": frame_id, "control_step": gate["end_step"],
                         "checks": [{k: check[k] for k in ("check_id", "status", "scope", "reason")}
                                    for check in latest["checks"]]})
        evidence_state.update({"motion_since_pose_estimate": True,
                               "current_effect_observation": images[-1].to_dict(),
                               "current_stage_evidence_status": "unresolved",
                               "reset_poses_are_not_a_new_post_motion_estimate": True})
        summary.update({"control_step": gate["end_step"],
                        "image_description": "first image is reset; second is the latest blocked-stage detector crop",
                        "planning_gate": {"program_allowed": False,
                                          "reason": "partial motion already occurred and local evidence gate blocked continuation; do not replay the full chain"}})
    request = PlanningRequest(
        task_id=TASK, objective="根据当前图像和感知证据，提出锤钉任务的受限技能计划，或建议重新观察/停止。",
        observation_summary=summary, skill_catalog=catalog,
        constraints=("historical plan-only evaluation: no execution authorized",
                     "copy perception-locked parameters exactly; never infer replacement poses from pixels",
                     "do not bypass local planning or runtime effect gates",
                     "no simulator object truth, nail depth, shell, Python or VLA"),
        prior_feedback=tuple(feedback), locked_parameters=locked,
        attempt_index=recorded["attempt_index"], images=tuple(images),
    )
    return PlanningSnapshot(request=request, provenance={
        "trace_path": str(trace_path), "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "source_event_id": event["event_id"], "snapshot_mode": mode,
        "export_policy": "allowlisted perception and stage evidence only; no whole trace or verifier memory",
    })
