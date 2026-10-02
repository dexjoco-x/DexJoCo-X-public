"""Hammer adapters for the task-neutral recovery supervisor."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from dexjoco.data.depth_capture import _env_viewer_and_camera_ids
from .memory import SkillExperience
from .recovery import PerceptionEvidenceError, RecoveryAction, RecoveryFeedback


def reset_fingerprint(env) -> str:
    """Equality-only reset audit, never a pose or controller input.

    Includes dynamics, camera/light state and joint state. No error is used to
    tune actions: a mismatch stops the run instead of reusing a stale estimate.
    """
    raw = env.unwrapped
    digest = hashlib.sha256(b"hammer_reset_v1")
    for owner, names in (
        (raw.data, ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")),
        (raw.model, ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_size", "geom_friction", "cam_pos", "cam_quat", "light_pos", "light_diffuse")),
    ):
        for name in names:
            array = np.ascontiguousarray(getattr(owner, name))
            digest.update(name.encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def record_recovery_decision(session, trace, feedback, *, step=0, **options):
    with trace.span("recovery.decide", category="recovery", start_step=step,
                    inputs={"feedback": feedback.to_dict()}) as span:
        decision = session.decide(feedback, **options)
        span.succeed(end_step=step, outputs=decision)
    return decision


def observe_hammer_with_recovery(
    *, perception, env, observation, cameras, session, trace, trace_path,
    skill_memory=None, experiment_id=None, study_id=None,
):
    """Try fresh, predeclared camera views of the SAME stationary reset.

    Only typed evidence failures are retryable. A GPU error, bad mask shape,
    unknown camera, or changed simulation state escapes as an infrastructure
    error; none is silently retried as if the scene was merely hard to see.
    """
    observation_attempts = []
    if hasattr(perception, "last_model_input_crop"):
        perception.last_model_input_crop = None
    primary = getattr(perception, "camera_key", "default")
    views = (primary, *cameras)
    for index, camera in enumerate(views):
        session.begin_observation(camera)
        backend = copy.copy(perception)
        backend.camera_key = camera
        current = observation
        if index:
            with trace.span("recovery.observe_new_view", category="sensor", start_step=0,
                            inputs={"camera_key": camera, "robot_motion_requested": False}) as span:
                before = reset_fingerprint(env)
                viewer, camera_ids = _env_viewer_and_camera_ids(env.unwrapped, [camera])
                if viewer is None or camera_ids is None:
                    raise RuntimeError(f"camera unavailable: {camera}")
                rgb = viewer.render(render_mode="rgb_array", camera_id=int(camera_ids[0]))
                current = {**observation, camera: np.asarray(rgb)}
                if reset_fingerprint(env) != before:
                    raise RuntimeError("scene changed during read-only reobservation")
                span.succeed(end_step=0, outputs={"same_scene_verified": True, "rgb_shape": list(rgb.shape)})
        row = {"index": index + 1, "camera_key": camera}
        try:
            with trace.span("perception.hammer_nail_view", category="perception", start_step=0,
                            inputs={"camera_key": camera, "observation_index": index + 1}) as span:
                estimate = backend.estimate(env, current)
                span.succeed(end_step=0, outputs={"scene_estimate": estimate.to_dict()})
        except PerceptionEvidenceError as exc:
            row.update(status="rejected", error={"type": type(exc).__name__, "message": str(exc)})
            provider = getattr(backend, "mask_provider", None)
            row["model_diagnostics"] = copy.deepcopy(getattr(provider, "last_diagnostics", {}))
            row["model_input_crop_xyxy"] = getattr(backend, "last_model_input_crop", None)
            observation_attempts.append(row)
            feedback = RecoveryFeedback("perception", "insufficient_or_ambiguous_evidence", str(exc)[:2048])
            decision = record_recovery_decision(session, trace, feedback, another_view=index + 1 < len(views))
            if skill_memory is not None:
                refs = [trace_path]
                artifact = row["model_diagnostics"].get("artifact_dir")
                if artifact:
                    refs.append(Path(artifact))
                experience = SkillExperience.create(
                    task_id=env.task_id, skill_name="perception.hammer_nail", skill_version="2",
                    success=False, parameters={"camera_key": camera},
                    scene_summary={"observation_index": index + 1},
                    skill_metrics={"control_actions_executed": 0, "recovery_decision": decision},
                    verification={"status": "not_run", "reason": "perception_rejected"},
                    trace_id=trace.trace_id, trace_path=trace_path, experiment_id=experiment_id,
                    study_id=study_id, failure_stage="perception", failure_codes=(feedback.code,),
                    artifact_refs=refs, eligible_as_training_data=False,
                    tags=("autocollect", "vision", "recovery", "no_control"),
                )
                row["memory_record"] = str(skill_memory.append(experience))
            if decision["action"] != RecoveryAction.REOBSERVE.value:
                exc.observation_attempts = observation_attempts
                raise
        else:
            session.accept_observation()
            row.update(status="accepted", confidence=estimate.confidence)
            observation_attempts.append(row)
            return estimate, observation_attempts
        finally:
            # Keep crop coordinates aligned with the provider's latest image,
            # including rejected views. Never carry them from another seed.
            if hasattr(backend, "last_model_input_crop"):
                perception.last_model_input_crop = backend.last_model_input_crop
    raise RuntimeError("observation recovery ended without a terminal result")
