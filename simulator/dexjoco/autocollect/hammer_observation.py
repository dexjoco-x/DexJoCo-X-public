"""Live, stationary MuJoCo RGB-D adapter for the no-motion planning loop."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np

from .agent import PlanningImage, PlanningRequest
from .hammer_chain import HAMMER_CHAIN, make_hammer_phase_manifests
from .hammer_recovery import reset_fingerprint
from .observation_loop import ObservationReceipt, utc_now
from .planning_snapshot import TASK, _image, validated_hammer_scene
from .recovery import PerceptionEvidenceError


def stationary_token(env):
    # Simulator equality checks only. Never export this hash or joint/object
    # values as perception/plan inputs. Even a time-only physics step invalidates it.
    state = reset_fingerprint(env)
    return hashlib.sha256(f"{state}:{float(env.unwrapped.data.time).hex()}".encode()).hexdigest()


def png_bytes(rgb):
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def live_hammer_request(*, image, scene_estimate, history, attempt_index):
    """Only the latest view may supply locked poses; old accepted poses expire."""
    manifests = make_hammer_phase_manifests(task_id=TASK, expected_action_dim=12)
    scene, locked = (None, {}) if scene_estimate is None else validated_hammer_scene(scene_estimate)
    if scene is not None:
        scene["confidence_note"] = "detector score is not a calibrated grasp-success probability"
    if not history or history[-1]["observation_id"] != image.frame_id:
        raise ValueError("latest perception receipt must match the supplied frame")
    current = history[-1]
    if ((scene is not None) != (current["status"] == "accepted")
            or current["camera"] != image.camera):
        raise ValueError("current perception status/camera disagrees with the image or pose estimate")
    if scene_estimate is not None:
        diagnostic = scene_estimate["diagnostics"]
        if diagnostic["camera_key"] != image.camera:
            raise ValueError("accepted pose source camera differs from current image")
        from PIL import Image
        with Image.open(io.BytesIO(image.png)) as picture:
            pixels = np.asarray(picture.convert("RGB"))
        provider = diagnostic["mask_provider"]["diagnostics"]
        if (provider.get("uses_simulator_labels") is not False
                or provider.get("input_shape") != list(pixels.shape)
                or provider.get("input_rgb_sha256") != hashlib.sha256(pixels.tobytes()).hexdigest()):
            raise ValueError("accepted poses do not refer to the supplied detector input image")
    feedback = []
    for older in history:
        if older["status"] != "rejected":
            continue
        superseded = scene is not None and older["observation_id"] != image.frame_id
        feedback.append({"stage": "perception", "code": "insufficient_or_ambiguous_evidence",
                         "reason": older["reason"], "camera": older["camera"],
                         "observation_id": older["observation_id"],
                         "temporal_status": "superseded_for_reset_pose_source" if superseded else "current_unresolved",
                         "current_blocker": not superseded, "physical_problem_resolved": False,
                         "superseded_by_observation_id": image.frame_id if superseded else None})
    return PlanningRequest(
        task_id=TASK,
        objective=("在当前新建的静止仿真场景中，根据最新图像和感知结果提出受限锤钉计划。"
                   "如果当前证据不足且仍允许观察，可以请求下一预设相机重新观察；"
                   "机会耗尽则停止。只生成计划，不执行动作，也不要宣布任务或抓取成功。"),
        observation_summary={
            "source_kind": "live_simulation_rgbd", "snapshot_mode": "live_reset",
            "historical_snapshot": False, "control_step": 0, "physical_effect_verified": False,
            "current_observation": {key: current[key] for key in
                                    ("observation_id", "camera", "index", "status", "image_scope")},
            "scene_estimate": scene, "required_skill_chain": HAMMER_CHAIN.to_dict(),
            "evidence_state": {"current_pose_observation": {"status": "accepted_estimate" if scene else "missing_or_ambiguous",
                                                           "frame_id": image.frame_id, "camera": image.camera},
                               "unresolved_perception_feedback_count": sum(row["current_blocker"] for row in feedback),
                               "physical_effect_status": "unverified"},
            "planning_gate": {"program_allowed": scene is not None,
                              "reason": ("latest sensor poses pass provenance and workspace checks; conditional plan only"
                                         if scene is not None else "latest perception rejected; no pose fallback permitted")},
        },
        skill_catalog=tuple(m.to_dict() for m in sorted(manifests, key=lambda m: m.name)),
        locked_parameters=locked, prior_feedback=tuple(feedback), attempt_index=attempt_index,
        constraints=("no robot motion or real hardware; a program is a proposal only",
                     "copy current perception-locked poses exactly; no stale pose or simulator-truth fallback",
                     "reobserve uses only the next existing fixed camera; no arm motion to seek a view",
                     "do not invent a source demonstration or claim its physical preconditions were verified",
                     "no Python, shell, VLA, task success labels or evaluator queries"),
        images=(image,),
    )


class StationaryHammerObserver:
    """An already-reset environment; no reset, step or verifier calls here."""

    def __init__(self, *, env, perception, cameras, attempt_id, output):
        from dexjoco.data.depth_capture import _env_viewer_and_camera_ids
        if perception.privileged_calibration_only or not perception.requires_rgb:
            raise ValueError("live observer requires nonprivileged RGB-D perception")
        if (not cameras or len(cameras) > 3 or len(set(cameras)) != len(cameras)
                or any(camera not in ("random_camera", "front", "ego_left", "ego_right", "wrist") for camera in cameras)):
            raise ValueError("invalid declared hammer camera schedule")
        self.env, self.perception = env, perception
        self.cameras, self.attempt_id = tuple(cameras), attempt_id
        self.output = Path(output)
        self.viewer, camera_ids = _env_viewer_and_camera_ids(env.unwrapped, cameras)
        if self.viewer is None or camera_ids is None or len(set(camera_ids)) != len(cameras):
            raise ValueError("camera views must exist and resolve to distinct physical cameras")
        self.camera_ids = tuple(camera_ids)
        self.token = stationary_token(env)
        self.history = []
        self.latest_scene_estimate = None

    def assert_stationary(self):
        if stationary_token(self.env) != self.token:
            raise RuntimeError("simulation changed since reset; discard all observation/plan evidence")
        return self.token

    def __call__(self, index, trace):
        if index != len(self.history) + 1 or not 1 <= index <= len(self.cameras):
            raise ValueError("sensor views must be consumed exactly once in declared order")
        self.assert_stationary()
        self.latest_scene_estimate = None
        started = utc_now()
        camera = self.cameras[index - 1]
        observation_id = f"view_{index}_{uuid.uuid4().hex}"
        folder = self.output / observation_id
        folder.mkdir(parents=True, exist_ok=False)
        with trace.span("sensor.render_rgb", category="sensor", start_step=0,
                        inputs={"camera": camera, "observation_id": observation_id}) as span:
            rgb = np.asarray(self.viewer.render(render_mode="rgb_array", camera_id=int(self.camera_ids[index - 1]))).copy()
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("live renderer must return uint8 HxWx3 RGB")
            raw_png = png_bytes(rgb)
            (folder / "rgb.png").write_bytes(raw_png)
            self.assert_stationary()
            span.succeed(end_step=0, outputs={"shape": list(rgb.shape), "png_sha256": hashlib.sha256(raw_png).hexdigest()})
        backend = copy.copy(self.perception)
        backend.camera_key = camera
        backend.mask_provider.diagnostic_dir = folder / "perception"
        backend.mask_provider.clear_prediction_evidence()
        scene = None
        row = {"index": index, "camera": camera, "observation_id": observation_id,
               "status": "accepted", "reason": None, "image_scope": "full_camera_rgb"}
        try:
            with trace.span("perception.hammer_nail_view", category="perception", start_step=0,
                            inputs={"camera_key": camera, "observation_index": index,
                                    "observation_id": observation_id}) as span:
                # Intentionally omit raw observation['state'], rewards and done flags.
                scene = backend.estimate(self.env, {camera: rgb}).to_dict()
                self.assert_stationary()
                span.succeed(end_step=0, outputs={"scene_estimate": scene})
        except PerceptionEvidenceError as exc:
            row.update(status="rejected", reason=str(exc)[:512])
        self.assert_stationary()
        diagnostic = copy.deepcopy(backend.mask_provider.last_diagnostics)
        if diagnostic.get("artifact_dir"):
            image = _image(diagnostic, root=folder, frame_id=observation_id, camera=camera)
            row["image_scope"] = "detector_input_crop" if backend.last_model_input_crop else "full_camera_rgb"
        else:
            image = PlanningImage(frame_id=observation_id, camera=camera, png=raw_png)
        # Successful poses must refer to the actual model-input image, not a fallback full image.
        history = [*self.history, row]
        request = live_hammer_request(image=image, scene_estimate=scene, history=history, attempt_index=index)
        self.history.append(row)
        self.latest_scene_estimate = copy.deepcopy(scene)
        (folder / "evidence.json").write_text(json.dumps({**row, "image": image.to_dict(),
            "scene_estimate": scene, "model_diagnostics": diagnostic, "same_scene_verified": True,
            "uses_object_truth_for_perception": False}, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        return ObservationReceipt(request=request, attempt_id=self.attempt_id, observation_id=observation_id,
                                  camera=camera, index=index, state_token=self.token,
                                  started_at_utc=started, finished_at_utc=utc_now())
