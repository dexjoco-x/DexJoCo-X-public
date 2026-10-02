"""Explicit privileged-teacher collection, without a planner or vision model.

Only the separate privileged.zarr contains simulator/object state. replay.zarr
contains 13-D robot proprioception and native 12-D actions; videos contain the
corresponding PRE-action observations. Existing nonprivileged runners are not
modified by this opt-in workflow.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import zarr

from dexjoco.data.episode_store import ZarrEpisodeStore
from .core import LowDimEpisodeRecorder, SkillContext, SkillExecutor, _json_safe
from .hammer import HammerNailTaskPlugin
from .hammer_recovery import reset_fingerprint
from .hammer_retarget import TwoFrameHammerRetargetSkill, make_hammer_frame_retarget_manifest

TASK = "hammer_nail_inspire_native_delta"
CAMERAS = ("front", "ego_left", "ego_right", "wrist")
PROTOCOL = "privileged_hammer_teacher_v1"


def write_json(path, value):
    """Replace only a file owned by this run, with no partially written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".json_", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_simulator_truth(env):
    """Live body transforms, NOT the stale *_ori_pose reset-reference fields."""
    raw = env.unwrapped
    data = raw._data
    def pose(name):
        body = data.body(name)
        value = np.r_[body.xpos, body.xquat].astype(np.float64)
        if not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]), 1., atol=1e-5):
            raise ValueError("invalid simulator body pose")
        return value
    return {"hammer_pose7": pose("hammer_body"), "nail_pose7": pose("nail"),
        "nail_depth": np.asarray(raw.get_nail_depth()), "simulation_time": np.asarray(data.time),
        **{key: np.asarray(getattr(data, key)).copy()
           for key in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")}}


def training_observation(observation):
    full = np.asarray(observation["state"], dtype=np.float64)
    if full.shape != (28,) or not np.isfinite(full).all():
        raise ValueError("this teacher supports only the declared 28-D Inspire hammer observation")
    return {"state": full[:13].copy()}


class TeacherRecorder:
    """Stream RGB to video; keep compact arrays until episode finalization."""
    def __init__(self, directory, *, control_hz=20, writer_factory=None):
        self.directory = Path(directory)
        self.control_hz = control_hz
        self.lowdim = LowDimEpisodeRecorder(control_hz)
        self.truth = []
        self.writers = {}
        self.image_shapes = {}
        self.closed = False
        self.hammer_contact_steps = 0
        if writer_factory is None:
            import imageio.v2 as imageio
            writer_factory = lambda p: imageio.get_writer(str(p), fps=control_hz,
                codec="libx264", quality=8, macro_block_size=None,
                ffmpeg_params=["-threads", "1"])
        self.writer_factory = writer_factory

    @property
    def steps(self):
        return self.lowdim.steps

    def append(self, observation, action, truth, *, reward, terminated, truncated, info):
        if self.closed:
            raise ValueError("recorder is closed")
        public = training_observation(observation)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("native Inspire action must have 12 finite values")
        frames = {}
        for camera in CAMERAS:
            source_key = "random_camera" if camera == "front" and "random_camera" in observation else camera
            frame = np.asarray(observation[source_key])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32:
                raise ValueError(f"missing or invalid real RGB for {camera}")
            if camera in self.image_shapes and tuple(frame.shape) != self.image_shapes[camera]:
                raise ValueError("image shape changed")
            frames[camera] = frame
        if not truth or any(not np.isfinite(np.asarray(value)).all() for value in truth.values()):
            raise ValueError("privileged audit contains non-finite values")
        if self.truth and (set(truth) != set(self.truth[0]) or any(
                np.asarray(truth[k]).shape != self.truth[0][k].shape for k in truth)):
            raise ValueError("privileged audit layout changed")
        for camera, frame in frames.items():
            if camera not in self.writers:
                (self.directory / "videos").mkdir(exist_ok=True)
                self.writers[camera] = self.writer_factory(self.directory / "videos" / f"{camera}.mp4")
                self.image_shapes[camera] = tuple(frame.shape)
            self.writers[camera].append_data(frame)
        self.lowdim.append(observation_before=public, action=action, reward=reward,
            terminated=terminated, truncated=truncated)
        self.truth.append({k: np.asarray(v).copy() for k, v in truth.items()})
        self.hammer_contact_steps += int(bool(info.get("hammer_hit", False)))

    def close(self):
        if not self.closed:
            self.closed = True
            errors = []
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]

    def save(self, metadata):
        self.close()
        if not self.steps:
            return
        public = self.lowdim.episode_data()
        # Outcome labels are not input features. Keep rewards and terminal flags
        # in the private audit, not alongside the robot-state training input.
        labels = {k: public.pop(k) for k in ("reward", "terminated", "truncated")}
        private = {k: np.stack([row[k] for row in self.truth]) for k in self.truth[0]}
        private.update(labels)
        for name, arrays in (("replay.zarr", public), ("privileged.zarr", private)):
            store = ZarrEpisodeStore.create_empty(zarr.DirectoryStore(str(self.directory / name)))
            store.append_episode(arrays, compressors="disk")
            store.root.attrs["dexjoco_collection_metadata"] = _json_safe(metadata)


class RecordingTeacherEnvironment:
    def __init__(self, env, observation, recorder):
        self.env, self.observation, self.recorder = env, observation, recorder

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        before = self.observation
        truth = read_simulator_truth(self.env)
        result = self.env.step(action)  # Original finite-action guard stays active.
        after, reward, terminated, truncated, info = result
        # GymAgentEnvironment normalizes accepted near-bound values. Save the
        # actual bounded command, not a slightly out-of-range requested value.
        actual = np.clip(np.asarray(action, dtype=np.float64),
                         self.env.action_space.low, self.env.action_space.high)
        self.recorder.append(before, actual, truth, reward=reward,
            terminated=terminated, truncated=truncated, info=info)
        self.observation = after
        return result


def validate_episode(directory, *, check_videos=True):
    """Read back stored arrays and every video frame before accepting data."""
    directory = Path(directory)
    public = zarr.open_group(str(directory / "replay.zarr"), mode="r")
    private = zarr.open_group(str(directory / "privileged.zarr"), mode="r")
    if set(public["data"].keys()) != {"state", "action", "timestamp"}:
        raise ValueError("unexpected training fields (possible privileged-state leakage)")
    n = len(public["data/action"])
    if n <= 0 or public["data/state"].shape != (n, 13) or public["data/action"].shape != (n, 12):
        raise ValueError("invalid native-delta state/action layout")
    for store in (public, private):
        if list(store["meta/episode_ends"][:]) != [n]:
            raise ValueError("episode boundary mismatch")
        for array in store["data"].values():
            if array.shape[0] != n or not np.isfinite(array[:]).all():
                raise ValueError("array length mismatch or non-finite values")
    times = public["data/timestamp"][:]
    hz = public.attrs["dexjoco_collection_metadata"]["control_hz"]
    if not np.allclose(times, np.arange(n) / hz, atol=1e-10, rtol=0):
        raise ValueError("incorrect action timestamps")
    if n > 1 and not np.allclose(np.diff(private["data/simulation_time"][:]), 1 / hz, atol=1e-7, rtol=0):
        raise ValueError("simulation and recording frequencies differ")
    result = {"steps": n, "state_shape": [n, 13], "action_shape": [n, 12],
        "training_has_object_truth": False, "video_frames": {}}
    if check_videos:
        import cv2
        for camera in CAMERAS:
            capture = cv2.VideoCapture(str(directory / "videos" / f"{camera}.mp4"))
            count, shape = 0, None
            try:
                if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), hz):
                    raise ValueError(f"invalid video or FPS: {camera}")
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if shape is not None and frame.shape != shape:
                        raise ValueError("video shape changed")
                    shape = frame.shape
                    count += 1
            finally:
                capture.release()
            if count != n:
                raise ValueError(f"{camera}: decoded {count} frames for {n} actions")
            result["video_frames"][camera] = count
    return result


def attempt_name(seed, rank):
    return f"seed_{seed}_candidate_{rank:02d}"


def existing_attempt(root, seed, rank):
    name = attempt_name(seed, rank)
    paths = [p for p in (Path(root) / "attempts" / name, Path(root) / "success" / name) if p.exists()]
    if len(paths) > 1:
        raise ValueError("duplicate attempt paths; refusing ambiguous resume")
    if not paths:
        return None
    directory = paths[0]
    if not (directory / "result.json").exists():
        return {"seed": seed, "candidate_rank": rank, "status": "interrupted",
            "episode_dir": str(directory), "training_eligible": False,
            "reason": "incomplete previous attempt preserved; never counted as success"}
    row = read_json(directory / "result.json")
    if row["seed"] != seed or row["candidate_rank"] != rank:
        raise ValueError("attempt identity mismatch")
    if row["status"] == "success":
        if directory.parent.name != "success":
            return {**row, "status": "interrupted", "training_eligible": False,
                "episode_dir": str(directory), "reason": "interrupted before success commit; retained, not counted"}
        if not row["training_eligible"]:
            raise ValueError("uncommitted success directory")
        for name, expected in row["artifact_sha256"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError("accepted data changed; refusing resume")
    return row


def run_attempt(root, config, library, *, seed, rank):
    root = Path(root)
    directory = root / "attempts" / attempt_name(seed, rank)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "started.json", {"seed": seed, "candidate_rank": rank,
        "protocol": PROTOCOL, "privileged_teacher": True, "started_unix": time.time()})
    row = {"seed": seed, "candidate_rank": rank, "status": "running", "training_eligible": False,
        "privileged_teacher": True, "agent_used": False, "vla_used": False, "api_calls": 0}
    env, recorder, context = None, None, None
    started = time.monotonic()
    interrupted = False
    try:
        plugin = HammerNailTaskPlugin(randomize_scene=config["randomize_scene"],
            randomize_dynamics=config["randomize_dynamics"], capture_rgb=True)
        env = plugin.make_environment(seed=seed)
        observation, _ = env.reset(seed=seed)
        initial_truth = read_simulator_truth(env)
        if abs(float(initial_truth["nail_depth"])) > 1e-9:
            raise ValueError("nail is already inserted at reset")
        fingerprint = reset_fingerprint(env)
        scene_file = root / "scenes" / f"seed_{seed}.json"
        scene = {"seed": seed, "reset_fingerprint": fingerprint,
            "reset_reference_state": observation["state"], "hammer_pose7": initial_truth["hammer_pose7"],
            "nail_pose7": initial_truth["nail_pose7"]}
        if scene_file.exists():
            if read_json(scene_file)["reset_fingerprint"] != fingerprint:
                raise ValueError("same-seed reset changed between candidates")
        else:
            write_json(scene_file, scene)
        candidates = library.select_scene(initial_truth["hammer_pose7"], initial_truth["nail_pose7"],
                                          max_candidates=config["max_candidates"])
        candidate = candidates[rank - 1]
        plugin.validate_episode(candidate.episode)
        parameters = {"target_hammer_pose7": initial_truth["hammer_pose7"],
            "target_nail_pose7": initial_truth["nail_pose7"], "target_pose_source": "explicit_simulator_truth_teacher"}
        write_json(directory / "teacher.json", {"source_demo": str(candidate.episode.source),
            "candidate_distance": candidate.distance, "parameters": parameters,
            "reset_fingerprint": fingerprint, "reset_reference_state": observation["state"],
            "uses_current_body_transforms": True, "restores_source_scene": False})
        recorder = TeacherRecorder(directory)
        recording_env = RecordingTeacherEnvironment(env, observation, recorder)
        context = SkillContext(recording_env, TASK, plugin.config, observation,
            maximum_total_steps=config["max_steps"], deadline_monotonic=started + config["max_attempt_seconds"])
        executor = SkillExecutor(require_manifests=True)
        skill = TwoFrameHammerRetargetSkill(candidate)
        executor.register(skill, manifest=make_hammer_frame_retarget_manifest(task_id=TASK, expected_action_dim=12))
        result = executor.execute(skill.name, context, parameters)
        verification = env.verify()
        success = bool(result.success and verification.success
                       and time.monotonic() < context.deadline_monotonic)
        final_truth = read_simulator_truth(env)
        np.savez_compressed(directory / "final_privileged.npz", **final_truth,
                            robot_state=np.asarray(context.observation["state"][:13]))
        row.update(status="success" if success else "failure", skill=result.to_dict(),
            verification=verification.to_dict(), source_demo=str(candidate.episode.source),
            reset_fingerprint=fingerprint, safety=env.safety_summary,
            hammer_contact_steps=recorder.hammer_contact_steps)
    except KeyboardInterrupt:
        row.update(status="interrupted", reason="user interrupted; incomplete data not accepted")
        interrupted = True
    except Exception as exc:
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        metadata = {"protocol": PROTOCOL, "task_id": TASK, "control_hz": 20,
            "action_semantics": "tcp_delta_pose_current_actual",
            "rotation_semantics": "world_left_multiply_rotvec",
            "state_layout": "tcp_pose_wxyz_7 + inspire_joint_positions_6",
            "action_layout": "world_delta_xyz_3 + world_left_rotvec_3 + inspire_joint_targets_6",
            "observation_action_alignment": "observation before action; T images for T actions",
            "privileged_teacher": True, "object_truth_in_training_state": False,
            "camera_mapping": {c: c for c in CAMERAS}, "training_input_keys": ["state", *CAMERAS]}
        try:
            if recorder is not None:
                recorder.save(metadata)
                row["steps"] = recorder.steps
                if recorder.steps:
                    row["data_validation"] = validate_episode(directory)
                elif row["status"] == "success":
                    raise ValueError("an empty trajectory cannot be accepted")
        except Exception as exc:
            row.update(status="error", reason=f"data finalization: {type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                env.close()
        row["duration_seconds"] = time.monotonic() - started
    if row["status"] == "success":
        row["training_eligible"] = True
        row["artifact_sha256"] = {str(p.relative_to(directory)): file_sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}
        target = root / "success" / directory.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            raise FileExistsError("refusing to overwrite accepted episode")
        row["episode_dir"] = str(target.resolve())
        write_json(directory / "result.json", row)
        os.rename(directory, target)
    else:
        row["episode_dir"] = str(directory.resolve())
        write_json(directory / "result.json", row)
    if interrupted:
        raise KeyboardInterrupt
    return row


def summarize(config, rows):
    successes = [row for row in rows if row["status"] == "success"]
    return {"protocol": PROTOCOL, "task_id": TASK,
        "status": "target_reached" if len(successes) >= config["target_successes"] else "target_not_reached",
        "target_successes": config["target_successes"], "accepted_episodes": len(successes),
        "scenes_attempted": len({r["seed"] for r in rows}), "candidate_attempts": len(rows),
        "accepted_steps": sum(r["steps"] for r in successes), "results": rows,
        "training_episode_dirs": [r["episode_dir"] for r in successes],
        "privileged_teacher": True, "agent_used": False, "vla_used": False, "api_calls": 0}


def run_batch(root, config, library):
    rows = []
    root = Path(root)
    for seed in range(config["seed_start"], config["seed_start"] + config["max_scenes"]):
        for rank in range(1, min(config["max_candidates"], len(library.episodes)) + 1):
            previous = existing_attempt(root, seed, rank)
            row = previous if previous is not None else run_attempt(root, config, library, seed=seed, rank=rank)
            rows.append(row)
            report = summarize(config, rows)
            write_json(root / "batch_summary.json", report)
            print(f"seed={seed} candidate={rank}: {row['status']}; "
                  f"accepted={report['accepted_episodes']}/{config['target_successes']}", flush=True)
            if row["status"] == "error" and previous is None:
                raise RuntimeError(f"collection stopped on infrastructure error: {row.get('reason')}")
            if report["status"] == "target_reached":
                return report
            if row["status"] == "success":
                break
    return summarize(config, rows)
