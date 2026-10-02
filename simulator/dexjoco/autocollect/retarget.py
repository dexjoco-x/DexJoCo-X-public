"""Object-frame trajectory retrieval and retargeting for pick-bucket."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .core import SkillContext, SkillResult, SkillStatus
from .registry import ParameterKind, SkillManifest, SkillParameterSpec
from .replay import ReplayEpisode, load_replay_episode


TCP_SLICE = slice(0, 7)
BUCKET_SLICE = slice(13, 20)
FOOD_SLICE = slice(20, 27)
MINIMUM_PICK_BUCKET_STATE_DIM = 28


def make_object_frame_retarget_manifest(
    *,
    task_id: str,
    expected_action_dim: int,
) -> SkillManifest:
    """Describe the callable contract of object-frame grasp/lift retargeting."""

    return SkillManifest(
        name="grasp_lift.object_frame_retarget",
        version="1",
        description=(
            "Retrieve a demonstrated grasp/lift trajectory, transform its TCP "
            "motion into the perceived bucket frame, close the hand, lift, and hold."
        ),
        supported_tasks=(str(task_id),),
        parameters=(
            SkillParameterSpec(
                name="target_bucket_pose7",
                kind=ParameterKind.ARRAY,
                description=(
                    "Perceived world-frame bucket pose [x,y,z,qw,qx,qy,qz]."
                ),
                shape=(7,),
                finite=True,
            ),
            SkillParameterSpec(
                name="target_pose_source",
                kind=ParameterKind.STRING,
                description="Name of the perception backend that produced the pose.",
            ),
        ),
        required_observations=("state",),
        action_dimension=int(expected_action_dim),
        required_perception=("bucket_pose7", "food_pose7"),
        preconditions=(
            "a ranked source demonstration has already been selected",
            "the perceived bucket pose is finite and expressed in world coordinates",
        ),
        expected_effects=(
            "the hand grasps the bucket handle",
            "the bucket and enclosed payload are lifted and held",
        ),
        safety_constraints=(
            f"every action has exactly {int(expected_action_dim)} finite values",
            "TCP translation and rotation deltas are clipped to environment limits",
            "task success is judged by an independent post-rollout verifier",
        ),
        implementation=(
            "dexjoco.autocollect.retarget.ObjectFrameRetargetSkill"
        ),
        controller_family="object_frame_trajectory_retarget",
        vla_used=False,
        tags=("grasp", "lift", "inspire", "no_vla"),
    )


def _pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"pose must be a finite 7-vector, got {pose.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = Rotation.from_quat(
        pose[3:7], scalar_first=True
    ).as_matrix()
    return matrix


def _matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"matrix must be finite with shape (4,4), got {matrix.shape}"
        )
    return np.concatenate(
        [
            matrix[:3, 3],
            Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True),
        ]
    )


def _pose_error6(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    current_matrix = _pose7_to_matrix(current)
    desired_matrix = _pose7_to_matrix(desired)
    return np.concatenate(
        [
            desired_matrix[:3, 3] - current_matrix[:3, 3],
            Rotation.from_matrix(
                desired_matrix[:3, :3] @ current_matrix[:3, :3].T
            ).as_rotvec(),
        ]
    )


def _pick_bucket_feature_from_poses(
    bucket_pose7: np.ndarray,
    food_pose7: np.ndarray,
) -> np.ndarray:
    bucket = np.asarray(bucket_pose7, dtype=np.float64).reshape(-1)
    food = np.asarray(food_pose7, dtype=np.float64).reshape(-1)
    if (
        bucket.shape != (7,)
        or food.shape != (7,)
        or not np.all(np.isfinite(bucket))
        or not np.all(np.isfinite(food))
    ):
        raise ValueError("bucket_pose7 and food_pose7 must be finite 7-vectors")
    bucket_rotation = Rotation.from_quat(bucket[3:7], scalar_first=True)
    bucket_matrix = bucket_rotation.as_matrix()
    bucket_yaw = float(np.arctan2(bucket_matrix[1, 0], bucket_matrix[0, 0]))
    food_in_bucket = bucket_rotation.inv().apply(food[:3] - bucket[:3])
    food_matrix = Rotation.from_quat(
        food[3:7], scalar_first=True
    ).as_matrix()
    food_yaw = float(np.arctan2(food_matrix[1, 0], food_matrix[0, 0]))
    relative_food_yaw = food_yaw - bucket_yaw
    return np.concatenate(
        [
            bucket[:3],
            [np.cos(bucket_yaw), np.sin(bucket_yaw)],
            food_in_bucket,
            [np.cos(relative_food_yaw), np.sin(relative_food_yaw)],
        ]
    )


def _pick_bucket_feature(state: np.ndarray) -> np.ndarray:
    """Build the retrieval feature from one flattened pick-bucket state."""

    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.size < MINIMUM_PICK_BUCKET_STATE_DIM or not np.all(np.isfinite(state)):
        raise ValueError(
            "pick-bucket state must be finite and contain TCP, hand, bucket, "
            f"food, and table fields; got shape {state.shape}"
        )
    return _pick_bucket_feature_from_poses(
        state[BUCKET_SLICE],
        state[FOOD_SLICE],
    )


@dataclass(frozen=True)
class RetargetCandidate:
    episode: ReplayEpisode
    distance: float
    rank: int


class PickBucketTrajectoryLibrary:
    """Validated demonstration library used only for skill retrieval."""

    DEFAULT_FEATURE_SCALES = np.asarray(
        [0.05, 0.05, 0.05, 0.35, 0.35, 0.06, 0.06, 0.03, 0.35, 0.35],
        dtype=np.float64,
    )

    def __init__(
        self,
        episodes: Iterable[ReplayEpisode],
        *,
        expected_action_dim: int = 12,
        feature_scales: np.ndarray | None = None,
    ) -> None:
        unique: list[ReplayEpisode] = []
        seen_initial_states: set[tuple[str, tuple[int, ...], bytes]] = set()
        for episode in episodes:
            if episode.actions.shape[1] != expected_action_dim:
                raise ValueError(
                    f"{episode.source} has action dim {episode.actions.shape[1]}, "
                    f"expected {expected_action_dim}"
                )
            if episode.states.shape[1] < MINIMUM_PICK_BUCKET_STATE_DIM:
                raise ValueError(
                    f"{episode.source} has state dim {episode.states.shape[1]}, "
                    f"expected at least {MINIMUM_PICK_BUCKET_STATE_DIM}"
                )
            initial = np.ascontiguousarray(episode.initial_state)
            fingerprint = (initial.dtype.str, initial.shape, initial.tobytes())
            if fingerprint in seen_initial_states:
                continue
            seen_initial_states.add(fingerprint)
            unique.append(episode)
        if not unique:
            raise ValueError("trajectory library is empty")

        scales = (
            self.DEFAULT_FEATURE_SCALES.copy()
            if feature_scales is None
            else np.asarray(feature_scales, dtype=np.float64).reshape(-1)
        )
        if scales.shape != self.DEFAULT_FEATURE_SCALES.shape:
            raise ValueError(
                f"feature scales must have shape {self.DEFAULT_FEATURE_SCALES.shape}"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("feature scales must be finite and positive")

        self.episodes = tuple(unique)
        self.feature_scales = scales
        self.features = np.stack(
            [_pick_bucket_feature(episode.initial_state) for episode in unique]
        )

    @classmethod
    def from_directory(
        cls,
        source_dir: str | Path,
        *,
        expected_action_dim: int = 12,
    ) -> "PickBucketTrajectoryLibrary":
        source_dir = Path(source_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"trajectory library not found: {source_dir}")

        def natural_key(path: Path) -> tuple[int, str]:
            match = re.search(r"_demo_(\d+)_", path.name)
            return (int(match.group(1)) if match else 2**31 - 1, path.name)

        paths = sorted(
            [
                path
                for path in source_dir.iterdir()
                if path.is_dir() and (path / "replay.zarr").is_dir()
            ],
            key=natural_key,
        )
        return cls(
            [load_replay_episode(path) for path in paths],
            expected_action_dim=expected_action_dim,
        )

    def select(
        self,
        state: np.ndarray,
        *,
        max_candidates: int = 5,
    ) -> tuple[RetargetCandidate, ...]:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.size < MINIMUM_PICK_BUCKET_STATE_DIM:
            raise ValueError(
                f"pick-bucket state needs {MINIMUM_PICK_BUCKET_STATE_DIM} values"
            )
        return self.select_scene(
            state[BUCKET_SLICE],
            state[FOOD_SLICE],
            max_candidates=max_candidates,
        )

    def select_scene(
        self,
        bucket_pose7: np.ndarray,
        food_pose7: np.ndarray,
        *,
        max_candidates: int = 5,
    ) -> tuple[RetargetCandidate, ...]:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        query = _pick_bucket_feature_from_poses(bucket_pose7, food_pose7)
        distances = np.linalg.norm(
            (self.features - query[None, :]) / self.feature_scales[None, :],
            axis=1,
        )
        count = min(int(max_candidates), len(self.episodes))
        indices = np.argsort(distances, kind="stable")[:count]
        return tuple(
            RetargetCandidate(
                episode=self.episodes[int(index)],
                distance=float(distances[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        )


class ObjectFrameRetargetSkill:
    """Retarget a successful native-delta trajectory to the current bucket."""

    name = "grasp_lift.object_frame_retarget"

    def __init__(
        self,
        candidate: RetargetCandidate,
        *,
        translation_feedback_gain: float = 0.35,
        rotation_feedback_gain: float = 0.25,
        grip_settle_steps: int = 5,
        tail_lift_steps: int = 12,
        tail_lift_step_m: float = 0.01,
        tail_lift_distance_m: float = 0.08,
        tail_feedback_gain: float = 0.75,
        top_hold_steps: int = 10,
    ) -> None:
        self.candidate = candidate
        self.translation_feedback_gain = float(translation_feedback_gain)
        self.rotation_feedback_gain = float(rotation_feedback_gain)
        self.grip_settle_steps = int(grip_settle_steps)
        self.tail_lift_steps = int(tail_lift_steps)
        self.tail_lift_step_m = float(tail_lift_step_m)
        self.tail_lift_distance_m = float(tail_lift_distance_m)
        self.tail_feedback_gain = float(tail_feedback_gain)
        self.top_hold_steps = int(top_hold_steps)
        if self.translation_feedback_gain < 0.0 or self.rotation_feedback_gain < 0.0:
            raise ValueError("feedback gains must be non-negative")
        if self.grip_settle_steps < 0:
            raise ValueError("grip_settle_steps must be non-negative")
        if self.tail_lift_steps < 0:
            raise ValueError("tail_lift_steps must be non-negative")
        if self.tail_lift_step_m <= 0.0:
            raise ValueError("tail_lift_step_m must be positive")
        if self.tail_lift_distance_m <= 0.0:
            raise ValueError("tail_lift_distance_m must be positive")
        if self.tail_feedback_gain <= 0.0:
            raise ValueError("tail_feedback_gain must be positive")
        if self.top_hold_steps < 0:
            raise ValueError("top_hold_steps must be non-negative")

    def _enabled_phases(self):
        return frozenset({"source", "settle", "lift", "hold"})

    def execute(
        self,
        context: SkillContext,
        parameters: Mapping[str, Any],
    ) -> SkillResult:
        started = time.perf_counter()
        episode = self.candidate.episode
        planning_context = (
            context.trace.span(
                "planning.object_frame_retarget",
                category="planning",
                start_step=context.total_steps,
                inputs={
                    "candidate_rank": self.candidate.rank,
                    "candidate_distance": self.candidate.distance,
                    "source_demo": str(episode.source),
                    "target_pose_source": parameters.get(
                        "target_pose_source", "perception"
                    ),
                },
            )
            if context.trace is not None
            else nullcontext()
        )
        with planning_context as planning_span:
            target_bucket_value = parameters.get("target_bucket_pose7")
            if target_bucket_value is None:
                raise ValueError(
                    "target_bucket_pose7 is required; object-frame retargeting "
                    "must receive an explicit perception result"
                )
            target_bucket_pose = np.asarray(
                target_bucket_value, dtype=np.float64
            ).reshape(-1)
            target_pose_source = str(
                parameters.get("target_pose_source", "perception")
            )
            source_bucket = _pose7_to_matrix(episode.initial_state[BUCKET_SLICE])
            target_bucket = _pose7_to_matrix(target_bucket_pose)
            source_to_target = target_bucket @ np.linalg.inv(source_bucket)
            rotation_delta = source_to_target[:3, :3]
            raw_env = context.env.unwrapped
            position_limit = float(
                getattr(raw_env, "_delta_position_limit", 0.02)
            )
            rotation_limit = float(
                getattr(raw_env, "_delta_rotation_limit", 0.03)
            )
            if planning_span is not None:
                planning_span.succeed(
                    end_step=context.total_steps,
                    outputs={
                        "target_bucket_pose7": target_bucket_pose,
                        "translation_delta_m": source_to_target[:3, 3],
                        "position_limit_m": position_limit,
                        "rotation_limit_rad": rotation_limit,
                    },
                )

        start_steps = context.total_steps
        phases = self._enabled_phases()
        source_horizon = len(episode.actions) if "source" in phases else 0
        settle_horizon = self.grip_settle_steps if "settle" in phases else 0
        lift_horizon = self.tail_lift_steps if "lift" in phases else 0
        hold_horizon = self.top_hold_steps if "hold" in phases else 0
        terminal_reference = np.asarray(context.observation["state"])[TCP_SLICE].copy()
        source_steps = 0
        settle_steps = 0
        tail_steps = 0
        hold_steps = 0
        source_context = (
            context.trace.span(
                "control.retargeted_source_horizon",
                category="control",
                start_step=context.total_steps,
                inputs={
                    "expected_steps": source_horizon,
                    "translation_feedback_gain": self.translation_feedback_gain,
                    "rotation_feedback_gain": self.rotation_feedback_gain,
                },
            )
            if context.trace is not None
            else nullcontext()
        )
        with source_context as source_span:
            for index, source_action in enumerate(episode.actions[:source_horizon]):
                if context.terminated or context.truncated:
                    break
                source_tcp = _pose7_to_matrix(episode.states[index, TCP_SLICE])
                desired_tcp = _matrix_to_pose7(source_to_target @ source_tcp)
                terminal_reference = desired_tcp.copy()
                current_tcp = np.asarray(context.observation["state"])[TCP_SLICE]
                error = _pose_error6(current_tcp, desired_tcp)
                feedforward = np.concatenate(
                    [
                        rotation_delta @ source_action[:3],
                        rotation_delta @ source_action[3:6],
                    ]
                )
                command = feedforward + np.concatenate(
                    [
                        self.translation_feedback_gain * error[:3],
                        self.rotation_feedback_gain * error[3:6],
                    ]
                )
                arm_action = np.concatenate(
                    [
                        np.clip(command[:3], -position_limit, position_limit),
                        np.clip(command[3:6], -rotation_limit, rotation_limit),
                    ]
                )
                action = np.concatenate([arm_action, source_action[6:]])
                _observation, _reward, terminated, truncated, _info = context.step(
                    action
                )
                source_steps += 1
                if terminated or truncated:
                    break
            if source_span is not None:
                source_outputs = {
                    "executed_steps": source_steps,
                    "terminated": context.terminated,
                    "truncated": context.truncated,
                    "final_tcp_pose7": np.asarray(context.observation["state"])[
                        TCP_SLICE
                    ],
                }
                if source_horizon == 0:
                    source_span.skip(end_step=context.total_steps, reason="source phase not selected")
                elif source_steps == source_horizon:
                    source_span.succeed(
                        end_step=context.total_steps,
                        outputs=source_outputs,
                    )
                else:
                    source_span.fail(
                        end_step=context.total_steps,
                        failure_code="source_horizon_interrupted",
                        reason="retargeted source trajectory ended early",
                        outputs=source_outputs,
                    )

        final_hand = episode.actions[-1, 6:].copy()
        zero_arm = np.zeros(6, dtype=np.float64)
        settle_context = (
            context.trace.span(
                "control.grip_settle",
                category="control",
                start_step=context.total_steps,
                inputs={"expected_steps": settle_horizon},
            )
            if context.trace is not None
            else nullcontext()
        )
        with settle_context as settle_span:
            while (
                not context.terminated
                and not context.truncated
                and settle_steps < settle_horizon
            ):
                context.step(np.concatenate([zero_arm, final_hand]))
                settle_steps += 1
            if settle_span is not None:
                settle_outputs = {
                    "executed_steps": settle_steps,
                    "terminated": context.terminated,
                    "truncated": context.truncated,
                }
                if settle_horizon == 0:
                    settle_span.skip(
                        end_step=context.total_steps,
                        reason="grip settling is disabled by configuration",
                        outputs=settle_outputs,
                    )
                elif settle_steps == settle_horizon:
                    settle_span.succeed(
                        end_step=context.total_steps,
                        outputs=settle_outputs,
                    )
                else:
                    settle_span.fail(
                        end_step=context.total_steps,
                        failure_code="grip_settle_interrupted",
                        reason="grip settling ended before its fixed horizon",
                        outputs=settle_outputs,
                    )

        lift_target_pose = np.asarray(context.observation["state"])[TCP_SLICE].copy()
        lift_target_pose[2] += self.tail_lift_distance_m
        if lift_horizon or hold_horizon:
            terminal_reference = lift_target_pose.copy()

        def terminal_action() -> np.ndarray:
            current_tcp = np.asarray(context.observation["state"])[TCP_SLICE]
            error = _pose_error6(current_tcp, lift_target_pose)
            command = np.concatenate(
                [
                    self.tail_feedback_gain * error[:3],
                    self.rotation_feedback_gain * error[3:6],
                ]
            )
            translation = np.clip(
                command[:3],
                -min(position_limit, self.tail_lift_step_m),
                min(position_limit, self.tail_lift_step_m),
            )
            rotation = np.clip(
                command[3:6],
                -rotation_limit,
                rotation_limit,
            )
            return np.concatenate([translation, rotation, final_hand])

        lift_context = (
            context.trace.span(
                "control.terminal_lift",
                category="control",
                start_step=context.total_steps,
                inputs={
                    "expected_steps": lift_horizon,
                    "target_distance_m": self.tail_lift_distance_m,
                    "translation_step_limit_m": self.tail_lift_step_m,
                    "feedback_gain": self.tail_feedback_gain,
                },
            )
            if context.trace is not None
            else nullcontext()
        )
        with lift_context as lift_span:
            while (
                not context.terminated
                and not context.truncated
                and tail_steps < lift_horizon
            ):
                context.step(terminal_action())
                tail_steps += 1
            if lift_span is not None:
                lift_outputs = {
                    "executed_steps": tail_steps,
                    "terminated": context.terminated,
                    "truncated": context.truncated,
                    "final_tcp_pose7": np.asarray(context.observation["state"])[
                        TCP_SLICE
                    ],
                }
                if lift_horizon == 0:
                    lift_span.skip(
                        end_step=context.total_steps,
                        reason="terminal lift is disabled by configuration",
                        outputs=lift_outputs,
                    )
                elif tail_steps == lift_horizon:
                    lift_span.succeed(
                        end_step=context.total_steps,
                        outputs=lift_outputs,
                    )
                else:
                    lift_span.fail(
                        end_step=context.total_steps,
                        failure_code="lift_interrupted",
                        reason="terminal lift ended before its fixed horizon",
                        outputs=lift_outputs,
                    )

        hold_context = (
            context.trace.span(
                "control.top_hold",
                category="control",
                start_step=context.total_steps,
                inputs={"expected_steps": hold_horizon},
            )
            if context.trace is not None
            else nullcontext()
        )
        with hold_context as hold_span:
            while (
                not context.terminated
                and not context.truncated
                and hold_steps < hold_horizon
            ):
                context.step(terminal_action())
                hold_steps += 1
            if hold_span is not None:
                hold_outputs = {
                    "executed_steps": hold_steps,
                    "terminated": context.terminated,
                    "truncated": context.truncated,
                    "final_tcp_pose7": np.asarray(context.observation["state"])[
                        TCP_SLICE
                    ],
                }
                if hold_horizon == 0:
                    hold_span.skip(
                        end_step=context.total_steps,
                        reason="top hold is disabled by configuration",
                        outputs=hold_outputs,
                    )
                elif hold_steps == hold_horizon:
                    hold_span.succeed(
                        end_step=context.total_steps,
                        outputs=hold_outputs,
                    )
                else:
                    hold_span.fail(
                        end_step=context.total_steps,
                        failure_code="hold_interrupted",
                        reason="top hold ended before its fixed horizon",
                        outputs=hold_outputs,
                    )

        steps = context.total_steps - start_steps
        execution_completed = (
            source_steps == source_horizon
            and settle_steps == settle_horizon
            and tail_steps == lift_horizon
            and hold_steps == hold_horizon
            and not context.terminated
            and not context.truncated
        )
        metrics = {
            "backend": "object_frame_retarget",
            "vla_used": False,
            "source_demo": str(episode.source),
            "candidate_rank": self.candidate.rank,
            "candidate_distance": self.candidate.distance,
            "source_steps_executed": source_steps,
            "grip_settle_steps_executed": settle_steps,
            "tail_lift_steps_executed": tail_steps,
            "tail_lift_distance_m": self.tail_lift_distance_m,
            "tail_feedback_gain": self.tail_feedback_gain,
            "top_hold_steps_executed": hold_steps,
            "execution_completed": execution_completed,
            "terminal_tcp_reference7": terminal_reference,
            "translation_feedback_gain": self.translation_feedback_gain,
            "rotation_feedback_gain": self.rotation_feedback_gain,
            "target_pose_source": target_pose_source,
        }
        if execution_completed:
            return SkillResult(
                skill_name=self.name,
                status=SkillStatus.SUCCEEDED,
                steps=steps,
                duration_seconds=time.perf_counter() - started,
                metrics=metrics,
            )
        return SkillResult(
            skill_name=self.name,
            status=SkillStatus.FAILED,
            steps=steps,
            duration_seconds=time.perf_counter() - started,
            failure_reason="retargeted trajectory ended before its fixed horizon",
            metrics=metrics,
        )
