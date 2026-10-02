"""Two-frame retrieval and trajectory retargeting for hammer-nail."""

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
from .retarget import TCP_SLICE, _matrix_to_pose7, _pose7_to_matrix, _pose_error6


HAMMER_SLICE = slice(13, 20)
NAIL_SLICE = slice(20, 27)
MINIMUM_HAMMER_NAIL_STATE_DIM = 28


def make_hammer_frame_retarget_manifest(
    *,
    task_id: str,
    expected_action_dim: int,
) -> SkillManifest:
    """Describe the non-VLA hammer skill exposed to the structured Agent."""

    return SkillManifest(
        name="hammer_nail.two_frame_retarget",
        version="1",
        description=(
            "Retrieve a successful native-delta hammer trajectory, follow the "
            "perceived hammer frame during acquisition, then blend into the "
            "perceived nail frame for transport and striking."
        ),
        supported_tasks=(str(task_id),),
        parameters=(
            SkillParameterSpec(
                name="target_hammer_pose7",
                kind=ParameterKind.ARRAY,
                description=(
                    "Perceived reset hammer pose [x,y,z,qw,qx,qy,qz]."
                ),
                shape=(7,),
                finite=True,
            ),
            SkillParameterSpec(
                name="target_nail_pose7",
                kind=ParameterKind.ARRAY,
                description=(
                    "Perceived reset nail pose [x,y,z,qw,qx,qy,qz]."
                ),
                shape=(7,),
                finite=True,
            ),
            SkillParameterSpec(
                name="target_pose_source",
                kind=ParameterKind.STRING,
                description="Perception backend that produced both target poses.",
            ),
        ),
        required_observations=("state",),
        action_dimension=int(expected_action_dim),
        required_perception=("hammer_pose7", "nail_pose7"),
        preconditions=(
            "a ranked successful hammer demonstration has been selected",
            "hammer and nail poses come from one explicit perception result",
            "the hammer is initially resting in the calibrated reset orientation",
        ),
        expected_effects=(
            "the hand acquires the hammer handle",
            "the hammer face is transported above the perceived nail",
            "the source strike motion is replayed relative to the nail frame",
        ),
        safety_constraints=(
            f"every action has exactly {int(expected_action_dim)} finite values",
            "translation and rotation commands are clipped to native-delta limits",
            "the two object frames are blended over a fixed finite horizon",
            "nail depth is used only by the independent post-rollout verifier",
        ),
        implementation=(
            "dexjoco.autocollect.hammer_retarget.TwoFrameHammerRetargetSkill"
        ),
        controller_family="two_object_frame_trajectory_retarget",
        vla_used=False,
        tags=("hammer", "nail", "inspire", "no_vla", "random_scene"),
    )


def _hammer_nail_feature_from_poses(
    hammer_pose7: np.ndarray,
    nail_pose7: np.ndarray,
) -> np.ndarray:
    hammer = np.asarray(hammer_pose7, dtype=np.float64).reshape(-1)
    nail = np.asarray(nail_pose7, dtype=np.float64).reshape(-1)
    if (
        hammer.shape != (7,)
        or nail.shape != (7,)
        or not np.all(np.isfinite(hammer))
        or not np.all(np.isfinite(nail))
    ):
        raise ValueError("hammer_pose7 and nail_pose7 must be finite 7-vectors")
    hammer_rotation = Rotation.from_quat(hammer[3:7], scalar_first=True)
    rotation_matrix = hammer_rotation.as_matrix()
    handle_axis = rotation_matrix[:, 2]
    handle_yaw = float(np.arctan2(handle_axis[1], handle_axis[0]))
    nail_in_hammer = hammer_rotation.inv().apply(nail[:3] - hammer[:3])
    return np.concatenate(
        [
            hammer[:3],
            [np.cos(handle_yaw), np.sin(handle_yaw)],
            nail_in_hammer,
        ]
    )


@dataclass(frozen=True)
class HammerRetargetCandidate:
    episode: ReplayEpisode
    distance: float
    rank: int


class HammerNailTrajectoryLibrary:
    """Validated successful demonstrations indexed by both object frames."""

    DEFAULT_FEATURE_SCALES = np.asarray(
        [0.05, 0.05, 0.05, 0.35, 0.35, 0.08, 0.08, 0.05],
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
            if episode.actions.shape[1] != int(expected_action_dim):
                raise ValueError(
                    f"{episode.source} has action dim {episode.actions.shape[1]}, "
                    f"expected {expected_action_dim}"
                )
            if episode.states.shape[1] < MINIMUM_HAMMER_NAIL_STATE_DIM:
                raise ValueError(
                    f"{episode.source} has state dim {episode.states.shape[1]}, "
                    f"expected at least {MINIMUM_HAMMER_NAIL_STATE_DIM}"
                )
            initial = np.ascontiguousarray(episode.initial_state)
            fingerprint = (initial.dtype.str, initial.shape, initial.tobytes())
            if fingerprint in seen_initial_states:
                continue
            seen_initial_states.add(fingerprint)
            unique.append(episode)
        if not unique:
            raise ValueError("hammer-nail trajectory library is empty")

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
            [
                _hammer_nail_feature_from_poses(
                    episode.initial_state[HAMMER_SLICE],
                    episode.initial_state[NAIL_SLICE],
                )
                for episode in unique
            ]
        )

    @classmethod
    def from_directory(
        cls,
        source_dir: str | Path,
        *,
        expected_action_dim: int = 12,
    ) -> "HammerNailTrajectoryLibrary":
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

    def select_scene(
        self,
        hammer_pose7: np.ndarray,
        nail_pose7: np.ndarray,
        *,
        max_candidates: int = 5,
    ) -> tuple[HammerRetargetCandidate, ...]:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        query = _hammer_nail_feature_from_poses(hammer_pose7, nail_pose7)
        distances = np.linalg.norm(
            (self.features - query[None, :]) / self.feature_scales[None, :],
            axis=1,
        )
        count = min(int(max_candidates), len(self.episodes))
        indices = np.argsort(distances, kind="stable")[:count]
        return tuple(
            HammerRetargetCandidate(
                episode=self.episodes[int(index)],
                distance=float(distances[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        )


@dataclass(frozen=True)
class HammerPhasePlan:
    closest_approach_step: int
    transition_start_step: int
    transition_end_step: int
    total_steps: int

    def __post_init__(self) -> None:
        values = (
            self.closest_approach_step,
            self.transition_start_step,
            self.transition_end_step,
            self.total_steps,
        )
        if any(int(value) != value for value in values):
            raise ValueError("phase plan indices must be integers")
        if not (
            0 <= self.closest_approach_step < self.total_steps
            and self.closest_approach_step <= self.transition_start_step
            and self.transition_start_step < self.transition_end_step
            and self.transition_end_step <= self.total_steps
        ):
            raise ValueError("invalid hammer phase ordering")

    def to_dict(self) -> dict[str, int]:
        return {
            "closest_approach_step": self.closest_approach_step,
            "transition_start_step": self.transition_start_step,
            "transition_end_step": self.transition_end_step,
            "total_steps": self.total_steps,
        }


def infer_hammer_phase_plan(
    episode: ReplayEpisode,
    *,
    grasp_settle_steps: int = 8,
    frame_blend_steps: int = 24,
) -> HammerPhasePlan:
    """Infer a conservative hammer-to-nail frame switch from source TCP motion.

    The recorded object-pose fields are reset poses and intentionally remain
    constant after grasp.  Therefore phase detection uses the source TCP's
    closest approach to the initial hammer, followed by a fixed settling and
    blending window.  No target-scene or verifier information enters it.
    """

    total_steps = int(len(episode.actions))
    if total_steps < 3:
        raise ValueError("hammer source trajectory needs at least three steps")
    grasp_settle_steps = int(grasp_settle_steps)
    frame_blend_steps = int(frame_blend_steps)
    if grasp_settle_steps < 0:
        raise ValueError("grasp_settle_steps must be non-negative")
    if frame_blend_steps <= 0:
        raise ValueError("frame_blend_steps must be positive")
    tcp_positions = np.asarray(episode.states[:, TCP_SLICE.start : 3])
    hammer_position = np.asarray(
        episode.initial_state[HAMMER_SLICE.start : HAMMER_SLICE.start + 3]
    )
    search_end = max(1, min(total_steps, int(np.ceil(total_steps * 0.8))))
    distances = np.linalg.norm(
        tcp_positions[:search_end] - hammer_position[None, :],
        axis=1,
    )
    closest = int(np.argmin(distances))
    transition_start = min(total_steps - 1, closest + grasp_settle_steps)
    transition_end = min(total_steps, transition_start + frame_blend_steps)
    if transition_end <= transition_start:
        transition_start = total_steps - 1
        transition_end = total_steps
    return HammerPhasePlan(
        closest_approach_step=closest,
        transition_start_step=transition_start,
        transition_end_step=transition_end,
        total_steps=total_steps,
    )


def _interpolate_rigid_transform(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate translation and shortest-path rotation between transforms."""

    first = np.asarray(first, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second, dtype=np.float64).reshape(4, 4)
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("transform interpolation alpha must be within [0, 1]")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
    relative_rotvec = Rotation.from_matrix(
        second[:3, :3] @ first[:3, :3].T
    ).as_rotvec()
    result[:3, :3] = (
        Rotation.from_rotvec(alpha * relative_rotvec).as_matrix()
        @ first[:3, :3]
    )
    return result


class TwoFrameHammerRetargetSkill:
    """Acquire relative to the hammer, then strike relative to the nail."""

    name = "hammer_nail.two_frame_retarget"

    def __init__(
        self,
        candidate: HammerRetargetCandidate,
        *,
        translation_feedback_gain: float = 0.35,
        rotation_feedback_gain: float = 0.25,
        grasp_settle_steps: int = 8,
        frame_blend_steps: int = 24,
    ) -> None:
        self.candidate = candidate
        self.translation_feedback_gain = float(translation_feedback_gain)
        self.rotation_feedback_gain = float(rotation_feedback_gain)
        self.grasp_settle_steps = int(grasp_settle_steps)
        self.frame_blend_steps = int(frame_blend_steps)
        if (
            not np.isfinite(self.translation_feedback_gain)
            or self.translation_feedback_gain < 0.0
            or not np.isfinite(self.rotation_feedback_gain)
            or self.rotation_feedback_gain < 0.0
        ):
            raise ValueError("feedback gains must be finite and non-negative")
        if self.grasp_settle_steps < 0:
            raise ValueError("grasp_settle_steps must be non-negative")
        if self.frame_blend_steps <= 0:
            raise ValueError("frame_blend_steps must be positive")

    def _phase_ranges(self, plan: HammerPhasePlan):
        return (
            ("control.acquire_hammer_frame", 0, plan.transition_start_step, "hammer"),
            ("control.blend_hammer_to_nail_frame", plan.transition_start_step,
             plan.transition_end_step, "blend"),
            ("control.transport_and_strike_nail_frame", plan.transition_end_step,
             plan.total_steps, "nail"),
        )

    def execute(
        self,
        context: SkillContext,
        parameters: Mapping[str, Any],
    ) -> SkillResult:
        started = time.perf_counter()
        hammer_value = parameters.get("target_hammer_pose7")
        nail_value = parameters.get("target_nail_pose7")
        if hammer_value is None or nail_value is None:
            raise ValueError(
                "target_hammer_pose7 and target_nail_pose7 are required; "
                "hammer retargeting must receive an explicit perception result"
            )
        target_hammer_pose = np.asarray(hammer_value, dtype=np.float64).reshape(-1)
        target_nail_pose = np.asarray(nail_value, dtype=np.float64).reshape(-1)
        target_pose_source = str(parameters.get("target_pose_source", "perception"))
        episode = self.candidate.episode
        source_hammer = _pose7_to_matrix(episode.initial_state[HAMMER_SLICE])
        source_nail = _pose7_to_matrix(episode.initial_state[NAIL_SLICE])
        hammer_transform = _pose7_to_matrix(target_hammer_pose) @ np.linalg.inv(
            source_hammer
        )
        nail_transform = _pose7_to_matrix(target_nail_pose) @ np.linalg.inv(
            source_nail
        )
        plan = infer_hammer_phase_plan(
            episode,
            grasp_settle_steps=self.grasp_settle_steps,
            frame_blend_steps=self.frame_blend_steps,
        )
        raw_env = context.env.unwrapped
        position_limit = float(getattr(raw_env, "_delta_position_limit", 0.02))
        rotation_limit = float(getattr(raw_env, "_delta_rotation_limit", 0.03))

        planning_context = (
            context.trace.span(
                "planning.hammer_two_frame_retarget",
                category="planning",
                start_step=context.total_steps,
                inputs={
                    "candidate_rank": self.candidate.rank,
                    "candidate_distance": self.candidate.distance,
                    "source_demo": str(episode.source),
                    "target_pose_source": target_pose_source,
                },
            )
            if context.trace is not None
            else nullcontext()
        )
        with planning_context as planning_span:
            if planning_span is not None:
                planning_span.succeed(
                    end_step=context.total_steps,
                    outputs={
                        "target_hammer_pose7": target_hammer_pose,
                        "target_nail_pose7": target_nail_pose,
                        "hammer_frame_translation_delta_m": hammer_transform[:3, 3],
                        "nail_frame_translation_delta_m": nail_transform[:3, 3],
                        "phase_plan": plan.to_dict(),
                        "position_limit_m": position_limit,
                        "rotation_limit_rad": rotation_limit,
                    },
                )

        ranges = self._phase_ranges(plan)
        start_steps = context.total_steps
        terminal_reference = None
        phase_counts: dict[str, int] = {}
        for phase_name, begin, end, mode in ranges:
            phase_context = (
                context.trace.span(
                    phase_name,
                    category="control",
                    start_step=context.total_steps,
                    inputs={
                        "source_begin_step": begin,
                        "source_end_step": end,
                        "frame_mode": mode,
                    },
                )
                if context.trace is not None
                else nullcontext()
            )
            count = 0
            with phase_context as phase_span:
                for index in range(begin, end):
                    if context.terminated or context.truncated:
                        break
                    if mode == "hammer":
                        transform = hammer_transform
                    elif mode == "nail":
                        transform = nail_transform
                    else:
                        denominator = max(1, end - begin - 1)
                        alpha = float(index - begin) / float(denominator)
                        transform = _interpolate_rigid_transform(
                            hammer_transform,
                            nail_transform,
                            alpha,
                        )
                    source_tcp = _pose7_to_matrix(episode.states[index, TCP_SLICE])
                    desired_tcp = _matrix_to_pose7(transform @ source_tcp)
                    terminal_reference = desired_tcp.copy()
                    current_tcp = np.asarray(
                        context.observation["state"], dtype=np.float64
                    )[TCP_SLICE]
                    error = _pose_error6(current_tcp, desired_tcp)
                    source_action = episode.actions[index]
                    transformed_feedforward = np.concatenate(
                        [
                            transform[:3, :3] @ source_action[:3],
                            transform[:3, :3] @ source_action[3:6],
                        ]
                    )
                    command = transformed_feedforward + np.concatenate(
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
                    context.step(np.concatenate([arm_action, source_action[6:]]))
                    count += 1
                phase_counts[mode] = count
                outputs = {
                    "executed_steps": count,
                    "expected_steps": end - begin,
                    "terminated": context.terminated,
                    "truncated": context.truncated,
                }
                if phase_span is not None:
                    if end == begin:
                        phase_span.skip(
                            end_step=context.total_steps,
                            reason="phase has an empty source range",
                            outputs=outputs,
                        )
                    elif count == end - begin:
                        phase_span.succeed(
                            end_step=context.total_steps,
                            outputs=outputs,
                        )
                    else:
                        phase_span.fail(
                            end_step=context.total_steps,
                            failure_code="hammer_phase_interrupted",
                            reason=f"{mode} phase ended before its fixed horizon",
                            outputs=outputs,
                        )

        steps = context.total_steps - start_steps
        requested_steps = sum(end - begin for _, begin, end, _ in ranges)
        execution_completed = steps == requested_steps
        metrics = {
            "backend": "two_object_frame_retarget",
            "vla_used": False,
            "source_demo": str(episode.source),
            "candidate_rank": self.candidate.rank,
            "candidate_distance": self.candidate.distance,
            "source_steps_available": len(episode.actions),
            "source_steps_requested": requested_steps,
            "source_steps_executed": steps,
            "phase_plan": plan.to_dict(),
            "phase_steps_executed": phase_counts,
            "translation_feedback_gain": self.translation_feedback_gain,
            "rotation_feedback_gain": self.rotation_feedback_gain,
            "grasp_settle_steps": self.grasp_settle_steps,
            "frame_blend_steps": self.frame_blend_steps,
            "target_pose_source": target_pose_source,
            "execution_completed": execution_completed,
            "terminal_tcp_reference7": terminal_reference,
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
            failure_reason="retargeted hammer trajectory ended before fixed horizon",
            metrics=metrics,
        )
