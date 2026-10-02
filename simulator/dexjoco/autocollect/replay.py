"""Recorded-action skills used as the first deterministic execution backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import zarr

from .core import SkillContext, SkillResult, SkillStatus


@dataclass(frozen=True)
class ReplayEpisode:
    source: Path
    actions: np.ndarray
    states: np.ndarray
    timestamps: np.ndarray | None
    metadata: Mapping[str, Any]

    @property
    def initial_state(self) -> np.ndarray:
        return self.states[0]


def _resolve_zarr_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir() and path.name != "replay.zarr":
        path = path / "replay.zarr"
    if not path.is_dir():
        raise FileNotFoundError(f"replay.zarr directory not found: {path}")
    return path


def load_replay_episode(path: str | Path) -> ReplayEpisode:
    """Load and validate one single-episode DexJoCo Zarr recording."""

    zarr_path = _resolve_zarr_path(path)
    root = zarr.open_group(str(zarr_path), mode="r")
    if "data" not in root or "action" not in root["data"]:
        raise ValueError(f"{zarr_path} does not contain data/action")
    if "state" not in root["data"]:
        raise ValueError(f"{zarr_path} does not contain data/state")

    actions = np.asarray(root["data/action"], dtype=np.float64)
    states = np.asarray(root["data/state"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"actions must have shape [T,A], got {actions.shape}")
    if states.ndim != 2 or states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"state/action length mismatch: state={states.shape}, action={actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("recorded actions contain NaN or Inf")
    if not np.all(np.isfinite(states)):
        raise ValueError("recorded states contain NaN or Inf")

    timestamps = None
    if "timestamp" in root["data"]:
        timestamps = np.asarray(root["data/timestamp"], dtype=np.float64)
        if timestamps.shape != (actions.shape[0],):
            raise ValueError(
                f"timestamp shape must be {(actions.shape[0],)}, got {timestamps.shape}"
            )
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("recorded timestamps must be strictly increasing")

    metadata = root.attrs.get("dexjoco_collection_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return ReplayEpisode(
        source=zarr_path.parent,
        actions=actions,
        states=states,
        timestamps=timestamps,
        metadata=dict(metadata),
    )


class ReplayActionSkill:
    """Execute a validated action sequence until task success or exhaustion."""

    def __init__(
        self,
        episode: ReplayEpisode,
        *,
        name: str = "grasp_lift.replay",
        expected_action_dim: int | None = None,
        success_info_key: str = "succeed",
    ) -> None:
        self.name = name
        self.episode = episode
        self.expected_action_dim = expected_action_dim
        self.success_info_key = success_info_key

    def execute(
        self,
        context: SkillContext,
        parameters: Mapping[str, Any],
    ) -> SkillResult:
        max_steps_value = parameters.get("max_steps")
        max_steps = (
            len(self.episode.actions)
            if max_steps_value is None
            else int(max_steps_value)
        )
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        actions = self.episode.actions[:max_steps]
        if self.expected_action_dim is not None and actions.shape[1] != int(
            self.expected_action_dim
        ):
            raise ValueError(
                f"recorded action dim {actions.shape[1]} does not match task dim "
                f"{self.expected_action_dim}"
            )

        started = time.perf_counter()
        start_steps = context.total_steps
        succeeded = False
        terminal_reason = "recorded action sequence exhausted before success"
        for action in actions:
            if context.terminated or context.truncated:
                terminal_reason = "environment ended before the next recorded action"
                break
            _observation, _reward, terminated, truncated, info = context.step(action)
            if bool(info.get(self.success_info_key, False)):
                succeeded = True
                break
            if terminated:
                terminal_reason = "environment terminated without task success"
                break
            if truncated:
                terminal_reason = "environment truncated without task success"
                break

        steps = context.total_steps - start_steps
        metrics = {
            "actions_available": int(len(self.episode.actions)),
            "actions_executed": steps,
            "source_demo": str(self.episode.source),
            "vla_used": False,
            "backend": "recorded_native_action",
        }
        if succeeded:
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
            failure_reason=terminal_reason,
            metrics=metrics,
        )


class NativeDeltaRetreatSkill:
    """Best-effort upward retreat for a failed single-arm native-delta skill."""

    def __init__(
        self,
        *,
        arm_action_dim: int,
        hand_action_dim: int,
        name: str = "recover.retreat",
        default_distance_m: float = 0.08,
    ) -> None:
        if arm_action_dim != 6:
            raise ValueError("NativeDeltaRetreatSkill requires a 6-D arm delta")
        if hand_action_dim <= 0:
            raise ValueError("hand_action_dim must be positive")
        self.name = name
        self.arm_action_dim = arm_action_dim
        self.hand_action_dim = hand_action_dim
        self.default_distance_m = float(default_distance_m)

    def _current_hand_target(self, context: SkillContext) -> np.ndarray:
        raw_env = context.env.unwrapped
        joint_ids = np.asarray(raw_env._allegro_dof_ids, dtype=int).reshape(-1)
        if joint_ids.size < self.hand_action_dim:
            raise ValueError("environment exposes fewer hand joints than expected")
        return np.asarray(
            raw_env._data.qpos[joint_ids[: self.hand_action_dim]],
            dtype=np.float64,
        ).copy()

    def execute(
        self,
        context: SkillContext,
        parameters: Mapping[str, Any],
    ) -> SkillResult:
        started = time.perf_counter()
        if context.terminated or context.truncated:
            return SkillResult(
                skill_name=self.name,
                status=SkillStatus.ABORTED,
                steps=0,
                duration_seconds=time.perf_counter() - started,
                failure_reason="environment already ended; reset is required",
            )

        distance = float(parameters.get("distance_m", self.default_distance_m))
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("distance_m must be finite and positive")
        raw_env = context.env.unwrapped
        per_step_limit = float(getattr(raw_env, "_delta_position_limit", 0.02))
        requested_step = float(parameters.get("step_m", per_step_limit))
        step_m = min(requested_step, per_step_limit)
        if not np.isfinite(step_m) or step_m <= 0.0:
            raise ValueError("step_m must be finite and positive")

        hand = self._current_hand_target(context)
        remaining = distance
        start_steps = context.total_steps
        while remaining > 1e-9:
            dz = min(step_m, remaining)
            action = np.concatenate(
                [np.asarray([0.0, 0.0, dz, 0.0, 0.0, 0.0]), hand]
            )
            context.step(action)
            remaining -= dz
            if context.terminated or context.truncated:
                break

        steps = context.total_steps - start_steps
        completed = remaining <= 1e-9
        metrics = {
            "requested_distance_m": distance,
            "commanded_distance_m": distance - max(remaining, 0.0),
            "vla_used": False,
            "backend": "native_delta_retreat",
        }
        if completed:
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
            failure_reason="environment ended during retreat",
            metrics=metrics,
        )
