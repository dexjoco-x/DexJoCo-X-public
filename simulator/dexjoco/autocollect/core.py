"""Core interfaces shared by deterministic and learned atomic skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from .registry import SkillManifest
    from .trace import TraceRecorder


class SkillStatus(str, Enum):
    """Terminal status of one atomic-skill invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class ExecutionBudgetExceeded(RuntimeError):
    """Raised before an action when a declared program limit is exhausted."""


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/container values into JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True)
class SkillResult:
    """Structured result returned by every atomic skill.

    ``failure_reason`` is mandatory for non-successful results.  This keeps a
    future planner from having to infer why execution stopped from raw logs.
    """

    skill_name: str
    status: SkillStatus
    steps: int
    duration_seconds: float
    failure_reason: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_name:
            raise ValueError("skill_name must not be empty")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.status is SkillStatus.SUCCEEDED and self.failure_reason is not None:
            raise ValueError("a successful skill cannot have a failure_reason")
        if self.status is not SkillStatus.SUCCEEDED and not self.failure_reason:
            raise ValueError("a failed or aborted skill requires failure_reason")

    @property
    def success(self) -> bool:
        return self.status is SkillStatus.SUCCEEDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "status": self.status.value,
            "success": self.success,
            "steps": self.steps,
            "duration_seconds": self.duration_seconds,
            "failure_reason": self.failure_reason,
            "metrics": _json_safe(self.metrics),
        }


class AtomicSkill(Protocol):
    """Minimal interface implemented by every callable robot skill."""

    name: str

    def execute(
        self,
        context: "SkillContext",
        parameters: Mapping[str, Any],
    ) -> SkillResult:
        """Execute the skill once and return a terminal result."""


class LowDimEpisodeRecorder:
    """Record state/action transitions without rendering or storing videos."""

    def __init__(self, control_hz: float):
        control_hz = float(control_hz)
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive")
        self.control_hz = control_hz
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []

    @property
    def steps(self) -> int:
        return len(self._actions)

    def append(
        self,
        *,
        observation_before: Mapping[str, Any],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        state = np.asarray(observation_before["state"], dtype=np.float64).reshape(-1)
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(state)):
            raise ValueError("observation state contains NaN or Inf")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains NaN or Inf")
        if self._states and state.shape != self._states[0].shape:
            raise ValueError(
                f"state shape changed from {self._states[0].shape} to {state.shape}"
            )
        if self._actions and action_array.shape != self._actions[0].shape:
            raise ValueError(
                "action shape changed from "
                f"{self._actions[0].shape} to {action_array.shape}"
            )
        self._states.append(state.copy())
        self._actions.append(action_array.copy())
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))

    def episode_data(self) -> dict[str, np.ndarray]:
        if not self._actions:
            raise ValueError("cannot serialize an empty episode")
        steps = self.steps
        return {
            "state": np.stack(self._states, axis=0),
            "action": np.stack(self._actions, axis=0),
            "timestamp": np.arange(steps, dtype=np.float64) / self.control_hz,
            "reward": np.asarray(self._rewards, dtype=np.float32),
            "terminated": np.asarray(self._terminated, dtype=np.bool_),
            "truncated": np.asarray(self._truncated, dtype=np.bool_),
        }


@dataclass
class SkillContext:
    """Mutable episode state shared by sequential skills."""

    env: Any
    task_id: str
    config: Any
    observation: Mapping[str, Any]
    recorder: LowDimEpisodeRecorder | None = None
    trace: "TraceRecorder | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_info: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    total_steps: int = 0
    maximum_total_steps: int | None = None
    deadline_monotonic: float | None = None
    step_observer: Any = None  # Trusted read-only host callback; never planner code.

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Mapping[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the environment and update the shared episode state."""

        if (
            self.maximum_total_steps is not None
            and self.total_steps >= self.maximum_total_steps
        ):
            raise ExecutionBudgetExceeded(
                f"program action budget of {self.maximum_total_steps} steps is exhausted"
            )
        if (
            self.deadline_monotonic is not None
            and time.monotonic() >= self.deadline_monotonic
        ):
            raise ExecutionBudgetExceeded("program wall-time budget is exhausted")
        observation_before = self.observation
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.recorder is not None:
            self.recorder.append(
                observation_before=observation_before,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
        self.observation = observation
        self.last_info = dict(info)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.total_steps += 1
        if self.step_observer is not None:
            step = self.total_steps
            proprioception = np.asarray(self.observation["state"])[:13].copy()
            self.step_observer(self)
            if self.total_steps != step or not np.array_equal(np.asarray(self.observation["state"])[:13], proprioception):
                raise RuntimeError("step observer changed control state")
        return observation, float(reward), self.terminated, self.truncated, self.last_info


class SkillExecutor:
    """Whitelisted registry through which a future planner invokes skills."""

    def __init__(self, *, require_manifests: bool = False) -> None:
        self._skills: dict[str, AtomicSkill] = {}
        self._manifests: dict[str, "SkillManifest"] = {}
        self.require_manifests = bool(require_manifests)

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def manifest(self, skill_name: str) -> "SkillManifest | None":
        """Return the public contract for a registered skill, when present."""

        return self._manifests.get(skill_name)

    def catalog(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic, JSON-safe contracts for planner inspection."""

        return tuple(
            self._manifests[name].json_safe()
            for name in sorted(self._manifests)
        )

    def register(
        self,
        skill: AtomicSkill,
        *,
        manifest: "SkillManifest | None" = None,
    ) -> None:
        name = str(skill.name).strip()
        if not name:
            raise ValueError("skill name must not be empty")
        if name in self._skills:
            raise ValueError(f"skill {name!r} is already registered")
        if self.require_manifests and manifest is None:
            raise ValueError(f"skill {name!r} requires a manifest")
        if manifest is not None and manifest.name != name:
            raise ValueError(
                f"skill name {name!r} does not match manifest {manifest.name!r}"
            )
        self._skills[name] = skill
        if manifest is not None:
            self._manifests[name] = manifest

    def execute(
        self,
        skill_name: str,
        context: SkillContext,
        parameters: Mapping[str, Any] | None = None,
    ) -> SkillResult:
        if skill_name not in self._skills:
            choices = ", ".join(self.skill_names)
            raise KeyError(f"unknown skill {skill_name!r}; registered: {choices}")

        started = time.perf_counter()
        start_steps = context.total_steps
        try:
            invocation_parameters = dict(parameters or {})
            manifest = self._manifests.get(skill_name)
            if manifest is not None:
                invocation_parameters = manifest.validate_invocation(
                    task_id=context.task_id,
                    parameters=invocation_parameters,
                )
            result = self._skills[skill_name].execute(
                context,
                invocation_parameters,
            )
        except Exception as exc:
            return SkillResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED,
                steps=context.total_steps - start_steps,
                duration_seconds=time.perf_counter() - started,
                failure_reason=f"{type(exc).__name__}: {exc}",
                metrics={"exception_type": type(exc).__name__},
            )

        if result.skill_name != skill_name:
            raise ValueError(
                f"skill returned name {result.skill_name!r}; expected {skill_name!r}"
            )
        return result
