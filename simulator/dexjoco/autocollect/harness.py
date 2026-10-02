"""Task-agnostic execution harness for agent-driven data collection.

The harness deliberately separates four concerns that used to be intertwined
inside task-specific runners:

* resetting an environment;
* rejecting malformed or explicitly out-of-bounds actions;
* executing one whitelisted skill rollout; and
* asking an independent verifier to judge the final state.

It contains no planner, language model, VLA, or task oracle.  A future coding
agent can therefore use the same interface without receiving privileged task
state through the execution API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .core import SkillContext, SkillExecutor, SkillResult, SkillStatus, _json_safe
from .verifiers import VerificationResult

if TYPE_CHECKING:
    from .trace import TraceRecorder


@dataclass(frozen=True)
class ResetReport:
    """Serializable description of one environment reset.

    The observation itself stays in memory.  Only its public keys and shapes
    enter the report so the experiment ledger remains compact and does not
    accidentally become an oracle-state channel.
    """

    task_id: str
    seed: int
    duration_seconds: float
    observation_keys: tuple[str, ...]
    observation_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "observation_keys": list(self.observation_keys),
            "observation_shapes": _json_safe(self.observation_shapes),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class SafetyCheck:
    """Result of checking one action before it reaches the environment."""

    allowed: bool
    reason: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allowed and self.reason is not None:
            raise ValueError("an allowed action cannot have a rejection reason")
        if not self.allowed and not self.reason:
            raise ValueError("a rejected action requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "metrics": _json_safe(self.metrics),
        }


class UnsafeActionError(ValueError):
    """Raised before stepping the robot when an action fails a hard check."""


class ActionSafetyGuard(Protocol):
    """Policy-independent hard guard for actions sent to a robot."""

    def check(self, action: np.ndarray) -> SafetyCheck:
        """Return whether the action may be executed."""


class FiniteActionSafetyGuard:
    """Reject wrong-sized, non-finite, and optionally out-of-bounds actions."""

    def __init__(
        self,
        *,
        expected_action_dim: int,
        lower: np.ndarray | float | None = None,
        upper: np.ndarray | float | None = None,
        absolute_tolerance: float = 0.0,
    ) -> None:
        expected_action_dim = int(expected_action_dim)
        if expected_action_dim <= 0:
            raise ValueError("expected_action_dim must be positive")
        self.expected_action_dim = expected_action_dim
        self.lower = self._normalise_bound(lower, "lower")
        self.upper = self._normalise_bound(upper, "upper")
        self.absolute_tolerance = float(absolute_tolerance)
        if (
            not np.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0.0
        ):
            raise ValueError("absolute_tolerance must be finite and non-negative")
        if self.lower is not None and self.upper is not None:
            if np.any(self.lower > self.upper):
                raise ValueError("lower action bounds must not exceed upper bounds")

    def _normalise_bound(
        self,
        value: np.ndarray | float | None,
        name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self.expected_action_dim, float(array), dtype=np.float64)
        else:
            array = array.reshape(-1)
        if array.shape != (self.expected_action_dim,):
            raise ValueError(
                f"{name} bound shape {array.shape} does not match "
                f"({self.expected_action_dim},)"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} action bounds must be finite")
        return array

    def check(self, action: np.ndarray) -> SafetyCheck:
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        metrics: dict[str, Any] = {
            "expected_action_dim": self.expected_action_dim,
            "actual_action_dim": int(array.size),
        }
        if array.size != self.expected_action_dim:
            return SafetyCheck(
                allowed=False,
                reason=(
                    f"action has {array.size} elements; expected "
                    f"{self.expected_action_dim}"
                ),
                metrics=metrics,
            )
        if not np.all(np.isfinite(array)):
            metrics["non_finite_indices"] = np.flatnonzero(~np.isfinite(array))
            return SafetyCheck(
                allowed=False,
                reason="action contains NaN or Inf",
                metrics=metrics,
            )
        if self.lower is not None and np.any(
            array < self.lower - self.absolute_tolerance
        ):
            metrics["below_lower_indices"] = np.flatnonzero(
                array < self.lower - self.absolute_tolerance
            )
            return SafetyCheck(
                allowed=False,
                reason="action violates an explicit lower safety bound",
                metrics=metrics,
            )
        if self.upper is not None and np.any(
            array > self.upper + self.absolute_tolerance
        ):
            metrics["above_upper_indices"] = np.flatnonzero(
                array > self.upper + self.absolute_tolerance
            )
            return SafetyCheck(
                allowed=False,
                reason="action violates an explicit upper safety bound",
                metrics=metrics,
            )
        return SafetyCheck(allowed=True, metrics=metrics)

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Clip only values already accepted within the declared tolerance."""

        array = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        check = self.check(array)
        if not check.allowed:
            raise UnsafeActionError(check.reason)
        if self.lower is not None:
            array = np.maximum(array, self.lower)
        if self.upper is not None:
            array = np.minimum(array, self.upper)
        return array

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "expected_action_dim": self.expected_action_dim,
            "lower": _json_safe(self.lower),
            "upper": _json_safe(self.upper),
            "absolute_tolerance": self.absolute_tolerance,
        }


class OutcomeVerifier(Protocol):
    """Independent task-success predicate used only after a rollout."""

    def verify(self, env: Any) -> VerificationResult:
        """Judge the current environment state without selecting actions."""


class AgentEnvironment(Protocol):
    """Minimal environment surface exposed to the rollout engine."""

    task_id: str

    @property
    def last_reset_report(self) -> ResetReport | None:
        """Return metadata for the most recent reset."""

    def reset(self, *, seed: int) -> tuple[Mapping[str, Any], dict[str, Any]]:
        """Reset to a reproducible initial state."""

    def safety_check(self, action: np.ndarray) -> SafetyCheck:
        """Check an action without executing it."""

    def step(self, action: np.ndarray) -> Any:
        """Execute an action only if the safety check permits it."""

    def verify(self) -> VerificationResult:
        """Independently judge the current task state."""

    def close(self) -> None:
        """Release environment resources."""


class TaskPlugin(Protocol):
    """Task-specific factory consumed by the generic experiment harness."""

    name: str
    task_id: str

    def make_environment(self, *, seed: int) -> AgentEnvironment:
        """Construct one isolated environment for a reproducible trial."""


class TaskSuccessTerminationAdapter:
    """Suppress only an early task-success termination during a fixed rollout.

    The original success value remains in ``info`` and an independent verifier
    still judges the final state.  Time limits, truncations, and non-success
    terminal conditions are never suppressed.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.reset(*args, **kwargs)

    def step(self, action: np.ndarray) -> Any:
        observation, reward, terminated, truncated, info = self._env.step(action)
        task_success = bool(info.get("succeed", False))
        if terminated and task_success and not truncated:
            terminated = False
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()


class GymAgentEnvironment:
    """Adapt a Gym-style DexJoCo environment to :class:`AgentEnvironment`."""

    def __init__(
        self,
        *,
        task_id: str,
        env: Any,
        verifier: OutcomeVerifier,
        safety_guard: ActionSafetyGuard,
        reset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        task_id = str(task_id).strip()
        if not task_id:
            raise ValueError("task_id must not be empty")
        self.task_id = task_id
        self._env = env
        self._verifier = verifier
        self._safety_guard = safety_guard
        self._reset_metadata = dict(reset_metadata or {})
        self._last_reset_report: ResetReport | None = None
        self._safety_checks = 0
        self._rejected_actions = 0
        self._normalized_actions = 0

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    @property
    def last_reset_report(self) -> ResetReport | None:
        return self._last_reset_report

    @property
    def safety_summary(self) -> dict[str, Any]:
        guard_dict = getattr(self._safety_guard, "to_dict", None)
        return {
            "checks": self._safety_checks,
            "rejected_actions": self._rejected_actions,
            "normalized_actions": self._normalized_actions,
            "guard": guard_dict() if callable(guard_dict) else {
                "type": type(self._safety_guard).__name__
            },
        }

    def __getattr__(self, name: str) -> Any:
        # Preserve task-specific read-only helpers such as nominal_dynamics.
        return getattr(self._env, name)

    def reset(self, *, seed: int) -> tuple[Mapping[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        self._safety_checks = 0
        self._rejected_actions = 0
        self._normalized_actions = 0
        observation, info = self._env.reset(seed=int(seed))
        if not isinstance(observation, Mapping):
            raise TypeError("environment reset observation must be a mapping")
        shapes = {
            str(key): tuple(np.asarray(value).shape)
            for key, value in observation.items()
            if isinstance(value, (np.ndarray, list, tuple, np.generic))
        }
        self._last_reset_report = ResetReport(
            task_id=self.task_id,
            seed=int(seed),
            duration_seconds=time.perf_counter() - started,
            observation_keys=tuple(sorted(str(key) for key in observation)),
            observation_shapes=shapes,
            metadata=self._reset_metadata,
        )
        return observation, dict(info)

    def safety_check(self, action: np.ndarray) -> SafetyCheck:
        self._safety_checks += 1
        check = self._safety_guard.check(action)
        if not check.allowed:
            self._rejected_actions += 1
        return check

    def step(self, action: np.ndarray) -> Any:
        check = self.safety_check(action)
        if not check.allowed:
            raise UnsafeActionError(check.reason)
        requested = np.asarray(action, dtype=np.float64).reshape(-1)
        normalize = getattr(self._safety_guard, "normalize", None)
        safe_action = normalize(requested) if callable(normalize) else requested
        if not np.array_equal(safe_action, requested):
            self._normalized_actions += 1
        return self._env.step(safe_action)

    def verify(self) -> VerificationResult:
        return self._verifier.verify(self)

    def close(self) -> None:
        self._env.close()


@dataclass(frozen=True)
class RolloutResult:
    """One skill execution and the independent judgment of its outcome."""

    skill: SkillResult
    verification: VerificationResult
    reset: ResetReport
    failure_stage: str | None = None

    @property
    def success(self) -> bool:
        return self.skill.success and self.verification.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.success else "fail",
            "failure_stage": self.failure_stage,
            "reset": self.reset.to_dict(),
            "skill": self.skill.to_dict(),
            "verification": self.verification.to_dict(),
        }


class SkillRolloutEngine:
    """Run one whitelisted skill and judge it with a separate verifier."""

    def run(
        self,
        *,
        environment: AgentEnvironment,
        executor: SkillExecutor,
        context: SkillContext,
        skill_name: str,
        parameters: Mapping[str, Any] | None = None,
        trace: "TraceRecorder | None" = None,
    ) -> RolloutResult:
        reset_report = environment.last_reset_report
        if reset_report is None:
            raise RuntimeError("environment must be reset before a rollout")

        if trace is None:
            skill_result = executor.execute(skill_name, context, parameters)
        else:
            manifest = executor.manifest(skill_name)
            with trace.span(
                "skill.execute",
                category="skill",
                start_step=context.total_steps,
                inputs={
                    "skill_name": skill_name,
                    "parameters": parameters or {},
                    "manifest": (
                        manifest.to_dict() if manifest is not None else None
                    ),
                },
            ) as span:
                skill_result = executor.execute(skill_name, context, parameters)
                if skill_result.success:
                    span.succeed(
                        end_step=context.total_steps,
                        outputs={"skill_result": skill_result.to_dict()},
                    )
                else:
                    exception_type = skill_result.metrics.get("exception_type")
                    failure_code = (
                        "unsafe_action"
                        if exception_type
                        in {"UnsafeActionError", "ExecutionBudgetExceeded"}
                        else "skill_failed"
                    )
                    span.fail(
                        end_step=context.total_steps,
                        failure_code=failure_code,
                        reason=str(skill_result.failure_reason),
                        outputs={"skill_result": skill_result.to_dict()},
                    )
        execution_succeeded = skill_result.success
        if trace is None:
            verification = environment.verify()
        else:
            with trace.span(
                "verification.task",
                category="verification",
                start_step=context.total_steps,
                inputs={"task_id": environment.task_id},
            ) as span:
                verification = environment.verify()
                if verification.success:
                    span.succeed(
                        end_step=context.total_steps,
                        outputs={"verification": verification.to_dict()},
                    )
                else:
                    reason = str(verification.reason)
                    if "outside" in reason:
                        failure_code = "payload_outside"
                    elif "nail" in reason and "depth" in reason and "below" in reason:
                        failure_code = "insufficient_nail_depth"
                    elif "lift" in reason and "below" in reason:
                        failure_code = "insufficient_lift"
                    else:
                        failure_code = "verification_failed"
                    span.fail(
                        end_step=context.total_steps,
                        failure_code=failure_code,
                        reason=reason,
                        outputs={"verification": verification.to_dict()},
                    )

        failure_stage: str | None = None
        if execution_succeeded and not verification.success:
            skill_result = SkillResult(
                skill_name=skill_result.skill_name,
                status=SkillStatus.FAILED,
                steps=skill_result.steps,
                duration_seconds=skill_result.duration_seconds,
                failure_reason=(
                    "skill execution completed but the independent verifier failed: "
                    f"{verification.reason}"
                ),
                metrics=skill_result.metrics,
            )
            failure_stage = "verification"
        elif not execution_succeeded:
            if skill_result.metrics.get("exception_type") in {
                "UnsafeActionError",
                "ExecutionBudgetExceeded",
            }:
                failure_stage = "safety"
            else:
                failure_stage = "policy"

        return RolloutResult(
            skill=skill_result,
            verification=verification,
            reset=reset_report,
            failure_stage=failure_stage,
        )
