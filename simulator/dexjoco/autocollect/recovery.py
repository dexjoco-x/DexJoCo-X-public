"""Task-neutral, bounded recovery decisions; no robot/simulator/model imports.

The recovery supervisor can request another observation or a new manifested
plan, but never supplies poses, lowers thresholds, or emits control actions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Callable, Mapping, Any


class PerceptionEvidenceError(RuntimeError):
    """Expected lack/ambiguity of sensor evidence; a new view may help.

    Model/device errors and invalid data contracts must NOT use this type.
    """


class RecoveryBudgetExceeded(TimeoutError):
    """A declared scene budget ended, not an arbitrary network/device timeout."""


class RecoveryAction(str, Enum):
    REOBSERVE = "reobserve"
    RESET_AND_REPLAN = "reset_and_replan"
    COMPLETE = "complete"
    SKIP = "skip"
    ABORT = "abort"


def should_abort_collection(result: Mapping[str, Any]) -> bool:
    """Safety/infrastructure failures stop the batch, not just the current seed."""
    recovery = result.get("recovery") or {}
    return result.get("status") == "error" or recovery.get("terminal_action") == "abort"


@dataclass(frozen=True)
class RecoveryLimits:
    maximum_observations: int = 3
    maximum_executions: int = 3
    maximum_total_steps: int = 3000
    maximum_scene_seconds: float = 180.0

    def __post_init__(self):
        for value in (self.maximum_observations, self.maximum_executions, self.maximum_total_steps):
            if type(value) is not int or value < 1:
                raise ValueError("recovery count/step limits must be positive integers")
        if not math.isfinite(self.maximum_scene_seconds) or self.maximum_scene_seconds <= 0:
            raise ValueError("scene time limit must be finite and positive")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RecoveryFeedback:
    stage: str
    code: str
    reason: str

    def __post_init__(self):
        if self.stage not in {"perception", "policy", "verification", "stage_check", "stage_evidence", "safety", "infrastructure", "budget", "success"}:
            raise ValueError("unknown recovery feedback stage")
        if not self.code or not self.reason or len(self.reason) > 2048:
            raise ValueError("feedback requires a bounded code/reason")

    def to_dict(self):
        return asdict(self)


class RecoverySession:
    """One scene's monotonic resource accounting and conservative supervisor.

    Wall-time bounds are cooperative at operation boundaries and control
    steps, not a hard timeout capable of interrupting a hung GPU/renderer.
    """

    def __init__(self, limits: RecoveryLimits, *, clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self._clock = clock
        self.started = clock()
        self.deadline_monotonic = self.started + limits.maximum_scene_seconds
        self.observation_ids: list[str] = []
        self.execution_ids: list[str] = []
        self.total_steps = 0
        self.history: list[dict] = []
        self.terminal_action: RecoveryAction | None = None
        self.observation_ready = False

    @property
    def remaining_steps(self):
        return max(0, self.limits.maximum_total_steps - self.total_steps)

    @property
    def expired(self):
        return self._clock() >= self.deadline_monotonic

    def _check_active(self):
        if self.terminal_action is not None:
            raise RuntimeError("recovery session is already terminal")
        if self.expired or not self.remaining_steps:
            raise RecoveryBudgetExceeded("scene recovery budget exhausted")

    def begin_observation(self, observation_id: str):
        self._check_active()
        if not observation_id or observation_id in self.observation_ids:
            raise ValueError("observation mode must be nonempty and not previously tried")
        if len(self.observation_ids) >= self.limits.maximum_observations:
            raise RecoveryBudgetExceeded("observation budget exhausted")
        self.observation_ids.append(observation_id)
        self.observation_ready = False

    def accept_observation(self):
        self._check_active()
        if not self.observation_ids:
            raise RuntimeError("cannot accept an observation that was never requested")
        self.observation_ready = True

    def begin_execution(self, execution_id: str):
        self._check_active()
        if not self.observation_ready:
            raise RuntimeError("execution requires an accepted observation")
        if not execution_id or execution_id in self.execution_ids:
            raise ValueError("execution candidate must be nonempty and not previously tried")
        if len(self.execution_ids) >= self.limits.maximum_executions:
            raise RecoveryBudgetExceeded("execution budget exhausted")
        self.execution_ids.append(execution_id)

    def record_steps(self, steps: int):
        if type(steps) is not int or steps < 0:
            raise ValueError("recorded control steps must be a nonnegative integer")
        self.total_steps += steps

    def decide(
        self, feedback: RecoveryFeedback, *, another_view: bool = False,
        another_candidate: bool = False, reset_supported: bool = False,
    ) -> dict:
        if self.terminal_action is not None:
            raise RuntimeError("cannot decide after recovery terminated")
        # Safety/infrastructure failures never trigger automatic actuation.
        if feedback.stage in {"safety", "infrastructure"}:
            action, reason = RecoveryAction.ABORT, "nonrecoverable_failure"
        elif self.expired or feedback.stage == "budget" or self.total_steps > self.limits.maximum_total_steps:
            action, reason = RecoveryAction.SKIP, "scene_budget_exhausted"
        elif feedback.stage == "success":
            action, reason = RecoveryAction.COMPLETE, "independent_verifier_passed"
        elif feedback.stage == "stage_evidence":
            action, reason = RecoveryAction.SKIP, "required_stage_evidence_unavailable"
        elif not self.remaining_steps:
            action, reason = RecoveryAction.SKIP, "scene_budget_exhausted"
        elif feedback.stage == "perception":
            available = another_view and len(self.observation_ids) < self.limits.maximum_observations
            action = RecoveryAction.REOBSERVE if available else RecoveryAction.SKIP
            reason = "try_unseen_sensor_view" if available else "observation_options_exhausted"
        else:
            available = another_candidate and reset_supported and len(self.execution_ids) < self.limits.maximum_executions
            action = RecoveryAction.RESET_AND_REPLAN if available else RecoveryAction.SKIP
            reason = "retry_untried_candidate_after_verified_reset" if available else "execution_options_exhausted"
        row = {
            "sequence": len(self.history), "feedback": feedback.to_dict(),
            "action": action.value, "reason": reason,
            "observations_used": len(self.observation_ids),
            "executions_used": len(self.execution_ids), "control_steps_used": self.total_steps,
        }
        self.history.append(row)
        if action in {RecoveryAction.COMPLETE, RecoveryAction.SKIP, RecoveryAction.ABORT}:
            self.terminal_action = action
        return row

    def to_dict(self):
        return {
            "policy": "bounded_rule_supervisor_v1", "llm_used": False,
            "limits": self.limits.to_dict(), "observation_ids": list(self.observation_ids),
            "execution_ids": list(self.execution_ids), "total_control_steps": self.total_steps,
            "terminal_action": self.terminal_action.value if self.terminal_action else None,
            "decisions": list(self.history),
        }
