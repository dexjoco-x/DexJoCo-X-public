"""Bounded post-stage checks on data-only snapshots, never simulator objects.

Checks and refresh callbacks are trusted host adapters, not planner-supplied
code. Refresh may add sensor evidence but may not step/reset the robot. Time
limits are cooperative: they are checked before and after each callback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import math
import time
from typing import Callable, Protocol
import uuid

import numpy as np

from .core import SkillResult, SkillStatus, _json_safe
from .retarget import _pose_error6


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StageSnapshot:
    task_id: str
    stage_id: str
    observation_id: str
    control_step: int
    proprioception: tuple[float, ...]
    reference_tcp_pose7: tuple[float, ...]
    # New sensor adapters can subclass this immutable, data-only snapshot.
    # The built-in adapter deliberately excludes reset object poses and env.


@dataclass(frozen=True)
class StageCheckResult:
    check_id: str
    status: CheckStatus
    scope: str
    reason: str
    measurements: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, CheckStatus):
            raise ValueError("check status must be pass, fail or unknown")
        if not self.check_id or not self.reason or self.scope not in {"motion_tracking", "object_effect"}:
            raise ValueError("checks require an ID, reason and explicit evidence scope")
        if any(not math.isfinite(value) for _, value in self.measurements):
            raise ValueError("check measurements must be finite")

    def to_dict(self):
        return {**asdict(self), "status": self.status.value,
                "measurements": dict(self.measurements)}


class StageCheck(Protocol):
    check_id: str
    scope: str

    def evaluate(self, snapshot: StageSnapshot) -> StageCheckResult: ...

    def configuration(self) -> dict: ...


@dataclass(frozen=True)
class StageObservationAdvice:
    """A routing suggestion; never evidence that an unknown check passed."""
    action: str
    reason: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.action not in ("reobserve", "stop") or not isinstance(self.reason, str) or not 1 <= len(self.reason) <= 2048:
            raise ValueError("stage observation advice must be bounded reobserve/stop")

    def to_dict(self):
        return {"action": self.action, "reason": self.reason, "metadata": _json_safe(self.metadata)}


@dataclass(frozen=True)
class TcpEndpointCheck:
    maximum_position_error_m: float = 0.05
    maximum_orientation_error_rad: float = 0.35
    check_id: str = "tcp_endpoint_tracking"
    scope: str = field(default="motion_tracking", init=False)

    def __post_init__(self):
        for value in (self.maximum_position_error_m, self.maximum_orientation_error_rad):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("TCP check thresholds must be finite and positive")

    def configuration(self):
        return asdict(self)

    def evaluate(self, snapshot):
        pose = np.asarray(snapshot.proprioception[:7])
        reference = np.asarray(snapshot.reference_tcp_pose7)
        if (pose.shape != (7,) or reference.shape != (7,)
                or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(reference))):
            return StageCheckResult(self.check_id, CheckStatus.UNKNOWN, self.scope,
                                    "finite TCP observation or executed reference is missing")
        error = _pose_error6(pose, reference)
        position, orientation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        passed = (position <= self.maximum_position_error_m
                  and orientation <= self.maximum_orientation_error_rad)
        return StageCheckResult(self.check_id, CheckStatus.PASS if passed else CheckStatus.FAIL,
            self.scope, "TCP endpoint within declared tolerance" if passed else "TCP endpoint outside declared tolerance",
            (("position_error_m", position), ("orientation_error_rad", orientation)))


@dataclass(frozen=True)
class MissingObjectEffectCheck:
    """Explicit fail-closed placeholder, NEVER a grasp verifier."""
    check_id: str = "object_acquired"
    scope: str = field(default="object_effect", init=False)

    def configuration(self):
        return {**asdict(self), "backend": "not_implemented"}

    def evaluate(self, snapshot):
        return StageCheckResult(self.check_id, CheckStatus.UNKNOWN, self.scope,
                                "no nonprivileged object-effect checker is configured")


class StageGate:
    """Host-owned check catalog and finite rule routing at stage boundaries."""

    def __init__(self, checks: dict[str, tuple[StageCheck, ...]], *,
                 maximum_reobservations: int = 1,
                 observe: Callable[[StageSnapshot], StageSnapshot] | None = None,
                 refresh: Callable[[StageSnapshot], StageSnapshot] | None = None,
                 unknown_router: Callable[[StageSnapshot, tuple[StageCheckResult, ...], int], StageObservationAdvice] | None = None):
        if type(maximum_reobservations) is not int or not 0 <= maximum_reobservations <= 8:
            raise ValueError("stage reobservations must be an integer in [0, 8]")
        if not checks or any(not key or not value for key, value in checks.items()):
            raise ValueError("each gated stage requires at least one check")
        for group in checks.values():
            if len({check.check_id for check in group}) != len(group):
                raise ValueError("check IDs must be unique within a stage")
        self.checks = {key: tuple(value) for key, value in checks.items()}
        self.maximum_reobservations = maximum_reobservations
        self.refresh = refresh
        self.observe = observe
        self.unknown_router = unknown_router

    def configuration(self):
        return {"policy": "bounded_post_stage_rules_v1", "llm_used": False,
                "maximum_reobservations_per_stage": self.maximum_reobservations,
                "refresh_available": self.refresh is not None,
                "initial_observer_available": self.observe is not None,
                "unknown_routing": "trusted_callback" if self.unknown_router is not None else "local_rules",
                "checks": {key: [check.configuration() for check in group]
                           for key, group in self.checks.items()}}

    def validate_stages(self, stage_ids):
        if set(self.checks) != set(stage_ids):
            raise ValueError("gate catalog must cover every program stage exactly")

    def evaluate(self, *, stage_id, context, skill_result):
        history = []
        initial_step = context.total_steps
        initial_state = tuple(np.asarray(context.observation["state"], dtype=float).reshape(-1)[:13])
        reference = tuple(np.asarray(skill_result.metrics.get("terminal_tcp_reference7", ()), dtype=float))
        snapshot = StageSnapshot(context.task_id, stage_id, uuid.uuid4().hex,
                                 initial_step, initial_state, reference)
        seen_ids = set()
        checked = []
        action, reason, exception_type = "abort", "stage check did not finish", None

        def budget_check():
            if context.deadline_monotonic is not None and time.monotonic() >= context.deadline_monotonic:
                raise TimeoutError("post-stage check exhausted the shared program deadline")

        def control_invariant():
            if context.total_steps != initial_step or tuple(np.asarray(context.observation["state"])[:13]) != initial_state:
                raise ValueError("stage checks and observation refresh must not advance or reset robot control")

        try:
            if self.observe is not None:
                budget_check()
                snapshot = self.observe(snapshot)
                budget_check()
                control_invariant()
            for attempt in range(self.maximum_reobservations + 1):
                budget_check()
                if (snapshot.task_id != context.task_id or snapshot.stage_id != stage_id
                        or snapshot.control_step != initial_step
                        or not snapshot.observation_id or snapshot.observation_id in seen_ids
                        or snapshot.reference_tcp_pose7 != reference
                        or snapshot.proprioception != initial_state):
                    raise ValueError("stage observation is stale, reused, or belongs to another execution")
                seen_ids.add(snapshot.observation_id)
                checked = []
                routing_advice = None
                for check in self.checks[stage_id]:
                    outcome = check.evaluate(snapshot)
                    budget_check()
                    control_invariant()
                    if not isinstance(outcome, StageCheckResult) or (outcome.check_id, outcome.scope) != (check.check_id, check.scope):
                        raise ValueError("checker returned an incompatible evidence contract")
                    checked.append(outcome)
                if any(item.status is CheckStatus.FAIL for item in checked):
                    action = "reset_and_replan"
                    reason = "a required stage check failed; do not execute the suffix"
                elif any(item.status is CheckStatus.UNKNOWN for item in checked):
                    remaining = self.maximum_reobservations - attempt if self.refresh is not None else 0
                    if self.unknown_router is None:
                        action = "reobserve" if remaining else "skip"
                        reason = "required evidence is unknown"
                    else:
                        routing_advice = self.unknown_router(snapshot, tuple(checked), remaining)
                        budget_check()
                        control_invariant()
                        if not isinstance(routing_advice, StageObservationAdvice):
                            raise ValueError("stage router returned an invalid advice contract")
                        if routing_advice.action == "reobserve" and not remaining:
                            raise ValueError("stage router exceeded the observation budget")
                        action = "reobserve" if routing_advice.action == "reobserve" else "skip"
                        reason = routing_advice.reason
                else:
                    action, reason = "continue", "all required stage checks passed"
                history.append({"observation_id": snapshot.observation_id, "control_step": initial_step,
                                "artifact_refs": list(getattr(snapshot, "artifact_refs", ())),
                                "checks": [item.to_dict() for item in checked], "action": action,
                                "reason": reason})
                if routing_advice is not None:
                    history[-1]["routing_advice"] = routing_advice.to_dict()
                if action != "reobserve":
                    break
                snapshot = self.refresh(snapshot)
                budget_check()
                control_invariant()
        except Exception as exc:
            # An arbitrary checker/driver timeout is infrastructure, not budget.
            expired = context.deadline_monotonic is not None and time.monotonic() >= context.deadline_monotonic
            action = "skip" if expired else "abort"
            reason = f"{type(exc).__name__}: {exc}"
            exception_type = "ExecutionBudgetExceeded" if expired else type(exc).__name__
        effects = [item.check_id for item in checked
                   if item.scope == "object_effect" and item.status is CheckStatus.PASS]
        return {"stage_id": stage_id, "action": action, "reason": reason,
                "observation_attempts": history, "exception_type": exception_type,
                "physical_effect_verified": action == "continue" and bool(effects),
                "verified_effects": effects if action == "continue" else []}


def checked_skill_result(result: SkillResult, gates):
    """Do not store task success if a required gate rejected completed motion."""
    blocked = next((gate for gate in gates if gate["action"] != "continue"), None)
    metrics = {**result.metrics, "stage_gates": list(gates)}
    if blocked is None:
        return replace(result, metrics=metrics)
    if blocked["exception_type"]:
        metrics["exception_type"] = blocked["exception_type"]
    return replace(result, status=SkillStatus.FAILED, failure_reason=blocked["reason"], metrics=metrics)


def make_stage_gate(mode, stage_ids):
    """Fixed experimental modes; no planner-controlled check removal/thresholds."""
    if mode == "off":
        return None
    if mode not in {"tcp_tracking", "require_object_effect"}:
        raise ValueError("unknown stage check mode")
    checks = {stage_id: (TcpEndpointCheck(),) for stage_id in stage_ids}
    if mode == "require_object_effect":
        first = next(iter(checks))
        checks[first] += (MissingObjectEffectCheck(),)
    return StageGate(checks)
