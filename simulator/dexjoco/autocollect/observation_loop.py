"""Bounded plan -> observe -> plan loop for a stationary reset scene.

Task adapters own sensing; the loop never executes a skill or environment step.
Historical replay, post-motion continuation and autonomous reset are excluded.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .agent import PlannerDeferred, PlanningRequest
from .core import SkillContext
from .recovery import RecoveryBudgetExceeded
from .trace import TraceRecorder


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ObservationLoopLimits:
    maximum_observations: int = 3  # Includes the initial observation.
    maximum_planner_calls: int = 3
    maximum_seconds: float = 180.0

    def __post_init__(self):
        for count in (self.maximum_observations, self.maximum_planner_calls):
            if type(count) is not int or not 1 <= count <= 3:
                raise ValueError("observation loop counts must be integers in 1..3")
        if not math.isfinite(self.maximum_seconds) or not 0 < self.maximum_seconds <= 300:
            raise ValueError("observation loop duration must be in (0, 300] seconds")


@dataclass(frozen=True)
class ObservationReceipt:
    request: PlanningRequest
    attempt_id: str
    observation_id: str
    camera: str
    index: int
    state_token: str  # Equality-only audit, never sent to the language model.
    started_at_utc: str
    finished_at_utc: str

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "request"}


def validate_receipt(receipt, *, index, attempt_id, token, seen_ids, seen_cameras, started, finished):
    if not isinstance(receipt, ObservationReceipt):
        raise ValueError("sensor callback must return an observation receipt")
    if (type(receipt.index) is not int or receipt.index != index or receipt.attempt_id != attempt_id
            or receipt.state_token != token or not receipt.observation_id
            or receipt.observation_id in seen_ids or receipt.camera in seen_cameras):
        raise ValueError("observation receipt is reused, foreign or refers to a different scene")
    begin, end = (datetime.fromisoformat(value) for value in (receipt.started_at_utc, receipt.finished_at_utc))
    if (begin.tzinfo is None or end.tzinfo is None
            or not datetime.fromisoformat(started) <= begin <= end <= datetime.fromisoformat(finished)):
        raise ValueError("observation receipt has stale or invalid capture timestamps")
    request = receipt.request
    summary = request.observation_summary
    current = summary.get("current_observation", {})
    if (summary.get("historical_snapshot") is not False or summary.get("snapshot_mode") != "live_reset"
            or summary.get("control_step") != 0 or summary.get("physical_effect_verified") is not False
            or current.get("observation_id") != receipt.observation_id or current.get("camera") != receipt.camera
            or len(request.images) != 1 or any(image.frame_id != receipt.observation_id or image.camera != receipt.camera
                                         for image in request.images)):
        raise ValueError("loop requires current reset images; historical/post-motion evidence is not supported")


def bounded_request(request, *, observations_remaining, calls_remaining, previous_advice):
    summary = deepcopy(dict(request.observation_summary))
    gate = summary.get("planning_gate", {})
    if type(gate.get("program_allowed")) is not bool:
        raise ValueError("task adapter must explicitly declare a local planning gate")
    allowed = gate.get("allowed_actions", ["program", "reobserve", "stop"])
    if not isinstance(allowed, (tuple, list)) or not allowed or any(
        action not in ("program", "reobserve", "stop") for action in allowed
    ):
        raise ValueError("invalid task adapter action gate")
    available = [action for action in allowed if not (
        (action == "program" and not gate["program_allowed"])
        or (action == "reobserve" and (observations_remaining <= 0 or calls_remaining <= 1)))]
    if "stop" not in available:
        raise ValueError("observation loop must always permit a terminal stop")
    summary["planning_gate"] = {**gate, "allowed_actions": available}
    summary["observation_opportunities_remaining"] = observations_remaining if calls_remaining > 1 else 0
    summary["planner_calls_remaining_including_current"] = calls_remaining
    summary["previous_planner_advice"] = previous_advice
    return replace(request, observation_summary=summary)


def run_observation_loop(*, agent, observe, assert_stationary, cameras, attempt_id, output,
                         limits=ObservationLoopLimits(), before_plan=None, redact=lambda value: value,
                         clock=time.monotonic, on_plan_validated=None):
    """Fresh observations only after a model reobserve decision (except initial).

    Time limits are cooperative operation-boundary checks, not GPU hard kills.
    ``assert_stationary`` must raise on change and return the same scene token.
    A model/transport/contract error terminates this run without a retry.
    """
    if not cameras or len(cameras) > 3 or len(set(cameras)) != len(cameras) or not attempt_id:
        raise ValueError("declare one to three unique views and an attempt ID")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite an existing observation loop")
    output.mkdir(parents=True, exist_ok=True)
    trace = TraceRecorder(attempt_id=attempt_id)
    report = {"schema_version": 1, "created_at_utc": utc_now(), "status": "running",
              "mode": "stationary_reset_observation_loop", "attempt_id": attempt_id,
              "limits": asdict(limits), "camera_schedule": list(cameras),
              "robot_actions": 0, "execution_enabled": False, "training_eligible": False,
              "task_success": None, "planner_calls_attempted": 0, "observations_attempted": 0,
              "observations_completed": 0, "observations": [], "rounds": []}
    started = clock()
    deadline = started + limits.maximum_seconds
    seen_ids, seen_cameras = set(), set()
    previous_advice = None
    token = None

    def checkpoint():
        report["elapsed_seconds"] = max(0, clock() - started)
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(redact(report), ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        temporary.replace(output / "summary.json")
        if trace.events:
            trace.write_jsonl(output / "trace.jsonl")

    def boundary():
        if clock() >= deadline:
            raise RecoveryBudgetExceeded("observation loop wall-time budget exhausted")
        current_token = assert_stationary()
        if token is not None and current_token != token:
            raise ValueError("scene changed across observation/planning boundaries")
        return current_token

    checkpoint()
    try:
        token = boundary()
        if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("stationary scene guard must supply a SHA-256 equality token")
        maximum_views = min(len(cameras), limits.maximum_observations)
        for index in range(1, maximum_views + 1):
            boundary()
            if report["planner_calls_attempted"] >= limits.maximum_planner_calls:
                raise RecoveryBudgetExceeded("planner call budget exhausted")
            report["observations_attempted"] += 1
            checkpoint()
            capture_started = utc_now()
            with trace.span("observation.refresh", category="sensor", start_step=0,
                            inputs={"index": index, "camera": cameras[index - 1],
                                    "trigger": "initial" if index == 1 else "model_reobserve"}) as span:
                receipt = observe(index, trace)
                validate_receipt(receipt, index=index, attempt_id=attempt_id, token=token,
                                 seen_ids=seen_ids, seen_cameras=seen_cameras,
                                 started=capture_started, finished=utc_now())
                if receipt.camera != cameras[index - 1]:
                    raise ValueError("sensor did not use the declared next camera")
                boundary()
                span.succeed(end_step=0, outputs=receipt.to_dict())
            seen_ids.add(receipt.observation_id)
            seen_cameras.add(receipt.camera)
            report["observations_completed"] += 1
            report["observations"].append(receipt.to_dict())
            remaining_calls = limits.maximum_planner_calls - report["planner_calls_attempted"]
            request = bounded_request(receipt.request, observations_remaining=maximum_views - index,
                                      calls_remaining=remaining_calls, previous_advice=previous_advice)
            folder = output / f"round_{index}"
            folder.mkdir()
            (folder / "request.json").write_text(json.dumps(redact(request.to_dict()), ensure_ascii=False, indent=2) + "\n")
            for image in request.images:
                (folder / "observation.png").write_bytes(image.png)
            if before_plan is not None:
                before_plan(request, folder)
            boundary()
            report["planner_calls_attempted"] += 1
            row = {"index": index, "observation_id": receipt.observation_id, "status": "requesting"}
            report["rounds"].append(row)
            checkpoint()
            try:
                decision = agent.plan_only(request=request,
                                           context=SkillContext(env=None, task_id=request.task_id,
                                                                config=None, observation={}), trace=trace)
                boundary()
                if on_plan_validated is not None:
                    # Trusted handoff of data only; it does not authorize execution.
                    on_plan_validated(decision, request, receipt)
                    boundary()
                row.update(status="plan_validated", action="program", decision=decision.to_dict())
                report["status"] = "plan_validated_no_execution"
                break
            except PlannerDeferred as deferred:
                boundary()
                row.update(status="deferred", action=deferred.action, advice=deferred.to_dict())
                if deferred.action == "stop":
                    report["status"] = "stopped_by_model"
                    break
                # Agent validation checked allowed_actions; guard again locally.
                if "reobserve" not in request.observation_summary["planning_gate"]["allowed_actions"]:
                    raise ValueError("reobservation exceeds the local action budget")
                previous_advice = {"action": "reobserve", "reason": deferred.reason,
                                   "based_on_observation_id": receipt.observation_id,
                                   "scope": "before_the_current_observation"}
            finally:
                row["model_response"] = getattr(agent.planner, "last_response", None)
            checkpoint()
        else:
            report["status"] = "budget_exhausted"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except RecoveryBudgetExceeded as exc:
        report.update(status="budget_exhausted", reason=str(exc))
    except Exception as exc:
        report.update(status="error", error={"type": type(exc).__name__, "message": str(redact(str(exc)))[:2048]})
    finally:
        if report["rounds"] and report["rounds"][-1]["status"] == "requesting":
            report["rounds"][-1]["status"] = "error_or_interrupted"
        checkpoint()
    return report
