"""Non-Turing-complete, resource-bounded skill programs for an Agent."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import time
from typing import Any

import numpy as np

from .core import SkillContext, SkillExecutor, SkillResult, _json_safe
from .chain import SkillChainContract
from .stage_checks import StageGate
from .harness import AgentEnvironment
from .trace import TraceRecorder
from .verifiers import VerificationResult


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _validate_json_value(
    value: Any,
    *,
    path: str,
    depth: int = 0,
    maximum_depth: int = 6,
) -> None:
    if depth > maximum_depth:
        raise ValueError(f"{path} exceeds maximum JSON nesting depth")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError(f"{path} contains an oversized string")
        return
    if isinstance(value, list):
        if len(value) > 4096:
            raise ValueError(f"{path} contains too many list elements")
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                maximum_depth=maximum_depth,
            )
        return
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ValueError(f"{path} contains too many object keys")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} contains a non-string or empty key")
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                maximum_depth=maximum_depth,
            )
        return
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True)
class ProgramLimits:
    """Hard limits applied before and during a generated skill program."""

    maximum_program_bytes: int = 64 * 1024
    maximum_invocations: int = 8
    maximum_total_steps: int = 2000
    maximum_wall_time_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.maximum_program_bytes <= 0:
            raise ValueError("maximum_program_bytes must be positive")
        if self.maximum_invocations <= 0:
            raise ValueError("maximum_invocations must be positive")
        if self.maximum_total_steps <= 0:
            raise ValueError("maximum_total_steps must be positive")
        if (
            not np.isfinite(self.maximum_wall_time_seconds)
            or self.maximum_wall_time_seconds <= 0.0
        ):
            raise ValueError("maximum_wall_time_seconds must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_program_bytes": self.maximum_program_bytes,
            "maximum_invocations": self.maximum_invocations,
            "maximum_total_steps": self.maximum_total_steps,
            "maximum_wall_time_seconds": self.maximum_wall_time_seconds,
        }


@dataclass(frozen=True)
class SkillInvocation:
    """One manifest-checked skill call in an Agent program."""

    invocation_id: str
    skill_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.invocation_id.strip() or not self.skill_name.strip():
            raise ValueError("invocation ID and skill name must not be empty")
        if len(self.invocation_id) > 128 or len(self.skill_name) > 256:
            raise ValueError("invocation ID or skill name is too long")
        _validate_json_value(self.parameters, path="parameters")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillInvocation":
        allowed = {"invocation_id", "skill_name", "parameters"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown invocation fields: {', '.join(unknown)}")
        if "invocation_id" not in payload or "skill_name" not in payload:
            raise ValueError("invocation_id and skill_name are required")
        if not isinstance(payload["invocation_id"], str) or not isinstance(
            payload["skill_name"], str
        ):
            raise TypeError("invocation_id and skill_name must be strings")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("invocation parameters must be a JSON object")
        return cls(
            invocation_id=payload["invocation_id"],
            skill_name=payload["skill_name"],
            parameters=dict(parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "skill_name": self.skill_name,
            "parameters": _json_safe(self.parameters),
        }


@dataclass(frozen=True)
class AgentProgram:
    """Strict sequential skill program; no imports, loops, shell, or Python."""

    program_id: str
    task_id: str
    invocations: tuple[SkillInvocation, ...]
    rationale: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only Agent program schema_version=1 is supported")
        if not self.program_id.strip() or not self.task_id.strip():
            raise ValueError("program_id and task_id must not be empty")
        if not self.invocations:
            raise ValueError("an Agent program must contain at least one invocation")
        identifiers = [item.invocation_id for item in self.invocations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("invocation IDs must be unique")
        if len(self.rationale) > 8192:
            raise ValueError("program rationale is too long")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        limits: ProgramLimits | None = None,
    ) -> "AgentProgram":
        limits = limits or ProgramLimits()
        allowed = {
            "schema_version",
            "program_id",
            "task_id",
            "invocations",
            "rationale",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown Agent program fields: {', '.join(unknown)}")
        required = {"schema_version", "program_id", "task_id", "invocations"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing Agent program fields: {', '.join(missing)}")
        raw_invocations = payload["invocations"]
        if not isinstance(raw_invocations, list):
            raise TypeError("Agent program invocations must be a JSON list")
        if len(raw_invocations) > limits.maximum_invocations:
            raise ValueError(
                f"Agent program contains {len(raw_invocations)} invocations; "
                f"limit is {limits.maximum_invocations}"
            )
        invocations = tuple(
            SkillInvocation.from_dict(item)
            for item in raw_invocations
            if isinstance(item, Mapping)
        )
        if len(invocations) != len(raw_invocations):
            raise TypeError("every invocation must be a JSON object")
        if isinstance(payload["schema_version"], bool) or not isinstance(
            payload["schema_version"], int
        ):
            raise TypeError("schema_version must be an integer")
        if not isinstance(payload["program_id"], str) or not isinstance(
            payload["task_id"], str
        ):
            raise TypeError("program_id and task_id must be strings")
        if "rationale" in payload and not isinstance(payload["rationale"], str):
            raise TypeError("program rationale must be a string")
        return cls(
            schema_version=payload["schema_version"],
            program_id=payload["program_id"],
            task_id=payload["task_id"],
            invocations=invocations,
            rationale=payload.get("rationale", ""),
        )

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        limits: ProgramLimits | None = None,
    ) -> "AgentProgram":
        limits = limits or ProgramLimits()
        encoded = text.encode("utf-8")
        if len(encoded) > limits.maximum_program_bytes:
            raise ValueError(
                f"Agent program is {len(encoded)} bytes; "
                f"limit is {limits.maximum_program_bytes}"
            )
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(payload, Mapping):
            raise TypeError("Agent program root must be a JSON object")
        _validate_json_value(payload, path="program")
        return cls.from_dict(payload, limits=limits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "task_id": self.task_id,
            "rationale": self.rationale,
            "invocations": [item.to_dict() for item in self.invocations],
        }


@dataclass(frozen=True)
class ProgramExecutionResult:
    program: AgentProgram
    skill_results: tuple[SkillResult, ...]
    verification: VerificationResult
    success: bool
    failure_stage: str | None
    steps: int
    duration_seconds: float
    skill_chain: Mapping[str, Any] | None = None
    stage_gates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.success else "fail",
            "failure_stage": self.failure_stage,
            "steps": self.steps,
            "duration_seconds": self.duration_seconds,
            "program": self.program.to_dict(),
            "skills": [result.to_dict() for result in self.skill_results],
            "verification": self.verification.to_dict(),
            "skill_chain": self.skill_chain,
            "stage_gates": list(self.stage_gates),
        }


def _verification_failure_code(reason: str | None) -> str:
    text = str(reason or "")
    if "outside" in text:
        return "payload_outside"
    if "nail" in text and "depth" in text and "below" in text:
        return "insufficient_nail_depth"
    if "lift" in text and "below" in text:
        return "insufficient_lift"
    return "verification_failed"


class SafeSkillProgramExecutor:
    """Interpret only manifest-backed skill calls under hard resource limits."""

    def __init__(
        self,
        *,
        skill_executor: SkillExecutor,
        limits: ProgramLimits | None = None,
        chain_contract: SkillChainContract | None = None,
        stage_gate: StageGate | None = None,
    ) -> None:
        self.skill_executor = skill_executor
        self.limits = limits or ProgramLimits()
        self.chain_contract = chain_contract
        self.stage_gate = stage_gate

    def validate(self, program: AgentProgram, context: SkillContext) -> None:
        if self.chain_contract is not None:
            self.chain_contract.validate_program(program)
        if self.stage_gate is not None:
            self.stage_gate.validate_stages(
                [stage.stage_id for stage in self.chain_contract.stages] if self.chain_contract else
                [invocation.invocation_id for invocation in program.invocations]
            )
        if program.task_id != context.task_id:
            raise ValueError(
                f"program task {program.task_id!r} does not match "
                f"context task {context.task_id!r}"
            )
        if len(program.invocations) > self.limits.maximum_invocations:
            raise ValueError("program exceeds invocation limit")
        for invocation in program.invocations:
            if invocation.skill_name not in self.skill_executor.skill_names:
                raise ValueError(
                    f"program requests unregistered skill {invocation.skill_name!r}"
                )
            manifest = self.skill_executor.manifest(invocation.skill_name)
            if manifest is None:
                raise ValueError(
                    f"program skill {invocation.skill_name!r} has no manifest"
                )
            manifest.validate_invocation(
                task_id=context.task_id,
                parameters=invocation.parameters,
            )

    def run(
        self,
        *,
        program: AgentProgram,
        environment: AgentEnvironment,
        context: SkillContext,
        trace: TraceRecorder | None = None,
    ) -> ProgramExecutionResult:
        self.validate(program, context)
        started = time.perf_counter()
        start_steps = context.total_steps
        previous_maximum = context.maximum_total_steps
        previous_deadline = context.deadline_monotonic
        context.maximum_total_steps = min(
            value
            for value in (
                previous_maximum,
                context.total_steps + self.limits.maximum_total_steps,
            )
            if value is not None
        )
        deadline = time.monotonic() + self.limits.maximum_wall_time_seconds
        context.deadline_monotonic = (
            min(previous_deadline, deadline)
            if previous_deadline is not None
            else deadline
        )
        skill_results: list[SkillResult] = []
        stage_gates: list[dict[str, Any]] = []
        failed_stage: str | None = None
        program_context = (
            trace.span(
                "program.execute",
                category="program",
                start_step=context.total_steps,
                inputs={
                    "program": program.to_dict(),
                    "limits": self.limits.to_dict(),
                    "catalog": self.skill_executor.catalog(),
                    "required_skill_chain": self.chain_contract.to_dict() if self.chain_contract else None,
                    "stage_check_configuration": self.stage_gate.configuration() if self.stage_gate else None,
                },
            )
            if trace is not None
            else nullcontext()
        )
        try:
            with program_context as program_span:
                for stage_index, invocation in enumerate(program.invocations):
                    manifest = self.skill_executor.manifest(invocation.skill_name)
                    invocation_context = (
                        trace.span(
                            "skill.execute",
                            category="skill",
                            start_step=context.total_steps,
                            inputs={
                                "invocation": invocation.to_dict(),
                                "manifest": (
                                    manifest.to_dict()
                                    if manifest is not None
                                    else None
                                ),
                            },
                        )
                        if trace is not None
                        else nullcontext()
                    )
                    with invocation_context as invocation_span:
                        result = self.skill_executor.execute(
                            invocation.skill_name,
                            context,
                            invocation.parameters,
                        )
                        skill_results.append(result)
                        if invocation_span is not None:
                            if result.success:
                                invocation_span.succeed(
                                    end_step=context.total_steps,
                                    outputs={"skill_result": result.to_dict()},
                                )
                            else:
                                exception_type = result.metrics.get("exception_type")
                                code = (
                                    "unsafe_action"
                                    if exception_type
                                    in {
                                        "UnsafeActionError",
                                        "ExecutionBudgetExceeded",
                                    }
                                    else "skill_failed"
                                )
                                invocation_span.fail(
                                    end_step=context.total_steps,
                                    failure_code=code,
                                    reason=str(result.failure_reason),
                                    outputs={"skill_result": result.to_dict()},
                                )
                    if not result.success:
                        failed_stage = (
                            "safety"
                            if result.metrics.get("exception_type")
                            in {"UnsafeActionError", "ExecutionBudgetExceeded"}
                            else "policy"
                        )
                        break

                    if self.stage_gate is not None:
                        stage_id = (self.chain_contract.stages[stage_index].stage_id
                                    if self.chain_contract else invocation.invocation_id)
                        gate_context = trace.span("stage.check_and_route", category="stage_check",
                            start_step=context.total_steps, inputs={"stage_id": stage_id}) if trace else nullcontext()
                        with gate_context as gate_span:
                            gate = self.stage_gate.evaluate(stage_id=stage_id, context=context, skill_result=result)
                            stage_gates.append(gate)
                            gate_refs = tuple(dict.fromkeys(ref for row in gate["observation_attempts"]
                                                            for ref in row.get("artifact_refs", ())))
                            if gate_span is not None:
                                if gate["action"] == "continue":
                                    gate_span.succeed(end_step=context.total_steps, outputs=gate, artifact_refs=gate_refs)
                                else:
                                    gate_span.fail(end_step=context.total_steps, failure_code="stage_check_blocked",
                                                   reason=gate["reason"], outputs=gate, artifact_refs=gate_refs)
                        if gate["action"] != "continue":
                            failed_stage = ("budget" if gate["exception_type"] == "ExecutionBudgetExceeded" else
                                            "infrastructure" if gate["action"] == "abort" else
                                            "stage_evidence" if gate["action"] == "skip" else "stage_check")
                            break

                verification_context = (
                    trace.span(
                        "verification.task",
                        category="verification",
                        start_step=context.total_steps,
                        inputs={"task_id": environment.task_id},
                    )
                    if trace is not None
                    else nullcontext()
                )
                with verification_context as verification_span:
                    verification = environment.verify()
                    if verification_span is not None:
                        if verification.success:
                            verification_span.succeed(
                                end_step=context.total_steps,
                                outputs={"verification": verification.to_dict()},
                            )
                        else:
                            verification_span.fail(
                                end_step=context.total_steps,
                                failure_code=_verification_failure_code(
                                    verification.reason
                                ),
                                reason=str(verification.reason),
                                outputs={"verification": verification.to_dict()},
                            )

                all_skills_succeeded = (
                    len(skill_results) == len(program.invocations)
                    and all(result.success for result in skill_results)
                )
                success = all_skills_succeeded and verification.success and failed_stage is None
                if failed_stage is None and not verification.success:
                    failed_stage = "verification"
                if program_span is not None:
                    outputs = {
                        "status": "pass" if success else "fail",
                        "skills_executed": len(skill_results),
                        "verification": verification.to_dict(),
                        "skill_chain": self.chain_contract.execution_report(skill_results, stage_gates) if self.chain_contract else None,
                        "stage_gates": stage_gates,
                    }
                    if success:
                        program_span.succeed(
                            end_step=context.total_steps,
                            outputs=outputs,
                        )
                    else:
                        program_span.fail(
                            end_step=context.total_steps,
                            failure_code=f"{failed_stage or 'program'}_failed",
                            reason=(
                                skill_results[-1].failure_reason
                                if skill_results and not skill_results[-1].success
                                else next((gate["reason"] for gate in stage_gates if gate["action"] != "continue"), None)
                                or verification.reason
                                or "program failed"
                            ),
                            outputs=outputs,
                        )
        finally:
            context.maximum_total_steps = previous_maximum
            context.deadline_monotonic = previous_deadline

        return ProgramExecutionResult(
            program=program,
            skill_results=tuple(skill_results),
            verification=verification,
            success=success,
            failure_stage=failed_stage,
            steps=context.total_steps - start_steps,
            duration_seconds=time.perf_counter() - started,
            skill_chain=self.chain_contract.execution_report(skill_results, stage_gates) if self.chain_contract else None,
            stage_gates=tuple(stage_gates),
        )
