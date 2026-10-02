"""Structured planner boundary and safe AutoCollect Agent controller."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import struct
import time
from typing import Any, Protocol

import numpy as np

from .core import SkillContext, SkillExecutor, _json_safe
from .chain import SkillChainContract
from .stage_checks import StageGate
from .harness import AgentEnvironment
from .program import (
    AgentProgram,
    ProgramExecutionResult,
    ProgramLimits,
    SafeSkillProgramExecutor,
)
from .registry import ParameterKind, SkillManifest
from .trace import TraceRecorder


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _parameter_schema(manifest_parameter: Mapping[str, Any]) -> dict[str, Any]:
    kind = manifest_parameter["kind"]
    if kind == ParameterKind.ARRAY.value:
        schema: dict[str, Any] = {"type": "array"}
        shape = manifest_parameter.get("shape")
        if shape and len(shape) == 1:
            schema["minItems"] = int(shape[0])
            schema["maxItems"] = int(shape[0])
            item_schema: dict[str, Any] = {"type": "number"}
            if manifest_parameter.get("minimum") is not None:
                item_schema["minimum"] = manifest_parameter["minimum"]
            if manifest_parameter.get("maximum") is not None:
                item_schema["maximum"] = manifest_parameter["maximum"]
            schema["items"] = item_schema
        return schema
    if kind == ParameterKind.STRING.value:
        schema = {"type": "string"}
        if manifest_parameter.get("choices"):
            schema["enum"] = list(manifest_parameter["choices"])
        return schema
    if kind == ParameterKind.NUMBER.value:
        schema = {"type": "number"}
    elif kind == ParameterKind.INTEGER.value:
        schema = {"type": "integer"}
    elif kind == ParameterKind.BOOLEAN.value:
        return {"type": "boolean"}
    else:
        raise ValueError(f"unsupported manifest parameter kind: {kind!r}")
    if manifest_parameter.get("minimum") is not None:
        schema["minimum"] = manifest_parameter["minimum"]
    if manifest_parameter.get("maximum") is not None:
        schema["maximum"] = manifest_parameter["maximum"]
    return schema


def agent_program_json_schema(
    *,
    task_id: str,
    manifests: tuple[SkillManifest, ...],
    limits: ProgramLimits,
    chain_contract: SkillChainContract | None = None,
) -> dict[str, Any]:
    """Build the exact structured-output schema offered to an LLM."""

    invocation_variants: list[dict[str, Any]] = []
    for manifest in manifests:
        if task_id not in manifest.supported_tasks:
            continue
        manifest_dict = manifest.to_dict()
        properties = {
            parameter["name"]: _parameter_schema(parameter)
            for parameter in manifest_dict["parameters"]
        }
        required = [
            parameter["name"]
            for parameter in manifest_dict["parameters"]
            if parameter["required"]
        ]
        invocation_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["invocation_id", "skill_name", "parameters"],
                "properties": {
                    "invocation_id": {"type": "string", "minLength": 1},
                    "skill_name": {"const": manifest.name},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": required,
                        "properties": properties,
                    },
                },
            }
        )
    if not invocation_variants:
        raise ValueError(f"no manifested skill supports task {task_id!r}")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DexJoCo Safe Agent Skill Program",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "program_id", "task_id", "invocations"],
        "properties": {
            "schema_version": {"const": 1},
            "program_id": {"type": "string", "minLength": 1},
            "task_id": {"const": task_id},
            "rationale": {"type": "string"},
            "invocations": {
                "type": "array",
                "minItems": 1,
                "maxItems": limits.maximum_invocations,
                "items": {"oneOf": invocation_variants},
            },
        },
    }
    if chain_contract is not None:
        if len(chain_contract.stages) > limits.maximum_invocations:
            raise ValueError("chain exceeds program invocation budget")
        schema["properties"]["invocations"] = {
            "type": "array", "minItems": len(chain_contract.stages),
            "maxItems": len(chain_contract.stages), "items": False,
            "prefixItems": [
                {"oneOf": [variant for variant in invocation_variants
                           if variant["properties"]["skill_name"]["const"] in stage.allowed_skills]}
                for stage in chain_contract.stages
            ],
        }
        if any(not stage["oneOf"] for stage in schema["properties"]["invocations"]["prefixItems"]):
            raise ValueError("chain stage has no available manifested skill")
    return schema


@dataclass(frozen=True)
class PlanningImage:
    """Explicit PNG attachment; traces contain its identity, never image bytes."""

    frame_id: str
    camera: str
    png: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not self.frame_id or not self.camera or max(len(self.frame_id), len(self.camera)) > 128:
            raise ValueError("planning image requires bounded frame and camera identifiers")
        if not isinstance(self.png, bytes) or not 24 <= len(self.png) <= 1024 * 1024:
            raise ValueError("planning image must be PNG bytes no larger than 1 MiB")
        if self.png[:8] != b"\x89PNG\r\n\x1a\n" or self.png[12:16] != b"IHDR":
            raise ValueError("planning image must be PNG")
        if any(not 1 <= size <= 2048 for size in struct.unpack(">II", self.png[16:24])):
            raise ValueError("planning image dimensions exceed bounds")

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "camera": self.camera,
                "mime_type": "image/png", "bytes": len(self.png),
                "sha256": hashlib.sha256(self.png).hexdigest()}


@dataclass(frozen=True)
class PlanningRequest:
    """Non-privileged task context passed to a planner."""

    task_id: str
    objective: str
    observation_summary: Mapping[str, Any]
    skill_catalog: tuple[Mapping[str, Any], ...]
    constraints: tuple[str, ...] = ()
    prior_feedback: tuple[Mapping[str, Any], ...] = ()
    locked_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    attempt_index: int = 1
    images: tuple[PlanningImage, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.objective.strip():
            raise ValueError("planning task and objective must not be empty")
        if not self.skill_catalog:
            raise ValueError("planning request requires a non-empty skill catalog")
        if self.attempt_index <= 0:
            raise ValueError("attempt_index must be positive")
        if len(self.objective) > 8192:
            raise ValueError("planning objective is too long")
        if len(self.images) > 2 or any(not isinstance(item, PlanningImage) for item in self.images):
            raise ValueError("planning request supports at most two explicit PNG images")
        if len({item.frame_id for item in self.images}) != len(self.images):
            raise ValueError("planning image frame IDs must be unique")
        # Canonical serialization rejects NaN/Inf and unsupported objects.
        _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "observation_summary": _json_safe(self.observation_summary),
            "skill_catalog": _json_safe(self.skill_catalog),
            "constraints": list(self.constraints),
            "prior_feedback": _json_safe(self.prior_feedback),
            "locked_parameters": _json_safe(self.locked_parameters),
            "attempt_index": self.attempt_index,
            "images": [item.to_dict() for item in self.images],
        }


@dataclass(frozen=True)
class PlannerDecision:
    """Auditable result of one planner call without storing hidden reasoning."""

    program: AgentProgram
    planner_name: str
    model_id: str
    duration_seconds: float
    request_sha256: str
    response_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_name": self.planner_name,
            "model_id": self.model_id,
            "duration_seconds": self.duration_seconds,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "metadata": _json_safe(self.metadata),
            "program": self.program.to_dict(),
        }


class StructuredPlanner(Protocol):
    """Planner interface; implementations may call a local or hosted LLM."""

    name: str
    model_id: str

    def plan(
        self,
        request: PlanningRequest,
        *,
        output_schema: Mapping[str, Any],
        limits: ProgramLimits,
    ) -> PlannerDecision:
        """Return one strictly parsed Agent program."""


class PlannerDeferred(RuntimeError):
    """A planner proposes observation or stopping, never an executable program."""

    def __init__(self, *, action: str, reason: str, metadata: Mapping[str, Any]):
        if action not in ("reobserve", "stop") or not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalid planner deferral")
        if len(reason) > 2048:
            raise ValueError("planner deferral reason is too long")
        super().__init__(reason)
        self.action = action
        self.reason = reason
        self.metadata = dict(metadata)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "metadata": self.metadata}


class CallableStructuredPlanner:
    """Provider-neutral adapter around a structured-output model callable."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        generate_fn: Callable[[Mapping[str, Any]], str | Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = str(name).strip()
        self.model_id = str(model_id).strip()
        self.generate_fn = generate_fn
        self.metadata = dict(metadata or {})
        if not self.name or not self.model_id:
            raise ValueError("planner name and model_id must not be empty")
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")

    def plan(
        self,
        request: PlanningRequest,
        *,
        output_schema: Mapping[str, Any],
        limits: ProgramLimits,
    ) -> PlannerDecision:
        payload = {
            "instruction": (
                "Return only one JSON object matching output_schema. Use only "
                "skills in planning_request.skill_catalog. Do not emit Python, "
                "shell commands, markdown fences, or undeclared parameters."
            ),
            "planning_request": request.to_dict(),
            "output_schema": output_schema,
        }
        started = time.perf_counter()
        response = self.generate_fn(payload)
        duration = time.perf_counter() - started
        if isinstance(response, str):
            program = AgentProgram.from_json(response, limits=limits)
            response_payload: Any = response
        elif isinstance(response, Mapping):
            encoded = _canonical_json(response)
            if len(encoded) > limits.maximum_program_bytes:
                raise ValueError("planner response exceeds maximum program bytes")
            program = AgentProgram.from_dict(response, limits=limits)
            response_payload = response
        else:
            raise TypeError("planner must return a JSON string or mapping")
        if program.task_id != request.task_id:
            raise ValueError(
                f"planner returned task {program.task_id!r}; "
                f"expected {request.task_id!r}"
            )
        return PlannerDecision(
            program=program,
            planner_name=self.name,
            model_id=self.model_id,
            duration_seconds=duration,
            request_sha256=_sha256(payload),
            response_sha256=_sha256(response_payload),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class AgentRunResult:
    decision: PlannerDecision
    execution: ProgramExecutionResult

    @property
    def success(self) -> bool:
        return self.execution.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.success else "fail",
            "decision": self.decision.to_dict(),
            "execution": self.execution.to_dict(),
        }


class AutoCollectAgent:
    """Plan once through a structured model boundary, then execute safely."""

    def __init__(
        self,
        *,
        planner: StructuredPlanner,
        skill_executor: SkillExecutor,
        limits: ProgramLimits | None = None,
        chain_contract: SkillChainContract | None = None,
        stage_gate: StageGate | None = None,
    ) -> None:
        self.planner = planner
        self.skill_executor = skill_executor
        self.limits = limits or ProgramLimits()
        self.chain_contract = chain_contract
        self.stage_gate = stage_gate
        if not self.skill_executor.catalog():
            raise ValueError("AutoCollectAgent requires manifested skills")

    def _manifests(self) -> tuple[SkillManifest, ...]:
        manifests = tuple(
            manifest
            for name in self.skill_executor.skill_names
            if (manifest := self.skill_executor.manifest(name)) is not None
        )
        if len(manifests) != len(self.skill_executor.skill_names):
            raise ValueError("every Agent-visible skill must have a manifest")
        return manifests

    def plan_only(
        self,
        *,
        request: PlanningRequest,
        context: SkillContext,
        trace: TraceRecorder | None = None,
    ) -> PlannerDecision:
        """Plan and statically validate, without reading or stepping an environment.

        Passing this boundary is NOT evidence of grasp success or permission to
        execute a stale plan. Runtime stage gates still apply during execution.
        """
        if request.task_id != context.task_id:
            raise ValueError("planner request and context task must match")
        if tuple(request.skill_catalog) != tuple(self.skill_executor.catalog()):
            raise ValueError("planning request catalog does not match executor catalog")
        gate = request.observation_summary.get("planning_gate")
        allowed_actions = gate.get("allowed_actions") if isinstance(gate, Mapping) else None
        if allowed_actions is not None and (
            not isinstance(allowed_actions, (list, tuple)) or not allowed_actions
            or any(action not in ("program", "reobserve", "stop") for action in allowed_actions)
        ):
            raise ValueError("invalid local allowed planning actions")

        def call_planner():
            try:
                result = self.planner.plan(request, output_schema=schema, limits=self.limits)
            except PlannerDeferred as deferred:
                if allowed_actions is not None and deferred.action not in allowed_actions:
                    raise ValueError("local planning action gate blocks the proposed deferral") from None
                raise
            if allowed_actions is not None and "program" not in allowed_actions:
                raise ValueError("local planning action gate blocks a motion program")
            return result

        schema = agent_program_json_schema(
            task_id=request.task_id,
            manifests=self._manifests(),
            limits=self.limits,
            chain_contract=self.chain_contract,
        )
        if trace is None:
            decision = call_planner()
        else:
            with trace.span(
                "agent.plan",
                category="agent",
                start_step=context.total_steps,
                inputs={
                    "planner_name": self.planner.name,
                    "model_id": self.planner.model_id,
                    "request": request.to_dict(),
                    "output_schema_sha256": _sha256(schema),
                },
            ) as planning_span:
                decision = call_planner()
                planning_span.succeed(
                    end_step=context.total_steps,
                    outputs={"decision": decision.to_dict()},
                )
        program_executor = SafeSkillProgramExecutor(
            skill_executor=self.skill_executor, limits=self.limits,
            chain_contract=self.chain_contract,
            stage_gate=self.stage_gate,
        )
        validation_context = (
            trace.span(
                "agent.validate_program",
                category="agent",
                start_step=context.total_steps,
                inputs={
                    "program_id": decision.program.program_id,
                    "locked_parameters": request.locked_parameters,
                },
            )
            if trace is not None
            else None
        )
        validation_manager: Any = (
            validation_context if validation_context is not None else nullcontext()
        )
        with validation_manager as validation_span:
            program_executor.validate(decision.program, context)
            gate = request.observation_summary.get("planning_gate")
            if gate is not None and (
                not isinstance(gate, Mapping) or gate.get("program_allowed") is not True
            ):
                raise ValueError("local planning evidence gate blocks a motion program")
            for invocation in decision.program.invocations:
                locked = request.locked_parameters.get(invocation.skill_name, {})
                for name, expected in locked.items():
                    if name not in invocation.parameters:
                        raise ValueError(
                            f"planner omitted locked parameter {name!r} for "
                            f"{invocation.skill_name!r}"
                        )
                    if _canonical_json(
                        invocation.parameters[name]
                    ) != _canonical_json(expected):
                        raise ValueError(
                            f"planner changed locked parameter {name!r} for "
                            f"{invocation.skill_name!r}"
                        )
            if validation_span is not None:
                validation_span.succeed(
                    end_step=context.total_steps,
                    outputs={"valid": True},
                )
        return decision

    def run(
        self,
        *,
        request: PlanningRequest,
        environment: AgentEnvironment,
        context: SkillContext,
        trace: TraceRecorder | None = None,
    ) -> AgentRunResult:
        if request.task_id != environment.task_id:
            raise ValueError("planner request and environment task must match")
        decision = self.plan_only(request=request, context=context, trace=trace)
        program_executor = SafeSkillProgramExecutor(
            skill_executor=self.skill_executor, limits=self.limits,
            chain_contract=self.chain_contract, stage_gate=self.stage_gate,
        )
        execution = program_executor.run(
            program=decision.program,
            environment=environment,
            context=context,
            trace=trace,
        )
        return AgentRunResult(decision=decision, execution=execution)
