"""Task-neutral contracts and execution handoffs for finite skill chains.

Completion receipts certify command execution, NOT physical task effects.
Only an independent effect/task verifier can certify an object-level outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import SkillResult, SkillStatus


@dataclass(frozen=True)
class ChainStage:
    stage_id: str
    allowed_skills: tuple[str, ...]

    def __post_init__(self):
        if not self.stage_id or not self.allowed_skills or any(not name for name in self.allowed_skills):
            raise ValueError("chain stages require an ID and allowed skills")
        if len(set(self.allowed_skills)) != len(self.allowed_skills):
            raise ValueError("a stage must not repeat allowed skills")

    def to_dict(self):
        return {"stage_id": self.stage_id, "allowed_skills": list(self.allowed_skills)}


@dataclass(frozen=True)
class SkillChainContract:
    chain_id: str
    stages: tuple[ChainStage, ...]

    def __post_init__(self):
        if not self.chain_id or not self.stages:
            raise ValueError("a skill chain must be named and nonempty")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("chain stage IDs must be unique")

    def validate_program(self, program):
        if len(program.invocations) != len(self.stages):
            raise ValueError("program must include every required chain stage exactly once")
        for stage, invocation in zip(self.stages, program.invocations):
            if invocation.skill_name not in stage.allowed_skills:
                raise ValueError(f"chain stage {stage.stage_id!r} does not allow {invocation.skill_name!r}")

    def to_dict(self):
        return {
            "chain_id": self.chain_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "failure_policy": "stop_suffix_and_return_to_recovery",
            "stage_success_means": "motion_execution_completed_not_physical_effect_verified",
        }

    def execution_report(self, results, stage_gates=()):
        if len(results) > len(self.stages):
            raise ValueError("more results than declared chain stages")
        rows = []
        gates_by_stage = {gate["stage_id"]: gate for gate in stage_gates}
        for index, stage in enumerate(self.stages):
            result = results[index] if index < len(results) else None
            rows.append({
                "stage_id": stage.stage_id,
                "status": "not_run" if result is None else "completed" if result.success else "failed",
                "skill_result": result.to_dict() if result is not None else None,
                "post_stage_gate": gates_by_stage.get(stage.stage_id),
                "physical_effect_verified": gates_by_stage.get(stage.stage_id, {}).get("physical_effect_verified", False),
            })
        return {**self.to_dict(), "stage_results": rows,
                "blocked_stage_id": next((gate["stage_id"] for gate in stage_gates if gate["action"] != "continue"), None),
                "failed_stage_id": next((row["stage_id"] for row in rows if row["status"] == "failed"), None)}

    def aggregate_motion_result(self, results, stage_gates=()):
        """Retain all steps/failures, never mistake the last stage for the whole chain."""
        from .stage_checks import checked_skill_result
        report = self.execution_report(results, stage_gates)
        failed = next((result for result in results if not result.success), None)
        completed = len(results) == len(self.stages) and failed is None
        result = SkillResult(
            skill_name=self.chain_id,
            status=SkillStatus.SUCCEEDED if completed else SkillStatus.FAILED,
            steps=sum(result.steps for result in results),
            duration_seconds=sum(result.duration_seconds for result in results),
            failure_reason=None if completed else (
                failed.failure_reason if failed is not None else "chain ended before all stages ran"
            ),
            metrics={"chain": report, "execution_completed": completed,
                     "physical_effect_verified": False,
                     **({"exception_type": failed.metrics["exception_type"]}
                        if failed is not None and "exception_type" in failed.metrics else {})},
        )
        return checked_skill_result(result, stage_gates) if stage_gates else result


class SkillHandoffError(ValueError):
    """A stage tried to skip, reuse, or change an execution handoff."""


class BoundStageHandoff:
    """One chain instance, one live context, locked parameters and sensor state.

    Checkpoint values are supplied by a trusted task adapter, never a planner.
    This intentionally does not support in-place retries after partial motion.
    """

    def __init__(self, stage_count: int):
        if type(stage_count) is not int or stage_count <= 0:
            raise ValueError("stage count must be a positive integer")
        self.stage_count = stage_count
        self.next_index = 0
        self.context = None
        self.signature = None
        self.checkpoint = None
        self.expected_step = None
        self.pending = False
        self.invalidated = False

    def enter(self, index: int, context: Any, *, signature: str, checkpoint: str):
        if self.invalidated or self.pending or index != self.next_index or index >= self.stage_count:
            raise SkillHandoffError("chain handoff is failed, reused, or out of order")
        if index == 0:
            self.context = context
            self.signature = signature
            self.expected_step = context.total_steps
            self.checkpoint = checkpoint
        if (context is not self.context or signature != self.signature
                or context.total_steps != self.expected_step or checkpoint != self.checkpoint):
            raise SkillHandoffError("chain context, parameters, or proprioceptive state changed between stages")
        self.pending = True

    def finish(self, context: Any, *, success: bool, checkpoint: str):
        if not self.pending or context is not self.context:
            raise SkillHandoffError("no matching stage is awaiting completion")
        self.pending = False
        if not success:
            self.invalidated = True
            return
        self.next_index += 1
        self.expected_step = context.total_steps
        self.checkpoint = checkpoint

    def invalidate(self):
        self.pending = False
        self.invalidated = True
