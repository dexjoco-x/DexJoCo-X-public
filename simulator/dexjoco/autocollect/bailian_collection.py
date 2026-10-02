"""Opt-in, first-stage-only Bailian integration for the existing collector.

Reset planning and post-motion observation routing share ONE API budget. An
unknown physical effect can only lead to fresh evidence or stopping, never a
model-granted permission to continue. No task-specific grasp thresholds change.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from .agent import PlannerDeferred, _canonical_json, agent_program_json_schema
from .bailian import BailianClient, BailianStructuredPlanner
from .collection_contracts import CollectionTaskSpec, FirstStageOnlyGate
from .collection_task_adapters import HammerCollectionAdapter
from .core import SkillContext
from .observation_loop import ObservationLoopLimits, run_observation_loop
from .stage_checks import CheckStatus, StageObservationAdvice


class CollectionPlanningStopped(RuntimeError):
    """An intentional pre-motion terminal outcome, not a failed grasp."""


class SingleUseResetPlan:
    """Reuse the validated answer, not a second HTTP call; revalidate at dispatch."""
    name = "single_use_validated_reset_plan"

    def __init__(self, *, decision, request, assert_stationary):
        self.decision, self.request = decision, request
        self.model_id = decision.model_id
        self.assert_stationary = assert_stationary
        self.request_digest = hashlib.sha256(_canonical_json(request.to_dict())).hexdigest()
        self.program_digest = hashlib.sha256(_canonical_json(decision.program.to_dict())).hexdigest()
        self.used = False

    def plan(self, request, *, output_schema, limits):
        if self.used:
            raise ValueError("a validated reset proposal may be dispatched only once")
        if (hashlib.sha256(_canonical_json(request.to_dict())).hexdigest() != self.request_digest
                or hashlib.sha256(_canonical_json(self.decision.program.to_dict())).hexdigest() != self.program_digest):
            raise ValueError("reset proposal or request changed after model validation")
        self.assert_stationary()
        self.used = True
        return replace(self.decision, metadata={**self.decision.metadata,
                       "dispatch_reuses_prior_api_answer": True, "dispatch_added_api_calls": 0})


class FirstStageIntegrationGate(FirstStageOnlyGate):
    """Compatibility name; new callers pass the task's explicit first stage."""
    def __init__(self, checks, *, first_stage_id=None, **kwargs):
        super().__init__(checks, first_stage_id=first_stage_id or next(iter(checks)), **kwargs)


class BailianCollectionController:
    """One scene/candidate, immutable reset plan, bounded post-stage advice."""

    def __init__(self, client: BailianClient, *, task_adapter=None):
        if client.calls_attempted or client.audit or not 2 <= client.budget.maximum_calls <= 5:
            raise ValueError("collector requires a fresh shared client with 2..5 total calls")
        self.client = client
        self.task_adapter = task_adapter if task_adapter is not None else HammerCollectionAdapter()
        self.task_spec = self.task_adapter.spec
        if not isinstance(self.task_spec, CollectionTaskSpec):
            raise ValueError("collection requires a trusted task specification")
        self.planner = BailianStructuredPlanner(client=client)
        self.agent = self.task_spec.make_planning_agent(self.planner)
        self.used = False
        self.output = None
        self.approved = None
        self.observer = None
        self.reset_report = None
        self.stage_decisions = []

    def configuration(self):
        return {"mode": "bailian_agent_first_stage_only_v1", "model": self.planner.model_id,
                "maximum_api_calls_whole_run": self.client.budget.maximum_calls,
                "maximum_reset_planner_calls": min(3, self.client.budget.maximum_calls - 1),
                "maximum_executable_stages": 1, "maximum_candidates": 1, "maximum_scenes": 1,
                "maximum_post_stage_reobservations": 1, "automatic_api_retries": 0,
                "vla_used": False, "formal_training_data_enabled": False,
                "task_contract": self.task_spec.configuration()}

    def source_objects(self):
        return (type(self), SingleUseResetPlan, FirstStageIntegrationGate, BailianClient,
                BailianStructuredPlanner, CollectionTaskSpec, FirstStageOnlyGate,
                *self.task_adapter.source_objects(), run_observation_loop)

    def write_json(self, path, value):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.client.redact(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        temporary.replace(path)

    def audit(self):
        report = {"configuration": self.configuration(), "calls_attempted": self.client.calls_attempted,
                  "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in self.client.audit),
                  "api_audit": self.client.audit, "billing_verified": False,
                  "reset_observation_report": self.reset_report,
                  "stage_decisions": self.stage_decisions, "formal_training_data_enabled": False}
        if self.output is not None:
            report["source_sha256"] = self.source_hashes
            report["report_path"] = str(self.output / "summary.json")
            self.write_json(self.output / "summary.json", report)
        return report

    def save_request(self, request, folder):
        schema = agent_program_json_schema(task_id=request.task_id, manifests=self.agent._manifests(),
                                           limits=self.agent.limits, chain_contract=self.agent.chain_contract)
        self.write_json(folder / "api_request.json", self.planner.build_payload(request, output_schema=schema))

    def prepare_reset(self, *, env, perception, cameras, session, output, attempt_id):
        if self.used:
            raise ValueError("Bailian collection controller is single-scene/single-candidate")
        self.used = True
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("bailian_collection.py", "bailian.py", "agent.py", "program.py", "stage_checks.py",
                         "observation_loop.py", "hammer_observation.py", "planning_snapshot.py",
                         "hammer_random_runner.py", "visual_effect_runtime.py", "visual_effects.py",
                         "collection_contracts.py", "collection_task_adapters.py")}
        self.audit()
        self.observer = self.task_adapter.make_reset_observer(env=env, perception=perception, cameras=cameras,
                                                 attempt_id=attempt_id, output=self.output / "reset_views")
        remaining_seconds = session.deadline_monotonic - time.monotonic()
        if remaining_seconds <= 0:
            raise CollectionPlanningStopped("scene deadline exhausted before reset observation")
        limits = ObservationLoopLimits(min(3, session.limits.maximum_observations, len(cameras)),
                                       min(3, self.client.budget.maximum_calls - 1), min(180, remaining_seconds))
        def observe(index, trace):
            session.begin_observation(cameras[index - 1])
            receipt = self.observer(index, trace)
            if self.observer.latest_scene_estimate is not None:
                session.accept_observation()
            request = self.task_spec.integration_request(receipt.request)
            return replace(receipt, request=request)
        def accept(decision, request, receipt):
            if self.observer.latest_scene_estimate is None:
                raise ValueError("a reset motion proposal requires the newest accepted sensor estimate")
            self.approved = SingleUseResetPlan(decision=decision, request=request,
                                               assert_stationary=self.observer.assert_stationary)
        try:
            self.reset_report = run_observation_loop(agent=self.agent, observe=observe,
                assert_stationary=self.observer.assert_stationary, cameras=cameras, attempt_id=attempt_id,
                output=self.output / "reset_loop", limits=limits, before_plan=self.save_request,
                redact=self.client.redact, on_plan_validated=accept)
        finally:
            self.audit()
        if self.reset_report["status"] == "interrupted":
            raise KeyboardInterrupt
        if self.reset_report["status"] == "error":
            raise RuntimeError(f"reset observation loop failed: {self.reset_report.get('error')}")
        if self.reset_report["status"] != "plan_validated_no_execution" or self.approved is None:
            raise CollectionPlanningStopped(self.reset_report["status"])
        self.observer.assert_stationary()
        # No derived result is allowed to be passed off as a fresh observation.
        return self.task_adapter.export_latest_scene(self.observer), deepcopy(self.observer.history)

    def make_stage_gate(self, runtime):
        if runtime.context.task_id != self.task_spec.task_id:
            raise ValueError("post-stage observer belongs to a different task")
        base = runtime.make_gate()
        return FirstStageIntegrationGate(base.checks, first_stage_id=self.task_spec.first_stage_id, maximum_reobservations=1,
            observe=runtime.observe, refresh=runtime.refresh,
            unknown_router=lambda snapshot, checks, remaining: self.route_unknown(runtime, snapshot, checks, remaining))

    def route_unknown(self, runtime, snapshot, checks, remaining):
        if not any(item.status is CheckStatus.UNKNOWN for item in checks):
            raise ValueError("model routing is only for unresolved post-stage evidence")
        if (snapshot.task_id != self.task_spec.task_id or snapshot.control_step != runtime.context.total_steps
                or snapshot.control_step <= 0 or snapshot.stage_id != self.task_spec.first_stage_id):
            raise ValueError("post-stage advice is bound to the current first-stage endpoint")
        calls_left = self.client.budget.maximum_calls - self.client.calls_attempted
        if calls_left <= 0:
            return StageObservationAdvice("stop", "shared API budget exhausted", {"llm_called": False})
        image = runtime.planning_image(snapshot)
        available = remaining if calls_left > 1 else 0
        request = self.task_spec.stage_request(snapshot=snapshot, checks=checks, image=image,
                                               observations_remaining=available)
        index = len(self.stage_decisions) + 1
        folder = self.output / f"stage_advice_{index}"
        folder.mkdir(exist_ok=False)
        self.write_json(folder / "request.json", request.to_dict())
        (folder / "observation.png").write_bytes(image.png)
        self.save_request(request, folder)
        row = {"index": index, "stage_id": snapshot.stage_id, "control_step": snapshot.control_step,
               "observation_id": snapshot.observation_id, "status": "requesting"}
        self.stage_decisions.append(row)
        token = self.task_adapter.stationary_token(runtime.context.env)
        try:
            self.audit()
            self.agent.plan_only(request=request, context=SkillContext(env=None, task_id=self.task_spec.task_id, config=None, observation={}))
            raise ValueError("post-stage model must not return a motion program")
        except PlannerDeferred as deferred:
            if self.task_adapter.stationary_token(runtime.context.env) != token:
                raise ValueError("simulation advanced while waiting for stage model advice")
            row.update(status="deferred", advice=deferred.to_dict())
            return StageObservationAdvice(deferred.action, deferred.reason,
                                          {"llm_called": True, "model": self.planner.model_id, "report_path": str(folder / "result.json")})
        except Exception as exc:
            row.update(status="error", error=str(self.client.redact(str(exc)))[:2048])
            raise
        finally:
            if row["status"] == "requesting":
                row.update(status="error", error="stage advice interrupted or stationary boundary violated")
            row["model_response"] = self.planner.last_response
            self.write_json(folder / "result.json", row)
            self.audit()
