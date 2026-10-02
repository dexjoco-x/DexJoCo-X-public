"""Host-owned task contracts for the bounded collection loop.

No task names, simulator access or provider dependencies belong in this module.
Reset sensing and post-motion evidence are distinct interfaces. A model may
select manifested skills; it cannot install an adapter or expand permissions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .agent import AutoCollectAgent, PlanningImage, PlanningRequest
from .chain import SkillChainContract
from .core import SkillExecutor
from .observation_loop import ObservationReceipt
from .program import ProgramLimits
from .registry import SkillManifest
from .stage_checks import CheckStatus, StageCheckResult, StageGate, StageSnapshot


class ResetObserver(Protocol):
    """Already-reset, stationary sensor adapter; never executes motion."""
    latest_scene_estimate: dict | None
    history: list[dict]

    def __call__(self, index: int, trace) -> ObservationReceipt: ...
    def assert_stationary(self) -> str: ...


class PostStageObserver(Protocol):
    """Binds a current endpoint image and read-only evidence to one live context."""
    context: object

    def make_gate(self) -> StageGate: ...
    def observe(self, snapshot: StageSnapshot) -> StageSnapshot: ...
    def refresh(self, snapshot: StageSnapshot) -> StageSnapshot: ...
    def planning_image(self, snapshot: StageSnapshot) -> PlanningImage: ...


class _ProposalOnlySkill:
    def __init__(self, name):
        self.name = name

    def execute(self, context, parameters):
        raise RuntimeError("planning contract has no motion executor; bind a validated task skill before dispatch")


@dataclass(frozen=True)
class CollectionTaskSpec:
    task_id: str
    chain: SkillChainContract
    manifests: tuple[SkillManifest, ...]
    reset_objective: str
    stage_objective: str

    def __post_init__(self):
        if not self.task_id or not self.reset_objective or not self.stage_objective:
            raise ValueError("task contract requires identity and explicit objectives")
        names = [m.name for m in self.manifests]
        required = {name for stage in self.chain.stages for name in stage.allowed_skills}
        if (len(names) != len(set(names)) or set(names) != required
                or any(self.task_id not in m.supported_tasks for m in self.manifests)):
            raise ValueError("task manifest catalog must cover exactly its own chain")
        if len({m.action_dimension for m in self.manifests}) != 1:
            raise ValueError("task chain must have one action dimension")
        if any(m.vla_used for m in self.manifests):
            raise ValueError("this collection integration does not authorize VLA skills")

    @property
    def first_stage_id(self):
        return self.chain.stages[0].stage_id

    def catalog(self):
        return tuple(m.to_dict() for m in sorted(self.manifests, key=lambda m: m.name))

    def configuration(self):
        return {"task_id": self.task_id, "chain_id": self.chain.chain_id,
                "first_stage_id": self.first_stage_id,
                "skills": [m.name for m in self.manifests],
                "observation_interfaces": ["stationary_reset", "current_post_stage"],
                "maximum_executable_stages": 1}

    def make_planning_agent(self, planner):
        executor = SkillExecutor(require_manifests=True)
        for manifest in self.manifests:
            executor.register(_ProposalOnlySkill(manifest.name), manifest=manifest)
        return AutoCollectAgent(planner=planner, skill_executor=executor, chain_contract=self.chain,
            limits=ProgramLimits(maximum_invocations=len(self.chain.stages), maximum_total_steps=1000,
                                 maximum_wall_time_seconds=180))

    def integration_request(self, request):
        if request.task_id != self.task_id or tuple(request.skill_catalog) != self.catalog():
            raise ValueError("reset observation belongs to a different task/catalog")
        if (request.observation_summary.get("snapshot_mode") != "live_reset"
                or request.observation_summary.get("control_step") != 0
                or request.observation_summary.get("historical_snapshot") is not False):
            raise ValueError("post-motion evidence must not enter the reset planning interface")
        execution_scope = "simulation only; propose the declared chain but ONLY stage 1 may execute after local validation"
        constraints = tuple(request.constraints)
        # Replace only the known legacy planning-only scope (or our own scope
        # on idempotent rebuild); never drop another task adapter's constraint.
        if constraints and constraints[0] in (execution_scope, "no robot motion or real hardware; a program is a proposal only"):
            constraints = constraints[1:]
        return replace(request, objective=self.reset_objective,
            observation_summary={**request.observation_summary,
                                 "execution_scope": "only_first_stage_after_local_validation"},
            constraints=(execution_scope, *constraints))

    def stage_request(self, *, snapshot, checks, image, observations_remaining):
        if (snapshot.task_id != self.task_id or snapshot.stage_id != self.first_stage_id
                or type(snapshot.control_step) is not int or snapshot.control_step <= 0
                or image.frame_id != snapshot.observation_id):
            raise ValueError("post-stage evidence must match this task and its current first-stage endpoint")
        if (not checks or any(not isinstance(item, StageCheckResult) for item in checks)
                or any(item.status is CheckStatus.FAIL for item in checks)
                or not any(item.status is CheckStatus.UNKNOWN for item in checks)):
            raise ValueError("model advice requires UNKNOWN evidence with no failed check")
        if type(observations_remaining) is not int or observations_remaining not in (0, 1):
            raise ValueError("post-stage observation opportunities must be 0 or 1")
        return PlanningRequest(task_id=self.task_id, objective=self.stage_objective,
            observation_summary={"snapshot_mode": "live_stage", "source_kind": "live_stage_rgbd",
                "historical_snapshot": False, "control_step": snapshot.control_step,
                "current_stage": snapshot.stage_id, "current_observation_id": snapshot.observation_id,
                "image_description": "current stage detector input (possibly workspace-masked crop), or raw camera RGB if inference was not reached",
                "stage_checks": [item.to_dict() for item in checks], "physical_effect_verified": False,
                "scene_estimate": None, "reset_poses_are_not_post_motion_estimates": True,
                "observation_opportunities_remaining": observations_remaining,
                "planning_gate": {"program_allowed": False,
                    "allowed_actions": ["reobserve", "stop"] if observations_remaining else ["stop"],
                    "reason": "current physical effect unknown; model cannot authorize motion"}},
            skill_catalog=self.catalog(), locked_parameters={}, images=(image,),
            constraints=("first-stage-only integration: no remaining motion stages authorized",
                         "use current image and checks only; never use reset poses or simulator object truth",
                         "reobserve means the next existing camera without control steps; exhausted budget means stop"))


class CollectionTaskAdapter(Protocol):
    """Task-owned geometry/provenance; the controller never reconstructs a pose."""
    spec: CollectionTaskSpec

    def make_reset_observer(self, **kwargs) -> ResetObserver: ...
    def export_latest_scene(self, observer: ResetObserver): ...
    def source_objects(self) -> tuple: ...
    def stationary_token(self, env) -> str: ...


class FirstStageOnlyGate(StageGate):
    def __init__(self, checks, *, first_stage_id, **kwargs):
        super().__init__(checks, **kwargs)
        if first_stage_id not in self.checks:
            raise ValueError("first stage must belong to the host check catalog")
        self.first_stage_id = first_stage_id

    def configuration(self):
        return {**super().configuration(), "policy": "bailian_unknown_routing_first_stage_only_v1",
                "llm_used": True, "maximum_executable_stages": 1, "first_stage_id": self.first_stage_id}

    def evaluate(self, *, stage_id, context, skill_result):
        if stage_id != self.first_stage_id:
            raise ValueError("integration run must never reach a second motion stage")
        result = super().evaluate(stage_id=stage_id, context=context, skill_result=skill_result)
        result["maximum_executable_stages"] = 1
        if result["action"] == "continue":
            result.update(action="skip", reason="first-stage-only integration limit; suffix is not authorized",
                          physical_effect_verified=False, verified_effects=[], integration_limit_reached=True)
        return result
