"""Random-reset AutoCollect runner using object-frame trajectory retargeting."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.tasks.resolver import make_task_config

from .agent import AutoCollectAgent, PlanningRequest, StructuredPlanner
from .bucket_chain import BUCKET_CHAIN, bind_bucket_phase_chain, make_bucket_phase_manifests
from .chain import SkillChainContract
from .stage_checks import StageGate, make_stage_gate
from .recovery import RecoveryAction, RecoveryFeedback, RecoveryLimits, RecoverySession
from .core import (
    LowDimEpisodeRecorder,
    SkillContext,
    SkillExecutor,
    SkillStatus,
)
from .experiments import (
    EnvironmentIdentity,
    ExperimentLedger,
    ExperimentSpec,
    ExperimentStatus,
    PolicyIdentity,
)
from .harness import (
    FiniteActionSafetyGuard,
    GymAgentEnvironment,
    SkillRolloutEngine,
    TaskSuccessTerminationAdapter,
)
from .memory import SkillExperience, SkillMemory
from .perception import (
    PickBucketPerception,
    SceneEstimate,
    perception_configuration,
    perception_source_objects,
)
from .program import ProgramLimits, SafeSkillProgramExecutor
from .retarget import (
    BUCKET_SLICE,
    FOOD_SLICE,
    ObjectFrameRetargetSkill,
    PickBucketTrajectoryLibrary,
    RetargetCandidate,
    make_object_frame_retarget_manifest,
)
from .storage import save_episode
from .trace import TraceRecorder
from .verifiers import PickBucketLiftVerifier


def _dummy_render() -> list[np.ndarray]:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    return [frame, frame, frame, frame]


class _FixedHorizonTaskAdapter(TaskSuccessTerminationAdapter):
    """Keep task-success signals from shortening an atomic-skill rollout.

    Pick-bucket marks ``terminated=True`` as soon as its simulator success
    predicate fires.  The adapter suppresses only that terminal condition so
    the controller executes its fixed action horizon.  The success value is
    not used to choose an action; the independent verifier judges the final
    state after execution.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        raw = env.unwrapped
        self._nominal_dynamics = {
            "bucket_body_mass_kg": float(
                raw.model.body_mass[int(raw._bucket_body_id)]
            ),
            "payload_body_mass_kg": float(
                raw.model.body_mass[int(raw._boxed_food_body_id)]
            ),
            "bucket_hinge_frictionloss": float(
                raw.model.dof_frictionloss[int(raw._bucket_dof_id)]
            ),
        }

    @property
    def nominal_dynamics(self) -> dict[str, float]:
        return dict(self._nominal_dynamics)

def _make_env(
    config: Any,
    seed: int,
    *,
    randomize_dynamics: bool,
    capture_rgb: bool = False,
) -> Any:
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        randomize_dynamics=randomize_dynamics,
        hz=int(config.control_hz),
        seed=int(seed),
    )
    raw_env = env.unwrapped
    raw_env.hz = 1_000_000
    if not capture_rgb:
        raw_env.render = _dummy_render
    return _FixedHorizonTaskAdapter(env)


def _dynamics_snapshot(env: Any) -> dict[str, Any]:
    """Record simulator dynamics for reproducibility, never for control."""

    raw = env.unwrapped
    nominal = env.nominal_dynamics
    actual = {
        "bucket_body_mass_kg": float(
            raw.model.body_mass[int(raw._bucket_body_id)]
        ),
        "payload_body_mass_kg": float(
            raw.model.body_mass[int(raw._boxed_food_body_id)]
        ),
        "bucket_hinge_frictionloss": float(
            raw.model.dof_frictionloss[int(raw._bucket_dof_id)]
        ),
    }
    multipliers = {
        key.removesuffix("_kg").removesuffix("_frictionloss") + "_multiplier": (
            actual[key] / nominal[key] if nominal[key] != 0.0 else None
        )
        for key in actual
    }
    return {
        "actual": actual,
        "nominal": nominal,
        "multipliers": multipliers,
        "used_by_controller": False,
        "purpose": "reproducibility_and_failure_analysis",
    }


def _same_dynamics(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return all(
        np.isclose(first["actual"][key], second["actual"][key])
        for key in first["actual"]
    )


class PickBucketTaskPlugin:
    """First task plugin for the task-agnostic AutoCollect harness.

    Task-specific environment construction and verification live here.  The
    rollout engine only sees the generic ``GymAgentEnvironment`` interface.
    """

    name = "pick_bucket.grasp_lift"

    def __init__(
        self,
        *,
        task_id: str,
        minimum_retained_lift_m: float,
        randomize_dynamics: bool,
        capture_rgb: bool = False,
    ) -> None:
        self.task_id = str(task_id)
        self.minimum_retained_lift_m = float(minimum_retained_lift_m)
        self.randomize_dynamics = bool(randomize_dynamics)
        self.capture_rgb = bool(capture_rgb)
        self.config = make_task_config(self.task_id)
        if (
            getattr(self.config, "action_semantics", None)
            != "tcp_delta_pose_current_actual"
        ):
            raise ValueError(f"{self.task_id!r} is not a supported native-delta task")
        self.expected_action_dim = int(
            self.config.arm_action_dim + self.config.hand_action_dim
        )

    def validate_library(self, library: PickBucketTrajectoryLibrary) -> None:
        for episode in library.episodes:
            if episode.actions.shape[1] != self.expected_action_dim:
                raise ValueError(
                    "library action dimension does not match task configuration"
                )

    def make_environment(self, *, seed: int) -> GymAgentEnvironment:
        base_env = _make_env(
            self.config,
            seed,
            randomize_dynamics=self.randomize_dynamics,
            capture_rgb=self.capture_rgb,
        )
        return GymAgentEnvironment(
            task_id=self.task_id,
            env=base_env,
            verifier=PickBucketLiftVerifier(
                minimum_lift_m=self.minimum_retained_lift_m
            ),
            # Value bounds will be added per embodiment.  Dimension and finite
            # checks are already hard failures and do not change the baseline.
            safety_guard=FiniteActionSafetyGuard(
                expected_action_dim=self.expected_action_dim
            ),
            reset_metadata={
                "task_plugin": self.name,
                "random_reset": True,
                "randomize_dynamics": self.randomize_dynamics,
                "capture_rgb_for_perception": self.capture_rgb,
            },
        )


def _trajectory_library_sha256(library: PickBucketTrajectoryLibrary) -> str:
    """Fingerprint all actions and initial states used for skill retrieval."""

    digest = hashlib.sha256(b"dexjoco_pick_bucket_library_v1\0")
    for index, episode in enumerate(library.episodes):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        for array in (episode.initial_state, episode.actions):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _wrapped_angle_error_deg(first: float, second: float) -> float:
    difference = np.arctan2(np.sin(first - second), np.cos(first - second))
    return float(abs(np.degrees(difference)))


def _perception_errors(
    estimate: SceneEstimate,
    truth_state: np.ndarray,
) -> dict[str, float]:
    truth_bucket = np.asarray(truth_state[BUCKET_SLICE], dtype=np.float64)
    truth_food = np.asarray(truth_state[FOOD_SLICE], dtype=np.float64)
    estimated_bucket_rotation = Rotation.from_quat(
        estimate.bucket_pose7[3:7], scalar_first=True
    ).as_matrix()
    truth_bucket_rotation = Rotation.from_quat(
        truth_bucket[3:7], scalar_first=True
    ).as_matrix()
    estimated_food_rotation = Rotation.from_quat(
        estimate.food_pose7[3:7], scalar_first=True
    ).as_matrix()
    truth_food_rotation = Rotation.from_quat(
        truth_food[3:7], scalar_first=True
    ).as_matrix()
    estimated_bucket_yaw = float(
        np.arctan2(estimated_bucket_rotation[1, 0], estimated_bucket_rotation[0, 0])
    )
    truth_bucket_yaw = float(
        np.arctan2(truth_bucket_rotation[1, 0], truth_bucket_rotation[0, 0])
    )
    estimated_food_yaw = float(
        np.arctan2(estimated_food_rotation[1, 0], estimated_food_rotation[0, 0])
    )
    truth_food_yaw = float(
        np.arctan2(truth_food_rotation[1, 0], truth_food_rotation[0, 0])
    )
    return {
        "bucket_position_error_m": float(
            np.linalg.norm(estimate.bucket_pose7[:3] - truth_bucket[:3])
        ),
        "bucket_yaw_error_deg": _wrapped_angle_error_deg(
            estimated_bucket_yaw,
            truth_bucket_yaw,
        ),
        "food_position_error_m": float(
            np.linalg.norm(estimate.food_pose7[:3] - truth_food[:3])
        ),
        "food_yaw_error_deg": _wrapped_angle_error_deg(
            estimated_food_yaw,
            truth_food_yaw,
        ),
    }


def _run_random_pick_bucket_grasp_lift(
    *,
    task_id: str,
    library: PickBucketTrajectoryLibrary,
    output_dir: str | Path,
    seed: int,
    perception: PickBucketPerception,
    max_candidates: int = 5,
    minimum_retained_lift_m: float = 0.17,
    save_failed: bool = False,
    randomize_dynamics: bool = False,
    experiment_id: str | None = None,
    study_id: str | None = None,
    skill_memory: SkillMemory | None = None,
    planner: StructuredPlanner | None = None,
    skill_mode: str = "monolithic",
    stage_check_mode: str = "off",
) -> dict[str, Any]:
    """Try ranked retargeted skills on one reproducible random reset."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    chain = BUCKET_CHAIN if skill_mode == "phase_chain" else None
    execution_name = chain.chain_id if chain else "grasp_lift.object_frame_retarget"
    stage_gate = make_stage_gate(stage_check_mode, [stage.stage_id for stage in BUCKET_CHAIN.stages])
    recovery = RecoverySession(RecoveryLimits(maximum_observations=1, maximum_executions=max_candidates)) if chain else None
    minimum_retained_lift_m = float(minimum_retained_lift_m)
    if (
        not np.isfinite(minimum_retained_lift_m)
        or minimum_retained_lift_m < 0.15
    ):
        raise ValueError("minimum_retained_lift_m must be finite and at least 0.15")
    if perception is None:
        raise ValueError("an explicit perception backend is required")
    plugin = PickBucketTaskPlugin(
        task_id=task_id,
        minimum_retained_lift_m=minimum_retained_lift_m,
        randomize_dynamics=randomize_dynamics,
        capture_rgb=bool(getattr(perception, "requires_rgb", False)),
    )
    plugin.validate_library(library)
    config = plugin.config

    initial_state: np.ndarray | None = None
    candidates: tuple[RetargetCandidate, ...] | None = None
    scene_estimate: SceneEstimate | None = None
    perception_errors: dict[str, float] | None = None
    dynamics_parameters: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    selected_episode_dir: Path | None = None
    trace_root = Path(output_dir).expanduser().resolve().parent / "attempt_traces"

    for candidate_index in range(max_candidates):
        if candidates is not None and candidate_index >= len(candidates):
            break
        trace = TraceRecorder(
            attempt_id=(
                f"{task_id}:seed={int(seed)}:candidate={candidate_index + 1}"
            ),
            experiment_id=experiment_id,
            study_id=study_id,
        )
        trace_path = (
            trace_root
            / f"seed_{int(seed)}"
            / (
                f"candidate_{candidate_index + 1:02d}_"
                f"{trace.trace_id}.jsonl"
            )
        )
        env = plugin.make_environment(seed=seed)
        attempt: dict[str, Any] | None = None
        success = False
        episode_dir: Path | None = None
        try:
            with trace.span(
                "candidate.attempt",
                category="attempt",
                start_step=0,
                inputs={
                    "task_id": task_id,
                    "seed": int(seed),
                    "candidate_index": candidate_index + 1,
                },
            ) as attempt_span:
                with trace.span(
                    "environment.reset",
                    category="environment",
                    start_step=0,
                    inputs={"seed": int(seed)},
                ) as reset_span:
                    observation, _info = env.reset(seed=int(seed))
                    current_initial_state = np.asarray(
                        observation["state"], dtype=np.float64
                    ).copy()
                    current_dynamics = _dynamics_snapshot(env)
                    if initial_state is None:
                        initial_state = current_initial_state
                        dynamics_parameters = current_dynamics
                    elif not np.array_equal(initial_state, current_initial_state):
                        raise RuntimeError(
                            "recreated environment did not reproduce the same "
                            "random initial state"
                        )
                    elif not _same_dynamics(
                        dynamics_parameters,
                        current_dynamics,
                    ):
                        raise RuntimeError(
                            "recreated environment did not reproduce the same dynamics"
                        )
                    assert env.last_reset_report is not None
                    reset_span.succeed(
                        end_step=0,
                        outputs={
                            "reset": env.last_reset_report.to_dict(),
                            "dynamics_audit": current_dynamics,
                        },
                    )

                if scene_estimate is None:
                    if recovery:
                        recovery.begin_observation("initial_depth")
                    with trace.span(
                        "perception.estimate_scene",
                        category="perception",
                        start_step=0,
                        inputs={"backend": perception.name},
                    ) as perception_span:
                        scene_estimate = perception.estimate(env, observation)
                        if recovery:
                            recovery.accept_observation()
                        perception_errors = _perception_errors(
                            scene_estimate,
                            initial_state,
                        )
                        perception_span.succeed(
                            end_step=0,
                            outputs={
                                "scene_estimate": scene_estimate.to_dict(),
                                "offline_truth_error_audit": {
                                    **perception_errors,
                                    "used_by_controller": False,
                                },
                            },
                        )
                    with trace.span(
                        "candidate.retrieve",
                        category="retrieval",
                        start_step=0,
                        inputs={
                            "library_size": len(library.episodes),
                            "max_candidates": int(max_candidates),
                        },
                    ) as retrieval_span:
                        candidates = library.select_scene(
                            scene_estimate.bucket_pose7,
                            scene_estimate.food_pose7,
                            max_candidates=max_candidates,
                        )
                        retrieval_span.succeed(
                            end_step=0,
                            outputs={
                                "candidates": [
                                    {
                                        "rank": item.rank,
                                        "distance": item.distance,
                                        "source_demo": str(item.episode.source),
                                    }
                                    for item in candidates
                                ]
                            },
                        )
                else:
                    with trace.span(
                        "perception.reuse_scene_estimate",
                        category="perception",
                        start_step=0,
                        inputs={"backend": scene_estimate.backend},
                    ) as reuse_span:
                        reuse_span.succeed(
                            end_step=0,
                            outputs={
                                "scene_estimate": scene_estimate.to_dict(),
                                "reason": (
                                    "candidate retries recreate the same seeded scene"
                                ),
                            },
                        )

                assert candidates is not None
                assert scene_estimate is not None
                assert perception_errors is not None
                assert dynamics_parameters is not None
                candidate = candidates[candidate_index]
                if recovery:
                    recovery.begin_execution(f"candidate_{candidate.rank}")
                with trace.span(
                    "candidate.select",
                    category="retrieval",
                    start_step=0,
                    inputs={"candidate_index": candidate_index + 1},
                ) as selection_span:
                    selection_span.succeed(
                        end_step=0,
                        outputs={
                            "candidate_rank": candidate.rank,
                            "candidate_distance": candidate.distance,
                            "source_demo": str(candidate.episode.source),
                        },
                    )

                recorder = LowDimEpisodeRecorder(
                    control_hz=float(config.control_hz)
                )
                context = SkillContext(
                    env=env,
                    task_id=task_id,
                    config=config,
                    observation=observation,
                    recorder=recorder,
                    trace=trace,
                    metadata={
                        "source_demo": str(candidate.episode.source),
                        "candidate_rank": candidate.rank,
                    },
                    maximum_total_steps=recovery.remaining_steps if recovery else None,
                    deadline_monotonic=recovery.deadline_monotonic if recovery else None,
                )
                skill_manifest = make_object_frame_retarget_manifest(
                    task_id=task_id,
                    expected_action_dim=plugin.expected_action_dim,
                )
                executor = SkillExecutor(require_manifests=True)
                if chain:
                    for skill, manifest in zip(bind_bucket_phase_chain(candidate), make_bucket_phase_manifests(
                            task_id=task_id, expected_action_dim=plugin.expected_action_dim)):
                        executor.register(skill, manifest=manifest)
                else:
                    executor.register(ObjectFrameRetargetSkill(candidate), manifest=skill_manifest)
                invocation_parameters = {
                    "target_bucket_pose7": scene_estimate.bucket_pose7,
                    "target_pose_source": scene_estimate.backend,
                }
                agent_decision: dict[str, Any] | None = None
                chain_report = None
                stage_gates = ()
                recovery_decision = None
                if planner is None:
                    rollout = SkillRolloutEngine().run(
                        environment=env,
                        executor=executor,
                        context=context,
                        skill_name=skill_manifest.name,
                        parameters=invocation_parameters,
                        trace=trace,
                    )
                    skill_result = rollout.skill
                    verification = rollout.verification
                    success = rollout.success
                    failure_stage = rollout.failure_stage
                    reset_report = rollout.reset
                else:
                    prior_feedback: tuple[dict[str, Any], ...] = ()
                    if skill_memory is not None:
                        memory_summary = skill_memory.failure_summary(
                            task_id=task_id,
                            skill_name=execution_name,
                        )
                        if memory_summary["failure_count"]:
                            prior_feedback = (memory_summary,)
                    if recovery:
                        prior_feedback += tuple(recovery.history)
                    request = PlanningRequest(
                        task_id=task_id,
                        objective=(
                            "grasp the perceived bucket handle, lift the bucket "
                            "while retaining the payload, and hold it for verification"
                        ),
                        observation_summary={
                            "scene_estimate": scene_estimate.to_dict(),
                            "required_skill_chain": chain.to_dict() if chain else None,
                            "stage_checks": stage_gate.configuration() if stage_gate else None,
                            "retrieved_candidate": {
                                "rank": candidate.rank,
                                "distance": candidate.distance,
                                "source_demo": str(candidate.episode.source),
                            },
                        },
                        skill_catalog=executor.catalog(),
                        constraints=(
                            "use only manifested skills in the required order",
                            "copy poses only from the supplied perception result",
                            "do not infer or request simulator object state",
                            "do not emit Python or shell commands",
                        ),
                        prior_feedback=prior_feedback,
                        locked_parameters={
                            name: invocation_parameters for name in executor.skill_names
                        },
                        attempt_index=candidate_index + 1,
                    )
                    agent_run = AutoCollectAgent(
                        planner=planner,
                        skill_executor=executor,
                        chain_contract=chain,
                        stage_gate=stage_gate,
                        limits=ProgramLimits(
                            maximum_invocations=3 if chain else 1,
                            maximum_total_steps=1000,
                            maximum_wall_time_seconds=120.0,
                        ),
                    ).run(
                        request=request,
                        environment=env,
                        context=context,
                        trace=trace,
                    )
                    execution = agent_run.execution
                    stage_gates = execution.stage_gates
                    skill_result = chain.aggregate_motion_result(execution.skill_results, stage_gates) if chain else execution.skill_results[-1]
                    chain_report = execution.skill_chain
                    verification = execution.verification
                    success = execution.success
                    failure_stage = execution.failure_stage
                    assert env.last_reset_report is not None
                    reset_report = env.last_reset_report
                    agent_decision = agent_run.decision.to_dict()
                if recovery:
                    recovery.record_steps(context.total_steps)
                    if not success and skill_result.metrics.get("exception_type") == "ExecutionBudgetExceeded":
                        failure_stage = "budget"
                    elif not success and skill_result.metrics.get("exception_type") and failure_stage != "safety":
                        failure_stage = "infrastructure"
                    recovery_decision = recovery.decide(RecoveryFeedback(
                        "success" if success else failure_stage or "policy", "candidate_finished",
                        ("independent verifier passed" if success else skill_result.failure_reason or verification.reason or "candidate failed")[:2048]),
                        another_candidate=candidate_index + 1 < len(candidates), reset_supported=True)
                    if success and recovery_decision["action"] != "complete":
                        success, failure_stage = False, "budget"
                        skill_result = replace(skill_result, status=SkillStatus.FAILED,
                                               failure_reason="scene budget expired before collection acceptance")
                is_last_candidate = (candidate_index == len(candidates) - 1
                                     or bool(recovery_decision and recovery_decision["action"] != "reset_and_replan"))
                should_save = success or (save_failed and is_last_candidate)
                with trace.span(
                    "storage.episode",
                    category="storage",
                    start_step=context.total_steps,
                    inputs={
                        "save_failed": bool(save_failed),
                        "should_save": should_save,
                        "low_dimensional_only": True,
                    },
                ) as storage_span:
                    if should_save:
                        episode_dir = save_episode(
                            output_dir=output_dir,
                            task_id=task_id,
                            seed=seed,
                            recorder=recorder,
                            skill_result=skill_result,
                            verification=verification,
                            source_demo=candidate.episode.source,
                            backend="object_frame_retarget",
                            extra_metadata={
                                "candidate_rank": candidate.rank,
                                "candidate_distance": candidate.distance,
                                "library_size": len(library.episodes),
                                "random_reset": True,
                                "randomize_dynamics": bool(randomize_dynamics),
                                "minimum_retained_lift_m": (
                                    minimum_retained_lift_m
                                ),
                                "dynamics_parameters": dynamics_parameters,
                                "initial_state": initial_state,
                                "perception": scene_estimate.to_dict(),
                                "perception_errors": perception_errors,
                                "reset": reset_report.to_dict(),
                                "safety": env.safety_summary,
                                "skill_manifest": skill_manifest.to_dict() if not chain else None,
                                "skill_mode": skill_mode, "skill_catalog": executor.catalog(),
                                "skill_chain": chain_report, "stage_gates": list(stage_gates),
                                "recovery": recovery.to_dict() if recovery else None,
                                "agent_decision": agent_decision,
                                "trace_schema_version": 1,
                                "trace_id": trace.trace_id,
                                "trace_path": str(trace_path),
                            },
                        )
                        storage_span.succeed(
                            end_step=context.total_steps,
                            outputs={"episode_dir": str(episode_dir)},
                            artifact_refs=(str(episode_dir),),
                        )
                    else:
                        storage_span.skip(
                            end_step=context.total_steps,
                            reason=(
                                "failed candidate remains trace-only because "
                                "failed episode retention is disabled"
                            ),
                            outputs={"trace_path": str(trace_path)},
                        )

                attempt = {
                    "candidate_rank": candidate.rank,
                    "candidate_distance": candidate.distance,
                    "source_demo": str(candidate.episode.source),
                    "status": "pass" if success else "fail",
                    "episode_dir": str(episode_dir) if episode_dir else None,
                    "skill": skill_result.to_dict(),
                    "skill_chain": chain_report, "stage_gates": list(stage_gates),
                    "recovery_decision": recovery_decision,
                    "verification": verification.to_dict(),
                    "perception_backend": scene_estimate.backend,
                    "failure_stage": failure_stage,
                    "reset": reset_report.to_dict(),
                    "safety": env.safety_summary,
                    "agent_decision": agent_decision,
                    "trace_id": trace.trace_id,
                    "trace_path": str(trace_path),
                }
                artifact_refs = (str(trace_path),)
                if episode_dir is not None:
                    artifact_refs += (str(episode_dir),)
                if success:
                    attempt_span.succeed(
                        end_step=context.total_steps,
                        outputs={
                            "status": "pass",
                            "candidate_rank": candidate.rank,
                        },
                        artifact_refs=artifact_refs,
                    )
                else:
                    candidate_failure_stage = failure_stage or "candidate"
                    attempt_span.fail(
                        end_step=context.total_steps,
                        failure_code=f"{candidate_failure_stage}_failed",
                        reason=(
                            skill_result.failure_reason
                            or verification.reason
                            or "candidate attempt failed"
                        ),
                        outputs={
                            "status": "fail",
                            "candidate_rank": candidate.rank,
                        },
                        artifact_refs=artifact_refs,
                    )
        finally:
            try:
                env.close()
            finally:
                if trace.events:
                    trace.write_jsonl(trace_path)

        if attempt is None:
            raise RuntimeError("candidate attempt ended without a structured result")
        attempt["trace_summary"] = trace.summary()
        if skill_memory is not None:
            artifacts: list[str | Path] = [trace_path]
            if episode_dir is not None:
                artifacts.append(episode_dir)
            experience = SkillExperience.create(
                task_id=task_id,
                skill_name=execution_name,
                skill_version=skill_manifest.version,
                success=success,
                parameters={
                    "candidate_rank": candidate.rank,
                    "candidate_distance": candidate.distance,
                    "source_demo": str(candidate.episode.source),
                    "target_pose_source": scene_estimate.backend,
                },
                scene_summary={
                    "perception_backend": scene_estimate.backend,
                    "perception_confidence": scene_estimate.confidence,
                    "perception_error_audit": perception_errors,
                    "randomize_dynamics": bool(randomize_dynamics),
                },
                skill_metrics=skill_result.metrics,
                verification=verification.to_dict(),
                trace_id=trace.trace_id,
                trace_path=trace_path,
                experiment_id=experiment_id,
                study_id=study_id,
                failure_stage=failure_stage,
                failure_codes=() if success else tuple(
                    event.failure_code
                    for event in trace.events
                    if event.failure_code is not None
                    and event.category not in {"attempt", "program"}
                ),
                artifact_refs=artifacts,
                eligible_as_training_data=success and episode_dir is not None,
                tags=("autocollect", "pick_bucket", "no_vla"),
            )
            memory_record = skill_memory.append(experience)
            attempt["experience_id"] = experience.experience_id
            attempt["memory_record"] = str(memory_record)
        attempts.append(attempt)
        if success:
            selected_episode_dir = episode_dir
            break
        if recovery_decision and recovery_decision["action"] != RecoveryAction.RESET_AND_REPLAN.value:
            break

    success = selected_episode_dir is not None
    return {
        "status": "pass" if success else "fail",
        "task_id": task_id,
        "task_plugin": plugin.name,
        "seed": int(seed),
        "vla_used": False,
        "skill_mode": skill_mode, "stage_check_mode": stage_check_mode,
        "recovery": recovery.to_dict() if recovery else None,
        "agent_planner": (
            {
                "name": planner.name,
                "model_id": planner.model_id,
            }
            if planner is not None
            else None
        ),
        "backend": "object_frame_retarget",
        "random_reset": True,
        "randomize_dynamics": bool(randomize_dynamics),
        "minimum_retained_lift_m": minimum_retained_lift_m,
        "dynamics_parameters": dynamics_parameters,
        "perception_backend": perception.name,
        "perception": scene_estimate.to_dict() if scene_estimate else None,
        "perception_errors": perception_errors,
        "library_size": len(library.episodes),
        "candidates_attempted": len(attempts),
        "attempt_trace_root": str(trace_root),
        "skill_memory_root": (
            str(skill_memory.root) if skill_memory is not None else None
        ),
        "selected_candidate_rank": (
            attempts[-1]["candidate_rank"] if success else None
        ),
        "episode_dir": str(selected_episode_dir) if selected_episode_dir else None,
        "attempts": attempts,
    }


def run_random_pick_bucket_grasp_lift(
    *,
    task_id: str,
    library: PickBucketTrajectoryLibrary,
    output_dir: str | Path,
    seed: int,
    perception: PickBucketPerception,
    max_candidates: int = 5,
    minimum_retained_lift_m: float = 0.17,
    save_failed: bool = False,
    randomize_dynamics: bool = False,
    experiment_ledger: ExperimentLedger | None = None,
    hypothesis: str = "object-frame retargeting is a viable first reusable skill",
    parent_experiment_id: str | None = None,
    study_id: str | None = None,
    skill_memory: SkillMemory | None = None,
    planner: StructuredPlanner | None = None,
    skill_mode: str = "monolithic",
    stage_check_mode: str = "off",
) -> dict[str, Any]:
    """Run one scene and optionally audit it in an experiment ledger."""

    if perception is None:
        raise ValueError("an explicit perception backend is required")
    if skill_mode not in {"monolithic", "phase_chain"}:
        raise ValueError("unknown skill mode")
    if skill_mode == "phase_chain" and planner is None:
        raise ValueError("phase_chain requires a structured planner")
    gate = make_stage_gate(stage_check_mode, [stage.stage_id for stage in BUCKET_CHAIN.stages])
    if gate and skill_mode != "phase_chain":
        raise ValueError("stage checks require phase_chain")
    planner_sources: tuple[Any, ...] = ()
    if planner is not None:
        planner_sources = (type(planner),)
        generate_fn = getattr(planner, "generate_fn", None)
        if callable(generate_fn):
            planner_sources += (generate_fn,)
    policy_configuration = {
        "skill_mode": skill_mode,
        "stage_check_configuration": gate.configuration() if gate else None,
        "required_skill_chain": BUCKET_CHAIN.to_dict() if skill_mode == "phase_chain" else None,
        "recovery_limits": RecoveryLimits(maximum_observations=1, maximum_executions=max_candidates).to_dict() if skill_mode == "phase_chain" else None,
        "max_candidates": int(max_candidates),
        "save_failed": bool(save_failed),
        "perception": (
            perception_configuration(perception)
            if perception is not None
            else None
        ),
        "library_size": len(library.episodes),
        "trajectory_library_sha256": _trajectory_library_sha256(library),
        "trace_schema_version": 1,
        "planner": (
            {"name": planner.name, "model_id": planner.model_id}
            if planner is not None
            else None
        ),
    }
    environment_configuration = {
        "task_id": task_id,
        "minimum_retained_lift_m": float(minimum_retained_lift_m),
        "randomize_dynamics": bool(randomize_dynamics),
    }
    run_arguments = {**policy_configuration, **environment_configuration}
    handle = None
    if experiment_ledger is not None:
        identity = PolicyIdentity.from_sources(
            name=BUCKET_CHAIN.chain_id if skill_mode == "phase_chain" else "grasp_lift.object_frame_retarget",
            version="1",
            sources=(
                ObjectFrameRetargetSkill,
                bind_bucket_phase_chain, SkillChainContract, StageGate, AutoCollectAgent,
                SafeSkillProgramExecutor, RecoverySession, _run_random_pick_bucket_grasp_lift,
                *perception_source_objects(perception),
                SkillRolloutEngine,
                TraceRecorder,
                SkillExecutor,
                make_object_frame_retarget_manifest,
                *planner_sources,
            ),
            configuration=policy_configuration,
        )
        environment_identity = EnvironmentIdentity.from_sources(
            name="pick_bucket.grasp_lift.environment",
            version="1",
            sources=(PickBucketTaskPlugin, PickBucketLiftVerifier, GymAgentEnvironment),
            configuration=environment_configuration,
        )
        handle = experiment_ledger.start(
            ExperimentSpec(
                task_id=task_id,
                seed=int(seed),
                policy=identity,
                hypothesis=hypothesis,
                environment=environment_identity,
                parameters=run_arguments,
                parent_experiment_id=parent_experiment_id,
                study_id=study_id,
                tags=("autocollect", "no_vla", "pick_bucket"),
            ),
            metadata={
                "framework": "enpire_style_environment_rollout_verification",
                "controller_uses_oracle_state": False,
                "structured_agent_enabled": planner is not None,
                "attempt_trace_root": str(
                    Path(output_dir).expanduser().resolve().parent
                    / "attempt_traces"
                ),
            },
        )

    try:
        result = _run_random_pick_bucket_grasp_lift(
            task_id=task_id,
            library=library,
            output_dir=output_dir,
            seed=seed,
            perception=perception,
            max_candidates=max_candidates,
            minimum_retained_lift_m=minimum_retained_lift_m,
            save_failed=save_failed,
            randomize_dynamics=randomize_dynamics,
            experiment_id=handle.experiment_id if handle is not None else None,
            study_id=study_id,
            skill_memory=skill_memory,
            planner=planner,
            skill_mode=skill_mode,
            stage_check_mode=stage_check_mode,
        )
    except Exception as exc:
        if handle is not None:
            experiment_ledger.finish(
                handle,
                status=ExperimentStatus.ERROR,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        raise

    if handle is not None:
        result["experiment_id"] = handle.experiment_id
        result["experiment_record"] = str(handle.record_path)
        experiment_ledger.finish(
            handle,
            status=(
                ExperimentStatus.PASSED
                if result["status"] == "pass"
                else ExperimentStatus.FAILED
            ),
            result=result,
        )
    return result
