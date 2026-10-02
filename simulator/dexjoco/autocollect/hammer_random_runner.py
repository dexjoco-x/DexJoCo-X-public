"""Random-reset hammer-nail collection through the structured Agent harness."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .agent import AutoCollectAgent, PlanningRequest, StructuredPlanner
from .bailian_collection import BailianCollectionController, CollectionPlanningStopped
from .chain import SkillChainContract
from .core import LowDimEpisodeRecorder, SkillContext, SkillExecutor, SkillStatus
from .experiments import (
    EnvironmentIdentity,
    ExperimentLedger,
    ExperimentSpec,
    ExperimentStatus,
    PolicyIdentity,
)
from .hammer import HammerNailTaskPlugin, _dummy_render
from .hammer_chain import HAMMER_CHAIN, bind_hammer_phase_chain, make_hammer_phase_manifests
from .hammer_perception import HammerNailPerception, HammerNailSceneEstimate
from .hammer_recovery import (
    observe_hammer_with_recovery, record_recovery_decision, reset_fingerprint,
)
from .hammer_retarget import (
    HAMMER_SLICE,
    NAIL_SLICE,
    HammerNailTrajectoryLibrary,
    TwoFrameHammerRetargetSkill,
    make_hammer_frame_retarget_manifest,
    infer_hammer_phase_plan,
)
from .harness import GymAgentEnvironment, SkillRolloutEngine
from .memory import SkillExperience, SkillMemory
from .perception import perception_configuration, perception_source_objects
from .program import ProgramLimits, SafeSkillProgramExecutor
from .recovery import (
    PerceptionEvidenceError, RecoveryAction, RecoveryBudgetExceeded,
    RecoveryFeedback, RecoveryLimits, RecoverySession,
)
from .storage import save_episode
from .stage_checks import StageGate, make_stage_gate
from .visual_effect_runtime import StageBaselineRejected, VisualEffectRuntime
from .visual_effects import VisualLiftCoMotionCheck
from .trace import TraceRecorder
from .verifiers import HammerNailDepthVerifier


class HammerPerceptionRejected(RuntimeError):
    """A scene rejected before retrieval/control, carrying its saved trace ID."""

    def __init__(self, cause: Exception, *, trace: TraceRecorder, trace_path: Path):
        super().__init__(str(cause))
        self.cause_type = type(cause).__name__
        self.trace_id = trace.trace_id
        self.trace_path = trace_path
        self.observation_attempts = getattr(cause, "observation_attempts", [])


def _trajectory_library_sha256(library: HammerNailTrajectoryLibrary) -> str:
    digest = hashlib.sha256(b"dexjoco_hammer_nail_library_v2\0")
    digest.update(np.ascontiguousarray(library.feature_scales).tobytes())
    for index, episode in enumerate(library.episodes):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        for array in (episode.states, episode.actions):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _rotation_error_deg(first_pose7: np.ndarray, second_pose7: np.ndarray) -> float:
    first = Rotation.from_quat(
        np.asarray(first_pose7, dtype=np.float64)[3:7], scalar_first=True
    )
    second = Rotation.from_quat(
        np.asarray(second_pose7, dtype=np.float64)[3:7], scalar_first=True
    )
    return float(np.degrees((first * second.inv()).magnitude()))


def _perception_errors(
    estimate: HammerNailSceneEstimate,
    truth_state: np.ndarray,
) -> dict[str, float]:
    """Offline audit only; returned values never enter retrieval or control."""

    truth = np.asarray(truth_state, dtype=np.float64).reshape(-1)
    truth_hammer = truth[HAMMER_SLICE]
    truth_nail = truth[NAIL_SLICE]
    return {
        "hammer_position_error_m": float(
            np.linalg.norm(estimate.hammer_pose7[:3] - truth_hammer[:3])
        ),
        "hammer_orientation_error_deg": _rotation_error_deg(
            estimate.hammer_pose7,
            truth_hammer,
        ),
        "nail_position_error_m": float(
            np.linalg.norm(estimate.nail_pose7[:3] - truth_nail[:3])
        ),
        "nail_orientation_error_deg": _rotation_error_deg(
            estimate.nail_pose7,
            truth_nail,
        ),
    }


def _run_random_hammer_nail(
    *,
    task_id: str,
    library: HammerNailTrajectoryLibrary,
    output_dir: str | Path,
    seed: int,
    perception: HammerNailPerception,
    max_candidates: int,
    minimum_insertion_depth_m: float,
    save_failed: bool,
    randomize_scene: bool,
    randomize_dynamics: bool,
    experiment_id: str | None,
    study_id: str | None,
    skill_memory: SkillMemory | None,
    planner: StructuredPlanner | None,
    skill_mode: str,
    stage_check_mode: str,
    recovery_session: RecoverySession,
    reobserve_cameras: tuple[str, ...],
    stage_cameras: tuple[str, ...],
    require_stage_baselines: bool,
    collection_controller: BailianCollectionController | None = None,
) -> dict[str, Any]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    chain_contract = HAMMER_CHAIN if skill_mode == "phase_chain" else None
    execution_name = HAMMER_CHAIN.chain_id if chain_contract else "hammer_nail.two_frame_retarget"
    visual_mode = stage_check_mode == "visual_lift_comotion"
    stage_gate = make_stage_gate("tcp_tracking" if visual_mode else stage_check_mode,
                                [stage.stage_id for stage in HAMMER_CHAIN.stages])
    plugin = HammerNailTaskPlugin(
        task_id=task_id,
        minimum_insertion_depth_m=minimum_insertion_depth_m,
        randomize_scene=randomize_scene,
        randomize_dynamics=randomize_dynamics,
        capture_rgb=bool(perception.requires_rgb),
    )
    for episode in library.episodes:
        plugin.validate_episode(episode)

    output_dir = Path(output_dir).expanduser().resolve()
    trace_root = output_dir.parent / "attempt_traces" / f"seed_{int(seed)}"
    trace_root.mkdir(parents=True, exist_ok=True)
    scene_estimate: HammerNailSceneEstimate | None = None
    perception_errors: dict[str, float] | None = None
    candidates = None
    attempts: list[dict[str, Any]] = []
    selected_episode_dir: Path | None = None
    selected_training_eligible = False
    privileged_calibration = bool(perception.privileged_calibration_only)
    initial_reset_fingerprint = None
    observation_attempts = []

    for candidate_index in range(min(max_candidates, len(library.episodes))):
        trace = TraceRecorder(
            attempt_id=(
                f"{task_id}:seed={int(seed)}:candidate={candidate_index + 1}"
            ),
            experiment_id=experiment_id,
            study_id=study_id,
        )
        trace_path = trace_root / f"candidate_{candidate_index + 1}.jsonl"
        env: GymAgentEnvironment = plugin.make_environment(seed=seed)
        attempt: dict[str, Any] | None = None
        episode_dir: Path | None = None
        success = False
        skill_manifest = make_hammer_frame_retarget_manifest(
            task_id=task_id,
            expected_action_dim=plugin.expected_action_dim,
        )
        try:
            with trace.span(
                "candidate.attempt",
                category="attempt",
                start_step=0,
                inputs={
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
                    observation, _reset_info = env.reset(seed=seed)
                    initial_state = np.asarray(
                        observation["state"], dtype=np.float64
                    ).copy()
                    assert env.last_reset_report is not None
                    current_fingerprint = reset_fingerprint(env)
                    if initial_reset_fingerprint is None:
                        initial_reset_fingerprint = current_fingerprint
                    elif current_fingerprint != initial_reset_fingerprint:
                        raise RuntimeError("candidate reset differs from the perceived scene; refusing stale poses")
                    reset_span.succeed(
                        end_step=0,
                        outputs={"reset": env.last_reset_report.to_dict(),
                                 "same_scene_fingerprint": current_fingerprint,
                                 "reset_matches_first_attempt": True},
                    )

                if scene_estimate is None:
                    with trace.span(
                        "perception.hammer_nail",
                        category="perception",
                        start_step=0,
                        inputs={"backend": perception.name},
                    ) as perception_span:
                        try:
                            if collection_controller is not None:
                                scene_estimate, observation_attempts = collection_controller.prepare_reset(
                                    env=env, perception=perception,
                                    cameras=(perception.camera_key, *reobserve_cameras), session=recovery_session,
                                    output=output_dir.parent / "bailian_collection" / f"seed_{seed}",
                                    attempt_id=trace.attempt_id,
                                )
                            else:
                                scene_estimate, observation_attempts = observe_hammer_with_recovery(
                                    perception=perception, env=env, observation=observation,
                                    cameras=reobserve_cameras, session=recovery_session,
                                    trace=trace, trace_path=trace_path, skill_memory=skill_memory,
                                    experiment_id=experiment_id, study_id=study_id,
                                )
                        except PerceptionEvidenceError as exc:
                            raise HammerPerceptionRejected(
                                exc, trace=trace, trace_path=trace_path,
                            ) from exc
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
                            scene_estimate.hammer_pose7,
                            scene_estimate.nail_pose7,
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
                                "reason": "candidate retry recreates the same seeded reset",
                            },
                        )

                # This prototype consumes reset RGB only. No per-step RGB/video
                # is recorded; disable expensive unused rendering for control.
                env.unwrapped.render = _dummy_render
                assert candidates is not None
                assert scene_estimate is not None
                assert perception_errors is not None
                candidate = candidates[candidate_index]
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

                recorder = LowDimEpisodeRecorder(control_hz=float(plugin.control_hz))
                context = SkillContext(
                    env=env,
                    task_id=task_id,
                    config=plugin.config,
                    observation=observation,
                    recorder=recorder,
                    trace=trace,
                    metadata={
                        "source_demo": str(candidate.episode.source),
                        "candidate_rank": candidate.rank,
                    },
                    maximum_total_steps=recovery_session.remaining_steps,
                    deadline_monotonic=recovery_session.deadline_monotonic,
                )
                visual_runtime = None
                if visual_mode:
                    plan = infer_hammer_phase_plan(candidate.episode)
                    visual_runtime = VisualEffectRuntime(
                        context=context,
                        stage_ends=dict(zip([stage.stage_id for stage in HAMMER_CHAIN.stages],
                                            [plan.transition_start_step, plan.transition_end_step, plan.total_steps])),
                        provider=perception.mask_provider,
                        artifact_root=output_dir.parent / "stage_vision" / f"seed_{seed}" / f"candidate_{candidate.rank}",
                        initial_search_position=scene_estimate.hammer_pose7[:3],
                        table_height_m=scene_estimate.diagnostics.get("estimated_table_height_m"),
                        cameras=stage_cameras,
                    )
                    visual_runtime.capture_due(context)
                    if require_stage_baselines:
                        with trace.span("perception.stage_baseline_preflight", category="perception", start_step=0) as baseline_span:
                            baseline_report = visual_runtime.require_baselines()
                            baseline_span.succeed(end_step=0, outputs=baseline_report)
                    context.step_observer = visual_runtime.capture_due
                    stage_gate = (collection_controller.make_stage_gate(visual_runtime)
                                  if collection_controller is not None else visual_runtime.make_gate())
                recovery_session.begin_execution(str(candidate.episode.source))
                executor = SkillExecutor(require_manifests=True)
                if chain_contract is not None:
                    manifests = make_hammer_phase_manifests(
                        task_id=task_id, expected_action_dim=plugin.expected_action_dim,
                    )
                    for skill, manifest in zip(bind_hammer_phase_chain(candidate), manifests):
                        executor.register(skill, manifest=manifest)
                else:
                    executor.register(TwoFrameHammerRetargetSkill(candidate), manifest=skill_manifest)
                invocation_parameters = {
                    "target_hammer_pose7": scene_estimate.hammer_pose7,
                    "target_nail_pose7": scene_estimate.nail_pose7,
                    "target_pose_source": scene_estimate.backend,
                }
                agent_decision: dict[str, Any] | None = None
                chain_report: dict[str, Any] | None = None
                stage_gates = ()
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
                    dispatch_planner = planner
                    prior_feedback = tuple(recovery_session.history)
                    if skill_memory is not None and collection_controller is None:
                        summary = skill_memory.failure_summary(
                            task_id=task_id,
                            skill_name=execution_name,
                        )
                        if summary["failure_count"]:
                            prior_feedback += (summary,)
                    request = (collection_controller.approved.request if collection_controller is not None else PlanningRequest(
                        task_id=task_id,
                        objective=(
                            "grasp the perceived hammer, align its face to the "
                            "perceived nail, and drive the nail into the board"
                        ),
                        observation_summary={
                            "scene_estimate": scene_estimate.to_dict(),
                            "required_skill_chain": chain_contract.to_dict() if chain_contract else None,
                            "stage_checks": stage_gate.configuration() if stage_gate else None,
                            "retrieved_candidate": {
                                "rank": candidate.rank,
                                "distance": candidate.distance,
                                "source_demo": str(candidate.episode.source),
                            },
                        },
                        skill_catalog=executor.catalog(),
                        constraints=(
                            "use only manifested skills in the required chain order when supplied",
                            "copy both poses only from the supplied perception result",
                            "do not request simulator object state or nail depth",
                            "do not emit Python or shell commands",
                        ),
                        prior_feedback=prior_feedback,
                        locked_parameters={
                            name: invocation_parameters for name in executor.skill_names
                        },
                        attempt_index=candidate_index + 1,
                    ))
                    if collection_controller is not None:
                        dispatch_planner = collection_controller.approved
                    agent_run = AutoCollectAgent(
                        planner=dispatch_planner,
                        skill_executor=executor,
                        chain_contract=chain_contract,
                        stage_gate=stage_gate,
                        limits=ProgramLimits(
                            maximum_invocations=len(chain_contract.stages) if chain_contract else 1,
                            maximum_total_steps=1000,
                            maximum_wall_time_seconds=180.0,
                        ),
                    ).run(
                        request=request,
                        environment=env,
                        context=context,
                        trace=trace,
                    )
                    execution = agent_run.execution
                    stage_gates = execution.stage_gates
                    if chain_contract is not None:
                        skill_result = chain_contract.aggregate_motion_result(execution.skill_results, stage_gates)
                        chain_report = skill_result.metrics["chain"]
                    else:
                        skill_result = execution.skill_results[-1]
                    verification = execution.verification
                    success = execution.success
                    failure_stage = execution.failure_stage
                    assert env.last_reset_report is not None
                    reset_report = env.last_reset_report
                    agent_decision = agent_run.decision.to_dict()

                recovery_session.record_steps(int(context.total_steps))
                if not success and skill_result.metrics.get("exception_type") == "ExecutionBudgetExceeded":
                    failure_stage = "budget"
                elif not success and skill_result.metrics.get("exception_type") and failure_stage != "safety":
                    failure_stage = "infrastructure"
                feedback = RecoveryFeedback(
                    "success" if success else (failure_stage or "policy"),
                    "task_succeeded" if success else f"{failure_stage or 'policy'}_failed",
                    ("independent task verification passed" if success else
                     skill_result.failure_reason or verification.reason or "candidate failed")[:2048],
                )
                recovery_decision = record_recovery_decision(
                    recovery_session, trace, feedback, step=context.total_steps,
                    another_candidate=candidate_index + 1 < len(candidates), reset_supported=True,
                )
                # Passing task verification after violating the scene budget
                # cannot be promoted to successful collection.
                if success and recovery_decision["action"] != RecoveryAction.COMPLETE.value:
                    success = False
                    failure_stage = "budget"
                    skill_result = replace(skill_result, status=SkillStatus.FAILED,
                                           failure_reason="scene budget expired before collection acceptance")
                is_last_candidate = recovery_decision["action"] != RecoveryAction.RESET_AND_REPLAN.value
                should_save = success or (save_failed and is_last_candidate)
                training_eligible = bool(
                    success and should_save and not privileged_calibration and collection_controller is None
                )
                with trace.span(
                    "storage.episode",
                    category="storage",
                    start_step=context.total_steps,
                    inputs={
                        "save_failed": bool(save_failed),
                        "should_save": should_save,
                        "privileged_calibration_only": privileged_calibration,
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
                            backend="hammer_two_frame_retarget",
                            extra_metadata={
                                "candidate_rank": candidate.rank,
                                "candidate_distance": candidate.distance,
                                "library_size": len(library.episodes),
                                "random_reset": True,
                                "randomize_scene": bool(randomize_scene),
                                "randomize_dynamics": bool(randomize_dynamics),
                                "minimum_insertion_depth_m": (
                                    minimum_insertion_depth_m
                                ),
                                "initial_state": initial_state,
                                "perception": scene_estimate.to_dict(),
                                "perception_errors": perception_errors,
                                "reset": reset_report.to_dict(),
                                "safety": env.safety_summary,
                                "skill_mode": skill_mode,
                                "skill_manifest": skill_manifest.to_dict() if not chain_contract else None,
                                "skill_catalog": executor.catalog(),
                                "skill_chain": chain_report,
                                "stage_gates": list(stage_gates),
                                "stage_visual_artifact_root": str(visual_runtime.root) if visual_runtime else None,
                                "agent_decision": agent_decision,
                                "recovery": recovery_session.to_dict(),
                                "observation_attempts": observation_attempts,
                                "trace_schema_version": 1,
                                "trace_id": trace.trace_id,
                                "trace_path": str(trace_path),
                                "privileged_calibration_only": privileged_calibration,
                                "eligible_as_training_data": training_eligible,
                            },
                        )
                        storage_span.succeed(
                            end_step=context.total_steps,
                            outputs={
                                "episode_dir": str(episode_dir),
                                "eligible_as_training_data": training_eligible,
                            },
                            artifact_refs=(str(episode_dir),),
                        )
                    else:
                        storage_span.skip(
                            end_step=context.total_steps,
                            reason="failed candidate retention is disabled",
                            outputs={"trace_path": str(trace_path)},
                        )

                attempt = {
                    "candidate_rank": candidate.rank,
                    "candidate_distance": candidate.distance,
                    "source_demo": str(candidate.episode.source),
                    "status": "pass" if success else "fail",
                    "episode_dir": str(episode_dir) if episode_dir else None,
                    "eligible_as_training_data": training_eligible,
                    "skill": skill_result.to_dict(),
                    "skill_chain": chain_report,
                    "stage_gates": list(stage_gates),
                    "stage_visual_artifact_root": str(visual_runtime.root) if visual_runtime else None,
                    "verification": verification.to_dict(),
                    "failure_stage": failure_stage,
                    "reset": reset_report.to_dict(),
                    "safety": env.safety_summary,
                    "agent_decision": agent_decision,
                    "recovery_decision": recovery_decision,
                    "trace_id": trace.trace_id,
                    "trace_path": str(trace_path),
                }
                refs = (str(trace_path),) + (
                    (str(episode_dir),) if episode_dir is not None else ()
                )
                if success:
                    attempt_span.succeed(
                        end_step=context.total_steps,
                        outputs={"status": "pass", "candidate_rank": candidate.rank},
                        artifact_refs=refs,
                    )
                else:
                    attempt_span.fail(
                        end_step=context.total_steps,
                        failure_code=f"{failure_stage or 'candidate'}_failed",
                        reason=(
                            skill_result.failure_reason
                            or verification.reason
                            or "hammer candidate failed"
                        ),
                        outputs={"status": "fail", "candidate_rank": candidate.rank},
                        artifact_refs=refs,
                    )
        finally:
            try:
                env.close()
            finally:
                if trace.events:
                    trace.write_jsonl(trace_path)

        if attempt is None:
            raise RuntimeError("hammer candidate ended without a structured result")
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
                    "randomize_scene": bool(randomize_scene),
                    "randomize_dynamics": bool(randomize_dynamics),
                    "privileged_calibration_only": privileged_calibration,
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
                    and event.category in {"skill", "control", "planning", "verification", "stage_check"}
                ),
                artifact_refs=artifacts,
                eligible_as_training_data=attempt["eligible_as_training_data"],
                tags=("autocollect", "hammer_nail", "no_vla"),
            )
            record_path = skill_memory.append(experience)
            attempt["experience_id"] = experience.experience_id
            attempt["memory_record"] = str(record_path)
        attempts.append(attempt)
        if success:
            selected_episode_dir = episode_dir
            selected_training_eligible = attempt["eligible_as_training_data"]
            break
        if recovery_session.terminal_action is not None:
            break

    success = selected_episode_dir is not None
    return {
        "status": "pass" if success else "fail",
        "task_id": task_id,
        "task_plugin": plugin.name,
        "seed": int(seed),
        "vla_used": False,
        "backend": "hammer_two_frame_retarget",
        "skill_mode": skill_mode,
        "stage_check_mode": stage_check_mode,
        "required_skill_chain": chain_contract.to_dict() if chain_contract else None,
        "random_reset": True,
        "randomize_scene": bool(randomize_scene),
        "randomize_dynamics": bool(randomize_dynamics),
        "minimum_insertion_depth_m": minimum_insertion_depth_m,
        "perception_backend": perception.name,
        "perception": scene_estimate.to_dict() if scene_estimate else None,
        "perception_errors": perception_errors,
        "privileged_calibration_only": privileged_calibration,
        "eligible_as_training_data": selected_training_eligible,
        "library_size": len(library.episodes),
        "candidates_attempted": len(attempts),
        "selected_candidate_rank": (
            attempts[-1]["candidate_rank"] if success else None
        ),
        "episode_dir": str(selected_episode_dir) if selected_episode_dir else None,
        "attempt_trace_root": str(trace_root),
        "skill_memory_root": (
            str(skill_memory.root) if skill_memory is not None else None
        ),
        "agent_planner": (
            {"name": planner.name, "model_id": planner.model_id}
            if planner is not None
            else None
        ),
        "attempts": attempts,
        "observation_attempts": observation_attempts,
        "recovery": recovery_session.to_dict(),
    }


def run_random_hammer_nail(
    *,
    task_id: str,
    library: HammerNailTrajectoryLibrary,
    output_dir: str | Path,
    seed: int,
    perception: HammerNailPerception,
    max_candidates: int = 5,
    minimum_insertion_depth_m: float = 0.04,
    save_failed: bool = False,
    randomize_scene: bool = False,
    randomize_dynamics: bool = False,
    experiment_ledger: ExperimentLedger | None = None,
    hypothesis: str = "two-frame retargeting transfers hammer demonstrations",
    parent_experiment_id: str | None = None,
    study_id: str | None = None,
    skill_memory: SkillMemory | None = None,
    planner: StructuredPlanner | None = None,
    skill_mode: str = "monolithic",
    stage_check_mode: str = "off",
    reobserve_cameras: tuple[str, ...] = (),
    stage_cameras: tuple[str, ...] | None = None,
    require_stage_baselines: bool = False,
    recovery_limits: RecoveryLimits | None = None,
    collection_controller: BailianCollectionController | None = None,
) -> dict[str, Any]:
    """Run one random hammer scene and preserve every candidate's evidence."""

    if perception is None:
        raise ValueError("an explicit hammer-nail perception backend is required")
    if skill_mode not in {"monolithic", "phase_chain"}:
        raise ValueError("skill_mode must be monolithic or phase_chain")
    if skill_mode == "phase_chain" and planner is None:
        raise ValueError("phase_chain requires a structured planner (template_agent is sufficient)")
    visual_mode = stage_check_mode == "visual_lift_comotion"
    if stage_cameras is not None and not visual_mode:
        raise ValueError("stage camera selection requires visual_lift_comotion checks")
    if type(require_stage_baselines) is not bool or (require_stage_baselines and not visual_mode):
        raise ValueError("require_stage_baselines must be boolean and requires visual_lift_comotion")
    stage_cameras = VisualEffectRuntime.validate_cameras(stage_cameras)
    if (set(stage_cameras) & {"fixed_front", "top"}
            and task_id != "hammer_nail_inspire_native_delta"):
        raise ValueError("fixed_front/top are currently mounted only for the Inspire hammer task")
    gate = make_stage_gate("tcp_tracking" if visual_mode else stage_check_mode,
                          [stage.stage_id for stage in HAMMER_CHAIN.stages])
    if gate is not None and skill_mode != "phase_chain":
        raise ValueError("stage checks require phase_chain mode")
    if visual_mode and (not perception.requires_rgb or perception.privileged_calibration_only
                        or getattr(perception, "mask_provider", None) is None):
        raise ValueError("visual lift/co-motion requires explicit nonprivileged RGB mask perception")
    primary_camera = getattr(perception, "camera_key", "default")
    all_cameras = (primary_camera, *reobserve_cameras)
    canonical_cameras = tuple("front" if camera == "random_camera" else camera for camera in all_cameras)
    if len(set(canonical_cameras)) != len(canonical_cameras):
        raise ValueError("reobservation cameras must be distinct physical views")
    if any(camera not in {"ego_left", "ego_right", "wrist"} for camera in reobserve_cameras):
        raise ValueError("reobservation uses only existing ego_left/ego_right/wrist cameras")
    if reobserve_cameras and (not perception.requires_rgb or perception.privileged_calibration_only):
        raise ValueError("camera recovery requires explicit nonprivileged RGB perception")
    recovery_limits = recovery_limits or RecoveryLimits(
        maximum_observations=len(all_cameras), maximum_executions=max_candidates,
    )
    if collection_controller is not None:
        from .planning_snapshot import TASK
        if (task_id != TASK or collection_controller.task_spec.task_id != task_id
                or collection_controller.task_spec.chain != HAMMER_CHAIN or skill_mode != "phase_chain" or not visual_mode
                or planner is not collection_controller.planner or max_candidates != 1
                or recovery_limits.maximum_executions != 1 or len(all_cameras) > 3
                or recovery_limits.maximum_observations > 3 or collection_controller.used):
            raise ValueError("Bailian integration requires one fresh scene/candidate, phase_chain, visual checks and at most 3 reset views")
    if not np.isfinite(minimum_insertion_depth_m) or minimum_insertion_depth_m <= 0:
        raise ValueError("minimum_insertion_depth_m must be finite and positive")
    planner_sources: tuple[Any, ...] = ()
    if planner is not None:
        planner_sources = (type(planner),)
        generate_fn = getattr(planner, "generate_fn", None)
        if callable(generate_fn):
            planner_sources += (generate_fn,)
    policy_configuration = {
        "skill_mode": skill_mode,
        "required_skill_chain": HAMMER_CHAIN.to_dict() if skill_mode == "phase_chain" else None,
        "stage_check_configuration": gate.configuration() if gate else None,
        "visual_effect_configuration": VisualEffectRuntime.configuration(stage_cameras) if visual_mode else None,
        "require_stage_baselines": require_stage_baselines,
        "max_candidates": int(max_candidates),
        "save_failed": bool(save_failed),
        "perception": perception_configuration(perception),
        "library_size": len(library.episodes),
        "trajectory_library_sha256": _trajectory_library_sha256(library),
        "trace_schema_version": 1,
        "recovery_limits": recovery_limits.to_dict(),
        "reobserve_cameras": list(reobserve_cameras),
        "planner": (
            {"name": planner.name, "model_id": planner.model_id}
            if planner is not None
            else None
        ),
    }
    if collection_controller is not None:
        policy_configuration["bailian_collection"] = collection_controller.configuration()
        planner_sources += collection_controller.source_objects()
    environment_configuration = {
        "task_id": task_id,
        "minimum_insertion_depth_m": float(minimum_insertion_depth_m),
        "randomize_scene": bool(randomize_scene),
        "randomize_dynamics": bool(randomize_dynamics),
    }
    handle = None
    if experiment_ledger is not None:
        identity = PolicyIdentity.from_sources(
            name=HAMMER_CHAIN.chain_id if skill_mode == "phase_chain" else "hammer_nail.two_frame_retarget",
            version="1",
            sources=(
                TwoFrameHammerRetargetSkill,
                bind_hammer_phase_chain, SkillChainContract, AutoCollectAgent, SafeSkillProgramExecutor,
                StageGate,
                VisualEffectRuntime, VisualLiftCoMotionCheck,
                *perception_source_objects(perception),
                SkillRolloutEngine,
                TraceRecorder,
                SkillExecutor,
                make_hammer_frame_retarget_manifest,
                _run_random_hammer_nail,
                RecoverySession, observe_hammer_with_recovery,
                *planner_sources,
            ),
            configuration=policy_configuration,
        )
        environment_identity = EnvironmentIdentity.from_sources(
            name="hammer_nail.insert.environment",
            version="1",
            sources=(
                HammerNailTaskPlugin,
                HammerNailDepthVerifier,
                GymAgentEnvironment,
            ),
            configuration=environment_configuration,
        )
        handle = experiment_ledger.start(
            ExperimentSpec(
                task_id=task_id,
                seed=int(seed),
                policy=identity,
                hypothesis=hypothesis,
                environment=environment_identity,
                parameters={**policy_configuration, **environment_configuration},
                parent_experiment_id=parent_experiment_id,
                study_id=study_id,
                tags=("autocollect", "hammer_nail", "no_vla"),
            ),
            metadata={
                "framework": "enpire_style_environment_rollout_verification",
                "controller_uses_oracle_object_pose": False,
                "controller_uses_privileged_segmentation": bool(
                    perception.privileged_calibration_only
                ),
                "structured_agent_enabled": planner is not None,
            },
        )

    recovery_session = RecoverySession(recovery_limits)
    try:
        result = _run_random_hammer_nail(
            task_id=task_id,
            library=library,
            output_dir=output_dir,
            seed=seed,
            perception=perception,
            max_candidates=max_candidates,
            minimum_insertion_depth_m=minimum_insertion_depth_m,
            save_failed=save_failed,
            randomize_scene=randomize_scene,
            randomize_dynamics=randomize_dynamics,
            experiment_id=handle.experiment_id if handle is not None else None,
            study_id=study_id,
            skill_memory=skill_memory,
            planner=planner,
            skill_mode=skill_mode,
            stage_check_mode=stage_check_mode,
            recovery_session=recovery_session,
            reobserve_cameras=reobserve_cameras,
            stage_cameras=stage_cameras,
            require_stage_baselines=require_stage_baselines,
            collection_controller=collection_controller,
        )
    except StageBaselineRejected as exc:
        recovery_session.decide(RecoveryFeedback("stage_evidence", "stage_baseline_rejected", str(exc)))
        result = {
            "status": "stage_baseline_rejected", "task_id": task_id, "seed": int(seed),
            "vla_used": False, "eligible_as_training_data": False,
            "privileged_calibration_only": False, "failure_stage": "stage_baseline",
            "perception_errors": None, "candidates_attempted": 0, "control_actions_executed": 0,
            "selected_candidate_rank": None, "attempts": [], "episode_dir": None,
            "reason": str(exc), "baseline_preflight": exc.report, "recovery": recovery_session.to_dict(),
        }
    except CollectionPlanningStopped as exc:
        recovery_session.decide(RecoveryFeedback("stage_evidence", "planning_stopped", str(exc)))
        result = {
            "status": "planning_stopped", "task_id": task_id, "seed": int(seed),
            "vla_used": False, "eligible_as_training_data": False,
            "privileged_calibration_only": False, "failure_stage": "planning",
            "perception_errors": None, "candidates_attempted": 0, "control_actions_executed": 0,
            "selected_candidate_rank": None, "attempts": [], "episode_dir": None,
            "reason": str(exc), "recovery": recovery_session.to_dict(),
        }
    except HammerPerceptionRejected as exc:
        # No candidate has been selected and no controller has run. Keep this
        # separate from grasp/strike failures so future proposals target vision.
        provider = getattr(perception, "mask_provider", None)
        diagnostics = dict(getattr(provider, "last_diagnostics", {}))
        result = {
            "status": "perception_rejected", "task_id": task_id, "seed": int(seed),
            "vla_used": False, "eligible_as_training_data": False,
            "privileged_calibration_only": bool(perception.privileged_calibration_only),
            "failure_stage": "perception", "perception_errors": None,
            "candidates_attempted": 0, "control_actions_executed": 0,
            "selected_candidate_rank": None, "attempts": [], "episode_dir": None,
            "trace_id": exc.trace_id, "trace_path": str(exc.trace_path),
            "error": {"type": exc.cause_type, "message": str(exc)},
            "perception_backend": perception.name, "model_diagnostics": diagnostics,
            "observation_attempts": exc.observation_attempts,
            "recovery": recovery_session.to_dict(),
        }
        if skill_memory is not None and not exc.observation_attempts:
            refs = [exc.trace_path]
            if diagnostics.get("artifact_dir"):
                refs.append(Path(diagnostics["artifact_dir"]))
            experience = SkillExperience.create(
                task_id=task_id, skill_name="perception.hammer_nail", skill_version="1",
                success=False, parameters={"perception_backend": perception.name},
                scene_summary={"seed": int(seed), "randomize_scene": bool(randomize_scene)},
                skill_metrics={"control_actions_executed": 0},
                verification={"status": "not_run", "reason": "perception_rejected"},
                trace_id=exc.trace_id, trace_path=exc.trace_path,
                experiment_id=handle.experiment_id if handle is not None else None,
                study_id=study_id, failure_stage="perception",
                failure_codes=("perception_rejected",), artifact_refs=refs,
                eligible_as_training_data=False, tags=("autocollect", "vision", "no_control"),
            )
            result["memory_record"] = str(skill_memory.append(experience))
            result["experience_id"] = experience.experience_id
    except RecoveryBudgetExceeded as exc:
        if recovery_session.terminal_action is None:
            recovery_session.decide(RecoveryFeedback("budget", "scene_budget_exhausted", str(exc)[:2048]))
        result = {
            "status": "budget_exhausted", "task_id": task_id, "seed": int(seed),
            "vla_used": False, "eligible_as_training_data": False,
            "perception_errors": None, "selected_candidate_rank": None,
            "candidates_attempted": len(recovery_session.execution_ids),
            "control_actions_executed": recovery_session.total_steps,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "recovery": recovery_session.to_dict(),
        }
    except Exception as exc:
        if recovery_session.terminal_action is None:
            recovery_session.decide(RecoveryFeedback("infrastructure", "unexpected_error", str(exc)[:2048] or type(exc).__name__))
        exc.autocollect_recovery = recovery_session.to_dict()
        if handle is not None:
            experiment_ledger.finish(
                handle,
                status=ExperimentStatus.ERROR,
                error={"type": type(exc).__name__, "message": str(exc),
                       "recovery": recovery_session.to_dict()},
            )
        raise

    if collection_controller is not None:
        result["bailian_collection"] = collection_controller.audit()
        result["control_actions_executed"] = recovery_session.total_steps
        result["eligible_as_training_data"] = False
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
