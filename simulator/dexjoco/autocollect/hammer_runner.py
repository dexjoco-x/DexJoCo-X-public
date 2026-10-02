"""Restored-demo validation runner for the hammer-nail task plugin."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from dexjoco.tasks.state_restorers import restore_initial_state

from .core import LowDimEpisodeRecorder, SkillContext, SkillExecutor
from .experiments import (
    EnvironmentIdentity,
    ExperimentLedger,
    ExperimentSpec,
    ExperimentStatus,
    PolicyIdentity,
)
from .hammer import HammerNailTaskPlugin, make_hammer_replay_manifest
from .harness import GymAgentEnvironment, SkillRolloutEngine
from .replay import ReplayActionSkill, ReplayEpisode, load_replay_episode
from .storage import save_episode
from .trace import TraceRecorder
from .verifiers import HammerNailDepthVerifier


def _episode_sha256(episode: ReplayEpisode) -> str:
    digest = hashlib.sha256(b"dexjoco_hammer_validation_episode_v1\0")
    for array in (episode.initial_state, episode.actions):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_restored_hammer_nail_validation(
    *,
    source_demo: str | Path,
    output_dir: str | Path,
    task_id: str = "hammer_nail_inspire_native_delta",
    seed: int = 42,
    minimum_insertion_depth_m: float = 0.04,
    max_steps: int | None = None,
    save_failed: bool = True,
    experiment_ledger: ExperimentLedger | None = None,
    hypothesis: str = "the generic harness supports a second task plugin",
    parent_experiment_id: str | None = None,
    study_id: str | None = None,
) -> dict[str, Any]:
    """Validate common harness plumbing on a restored successful hammer demo.

    This intentionally restores a recorded initial state.  It proves plugin,
    safety, trace, verification, and storage compatibility; it is not claimed
    to be a random-scene hammering policy or newly collected training data.
    """

    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be positive when provided")
    run_root = Path(output_dir).expanduser().resolve()
    episode = load_replay_episode(source_demo)
    plugin = HammerNailTaskPlugin(
        task_id=task_id,
        minimum_insertion_depth_m=minimum_insertion_depth_m,
        control_hz=20,
        randomize_scene=False,
        randomize_dynamics=False,
    )
    plugin.validate_episode(episode)
    manifest = make_hammer_replay_manifest(
        task_id=task_id,
        expected_action_dim=plugin.expected_action_dim,
    )
    policy_configuration = {
        "source_demo": str(episode.source),
        "source_episode_sha256": _episode_sha256(episode),
        "max_steps": int(max_steps) if max_steps is not None else None,
        "skill_manifest": manifest.to_dict(),
        "validation_only": True,
        "trace_schema_version": 1,
    }
    environment_configuration = {
        "task_id": task_id,
        "minimum_insertion_depth_m": float(minimum_insertion_depth_m),
        "control_hz": plugin.control_hz,
        "randomize_scene": False,
        "randomize_dynamics": False,
    }
    handle = None
    if experiment_ledger is not None:
        policy_identity = PolicyIdentity.from_sources(
            name=manifest.name,
            version=manifest.version,
            sources=(
                ReplayActionSkill,
                SkillExecutor,
                make_hammer_replay_manifest,
                TraceRecorder,
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
                policy=policy_identity,
                environment=environment_identity,
                hypothesis=hypothesis,
                parameters={
                    **policy_configuration,
                    **environment_configuration,
                },
                parent_experiment_id=parent_experiment_id,
                study_id=study_id,
                tags=(
                    "autocollect",
                    "no_vla",
                    "hammer_nail",
                    "restored_demo_validation",
                ),
            ),
            metadata={
                "framework": "enpire_style_environment_rollout_verification",
                "controller_uses_oracle_state": False,
                "restores_recorded_state": True,
                "eligible_as_random_scene_training_data": False,
            },
        )

    experiment_id = handle.experiment_id if handle is not None else None
    trace = TraceRecorder(
        attempt_id=f"{task_id}:seed={int(seed)}:restored_demo_validation",
        experiment_id=experiment_id,
        study_id=study_id,
    )
    trace_path = (
        run_root
        / "attempt_traces"
        / f"seed_{int(seed)}"
        / f"restored_demo_{trace.trace_id}.jsonl"
    )
    environment = plugin.make_environment(seed=int(seed))
    episode_dir: Path | None = None
    result: dict[str, Any] | None = None
    try:
        with trace.span(
            "candidate.attempt",
            category="attempt",
            start_step=0,
            inputs={
                "task_id": task_id,
                "seed": int(seed),
                "mode": "restored_demo_validation",
            },
        ) as attempt_span:
            with trace.span(
                "environment.reset",
                category="environment",
                start_step=0,
                inputs={"seed": int(seed)},
            ) as reset_span:
                environment.reset(seed=int(seed))
                assert environment.last_reset_report is not None
                reset_span.succeed(
                    end_step=0,
                    outputs={"reset": environment.last_reset_report.to_dict()},
                )
            with trace.span(
                "environment.restore_demo_state",
                category="environment",
                start_step=0,
                inputs={
                    "source_demo": str(episode.source),
                    "privileged_state_restore": True,
                    "purpose": "task_plugin_validation_only",
                },
            ) as restore_span:
                observation = restore_initial_state(
                    environment,
                    task_id,
                    plugin.config,
                    episode.initial_state,
                )
                restore_span.succeed(
                    end_step=0,
                    outputs={
                        "observation_keys": sorted(observation),
                        "eligible_as_random_scene_training_data": False,
                    },
                )

            recorder = LowDimEpisodeRecorder(control_hz=plugin.control_hz)
            context = SkillContext(
                env=environment,
                task_id=task_id,
                config=plugin.config,
                observation=observation,
                recorder=recorder,
                trace=trace,
                metadata={
                    "source_demo": str(episode.source),
                    "validation_only": True,
                },
            )
            executor = SkillExecutor(require_manifests=True)
            executor.register(
                ReplayActionSkill(
                    episode,
                    name=manifest.name,
                    expected_action_dim=plugin.expected_action_dim,
                ),
                manifest=manifest,
            )
            invocation = (
                {"max_steps": int(max_steps)} if max_steps is not None else {}
            )
            rollout = SkillRolloutEngine().run(
                environment=environment,
                executor=executor,
                context=context,
                skill_name=manifest.name,
                parameters=invocation,
                trace=trace,
            )
            should_save = rollout.success or save_failed
            with trace.span(
                "storage.episode",
                category="storage",
                start_step=context.total_steps,
                inputs={
                    "should_save": should_save,
                    "validation_only": True,
                    "low_dimensional_only": True,
                },
            ) as storage_span:
                if should_save:
                    episode_dir = save_episode(
                        output_dir=run_root / "episodes",
                        task_id=task_id,
                        seed=int(seed),
                        recorder=recorder,
                        skill_result=rollout.skill,
                        verification=rollout.verification,
                        source_demo=episode.source,
                        backend="recorded_native_action_validation",
                        extra_metadata={
                            "validation_only": True,
                            "eligible_as_random_scene_training_data": False,
                            "privileged_state_restore": True,
                            "skill_manifest": manifest.to_dict(),
                            "trace_schema_version": 1,
                            "trace_id": trace.trace_id,
                            "trace_path": str(trace_path),
                            "reset": rollout.reset.to_dict(),
                            "safety": environment.safety_summary,
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
                        reason="failed validation episode retention is disabled",
                        outputs={"trace_path": str(trace_path)},
                    )

            result = {
                "status": "pass" if rollout.success else "fail",
                "task_id": task_id,
                "task_plugin": plugin.name,
                "seed": int(seed),
                "vla_used": False,
                "backend": "recorded_native_action_validation",
                "validation_only": True,
                "eligible_as_random_scene_training_data": False,
                "privileged_state_restore": True,
                "source_demo": str(episode.source),
                "episode_dir": str(episode_dir) if episode_dir else None,
                "trace_id": trace.trace_id,
                "trace_path": str(trace_path),
                "skill_manifest": manifest.to_dict(),
                "skill": rollout.skill.to_dict(),
                "verification": rollout.verification.to_dict(),
                "failure_stage": rollout.failure_stage,
                "reset": rollout.reset.to_dict(),
                "safety": environment.safety_summary,
            }
            artifacts = (str(trace_path),)
            if episode_dir is not None:
                artifacts += (str(episode_dir),)
            if rollout.success:
                attempt_span.succeed(
                    end_step=context.total_steps,
                    outputs={"status": "pass"},
                    artifact_refs=artifacts,
                )
            else:
                failure_stage = rollout.failure_stage or "validation"
                attempt_span.fail(
                    end_step=context.total_steps,
                    failure_code=f"{failure_stage}_failed",
                    reason=(
                        rollout.skill.failure_reason
                        or rollout.verification.reason
                        or "hammer validation failed"
                    ),
                    outputs={"status": "fail"},
                    artifact_refs=artifacts,
                )
    except Exception as exc:
        if handle is not None:
            experiment_ledger.finish(
                handle,
                status=ExperimentStatus.ERROR,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        raise
    finally:
        try:
            environment.close()
        finally:
            if trace.events:
                trace.write_jsonl(trace_path)

    if result is None:
        raise RuntimeError("hammer validation ended without a structured result")
    result["trace_summary"] = trace.summary()
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
