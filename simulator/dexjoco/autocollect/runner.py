"""End-to-end no-video runner for the first deterministic grasp/lift skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dexjoco.tasks.resolver import make_task_config
from dexjoco.tasks.state_restorers import restore_initial_state

from .core import (
    LowDimEpisodeRecorder,
    SkillContext,
    SkillExecutor,
    SkillResult,
    SkillStatus,
)
from .replay import NativeDeltaRetreatSkill, ReplayActionSkill, load_replay_episode
from .storage import save_episode
from .verifiers import PickBucketLiftVerifier


def _dummy_render() -> list[np.ndarray]:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    return [frame, frame, frame, frame]


def run_pick_bucket_grasp_lift(
    *,
    task_id: str,
    source_demo: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    max_steps: int | None = None,
    save_failed: bool = True,
) -> dict[str, Any]:
    """Restore one scene, execute the no-VLA skill, verify, and save it."""

    episode = load_replay_episode(source_demo)
    config = make_task_config(task_id)
    if getattr(config, "action_semantics", None) != "tcp_delta_pose_current_actual":
        raise ValueError(
            f"{task_id!r} is not a native current-TCP delta task; this first "
            "AutoCollect runner intentionally supports only that safe contract"
        )
    arm_action_dim = int(config.arm_action_dim)
    hand_action_dim = int(config.hand_action_dim)
    action_dim = arm_action_dim + hand_action_dim
    if episode.actions.shape[1] != action_dim:
        raise ValueError(
            f"source action dim {episode.actions.shape[1]} does not match "
            f"task action dim {action_dim}"
        )

    control_hz = float(getattr(config, "control_hz", 20))
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        randomize_dynamics=False,
        hz=int(control_hz),
        seed=int(seed),
    )
    raw_env = env.unwrapped
    # No frames are consumed or written.  Removing real-time sleep also makes
    # smoke and batch tests deterministic and fast.
    raw_env.hz = 1_000_000
    raw_env.render = _dummy_render

    episode_dir: Path | None = None
    recovery_result: SkillResult | None = None
    try:
        env.reset(seed=int(seed))
        observation = restore_initial_state(
            env,
            task_id,
            config,
            episode.initial_state,
        )
        recorder = LowDimEpisodeRecorder(control_hz=control_hz)
        context = SkillContext(
            env=env,
            task_id=task_id,
            config=config,
            observation=observation,
            recorder=recorder,
            metadata={"source_demo": str(episode.source)},
        )
        executor = SkillExecutor()
        executor.register(
            ReplayActionSkill(
                episode,
                expected_action_dim=action_dim,
            )
        )
        executor.register(
            NativeDeltaRetreatSkill(
                arm_action_dim=arm_action_dim,
                hand_action_dim=hand_action_dim,
            )
        )

        skill_result = executor.execute(
            "grasp_lift.replay",
            context,
            {"max_steps": max_steps} if max_steps is not None else {},
        )
        verification = PickBucketLiftVerifier().verify(env)
        if skill_result.success and not verification.success:
            skill_result = SkillResult(
                skill_name=skill_result.skill_name,
                status=SkillStatus.FAILED,
                steps=skill_result.steps,
                duration_seconds=skill_result.duration_seconds,
                failure_reason=(
                    "environment reported success but the independent verifier failed: "
                    f"{verification.reason}"
                ),
                metrics=skill_result.metrics,
            )

        if not skill_result.success and not context.terminated and not context.truncated:
            recovery_result = executor.execute("recover.retreat", context)

        if skill_result.success or save_failed:
            episode_dir = save_episode(
                output_dir=output_dir,
                task_id=task_id,
                seed=seed,
                recorder=recorder,
                skill_result=skill_result,
                verification=verification,
                source_demo=episode.source,
                recovery_result=recovery_result,
            )

        return {
            "status": (
                "pass" if skill_result.success and verification.success else "fail"
            ),
            "task_id": task_id,
            "seed": int(seed),
            "vla_used": False,
            "backend": "recorded_native_action",
            "source_demo": str(episode.source),
            "episode_dir": str(episode_dir) if episode_dir is not None else None,
            "skill": skill_result.to_dict(),
            "verification": verification.to_dict(),
            "recovery": recovery_result.to_dict() if recovery_result else None,
        }
    finally:
        env.close()
