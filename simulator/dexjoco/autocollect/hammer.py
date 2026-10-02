"""Hammer-nail task plugin and deterministic validation-skill contract."""

from __future__ import annotations

import numpy as np

from dexjoco.tasks.resolver import make_task_config

from .harness import (
    FiniteActionSafetyGuard,
    GymAgentEnvironment,
    TaskSuccessTerminationAdapter,
)
from .registry import ParameterKind, SkillManifest, SkillParameterSpec
from .replay import ReplayEpisode
from .verifiers import HammerNailDepthVerifier


def _dummy_render() -> list[np.ndarray]:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    return [frame, frame, frame, frame]


class HammerNailTaskPlugin:
    """Second task plugin for the task-agnostic AutoCollect harness."""

    name = "hammer_nail.insert"

    def __init__(
        self,
        *,
        task_id: str = "hammer_nail_inspire_native_delta",
        minimum_insertion_depth_m: float = 0.04,
        control_hz: int = 20,
        randomize_scene: bool = False,
        randomize_dynamics: bool = False,
        capture_rgb: bool = False,
    ) -> None:
        self.task_id = str(task_id)
        self.minimum_insertion_depth_m = float(minimum_insertion_depth_m)
        self.control_hz = int(control_hz)
        self.randomize_scene = bool(randomize_scene)
        self.randomize_dynamics = bool(randomize_dynamics)
        self.capture_rgb = bool(capture_rgb)
        if (
            not np.isfinite(self.minimum_insertion_depth_m)
            or self.minimum_insertion_depth_m <= 0.0
        ):
            raise ValueError(
                "minimum_insertion_depth_m must be finite and positive"
            )
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")

        self.config = make_task_config(self.task_id)
        if (
            getattr(self.config, "action_semantics", None)
            != "tcp_delta_pose_current_actual"
        ):
            raise ValueError(f"{self.task_id!r} is not a supported native-delta task")
        self.expected_action_dim = int(
            self.config.arm_action_dim + self.config.hand_action_dim
        )

    def validate_episode(self, episode: ReplayEpisode) -> None:
        if episode.actions.shape[1] != self.expected_action_dim:
            raise ValueError(
                f"source action dimension {episode.actions.shape[1]} does not "
                f"match task dimension {self.expected_action_dim}"
            )
        if episode.states.shape[1] < 28:
            raise ValueError(
                "hammer-nail source state must contain TCP, hand, hammer, "
                "nail, and table fields"
            )

    def make_environment(self, *, seed: int) -> GymAgentEnvironment:
        base_env = self.config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=self.randomize_scene,
            randomize_dynamics=self.randomize_dynamics,
            hz=self.control_hz,
            seed=int(seed),
            success_depth=self.minimum_insertion_depth_m,
        )
        raw_env = base_env.unwrapped
        raw_env.hz = 1_000_000
        if not self.capture_rgb:
            raw_env.render = _dummy_render
        action_low = np.asarray(base_env.action_space.low, dtype=np.float64)
        action_high = np.asarray(base_env.action_space.high, dtype=np.float64)
        return GymAgentEnvironment(
            task_id=self.task_id,
            env=TaskSuccessTerminationAdapter(base_env),
            verifier=HammerNailDepthVerifier(
                minimum_depth_m=self.minimum_insertion_depth_m
            ),
            safety_guard=FiniteActionSafetyGuard(
                expected_action_dim=self.expected_action_dim,
                lower=action_low,
                upper=action_high,
                # Recorded float64 saturation values differ from float32 Gym
                # bounds by about 7e-10.  Accepted values are normalized back
                # to the exact bound before reaching MuJoCo.
                absolute_tolerance=1e-8,
            ),
            reset_metadata={
                "task_plugin": self.name,
                "random_reset": True,
                "randomize_scene": self.randomize_scene,
                "randomize_dynamics": self.randomize_dynamics,
                "capture_rgb_for_perception": self.capture_rgb,
                "success_termination_suppressed_during_rollout": True,
            },
        )


def make_hammer_replay_manifest(
    *,
    task_id: str,
    expected_action_dim: int,
) -> SkillManifest:
    """Manifest for exact-demo replay used to validate the second task plugin."""

    return SkillManifest(
        name="hammer_nail.replay_validation",
        version="1",
        description=(
            "Replay one recorded native-delta hammering trajectory from its "
            "restored demonstration state to validate the task plugin."
        ),
        supported_tasks=(str(task_id),),
        parameters=(
            SkillParameterSpec(
                name="max_steps",
                kind=ParameterKind.INTEGER,
                description="Optional upper bound on replayed actions.",
                required=False,
                minimum=1,
            ),
        ),
        required_observations=("state",),
        action_dimension=int(expected_action_dim),
        required_perception=(),
        preconditions=(
            "the environment has been restored to the demonstration initial state",
            "the action semantics match native current-TCP delta control",
        ),
        expected_effects=(
            "the hammer contacts the nail",
            "nail insertion depth reaches the independently declared threshold",
        ),
        safety_constraints=(
            f"every action has exactly {int(expected_action_dim)} finite values",
            "every action remains inside the environment action bounds",
            "the replay is validation-only and is not a random-scene policy",
        ),
        implementation="dexjoco.autocollect.replay.ReplayActionSkill",
        controller_family="recorded_native_action_validation",
        vla_used=False,
        tags=("hammer", "nail", "inspire", "no_vla", "validation_only"),
    )
