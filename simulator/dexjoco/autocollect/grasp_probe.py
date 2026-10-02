"""Single-use simulation grasp acquisition followed by a bounded lift probe.

Target-scene inputs are perception and robot proprioception only. Recorded
source object poses calibrate the demonstration, not the current scene. There
is no nail-frame transition, strike, reset, retry or success-oracle call here.
"""
from dataclasses import dataclass
import hashlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .core import SkillResult, SkillStatus
from .hammer_retarget import HAMMER_SLICE, infer_hammer_phase_plan
from .retarget import _pose7_to_matrix, _matrix_to_pose7, _pose_error6
from .registry import ParameterKind, SkillManifest, SkillParameterSpec


def grasp_probe_manifest(task_id):
    return SkillManifest(name="hammer.acquire_and_lift_probe", version="1",
        description="One bound demonstration acquisition and a bounded vertical lift probe; no strike.",
        supported_tasks=(task_id,), parameters=(
            SkillParameterSpec(name="target_hammer_pose7", kind=ParameterKind.ARRAY,
                description="Locked current RGB-D hammer pose, never a model-generated pose.", shape=(7,), finite=True),
            SkillParameterSpec(name="estimated_table_height_m", kind=ParameterKind.NUMBER,
                description="Depth-estimated table plane.", minimum=.86, maximum=1.4)),
        required_observations=("state",), action_dimension=12, required_perception=("hammer_pose7",),
        preconditions=("host single-use permission, current semantic review and RGB-D geometry",),
        expected_effects=("acquisition and lift motion completed; physical grasp must be checked separately",),
        safety_constraints=("single simulation attempt", "bounded native-delta commands", "no later strike stages"),
        implementation="dexjoco.autocollect.grasp_probe.BoundAcquireProbe",
        controller_family="object_frame_retarget_and_bounded_probe", vla_used=False, tags=("no_vla", "simulation_only"))


class BoundAcquireProbe:
    name = "hammer.acquire_and_lift_probe"

    def __init__(self, probe, permit, token_fn):
        self.probe, self.permit, self.token_fn = probe, permit, token_fn

    def execute(self, context, parameters):
        return self.probe.execute(context, parameters, permit=self.permit, token_fn=self.token_fn)


def checked_pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5)):
        raise ValueError("explicit finite unit-quaternion perception pose required")
    return pose.copy()


def select_acquire_episode(episodes, target_pose):
    target = checked_pose(target_pose)
    ranked = []
    for episode in episodes:
        source = checked_pose(episode.initial_state[HAMMER_SLICE])
        rotation = (Rotation.from_quat(target[3:], scalar_first=True)
                    * Rotation.from_quat(source[3:], scalar_first=True).inv()).magnitude()
        distance = float(np.sqrt(np.sum(((target[:3] - source[:3]) / .05) ** 2) + (rotation / .35) ** 2))
        ranked.append((distance, str(episode.source), episode))
    if not ranked:
        raise ValueError("empty acquisition library")
    distance, _, episode = min(ranked, key=lambda row: row[:2])
    return episode, distance


@dataclass
class SingleUseGraspPermit:
    """Host-side dispatch binding; a VLM JSON answer is not this permission."""
    context_identity: int
    stationary_receipt: object
    pose_sha256: str
    deadline_monotonic: float
    used: bool = False

    @classmethod
    def create(cls, context, target_pose, *, token_fn, deadline, semantic_rows, handle_id, head_id):
        pose = checked_pose(target_pose)
        rows = {row["candidate_id"]: row for row in semantic_rows}
        if (len(rows) != len(semantic_rows) or len(rows) != 2 or set(rows) != {handle_id, head_id}
                or rows[handle_id]["main_region"] != "target_part"
                or rows[head_id]["main_region"] != "other_part_same_object"
                or context.total_steps != 0 or time.monotonic() >= deadline):
            raise ValueError("current handle/head semantics or dispatch state do not support this trial")
        return cls(id(context), token_fn(), hashlib.sha256(pose.tobytes()).hexdigest(), deadline)

    def consume(self, context, target_pose, *, token_fn):
        pose = checked_pose(target_pose)
        if (self.used or id(context) != self.context_identity or context.total_steps != 0
                or time.monotonic() >= self.deadline_monotonic
                or token_fn() != self.stationary_receipt
                or hashlib.sha256(pose.tobytes()).hexdigest() != self.pose_sha256):
            raise ValueError("grasp permission is used, stale, expired or bound to another pose/context")
        self.used = True


class ProprioceptionOnlyEnvironment:
    """Hide reset object fields, task reward and oracle info from motion code."""
    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        # Used only for renderer/calibration, equality audit and control limits.
        return self.env.unwrapped

    def step(self, action):
        observation, _, terminated, truncated, _ = self.env.step(action)
        return {"state": np.asarray(observation["state"])[:13].copy()}, 0., terminated, truncated, {}


class AcquireAndLiftProbe:
    """One demonstrated acquisition prefix, then 5 cm vertical TCP probe.

Completion describes motion only. A separate sensor checker must evaluate the
object. This bounded controller is for the Inspire native-delta simulator, not
a general collision planner or real-hardware controller.
"""
    name = "hammer.acquire_and_lift_probe"

    def __init__(self, episode, *, lift_m=.05, lift_steps=40, hold_steps=20, maximum_acquire_steps=300):
        self.episode = episode
        self.lift_m, self.lift_steps, self.hold_steps = float(lift_m), lift_steps, hold_steps
        if (not .025 <= self.lift_m <= .06 or type(lift_steps) is not int or not 30 <= lift_steps <= 60
                or type(hold_steps) is not int or not 10 <= hold_steps <= 30
                or type(maximum_acquire_steps) is not int or not 1 <= maximum_acquire_steps <= 300
                or episode.actions.ndim != 2 or episode.actions.shape[1] != 12):
            raise ValueError("invalid bounded Inspire grasp/lift probe")
        self.acquire_steps = infer_hammer_phase_plan(episode).transition_start_step
        if not 1 <= self.acquire_steps <= maximum_acquire_steps:
            raise ValueError("acquisition prefix outside the declared step budget")
        self.sample_steps = (self.acquire_steps, self.acquire_steps + lift_steps // 2,
                             self.acquire_steps + lift_steps + hold_steps)

    def references(self, target_pose):
        target = checked_pose(target_pose)
        transform = _pose7_to_matrix(target) @ np.linalg.inv(_pose7_to_matrix(
            checked_pose(self.episode.initial_state[HAMMER_SLICE])))
        references = np.stack([_matrix_to_pose7(transform @ _pose7_to_matrix(pose))
                               for pose in self.episode.states[:self.acquire_steps, :7]])
        return transform, references

    @staticmethod
    def check_workspace(poses, *, table_height):
        points = np.asarray(poses, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 7 or not np.isfinite(points).all()
                or not np.isfinite(table_height)
                or np.any(points[:, :3] < [-.65, -.75, table_height])
                or np.any(points[:, :3] > [.50, .75, 1.85])):
            raise ValueError("TCP reference outside the fixed simulation workspace")

    def execute(self, context, parameters, *, permit, token_fn):
        # Reject extra stages/parameters, and never synthesize a nail pose.
        if set(parameters) != {"target_hammer_pose7", "estimated_table_height_m"}:
            raise ValueError("grasp probe accepts only perceived hammer pose and table height")
        pose = checked_pose(parameters["target_hammer_pose7"])
        table = float(parameters["estimated_table_height_m"])
        transform, references = self.references(pose)
        self.check_workspace(references, table_height=table)
        if np.linalg.norm(references[0, :3] - np.asarray(context.observation["state"])[:3]) > .30:
            raise ValueError("initial source reference is too far from current TCP")
        permit.consume(context, pose, token_fn=token_fn)
        started, begin = time.perf_counter(), context.total_steps
        position_limit = min(.01, float(context.env.unwrapped._delta_position_limit))
        rotation_limit = min(.03, float(context.env.unwrapped._delta_rotation_limit))
        if not np.isfinite([position_limit, rotation_limit]).all() or min(position_limit, rotation_limit) <= 0:
            raise ValueError("invalid native-delta controller limits")

        def command(reference, hand, feedforward=None):
            self.check_workspace(np.asarray(context.observation["state"][:7])[None], table_height=table)
            self.check_workspace(reference[None], table_height=table)
            error = _pose_error6(context.observation["state"][:7], reference)
            delta = np.r_[.35 * error[:3], .25 * error[3:]]
            if feedforward is not None:
                delta += feedforward
            action = np.r_[np.clip(delta[:3], -position_limit, position_limit),
                           np.clip(delta[3:], -rotation_limit, rotation_limit), hand]
            if action.shape != (12,) or not np.isfinite(action).all():
                raise ValueError("malformed acquisition action")
            context.step(action)

        for i, reference in enumerate(references):
            if context.terminated or context.truncated:
                break
            action = self.episode.actions[i]
            feedforward = np.r_[transform[:3, :3] @ action[:3], transform[:3, :3] @ action[3:6]]
            command(reference, action[6:], feedforward)
        acquired_motion = context.total_steps - begin == self.acquire_steps
        acquire_error = _pose_error6(context.observation["state"][:7], references[-1])
        if (not acquired_motion or np.linalg.norm(acquire_error[:3]) > .05
                or np.linalg.norm(acquire_error[3:]) > .35):
            return SkillResult(self.name, SkillStatus.FAILED, context.total_steps - begin,
                time.perf_counter() - started, "acquisition motion interrupted or TCP tracking outside tolerance",
                {"acquire_motion_completed": acquired_motion, "lift_executed": False,
                 "physical_grasp_verified": False, "vla_used": False})

        lift_start = checked_pose(context.observation["state"][:7])
        endpoint = lift_start.copy()
        endpoint[2] += self.lift_m
        self.check_workspace(endpoint[None], table_height=table)
        hand = self.episode.actions[self.acquire_steps - 1, 6:].copy()
        for i in range(1, self.lift_steps + self.hold_steps + 1):
            if context.terminated or context.truncated:
                break
            reference = lift_start.copy()
            reference[2] += self.lift_m * min(1., i / self.lift_steps)
            error = _pose_error6(context.observation["state"][:7], reference)
            if np.linalg.norm(error[:3]) > .05 or np.linalg.norm(error[3:]) > .35:
                break
            command(reference, hand)
        end_error = _pose_error6(context.observation["state"][:7], endpoint)
        completed = (context.total_steps - begin == self.sample_steps[-1]
                     and np.linalg.norm(end_error[:3]) <= .01 and np.linalg.norm(end_error[3:]) <= .10)
        return SkillResult(self.name, SkillStatus.SUCCEEDED if completed else SkillStatus.FAILED,
            context.total_steps - begin, time.perf_counter() - started,
            None if completed else "lift probe interrupted or endpoint tracking insufficient",
            {"acquire_motion_completed": True, "lift_executed": context.total_steps - begin > self.acquire_steps,
             "lift_requested_m": self.lift_m, "tcp_vertical_displacement_m": float(context.observation["state"][2] - lift_start[2]),
             "terminal_tcp_reference7": endpoint.tolist(), "physical_grasp_verified": False,
             "subsequent_stages_enabled": False, "vla_used": False})
