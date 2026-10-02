"""One task implementation for all hands, explicit native actuator actions."""
from __future__ import annotations
import hashlib
import json
import random
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from . import TARGET_TASKS, VERSION
from .models import path_for, PREFIX
from .success_policy import success_policy
# These copies expose native control gains and the corrected physics velocity
# timebase. ONE implementation is used for all seven hands, never chosen by hand.
from ..sim.envs.panda_hammer_nail_inspire_base_env import PandaHammerNailGymEnv
from ..sim.envs.panda_pick_bucket_inspire_base_env import PandaPickBucketGymEnv
from ..sim.envs.panda_bimanual_microwave_cook_env import PandaBimanualMicrowaveCookGymEnv
from ..sim.envs.panda_bimanual_photograph_env import PandaBimanualPhotographGymEnv
from ..sim.envs.panda_bimanual_unlock_ipad_env import PandaBimanualUnlockIpadGymEnv

DT = .05
HORIZONS = dict(hammer_nail=1000, pick_bucket=1200, bimanual_microwave_cook=1100,
                bimanual_photograph=1000, bimanual_unlock_ipad=1200,
                single_hand_hanoi=2000)
TIP_SUFFIXES = {'allegro': ('th_tip', 'ff_tip', 'mf_tip', 'rf_tip'), 'leap': ('thumb_fingertip', 'fingertip', 'fingertip_2', 'fingertip_3'), 'xhand': tuple(('xhand_right_hand_' + s for s in ('thumb_rota_tip', 'index_rota_tip', 'mid_tip', 'ring_tip', 'pinky_tip'))), 'inspire': tuple(('inspire_' + s for s in ('thumb_distal', 'index_intermediate', 'middle_intermediate', 'ring_intermediate', 'pinky_intermediate'))), 'wuji': ('wuji_IP_thumb', 'wuji_tipmount_index', 'wuji_tipmount_middle', 'wuji_tipmount_ring', 'wuji_tipmount_pinky'), 'sharpa_wave': tuple(('right_' + s + '_DP' for s in ('thumb', 'index', 'middle', 'ring', 'pinky'))), 'linkerhand_l20': tuple(('l20_' + s + '_distal' for s in ('thumb', 'index', 'middle', 'ring', 'pinky')))}


class Common:
    def _apply_builtin_contact_tuning(self):
        # The legacy initializer includes morphology-specific OBJECT tuning.
        # Native hand collision definitions are already in the assembled XML.
        pass

    def __init__(self, *, hand, seed=0, pixels=False):
        self.hand = hand
        self.pixels = bool(pixels)
        self.scene_seed = int(seed)
        path = path_for(self.task, hand)
        self.hand_metadata = json.loads(path.with_suffix(".json").read_text())
        if self.hand_metadata["xml_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("Model changed without rebuilding metadata")
        sides = self.hand_metadata["sides"]
        kwargs = dict(xml_path=path, seed=seed, control_dt=DT, physics_dt=.002,
                      hz=20, max_episode_steps=HORIZONS[self.task], randomize=False,
                      randomize_dynamics=False, hand_command_mode="actuator")
        if len(sides) == 1:
            spec = sides["right"]
            kwargs.update(hand_joint_names=tuple(spec["joint_names"]),
                          hand_actuator_names=tuple(spec["actuator_names"]),
                          hand_home=np.asarray(spec["home"]), hand_type="benchmark6")
            if self.task != "click_mouse":
                kwargs.update(arm_pos_gain_base=3600., arm_ori_gain_base=3600., arm_damping_ratio=1.)
        else:
            for side, spec in sides.items():
                kwargs[side+"_hand_joint_names"] = tuple(spec["joint_names"])
                kwargs[side+"_hand_actuator_names"] = tuple(spec["actuator_names"])
                kwargs[side+"_hand_home"] = np.asarray(spec["home"])
            kwargs["image_obs"] = bool(pixels)
        if self.task == "hammer_nail":
            kwargs.update(success_depth=.04, impact_velocity_timebase="physics_dt")
        if self.task == "bimanual_photograph":
            kwargs.update(target_angle_deg=18., camera_object_scale=1., camera_screen_update_interval=1,
                          shutter_fingertip_body_names=tuple(PREFIX+"right_native_"+s for s in TIP_SUFFIXES[hand]),
                          shutter_touch_contact_geom_names=("camera_body_top_cap", "camera_body_collision"))
        if self.task == "bimanual_unlock_ipad":
            kwargs.update(password=[1, 2, 3], ipad_screen_effect=False,
                          button_fingertip_body_names=tuple(PREFIX+"right_native_"+s for s in TIP_SUFFIXES[hand]),
                          button_touch_contact_geom_names=("ipad_collision_ll", "ipad_collision_lr", "ipad_collision_ul", "ipad_collision_ur"),
                          button_touch_trigger_digits=(1, 2, 3))
        super().__init__(**kwargs)
        self._n_substeps = round(DT / self._model.opt.timestep)
        if abs(self._n_substeps*self._model.opt.timestep-DT) > 1e-12:
            raise RuntimeError("Control interval must be integer physics steps")
        self.hz = 1_000_000  # remove wall-clock sleeping, NOT simulation timing
        self.contract = dict(version=VERSION, task=self.task, hand=hand, control_dt=DT,
                             physics_dt=float(self._model.opt.timestep), randomize_dynamics=False,
                             max_episode_steps=HORIZONS[self.task], action="world delta TCP rotvec6 + actuator position targets",
                             reference="cross_embodiment_v1 common objects and task criteria")
        # These are position-controller setpoint errors, NOT instantaneous
        # physical displacements. The legacy dual controller is strongly damped:
        # a 2 cm / .03 rad cap prevents it tracking ordinary demo velocities.
        high_gain = len(sides)==1
        self._translation_bound=.02 if high_gain else .08
        self._rotation_bound=.03 if high_gain else .30
        self.contract.update(translation_action_bound=self._translation_bound,
                             rotation_action_bound=self._rotation_bound)
        if high_gain:
            self.contract.update(arm_pos_gain=3600., arm_ori_gain=3600., arm_damping_ratio=1.)
        if self.task == "hammer_nail": self.contract["impact_velocity_timebase"] = "physics_dt"
        if self.task == "pick_bucket":
            self.contract['success_policy'] = success_policy(self.task)
        self.contract["reference"]=self.hand_metadata["reference"]
        self._full_hand_qpos = {side: np.asarray([int(self._model.joint(n).qposadr)
            for n in spec["state_joint_names"]]) for side, spec in sides.items()}
        self._native_ctrl_ids = {side: np.asarray([self._model.actuator(n).id
            for n in spec["actuator_names"]]) for side, spec in sides.items()}
        if len(sides) == 1:
            self._native_command_bounds = self._model.actuator_ctrlrange[self._native_ctrl_ids["right"]].copy()
        self._tcp_sites = {side: self._model.site("attachment_site"+("_"+side if len(sides)>1 else "")).id
                           for side in sides}
        # Shared versioned camera profile for collection, export and evaluation.
        from .cameras import apply_wrist_rig,RIG
        apply_wrist_rig(self)
        self.contract['wrist_camera_rig']=dict(RIG)
        from .contact_policy import POLICY_ID, validate_applied
        validate_applied(self._model, self.hand_metadata)
        self.contract['surface_friction_policy_id'] = POLICY_ID
        if self.hand_metadata['version']=='benchmark7_v3':
            from .geometry_revision import validate_geometry, POLICY_ID as GEOMETRY_POLICY_ID
            validate_geometry(self._model,self.hand_metadata)
            self.contract['geometry_policy_id']=GEOMETRY_POLICY_ID
        boxes = {side: spaces.Box(
            low=np.r_[[-self._translation_bound]*3, [-self._rotation_bound]*3, self._model.actuator_ctrlrange[ids, 0]].astype(np.float64),
            high=np.r_[[self._translation_bound]*3, [self._rotation_bound]*3, self._model.actuator_ctrlrange[ids, 1]].astype(np.float64),
            dtype=np.float64) for side, ids in self._native_ctrl_ids.items()}
        self.action_space = boxes["right"] if len(boxes) == 1 else spaces.Dict(boxes)

    def _prime_rgb_array_renderer(self):
        if self.pixels: return super()._prime_rgb_array_renderer()

    def render(self, *args, **kwargs):
        if self.pixels: return super().render(*args, **kwargs)
        # Diagnostic-only mode. The dataset writer must never save these.
        return [np.zeros((1, 1, 3), dtype=np.uint8)] * (5 if self.task.startswith("bimanual") else 4)

    def reset(self, *, seed=None, **kwargs):
        self.scene_seed = self.scene_seed if seed is None else int(seed)
        old_np, old_py = np.random.get_state(), random.getstate()
        try:
            np.random.seed(self.scene_seed); random.seed(self.scene_seed)
            self._last_successful_digit = None
            self._success_observed_step = None
            result = super().reset(seed=self.scene_seed, **kwargs)
            for side, ids in self._native_ctrl_ids.items():
                self._data.ctrl[ids] = self.hand_metadata["sides"][side]["home"]
            mujoco.mj_forward(self._model, self._data)
            from .contact_policy import validate_applied
            validate_applied(self._model, self.hand_metadata)
            if self.hand_metadata['version']=='benchmark7_v3':
                from .geometry_revision import validate_geometry
                validate_geometry(self._model,self.hand_metadata)
            return result
        finally:
            np.random.set_state(old_np); random.setstate(old_py)

    def tcp(self, side="right"):
        sid = self._tcp_sites[side]; pose = np.eye(4)
        pose[:3, 3] = self._data.site_xpos[sid]
        pose[:3, :3] = self._data.site_xmat[sid].reshape(3, 3)
        return pose

    def hold_action(self):
        return {side: np.r_[np.zeros(6), self._data.ctrl[ids]] for side, ids in self._native_ctrl_ids.items()}

    def step(self, action):
        """Actions use the independent actuator dimensions of each hand."""
        if not isinstance(action, dict):
            if len(self._native_ctrl_ids) != 1: raise ValueError("Dual action must have right and left entries")
            action = {"right": action}
        if set(action) != set(self._native_ctrl_ids): raise ValueError("Wrong active arm set")
        converted = {}
        for side, ids in self._native_ctrl_ids.items():
            value = np.asarray(action[side], dtype=np.float64)
            if value.shape != (6+len(ids),) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {side} action")
            # Refuse to record an action different from the action actually run.
            if np.any(np.abs(value[:3]) > self._translation_bound+1e-9) or np.any(np.abs(value[3:6]) > self._rotation_bound+1e-9):
                raise ValueError("TCP delta exceeds contract bounds")
            limits = self._model.actuator_ctrlrange[ids]
            if np.any(value[6:] < limits[:, 0]-1e-7) or np.any(value[6:] > limits[:, 1]+1e-7):
                raise ValueError("Hand action exceeds native actuator bounds")
            actual = self.tcp(side)
            rot = Rotation.from_rotvec(value[3:6]).as_matrix() @ actual[:3, :3]
            quat = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
            converted[side] = np.r_[actual[:3, 3]+value[:3], quat, value[6:]]
        result = super().step(converted if len(converted)>1 else converted["right"])
        for side, ids in self._native_ctrl_ids.items():
            if not np.allclose(self._data.ctrl[ids], np.asarray(action[side])[6:], atol=1e-10, rtol=0):
                raise RuntimeError("Recorded native hand action differs from executed actuator target")
        if not np.all(np.isfinite(self._data.qpos)) or not np.all(np.isfinite(self._data.qvel)):
            raise RuntimeError("Non-finite simulation state")
        return result


class Hammer(Common, PandaHammerNailGymEnv): task = "hammer_nail"
class Bucket(Common, PandaPickBucketGymEnv): task = "pick_bucket"
class Microwave(Common, PandaBimanualMicrowaveCookGymEnv): task = "bimanual_microwave_cook"
class Photograph(Common, PandaBimanualPhotographGymEnv): task = "bimanual_photograph"





class Ipad(Common, PandaBimanualUnlockIpadGymEnv):
    task = "bimanual_unlock_ipad"

    def _has_ipad_touch_contact(self):
        finger_ids = set(getattr(self, "_finger_geom_ids", ()))
        target_ids = set(getattr(self, "_touch_contact_geom_ids", ())).union(
            getattr(self, "_button_geom_ids", ())
        )
        if not finger_ids or not target_ids:
            return False
        for i in range(int(self._data.ncon)):
            contact = self._data.contact[i]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if (geom1 in finger_ids and geom2 in target_ids) or (
                geom2 in finger_ids and geom1 in target_ids
            ):
                return True
        return False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .ipad_task import contract
        self.contract['ipad_task_rules'] = contract()

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        from .ipad_task import SequenceGate, Measurements
        self._ipad_gate = SequenceGate()
        self._ipad_measurements = Measurements(self)
        return result

    def _after_ipad_physics_step(self):
        measurements = self._ipad_measurements.sample()
        self._ipad_gate.observe(measurements, self._data.time)
        if self._ipad_gate.failure:
            self._unlock_failed = True
            self._unlock_failure_reason = self._ipad_gate.failure
            self._screen_unlocked = False
            self._success_trigger_count = 0
            self._set_screen_locked()

    def _advance_unlock_sequence(self, pressed: set):
        """Apply the shared tap/release protocol to every hand."""
        if not self._button_geom_to_digit:
            self._pressed_last = pressed
            return []

        if self._button_release_required:
            if self._has_ipad_touch_contact():
                self._pressed_last = pressed
                return []
            self._button_release_required = False
            self._pressed_last = set()
            return []

        newly_pressed = pressed - self._pressed_last
        inputs = [
            (geom_id, self._button_geom_to_digit[geom_id])
            for geom_id in sorted(newly_pressed)
            if geom_id in self._button_geom_to_digit
        ]
        self._pressed_last = pressed
        if len(inputs) != 1:
            return []

        geom_id, digit = inputs[0]
        expected = self._unlock_sequence[self._unlock_index]
        if (
            self._last_successful_digit is not None
            and digit == self._last_successful_digit
            and digit != expected
        ):
            return []

        if digit == expected:
            self._unlock_index += 1
            self._last_successful_digit = int(digit)
            self._button_release_required = True
            if self._unlock_index >= len(self._unlock_sequence):
                self._set_screen_unlocked()
                self._hide_buttons()
                self._screen_unlocked = True
        else:
            self._unlock_failed = True
            self._unlock_failure_reason = f"wrong_digit_expected_{expected}_got_{digit}"
            self._last_wrong_digit = int(digit)
            self._unlock_index = 0
            self._last_successful_digit = None
            self._button_release_required = False
            self._set_screen_locked()
            self._screen_unlocked = False
        return [digit]


    def _update_unlock_sequence(self, pressed):
        gate = self._ipad_gate
        if not gate.ready or self._unlock_failed:
            return []
        previous = self._unlock_index
        result = self._advance_unlock_sequence(pressed)
        if self._unlock_failed:
            gate.fail(self._unlock_failure_reason, self._data.time)
        if self._unlock_index > previous:
            gate.mark_digit(self._unlock_index, self._data.time)
        return result

    def ipad_task_metrics(self):
        sensor = self._ipad_measurements
        return dict(**sensor.last, phase=self._ipad_gate.phase, failure=self._ipad_gate.failure,
            max_lift_m=sensor.maximum_lift, max_left_contact_points=sensor.maximum_left_contacts,
            opposed_grasp_samples=sensor.grasp_samples, physics_samples=sensor.samples,
            phase_history=list(self._ipad_gate.history))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(ipad_phase=self._ipad_gate.phase, ipad_task_metrics=self.ipad_task_metrics())
        return obs, reward, terminated, truncated, info

    def _compute_success(self):
        # The legacy getter also calls this function. Queries must not advance
        # a 10-step hold counter and manufacture success without robot steps.
        if self._unlock_failed or self._ipad_gate.phase != 'unlocked': return False
        if self._success_observed_step != self.env_step:
            self._success_trigger_count = self._success_trigger_count+1 if self._screen_unlocked else 0
            self._success_observed_step = self.env_step
        return self._screen_unlocked and self._success_trigger_count >= self._success_trigger_target

from .single_hand_hanoi import SingleHandHanoi

CLASSES = {c.task: c for c in (Hammer, Bucket, Microwave, Photograph, Ipad)}
CLASSES['single_hand_hanoi'] = SingleHandHanoi


def make_env(task, hand, seed=0, pixels=False):
    if task not in TARGET_TASKS:
        raise ValueError(
            f"Unsupported DexJoCo-X release task {task!r}; expected one of {TARGET_TASKS}"
        )
    env = CLASSES[task](hand=hand, seed=seed, pixels=pixels)
    env.reset(seed=seed)
    return env
