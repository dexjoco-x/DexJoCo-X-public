import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ..controllers import apply_arm_pose_control
from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from ..rendering import MujocoRenderer
from .arm_profiles import configure_bimanual_arm
from .base_hand_specs import (
    ALLEGRO_LEFT_BIMANUAL_ACTUATOR_NAMES,
    ALLEGRO_LEFT_BIMANUAL_JOINT_NAMES,
    ALLEGRO_RIGHT_BIMANUAL_ACTUATOR_NAMES,
    ALLEGRO_RIGHT_BIMANUAL_JOINT_NAMES,
    shadow_hand_e3m5_joint_targets_to_actuator_ctrl,
)

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_ipad.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))  # Origin
_ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32
)
_CARTESIAN_BOUNDS = np.asarray([[-0.8, -0.8, -0.8], [0.8, 0.8, 0.8]])
_STAND_SAMPLING_BOUNDS = np.array([[-0.35, 0.05], [-0.30, 0.1]])
_ALLEGRO_UNLOCK_HAND_POSE = np.asarray(
    [
        0.1, 0.1, 0.1, 0.0,  # ff (index extended)
        1.2, 1.2, 1.2, 1.2,  # mf curled
        1.2, 1.2, 1.2, 1.2,  # rf curled
        1.0, 1.0, 1.0, 1.0,  # thumb curled
    ],
    dtype=np.float32,
)
_DEFAULT_BUTTON_FINGERTIP_GEOM_NAMES = (
    "fingertip_ff_right",
    "fingertip_mf_right",
    "fingertip_rf_right",
    "thumbtip_right",
)


class PandaBimanualUnlockIpadGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        config=None,
        hz=30,
        password: list[int] | None = None,
        ipad_screen_effect: bool = False,
        max_episode_steps: int = 1200,
        xml_path: Path | None = None,
        right_hand_joint_names: tuple[str, ...] = ALLEGRO_RIGHT_BIMANUAL_JOINT_NAMES,
        left_hand_joint_names: tuple[str, ...] = ALLEGRO_LEFT_BIMANUAL_JOINT_NAMES,
        right_hand_actuator_names: tuple[str, ...] = ALLEGRO_RIGHT_BIMANUAL_ACTUATOR_NAMES,
        left_hand_actuator_names: tuple[str, ...] = ALLEGRO_LEFT_BIMANUAL_ACTUATOR_NAMES,
        right_hand_home: np.ndarray = _ALLEGRO_HOME,
        left_hand_home: np.ndarray = _ALLEGRO_HOME,
        hand_command_mode: str = "actuator",
        right_arm_home: np.ndarray = _PANDA_HOME,
        left_arm_home: np.ndarray = _PANDA_HOME,
        arm_nullspace_stiffness: float = 0.5,
        arm_max_ori_acceleration: float | None = None,
        button_fingertip_geom_names: tuple[str, ...] | None = _DEFAULT_BUTTON_FINGERTIP_GEOM_NAMES,
        button_fingertip_body_names: tuple[str, ...] | None = None,
        button_touch_contact_geom_names: tuple[str, ...] | None = None,
        button_touch_trigger_digits: tuple[int, ...] | None = None,
    ):
        self.hz = int(hz)
        self._action_scale = action_scale
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics
        self._ipad_screen_effect = bool(ipad_screen_effect)
        self._max_episode_steps = int(max_episode_steps)
        self._hand_command_mode = str(hand_command_mode).lower()
        self._right_hand_home = np.asarray(right_hand_home, dtype=np.float64)
        self._left_hand_home = np.asarray(left_hand_home, dtype=np.float64)
        self._arm_home_right = np.asarray(right_arm_home, dtype=np.float64)
        self._arm_home_left = np.asarray(left_arm_home, dtype=np.float64)
        self._arm_nullspace_stiffness = float(arm_nullspace_stiffness)
        self._arm_max_ori_acceleration = (
            None if arm_max_ori_acceleration is None else float(arm_max_ori_acceleration)
        )
        self._n_hand_right = int(len(right_hand_joint_names))
        self._n_hand_left = int(len(left_hand_joint_names))
        if xml_path is None:
            xml_path = _XML_PATH

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        # Seed the RNGs used by environment randomization.
        random.seed(seed)
        np.random.seed(seed)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.image_obs = image_obs
        self.env_step = 0
        self.intervened = False

        # Panda caches
        self._arm_profile_right, self._panda_right_dof_ids, self._panda_right_ctrl_ids = configure_bimanual_arm(
            self._model,
            "right",
        )
        self._site_right_id = self._model.site("attachment_site_right").id
        self._arm_profile_left, self._panda_left_dof_ids, self._panda_left_ctrl_ids = configure_bimanual_arm(
            self._model,
            "left",
        )
        self._site_left_id = self._model.site("attachment_site_left").id
        if self._arm_home_right.shape[0] != self._panda_right_dof_ids.shape[0]:
            self._arm_home_right = self._arm_profile_right.home_qpos.copy()
        if self._arm_home_left.shape[0] != self._panda_left_dof_ids.shape[0]:
            self._arm_home_left = self._arm_profile_left.home_qpos.copy()

        self._panda_dof_ids = np.concatenate(
            [self._panda_right_dof_ids, self._panda_left_dof_ids]
        )
        self._panda_ctrl_ids = np.concatenate(
            [self._panda_right_ctrl_ids, self._panda_left_ctrl_ids]
        )
        self._mocap_right_id = int(self._model.body("target_right").mocapid)
        self._mocap_left_id = int(self._model.body("target_left").mocapid)

        allegro_actuator_right_names = list(right_hand_actuator_names)
        allegro_actuator_left_names = list(left_hand_actuator_names)
        # 变量名保留 allegro 是为了少改原始环境；现在它表示“当前手型”。
        self._allegro_joint_right_names = list(right_hand_joint_names)
        self._allegro_joint_left_names = list(left_hand_joint_names)
        self._allegro_actuator_right_names = list(right_hand_actuator_names)
        self._allegro_actuator_left_names = list(left_hand_actuator_names)
        # Get actuator ids (fall back to mj_name2id if necessary)
        allegro_ids = []
        for name in allegro_actuator_right_names:
            aid = self._model.actuator(name).id
            allegro_ids.append(aid)

        for name in allegro_actuator_left_names:
            aid = self._model.actuator(name).id
            allegro_ids.append(aid)

        # print("Allegro actuator IDs:", allegro_ids)

        self._allegro_ctrl_ids = np.asarray(allegro_ids, dtype=int)
        self._allegro_ctrl_ids_right = self._allegro_ctrl_ids[: len(allegro_actuator_right_names)]
        self._allegro_ctrl_ids_left = self._allegro_ctrl_ids[len(allegro_actuator_right_names) :]

        self._allegro_dof_right_ids = np.asarray(
            [
                int(self._model.joint(n).qposadr)
                for n in self._allegro_joint_right_names
            ],
            dtype=int,
        )
        self._allegro_dof_left_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_left_names],
            dtype=int,
        )
        self._allegro_vel_right_ids = np.asarray(
            [int(self._model.joint(n).dofadr) for n in self._allegro_joint_right_names],
            dtype=int,
        )
        self._allegro_vel_left_ids = np.asarray(
            [int(self._model.joint(n).dofadr) for n in self._allegro_joint_left_names],
            dtype=int,
        )

        self._table_body_id = self._model.body("table").id
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._table_body_id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self._model.geom_size[gid, 1])
            for gid in self._table_leg_geom_ids
        }

        self._front_camera_id = int(self._model.camera("back").id)
        self._ego_left_camera_id = self._get_cam_id_by_name("left")
        self._ego_right_camera_id = self._get_cam_id_by_name("right")
        self._wrist_left_camera_id = self._get_cam_id_by_name("handcam_rgb_left")
        self._wrist_right_camera_id = self._get_cam_id_by_name("handcam_rgb_right")
        self.camera_id = (
            self._front_camera_id,
            self._ego_left_camera_id,
            self._ego_right_camera_id,
            self._wrist_left_camera_id,
            self._wrist_right_camera_id,
        )

        self._camera_params = np.load(_HERE / "replay_cameras_2.npy")
        self._num_preset_cameras = int(self._camera_params.shape[0])
        self._scene_center = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._orig_light_pos = self._model.light_pos.copy()
        self._orig_light_dir = self._model.light_dir.copy()
        self._table_geom_id = self._model.geom("table_visual").id
        self._texture_names = [
            "table_bamboo",
            "table_blue-wood",
            "table_brass-ambra",
            "table_ceramic",
            "table_cream-plaster",
            "table_dark_wood_planks_2",
            "table_dark-wood",
            "table_gray-plaster",
            "table_gray_wood_planks",
            "table_light-wood",
            "table_metal",
            "table_pink-plaster",
            "table_red-wood",
            "table_legs_metal",
            "table_steel-scratched",
            "table_walnut_wood_grain",
            "table_warm_wood_grain_2",
            "table_white-plaster",
            "table_wood_grain_1",
            "table_yellow-plaster",
        ]

        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,)),
                "gripper_pose": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self._n_hand_right + self._n_hand_left,),
                ),
                "stand_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "ipad_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )

        observation_space_dict = {"state": state_space}
        if self.image_obs:
            image_h = int(self._model.vis.global_.offheight)
            image_w = int(self._model.vis.global_.offwidth)

            image_spaces = {
                "random_camera" if self.randomize else "ego": spaces.Box(
                    0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                )
            }
            if self._ego_left_camera_id >= 0:
                image_spaces["ego_left"] = spaces.Box(
                    0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                )
            if self._ego_right_camera_id >= 0:
                image_spaces["ego_right"] = spaces.Box(
                    0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                )
            if self._wrist_left_camera_id >= 0:
                image_spaces["wrist_left"] = spaces.Box(
                    0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                )
            if self._wrist_right_camera_id >= 0:
                image_spaces["wrist_right"] = spaces.Box(
                    0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                )
            observation_space_dict["images"] = spaces.Dict(image_spaces)

        self.observation_space = spaces.Dict(observation_space_dict)

        # Action space
        self.action_space = spaces.Box(
            low=np.full(
                7 + max(self._n_hand_right, self._n_hand_left),
                -1.0,
                dtype=np.float32,
            ),
            high=np.full(
                7 + max(self._n_hand_right, self._n_hand_left),
                1.0,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        # --- Button interaction setup ---
        self._button_geom_ids = []
        self._button_geom_orig_matid = {}
        self._button_geom_orig_rgba = {}
        self._button_geom_to_digit = {}
        self._button_visual_geom_ids = []
        self._button_visual_orig_rgba = {}
        self._digit_geom_ids_by_digit = {}
        try:
            for gid in range(self._model.ngeom):
                gname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if gname and gname.startswith("btn") and gname.endswith("_cyl"):
                    self._button_geom_ids.append(gid)
                    self._button_geom_orig_matid[gid] = int(self._model.geom_matid[gid])
                    self._button_geom_orig_rgba[gid] = self._model.geom_rgba[gid].copy()
                    # Map btnX_cyl -> digit X
                    try:
                        num = int(gname.replace("btn", "").replace("_cyl", ""))
                        self._button_geom_to_digit[gid] = num
                    except Exception:
                        pass
                if gname and gname.startswith("btn") and gname.endswith("_digit"):
                    try:
                        num = int(gname.replace("btn", "").replace("_digit", ""))
                        self._digit_geom_ids_by_digit[num] = gid
                    except Exception:
                        pass
                # Track all button/digit visuals by material name
                try:
                    matid = int(self._model.geom_matid[gid])
                    mname = mujoco.mj_id2name(
                        self._model, mujoco.mjtObj.mjOBJ_MATERIAL, matid
                    )
                    if mname in ("mat_digit",) or (
                        mname and mname.startswith("mat_digit_")
                    ):
                        self._button_visual_geom_ids.append(gid)
                        self._button_visual_orig_rgba[gid] = self._model.geom_rgba[
                            gid
                        ].copy()
                except Exception:
                    pass
        except Exception:
            pass
            # print("[Warning] failed to collect button geoms:", e)

        self._mat_digit_ids = {}
        self._mat_digit_light_ids = {}
        for d in range(10):
            try:
                self._mat_digit_ids[d] = int(self._model.material(f"mat_digit_{d}").id)
            except Exception:
                self._mat_digit_ids[d] = -1
            try:
                self._mat_digit_light_ids[d] = int(
                    self._model.material(f"mat_digit_{d}_light").id
                )
            except Exception:
                self._mat_digit_light_ids[d] = -1

        # --- Screen unlock interaction ---
        if password is None:
            self._unlock_sequence = [1, 2, 3]
        else:
            assert all(0 <= p and p <= 9 for p in password), "Password digits must be between 0 and 9."
            self._unlock_sequence = list(password)

        self._unlock_index = 0
        self._unlock_failed = False
        self._pressed_last = set()
        self._buttons_hidden = False
        self._screen_unlocked = False
        self._unlock_pending = False
        self._unlock_delay_steps = 12
        self._unlock_delay_count = 0
        self._unlock_transition_steps = 18
        self._unlock_transition_step = 0
        self._unlock_final_pressed = set()
        self._screen_geom_id = -1
        self._mat_screen_locked_id = -1
        self._mat_screen_unlocked_id = -1
        try:
            self._screen_geom_id = int(self._model.geom("ipad_screen").id)
        except Exception:
            pass
        try:
            self._mat_screen_locked_id = int(self._model.material("Material_black").id)
        except Exception:
            pass
        try:
            self._mat_screen_unlocked_id = int(
                self._model.material("mat_screen_unlocked").id
            )
        except Exception:
            pass
        self._mat_screen_locked_rgba = self._get_material_rgba(
            self._mat_screen_locked_id, np.array([0.2, 0.2, 0.2, 1.0])
        )
        self._mat_screen_unlocked_rgba = self._get_material_rgba(
            self._mat_screen_unlocked_id, np.ones(4)
        )

        # Default unlock parameters and pose.
        self._unlock_params = {
            "sequence": list(self._unlock_sequence),
            "settle_steps": 10,
            "above_steps": 90,
            "press_steps": 50,
            "post_press_steps": 15,
            "release_steps": 40,
            "above_offset": 0.09,
            "press_offset": 0.004,
            "dwell_s": 0.2,
        }

        self._unlock_action_template = np.zeros(
            7 + self._n_hand_right, dtype=np.float32
        )
        if self._n_hand_right == _ALLEGRO_UNLOCK_HAND_POSE.size:
            self._unlock_action_template[7:] = _ALLEGRO_UNLOCK_HAND_POSE
        else:
            self._unlock_action_template[7:] = self._right_hand_home.astype(np.float32)
        self._success_trigger_count = 0
        self._success_trigger_target = 10

        self._model_geom_pos0 = self._model.geom_pos.copy()
        self._model_geom_size0 = self._model.geom_size.copy()
        self._model_site_pos0 = self._model.site_pos.copy()

        self._table_z = self._model.body("table").pos[2].copy()

        self._ipad_body_z0 = self._model.body("ipad_body").pos[2].copy()
        self._stand_body_z0 = self._model.body("ipad_stand").pos[2].copy()

        self._finger_geom_ids = self._collect_fingertip_geom_ids(
            button_fingertip_geom_names=button_fingertip_geom_names,
            button_fingertip_body_names=button_fingertip_body_names,
        )
        self._touch_contact_geom_ids = self._collect_geom_ids_by_name(
            button_touch_contact_geom_names
        )
        self._touch_trigger_digits = set(button_touch_trigger_digits or ())
        self._ipad_body_id = self._model.body("ipad_body").id
        self._ipad_mass0 = float(self._model.body_mass[self._ipad_body_id])
        self._ipad_mass_mul = (0.75, 1.25)

    def _collect_geom_ids_by_name(self, geom_names: tuple[str, ...] | None) -> set[int]:
        geom_ids: set[int] = set()
        for name in geom_names or ():
            try:
                geom_ids.add(int(self._model.geom(name).id))
            except Exception:
                pass
        return geom_ids

    def _collect_fingertip_geom_ids(
        self,
        button_fingertip_geom_names: tuple[str, ...] | None,
        button_fingertip_body_names: tuple[str, ...] | None,
    ) -> set[int]:
        fingertip_geom_ids: set[int] = set()
        for name in button_fingertip_geom_names or ():
            try:
                fingertip_geom_ids.add(int(self._model.geom(name).id))
            except Exception:
                pass

        body_ids = set()
        for body_name in button_fingertip_body_names or ():
            try:
                body_ids.add(int(self._model.body(body_name).id))
            except Exception:
                pass
        if body_ids:
            for gid in range(self._model.ngeom):
                if int(self._model.geom_bodyid[gid]) in body_ids:
                    fingertip_geom_ids.add(gid)

        return fingertip_geom_ids

    def _get_cam_id_by_name(self, name: str) -> int:
        """Return camera id from name; return -1 if not found."""
        try:
            cam = self._model.camera(name)
            return int(cam.id)
        except Exception:
            try:
                return int(
                    mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                )
            except Exception:
                return -1

    def randomize_lighting(self):
        model = self._model

        model.light_pos[:] = self._orig_light_pos
        model.light_dir[:] = self._orig_light_dir

        for i in range(model.nlight):
            model.light_pos[i, 0] += random.uniform(-0.3, 0.3)
            model.light_pos[i, 1] += random.uniform(-0.3, 0.3)
            model.light_dir[i, 0] += random.uniform(-0.4, 0.4)
            model.light_dir[i, 1] += random.uniform(-0.4, 0.4)
            model.light_diffuse[i] = [random.uniform(0.3, 0.8) for _ in range(3)]

        model.vis.headlight.ambient[:] = [random.uniform(0.3, 0.7) for _ in range(3)]
        model.vis.headlight.diffuse[:] = [random.uniform(0.2, 0.6) for _ in range(3)]

        # print("Randomized light positions:", model.light_pos)

    def randomize_desktop_texture(self):
        chosen_texture = random.choice(self._texture_names)
        mat_id = self._model.material(chosen_texture).id
        self._model.geom_matid[self._table_geom_id] = mat_id

    def randomize_camera(self):
        """Randomly choose one preset camera from replay_cameras.npy."""
        random_camera_idx = random.randint(0, self._num_preset_cameras - 1)
        self._apply_random_camera_to_front(random_camera_idx)

    def _prime_rgb_array_renderer(self):
        """Discard one offscreen frame per camera to avoid stale first-reset images."""
        for cam_id in self.camera_id:
            if cam_id >= 0:
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)

    def _apply_random_camera_to_front(self, camera_idx):
        camera = self._camera_params[camera_idx]
        azimuth = float(camera[0])
        elevation = float(-camera[1])
        distance = float(camera[2])

        azim_rad = np.deg2rad(azimuth)
        elev_rad = np.deg2rad(elevation)

        cam_offset = np.array(
            [
                -distance * np.cos(elev_rad) * np.cos(azim_rad),
                distance * np.cos(elev_rad) * np.sin(azim_rad),
                distance * np.sin(elev_rad),
            ],
            dtype=np.float64,
        )
        cam_pos = self._scene_center + cam_offset

        forward = -cam_offset
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        rot_matrix = np.column_stack([right, up, -forward])
        cam_quat_wxyz = R.from_matrix(rot_matrix).as_quat(scalar_first=True)

        self._model.cam_pos[self._front_camera_id] = cam_pos
        self._model.cam_quat[self._front_camera_id] = cam_quat_wxyz

    def _hand_targets_to_ctrl(self, joint_targets: np.ndarray, side: str) -> np.ndarray:
        if side == "right":
            ctrl_ids = self._allegro_ctrl_ids_right
            joint_names = tuple(self._allegro_joint_right_names)
            actuator_names = tuple(self._allegro_actuator_right_names)
        else:
            ctrl_ids = self._allegro_ctrl_ids_left
            joint_names = tuple(self._allegro_joint_left_names)
            actuator_names = tuple(self._allegro_actuator_left_names)

        joint_targets = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        if len(ctrl_ids) == len(joint_targets):
            return joint_targets
        if actuator_names and actuator_names[0].startswith("shadow_"):
            return shadow_hand_e3m5_joint_targets_to_actuator_ctrl(
                joint_targets,
                ctrl_ids,
                self._model.actuator_ctrlrange,
                joint_names=joint_names,
                actuator_names=actuator_names,
            )

        ctrl = np.zeros(len(ctrl_ids), dtype=np.float64)
        ctrl[: min(len(ctrl), len(joint_targets))] = joint_targets[: len(ctrl)]
        return ctrl

    # --------------------------
    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # reset table height
        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        table_ori_pos = self._model.body("table").pos
        table_ori_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_ori_pos

        dh = float(self.delta_h)

        for lgid in self._table_leg_geom_ids:
            base_center_z = float(self._model_geom_pos0[lgid, 2])
            base_half_len = self._table_leg_half_len0[lgid]

            new_center_z = base_center_z - 0.5 * dh
            new_half_len = base_half_len + 0.5 * dh

            self._model.geom_pos[lgid, 2] = new_center_z
            self._model.geom_size[lgid, 1] = new_half_len

        # Sample a new ipad position.
        stand_xy = np.random.uniform(*_STAND_SAMPLING_BOUNDS)
        stand_body_id = self._model.body("ipad_stand").id
        stand_pos = self._model.body("ipad_stand").pos
        stand_quat = self._model.body("ipad_stand").quat
        ori_stand_xy = np.array(self._model.body_pos[stand_body_id][:2])
        delta_xy = stand_xy - ori_stand_xy

        # print("delta_xy:", delta_xy)
        ipad_body_id = self._model.body("ipad_body").id
        ipad_pos = self._model.body_pos[ipad_body_id]
        ipad_quat = self._model.body("ipad_body").quat
        ipad_pos[:2] += delta_xy
        ipad_pos[2] = self._ipad_body_z0 + self.delta_h

        # print("stand_xy:", stand_xy)

        stand_pos[:2] = stand_xy
        stand_pos[2] = self._stand_body_z0 + self.delta_h
        self._ipad_ori_pose = np.array(
            list(ipad_pos) + list(ipad_quat), dtype=np.float64
        )
        self._stand_ori_pose = np.array(
            list(stand_pos) + list(stand_quat), dtype=np.float64
        )

        self._model.body_pos[stand_body_id] = self._stand_ori_pose[:3]
        self._data.jnt("ipad_freejoint").qpos[:3] = self._ipad_ori_pose[:3]

        # Reset arm to home position.
        self._data.qpos[self._panda_right_dof_ids] = self._arm_home_right
        self._data.qpos[self._panda_left_dof_ids] = self._arm_home_left
        self._data.qpos[self._allegro_dof_right_ids] = self._right_hand_home
        self._data.qpos[self._allegro_dof_left_ids] = self._left_hand_home
        self._data.qvel[self._allegro_vel_right_ids] = 0.0
        self._data.qvel[self._allegro_vel_left_ids] = 0.0
        self._data.ctrl[self._allegro_ctrl_ids_right] = self._hand_targets_to_ctrl(
            self._right_hand_home, "right"
        )
        self._data.ctrl[self._allegro_ctrl_ids_left] = self._hand_targets_to_ctrl(
            self._left_hand_home, "left"
        )
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        self._data.mocap_pos[self._mocap_right_id] = tcp_pos_right
        self._data.mocap_pos[self._mocap_left_id] = tcp_pos_left
        self._data.mocap_quat[self._mocap_right_id] = tcp_quat_right
        self._data.mocap_quat[self._mocap_left_id] = tcp_quat_left

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            mass = float(
                np.random.uniform(self._ipad_mass_mul[0], self._ipad_mass_mul[1])
            )
            self._model.body_mass[self._ipad_body_id] = self._ipad_mass0 * mass

        # print(
        #     "ipad mass="
        #     f"{self._model.body_mass[self._ipad_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0

        # Reset button visuals
        self._set_button_materials(set())
        self._unlock_index = 0
        self._unlock_failed = False
        self._unlock_failure_reason = ""
        self._last_wrong_digit = None
        self._success_trigger_count = 0
        self._pressed_last = set()
        self._button_release_required = False
        self._buttons_hidden = False
        self._unlock_pending = False
        self._unlock_delay_count = 0
        self._unlock_transition_step = 0
        self._unlock_final_pressed = set()
        self._set_screen_locked()
        self._show_buttons()
        self._screen_unlocked = False

        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(
        self, action
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        # Determine if action is a "zero op" (None / scalar 0 / size-1 array 0).
        is_zero_action = False
        if action is None:
            is_zero_action = True
        else:
            try:
                if np.isscalar(action) and action == 0:
                    is_zero_action = True
                elif (
                    isinstance(action, np.ndarray)
                    and action.size == 1
                    and action.item() == 0
                ):
                    is_zero_action = True
            except Exception:
                is_zero_action = False

        # ----- parse action when not zero -----
        if not is_zero_action:
            # print("Received action:", action)
            # action expected to be dict: {"right": arr23, "left": arr23}
            right = np.asarray(action["right"])
            left = np.asarray(action["left"])

            # Parse franka delta
            x_r, y_r, z_r, w_r, qx_r, qy_r, qz_r = right[0:7]
            x_l, y_l, z_l, w_l, qx_l, qy_l, qz_l = left[0:7]

            # Parse the right/left hand qpos targets for the active hand type.
            allegro_r = np.asarray(
                right[7 : 7 + self._n_hand_right], dtype=np.float64
            )
            allegro_l = np.asarray(left[7 : 7 + self._n_hand_left], dtype=np.float64)
            allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
        else:
            # zero action -> do not change mocap nor allegro
            x_r = y_r = z_r = w_r = qx_r = qy_r = qz_r = 0.0
            x_l = y_l = z_l = w_l = qx_l = qy_l = qz_l = 0.0
            allegro_angles = None  # Keep the current Allegro pose on zero action.

        # ----- read current mocap (keep backups) -----
        r_pos = self._data.mocap_pos[self._mocap_right_id].copy()
        l_pos = self._data.mocap_pos[self._mocap_left_id].copy()
        r_quat = self._data.mocap_quat[self._mocap_right_id].copy()
        l_quat = self._data.mocap_quat[self._mocap_left_id].copy()

        # construct target mocap arrays (if action given)
        tpos_r = np.asarray([x_r, y_r, z_r])
        tquat_r = np.array([w_r, qx_r, qy_r, qz_r])
        tpos_l = np.asarray([x_l, y_l, z_l])
        tquat_l = np.array([w_l, qx_l, qy_l, qz_l])

        # ----- apply mocap for right (keep original protection logic) -----
        if not is_zero_action and not (
            np.allclose(tpos_r, 0.0) and np.allclose(tquat_r, 0.0)
        ):
            # print("Applying mocap action - Right TCP pos:", tpos_r, "quat:", tquat_r)
            self._data.mocap_pos[self._mocap_right_id] = tpos_r
            self._data.mocap_quat[self._mocap_right_id] = tquat_r
        else:
            # keep original right pose
            self._data.mocap_pos[self._mocap_right_id] = r_pos
            self._data.mocap_quat[self._mocap_right_id] = r_quat

        # ----- apply mocap for left (preserve the current pose on zero action) -----
        if not is_zero_action and not (
            np.allclose(tpos_l, 0.0) and np.allclose(tquat_l, 0.0)
        ):
            self._data.mocap_pos[self._mocap_left_id] = tpos_l
            self._data.mocap_quat[self._mocap_left_id] = tquat_l
        else:
            # keep original left pose
            self._data.mocap_pos[self._mocap_left_id] = l_pos
            self._data.mocap_quat[self._mocap_left_id] = l_quat

        # print("Applied mocap pos right:", self._data.mocap_pos[0])
        # print("Applied mocap quat right:", self._data.mocap_quat[0])
        # print("Applied mocap pos left:", self._data.mocap_pos[1])
        # print("Applied mocap quat left:", self._data.mocap_quat[1])

        # ----- control loop  -----
        for _ in range(self._n_substeps):
            tau_right = apply_arm_pose_control(
                model=self._model,
                data=self._data,
                profile=self._arm_profile_right,
                site_id=self._site_right_id,
                dof_ids=self._panda_right_dof_ids,
                ctrl_ids=self._panda_right_ctrl_ids,
                pos=self._data.mocap_pos[self._mocap_right_id],
                ori=self._data.mocap_quat[self._mocap_right_id],
                joint=self._arm_home_right,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
                nullspace_stiffness=self._arm_nullspace_stiffness,
                max_ori_acceleration=self._arm_max_ori_acceleration,
            )
            self._data.ctrl[self._panda_right_ctrl_ids] = tau_right

            tau_left = apply_arm_pose_control(
                model=self._model,
                data=self._data,
                profile=self._arm_profile_left,
                site_id=self._site_left_id,
                dof_ids=self._panda_left_dof_ids,
                ctrl_ids=self._panda_left_ctrl_ids,
                pos=self._data.mocap_pos[self._mocap_left_id],
                ori=self._data.mocap_quat[self._mocap_left_id],
                joint=self._arm_home_left,
                gravity_comp=True,
                pos_gains=(400.0, 400.0, 400.0),
                damping_ratio=4,
                nullspace_stiffness=self._arm_nullspace_stiffness,
                max_ori_acceleration=self._arm_max_ori_acceleration,
            )
            self._data.ctrl[self._panda_left_ctrl_ids] = tau_left

            if allegro_angles is not None:
                if self._hand_command_mode == "qpos":
                    right_angles = allegro_angles[: self._n_hand_right]
                    left_angles = allegro_angles[
                        self._n_hand_right : self._n_hand_right + self._n_hand_left
                    ]
                    self._data.qpos[self._allegro_dof_right_ids] = right_angles
                    self._data.qpos[self._allegro_dof_left_ids] = left_angles
                    self._data.qvel[self._allegro_vel_right_ids] = 0.0
                    self._data.qvel[self._allegro_vel_left_ids] = 0.0
                right_angles = allegro_angles[: self._n_hand_right]
                left_angles = allegro_angles[
                    self._n_hand_right : self._n_hand_right + self._n_hand_left
                ]
                self._data.ctrl[self._allegro_ctrl_ids_right] = self._hand_targets_to_ctrl(
                    right_angles, "right"
                )
                self._data.ctrl[self._allegro_ctrl_ids_left] = self._hand_targets_to_ctrl(
                    left_angles, "left"
                )

            mujoco.mj_step(self._model, self._data)
            # Optional benchmark observer: reads physics, never changes the
            # action or object state. Kept outside the legacy visual try/except
            # so measurement failures cannot silently bypass task constraints.
            observer = getattr(self, '_after_ipad_physics_step', None)
            if observer is not None:
                observer()

        # Update button colors based on contacts
        pressed_digits = []
        try:
            pressed_digits = self._update_button_visuals()
            self._update_unlock_animation()
        except Exception:
            pass
            # print("[Warning] failed to update button visuals:", e)

        obs = self._compute_observation()

        self.env_step += 1
        terminated = False
        failure_reason = ""
        if self.env_step >= self._max_episode_steps or self._unlock_failed:
            terminated = True
            if self._unlock_failed:
                failure_reason = self._unlock_failure_reason or "wrong_digit"
            else:
                failure_reason = "max_episode_steps"

        # ---- human rendering ----
        if self.render_mode == "human":
            try:
                self._add_unlock_progress_overlay()
                self._viewer.render("human")
            except Exception:
                pass
                # print("[Warning] human render failed:", e)

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        success = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = terminated or success

        return (
            obs,
            rew,
            terminated,
            False,
            {
                "succeed": success,
                "grasp_penalty": 0.0,
                "pressed_digits": pressed_digits,
                "failure_reason": failure_reason,
                "unlock_index": int(self._unlock_index),
                "unlock_sequence": list(self._unlock_sequence),
                "last_wrong_digit": self._last_wrong_digit,
            },
        )

    def _set_button_materials(self, pressed_ids: set):
        """Set button materials based on pressed state."""
        if not self._button_geom_ids:
            return
        for gid in self._button_geom_ids:
            digit = self._button_geom_to_digit.get(gid)
            digit_gid = self._digit_geom_ids_by_digit.get(digit, None)
            if gid in pressed_ids:
                mat_light = self._mat_digit_light_ids.get(digit, -1)
                if mat_light >= 0:
                    self._model.geom_matid[gid] = mat_light
                    if digit_gid is not None:
                        self._model.geom_matid[digit_gid] = mat_light
                else:
                    rgba = self._model.geom_rgba[gid].copy()
                    rgba[:3] = [0.2, 0.4, 1.0]
                    rgba[3] = 1.0
                    self._model.geom_rgba[gid] = rgba
            else:
                mat_off = self._mat_digit_ids.get(digit, -1)
                if mat_off >= 0:
                    self._model.geom_matid[gid] = mat_off
                    if digit_gid is not None:
                        self._model.geom_matid[digit_gid] = mat_off
                else:
                    self._model.geom_rgba[gid] = self._button_geom_orig_rgba.get(
                        gid, self._model.geom_rgba[gid]
                    )

    def _update_button_visuals(self):
        """Detect contacts and toggle button color while pressed.
        Only count a button as pressed when the contacting geom is a fingertip geom.
        """
        if not self._button_geom_ids:
            return []
        if self._buttons_hidden:
            return []
        if self._unlock_pending:
            self._set_button_materials(self._unlock_final_pressed)
            return []

        button_set = set(self._button_geom_ids)
        pressed = set()

        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            # Only accept button <-> fingertip contacts.
            if g1 in button_set and g2 in self._finger_geom_ids:
                pressed.add(g1)

            elif g2 in button_set and g1 in self._finger_geom_ids:
                pressed.add(g2)

        pressed.update(self._compute_touch_trigger_button_ids())
        self._set_button_materials(pressed)
        pressed_digits = self._update_unlock_sequence(pressed)
        return pressed_digits

    def _compute_touch_trigger_button_ids(self) -> set[int]:
        if (
            not self._touch_contact_geom_ids
            or not self._finger_geom_ids
            or self._button_release_required
        ):
            return set()

        normal, _ = self._get_screen_normal_and_center()
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        pressed: set[int] = set()
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            if g1 in self._touch_contact_geom_ids and g2 in self._finger_geom_ids:
                contact_pos = np.asarray(c.pos, dtype=np.float64).copy()
            elif g2 in self._touch_contact_geom_ids and g1 in self._finger_geom_ids:
                contact_pos = np.asarray(c.pos, dtype=np.float64).copy()
            else:
                continue

            match_gid = self._match_contact_to_button(contact_pos, normal)
            if match_gid is not None:
                pressed.add(int(match_gid))

        return pressed

    def _match_contact_to_button(
        self, contact_pos: np.ndarray, normal: np.ndarray
    ) -> int | None:
        best_gid = None
        best_dist = float("inf")

        for gid, digit in self._button_geom_to_digit.items():
            if self._touch_trigger_digits and digit not in self._touch_trigger_digits:
                continue

            center = self._data.geom_xpos[gid].copy()
            radius = float(self._model.geom_size[gid][0]) * 1.15

            offset = contact_pos - center
            tangential = offset - float(np.dot(offset, normal)) * normal
            tangential_dist = float(np.linalg.norm(tangential))
            if tangential_dist <= radius and tangential_dist < best_dist:
                best_gid = int(gid)
                best_dist = tangential_dist

        return best_gid

    def _update_unlock_sequence(self, pressed: set):
        """Track rising-edge button presses for unlock sequence."""
        if not self._button_geom_to_digit:
            self._pressed_last = pressed
            return
        if self._button_release_required:
            if pressed:
                self._pressed_last = pressed
                return []
            self._button_release_required = False
        newly_pressed = pressed - self._pressed_last
        pressed_digits = []
        # TODO: only consider the first newly pressed button
        for gid in sorted(newly_pressed):
            digit = self._button_geom_to_digit.get(gid)
            if digit is None:
                continue
            pressed_digits.append(digit)
            expected = self._unlock_sequence[self._unlock_index]
            if digit == expected:
                self._unlock_index += 1
                self._button_release_required = True
                if self._unlock_index >= len(self._unlock_sequence):
                    if self._ipad_screen_effect:
                        self._start_unlock_animation({gid})
                    else:
                        self._set_screen_unlocked()
                        self._hide_buttons()
                        self._screen_unlocked = True
            else:
                # print("----!!!wrong input!!!----")
                # self.reset()
                # reset on wrong digit
                self._unlock_failed = True
                self._unlock_failure_reason = f"wrong_digit_expected_{expected}_got_{digit}"
                self._last_wrong_digit = int(digit)
                self._unlock_index = 0
                self._button_release_required = False
                self._set_screen_locked()
                self._screen_unlocked = False
        self._pressed_last = pressed
        return pressed_digits

    def _get_material_rgba(self, mat_id: int, fallback: np.ndarray) -> np.ndarray:
        if mat_id >= 0:
            try:
                return self._model.mat_rgba[mat_id].copy()
            except Exception:
                pass
        return fallback.astype(np.float32).copy()

    def _set_screen_locked(self):
        if self._screen_geom_id >= 0 and self._mat_screen_locked_id >= 0:
            self._model.geom_matid[self._screen_geom_id] = self._mat_screen_locked_id
        if self._mat_screen_unlocked_id >= 0:
            self._model.mat_rgba[self._mat_screen_unlocked_id] = (
                self._mat_screen_unlocked_rgba
            )

    def _set_screen_unlocked(self):
        if self._screen_geom_id >= 0 and self._mat_screen_unlocked_id >= 0:
            self._model.geom_matid[self._screen_geom_id] = self._mat_screen_unlocked_id
            self._model.mat_rgba[self._mat_screen_unlocked_id] = (
                self._mat_screen_unlocked_rgba
            )
        self._screen_unlocked = True

    def _start_unlock_animation(self, final_pressed: set):
        self._unlock_pending = True
        self._unlock_delay_count = int(self._unlock_delay_steps)
        self._unlock_transition_step = 0
        self._unlock_final_pressed = set(final_pressed)
        self._set_button_materials(self._unlock_final_pressed)

    def _update_unlock_animation(self):
        if not self._unlock_pending or self._screen_unlocked:
            return
        if self._unlock_delay_count > 0:
            self._unlock_delay_count -= 1
            self._set_button_materials(self._unlock_final_pressed)
            return

        steps = max(1, int(self._unlock_transition_steps))
        t = min(1.0, float(self._unlock_transition_step + 1) / float(steps))
        t = t * t * (3.0 - 2.0 * t)

        if self._screen_geom_id >= 0 and self._mat_screen_unlocked_id >= 0:
            self._model.geom_matid[self._screen_geom_id] = self._mat_screen_unlocked_id
            rgba = (
                self._mat_screen_locked_rgba * (1.0 - t)
                + self._mat_screen_unlocked_rgba * t
            )
            rgba[3] = self._mat_screen_unlocked_rgba[3]
            self._model.mat_rgba[self._mat_screen_unlocked_id] = rgba

        self._fade_buttons(1.0 - t)
        self._unlock_transition_step += 1

        if self._unlock_transition_step >= steps:
            self._unlock_pending = False
            self._set_screen_unlocked()
            self._hide_buttons()

    def _fade_buttons(self, alpha_scale: float):
        """Fade button/digit geoms while preserving their current materials."""
        alpha_scale = float(np.clip(alpha_scale, 0.0, 1.0))
        if not self._button_visual_geom_ids:
            return
        for gid in self._button_visual_geom_ids:
            base = self._button_visual_orig_rgba.get(gid, self._model.geom_rgba[gid])
            rgba = self._model.geom_rgba[gid].copy()
            rgba[3] = float(base[3]) * alpha_scale
            self._model.geom_rgba[gid] = rgba

    def _hide_buttons(self):
        """Hide all button and digit visuals (alpha=0)."""
        if not self._button_visual_geom_ids:
            return
        for gid in self._button_visual_geom_ids:
            rgba = self._model.geom_rgba[gid].copy()
            rgba[3] = 0.0
            self._model.geom_rgba[gid] = rgba
        self._buttons_hidden = True

    def _show_buttons(self):
        """Restore button and digit visuals."""
        if not self._button_visual_geom_ids:
            return
        for gid in self._button_visual_geom_ids:
            self._model.geom_rgba[gid] = self._button_visual_orig_rgba.get(
                gid, self._model.geom_rgba[gid]
            )

    def _compute_success(self):
        if self._unlock_failed:
            return False
        
        if self._screen_unlocked:
            self._success_trigger_count += 1
        else:
            self._success_trigger_count = 0

        return self._success_trigger_count >= self._success_trigger_target

        # if self._screen_geom_id >= 0 and self._mat_screen_unlocked_id >= 0:
        #     return int(self._model.geom_matid[self._screen_geom_id]) == self._mat_screen_unlocked_id
        # return False

    def get_unlock_params(self) -> dict:
        params = dict(self._unlock_params)
        params["action_template"] = self._unlock_action_template.copy()
        params["sequence"] = list(self._unlock_sequence)
        params["other_arm_home"] = getattr(
            self, "_other_arm_home", np.zeros_like(self._unlock_action_template)
        ).copy()
        return params

    def get_unlock_status(self) -> dict:
        screen_mat = (
            int(self._model.geom_matid[self._screen_geom_id])
            if self._screen_geom_id >= 0
            else -1
        )
        return {
            "unlocked": bool(self._compute_success()),
            "unlock_index": int(self._unlock_index),
            "screen_mat_id": screen_mat,
            "screen_mat_unlocked_id": int(self._mat_screen_unlocked_id),
        }

    def _find_button_geom_id(self, preferred: str = "btn1_cyl") -> tuple[int, str]:
        try:
            gid = self._model.geom(preferred).id
            return gid, preferred
        except Exception:
            pass
        for gid in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and name.startswith("btn") and name.endswith("_cyl"):
                return gid, name
        raise RuntimeError("No button geom found (expected names like btn*_cyl).")

    def _get_button_geom_id_by_digit(self, digit: int) -> int:
        name = f"btn{digit}_cyl"
        try:
            return self._model.geom(name).id
        except Exception as e:
            raise RuntimeError(f"Button geom not found: {name}") from e

    def _get_screen_normal_and_center(self) -> tuple[np.ndarray, np.ndarray]:
        """Return screen normal (world) and screen center using ipad_screen geom."""
        try:
            gid = self._model.geom("ipad_screen").id
            xmat = self._data.geom_xmat[gid].reshape(3, 3)
            normal = xmat[:, 2]
            center = self._data.geom_xpos[gid].copy()
            return normal, center
        except Exception:
            gid = self._model.geom("btn1_cyl").id
            xmat = self._data.geom_xmat[gid].reshape(3, 3)
            normal = xmat[:, 2]
            center = self._data.geom_xpos[gid].copy()
            return normal, center

    def _compute_press_targets(
        self,
        fingertip_id: int,
        button_id: int,
        normal: np.ndarray,
        above_offset: float,
        press_offset: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute above/press targets along a consistent normal direction."""
        tip = self._data.geom_xpos[fingertip_id].copy()
        center = self._data.geom_xpos[button_id].copy()
        n = normal / (np.linalg.norm(normal) + 1e-9)
        if np.dot(tip - center, n) < 0:
            n = -n
        target_above = center + n * float(above_offset)
        target_press = center - n * float(press_offset)
        return target_above, target_press

    # ==========================
    def _add_unlock_progress_overlay(self) -> None:
        viewer = self._viewer._get_viewer(render_mode="human")
        viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            f"num:{int(self._unlock_index)}",
            "",
        )

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            if cam_id >= 0:
                rendered_frames.append(
                    self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
                )
        return rendered_frames

    # Helper methods.
    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_pose_right = np.concatenate([tcp_pos_right, tcp_quat_right])

        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        tcp_pose_left = np.concatenate([tcp_pos_left, tcp_quat_left])

        tcp_pose = np.concatenate([tcp_pose_right, tcp_pose_left])

        allegro_right_qpos = self._data.qpos[self._allegro_dof_right_ids].astype(
            np.float32
        )
        allegro_left_qpos = self._data.qpos[self._allegro_dof_left_ids].astype(
            np.float32
        )
        allegro_qpos = np.concatenate([allegro_right_qpos, allegro_left_qpos])

        if self.image_obs:
            obs["images"] = {}
            rendered_frames = self.render()
            image_keys = ["random_camera" if self.randomize else "ego"]
            if self._ego_left_camera_id >= 0:
                image_keys.append("ego_left")
            if self._ego_right_camera_id >= 0:
                image_keys.append("ego_right")
            if self._wrist_left_camera_id >= 0:
                image_keys.append("wrist_left")
            if self._wrist_right_camera_id >= 0:
                image_keys.append("wrist_right")
            for key, frame in zip(image_keys, rendered_frames):
                obs["images"][key] = frame

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "stand_ori_pose": self._stand_ori_pose,
            "ipad_ori_pose": self._ipad_ori_pose,
            "table_delta_height": self.delta_h,
        }

        return obs

    def get_end_effector_pose_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # ---------- Right ----------
        pos_r = self._data.mocap_pos[self._mocap_right_id]
        quat_r = self._data.mocap_quat[self._mocap_right_id]
        quat_r = np.array([quat_r[1], quat_r[2], quat_r[3], quat_r[0]])
        rot_r = R.from_quat(quat_r).as_matrix()

        T_right = np.eye(4)
        T_right[:3, :3] = rot_r
        T_right[:3, 3] = pos_r

        # ---------- Left ----------

        pos_l = self._data.mocap_pos[self._mocap_left_id]
        quat_l = self._data.mocap_quat[self._mocap_left_id]
        quat_l = np.array([quat_l[1], quat_l[2], quat_l[3], quat_l[0]])
        rot_l = R.from_quat(quat_l).as_matrix()

        T_left = np.eye(4)
        T_left[:3, :3] = rot_l
        T_left[:3, 3] = pos_l

        return T_right, T_left


if __name__ == "__main__":
    # quick manual test
    env = PandaIpadGymEnv(render_mode="human")
    obs, info = env.reset()
    for i in range(200):
        # random actions: 7 franka + 16 allegro = 23
        action = np.random.uniform(-1, 1, 23)
        obs, rew, done, trunc, info = env.step(action)
    env.close()
