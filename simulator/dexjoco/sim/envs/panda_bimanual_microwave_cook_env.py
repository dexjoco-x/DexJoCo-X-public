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
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_microwave_cook.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
_ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32)
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[-0.35, -0.3], [-0.25, -0.4]])
_HOT_DOG_YAW_BOUNDS = (-20.0, 20.0)


class PandaBimanualMicrowaveCookGymEnv(MujocoGymEnv):
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
        max_episode_steps: int = 1100,
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
        arm_pos_gains: tuple[float, float, float] = (400.0, 400.0, 400.0),
    ):
        self.hz = int(hz)
        self._action_scale = action_scale
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics
        self._max_episode_steps = int(max_episode_steps)
        self._hand_command_mode = str(hand_command_mode).lower()
        self._right_hand_home = np.asarray(right_hand_home, dtype=np.float64)
        self._left_hand_home = np.asarray(left_hand_home, dtype=np.float64)
        self._arm_home_right = np.asarray(right_arm_home, dtype=np.float64)
        self._arm_home_left = np.asarray(left_arm_home, dtype=np.float64)
        self._arm_nullspace_stiffness = float(arm_nullspace_stiffness)
        self._arm_pos_gains = tuple(float(value) for value in arm_pos_gains)
        if len(self._arm_pos_gains) != 3:
            raise ValueError("arm_pos_gains must contain exactly three values")
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

        self._panda_dof_ids = np.concatenate([self._panda_right_dof_ids, self._panda_left_dof_ids])
        self._panda_ctrl_ids = np.concatenate([self._panda_right_ctrl_ids, self._panda_left_ctrl_ids])

        allegro_actuator_right_names = list(right_hand_actuator_names)
        allegro_actuator_left_names = list(left_hand_actuator_names)
        # 变量名沿用 allegro 是为了少改原始逻辑；这里表示“当前手型”。
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
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_right_names],
            dtype=int
        )
        self._allegro_dof_left_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_left_names],
            dtype=int
        )
        self._allegro_vel_right_ids = np.asarray(
            [int(self._model.joint(n).dofadr) for n in self._allegro_joint_right_names],
            dtype=int,
        )
        self._allegro_vel_left_ids = np.asarray(
            [int(self._model.joint(n).dofadr) for n in self._allegro_joint_left_names],
            dtype=int,
        )

        self._mocap_right_id = int(self._model.body("target_right").mocapid)
        self._mocap_left_id = int(self._model.body("target_left").mocapid)

        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,)),
                "gripper_pose": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self._n_hand_right + self._n_hand_left,),
                ),
                "hot_dog_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "microwave_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )
        observation_space_dict = {"state": state_space}

        # Action space
        self.action_space = spaces.Box(
            low=np.full(7 + self._n_hand_right, -1.0, dtype=np.float32),
            high=np.full(7 + self._n_hand_right, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        self._front_camera_id = int(self._model.camera("back").id)
        self._ego_left_camera_id = self._get_cam_id_by_name("left")
        self._ego_right_camera_id = self._get_cam_id_by_name("right")
        self._wrist_left_camera_id = self._get_cam_id_by_name("handcam_rgb_left")
        self._wrist_right_camera_id = self._get_cam_id_by_name("handcam_rgb_right")
        # print("ego_id:", self._front_camera_id)

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

        if self.image_obs:
            image_h = int(self._model.vis.global_.offheight)
            image_w = int(self._model.vis.global_.offwidth)

            observation_space_dict["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "wrist_right": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "ego_left": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "ego_right": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                }
            )

        self.observation_space = spaces.Dict(observation_space_dict)

        self._table_z = self._model.body("table").pos[2].copy()


        self._model_geom_pos0 = self._model.geom_pos.copy()
        self._model_geom_size0 = self._model.geom_size.copy()
        self._model_site_pos0 = self._model.site_pos.copy()
        self._hot_dog_body_z0 = self._model.body("hot_dog").pos[2].copy()
        self._microwave_body_z0 = self._model.body("microwave_object").pos[2].copy()

        #for dynamics randomization
        self._micro_joint_id = self._model.joint("microjoint").id
        self._micro_dof_id = int(self._model.jnt_dofadr[self._micro_joint_id])
        self._micro_friction0 = float(self._model.dof_frictionloss[self._micro_dof_id])
        self._micro_friction_mul = (0.75, 1.25)

        self._microwave_body_id = self._model.body("microwave_object").id
        self._hot_dog_body_id = self._model.body("hot_dog").id
        self._microwave_mass0 = float(self._model.body_mass[self._microwave_body_id])
        self._hot_dog_mass0 = float(self._model.body_mass[self._hot_dog_body_id])
        self._mass_mul = (0.75, 1.25)



    def _get_cam_id_by_name(self, name: str) -> int:
        """Return camera id from name; return -1 if not found."""
        try:
            cam = self._model.camera(name)
            return int(cam.id)
        except Exception:
            try:
                return int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name))
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
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_left_camera_id)
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_right_camera_id)
        self._viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)
        if self._ego_left_camera_id >= 0:
            self._viewer.render(render_mode="rgb_array", camera_id=self._ego_left_camera_id)
        if self._ego_right_camera_id >= 0:
            self._viewer.render(render_mode="rgb_array", camera_id=self._ego_right_camera_id)

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

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # reset table height
        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        table_ori_pos = self._model.body("table").pos
        table_ori_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_ori_pos

        leg_names = ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")
        dh = self.delta_h

        for lname in leg_names:
            lgid = self._model.geom(lname).id

            base_center_z = float(self._model_geom_pos0[lgid, 2])   # c0
            base_half_len = float(self._model_geom_size0[lgid, 1])   # h0 (cylinder: [radius, half_length])

            new_center_z = base_center_z - 0.5 * dh
            new_half_len = base_half_len + 0.5 * dh

            self._model.geom_pos[lgid, 2] = new_center_z
            self._model.geom_size[lgid, 1] = new_half_len

        # Sample a new hot_dog pose.
        hot_dog_base_quat = self._model.body("hot_dog").quat
        hot_dog_base_quat = np.array([hot_dog_base_quat[1], hot_dog_base_quat[2], hot_dog_base_quat[3], hot_dog_base_quat[0]])
        r_base = R.from_quat(hot_dog_base_quat)
        yaw_deg = np.random.uniform(*_HOT_DOG_YAW_BOUNDS)
        r_yaw = R.from_euler('z', yaw_deg, degrees=True)  # rotation representing yaw offset
        r_new = r_yaw * r_base
        hot_dog_ori_quat = r_new.as_quat()
        hot_dog_ori_quat = np.array([hot_dog_ori_quat[3], hot_dog_ori_quat[0], hot_dog_ori_quat[1], hot_dog_ori_quat[2]])
        hot_dog_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        hot_dog_z = self._hot_dog_body_z0 + self.delta_h
        hot_dog_ori_pos = (*hot_dog_xy, hot_dog_z)

        self._hot_dog_ori_pose = np.concatenate([hot_dog_ori_pos, hot_dog_ori_quat]).astype(np.float64)
        self._data.jnt("hot_dog_free").qpos = self._hot_dog_ori_pose

        microwave_ori_pos = self._model.body("microwave_object").pos
        microwave_ori_quat = self._model.body("microwave_object").quat
        microwave_ori_pos[2] = self._microwave_body_z0 + self.delta_h
        self._microwave_ori_pose = np.concatenate([microwave_ori_pos, microwave_ori_quat]).astype(np.float64)
        self._model.body("microwave_object").pos = self._microwave_ori_pose[:3]
        self._model.body("microwave_object").quat = self._microwave_ori_pose[3:]

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
        # print("Reset TCP right pos:", tcp_pos_right)
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        # print("Reset TCP left pos:", tcp_pos_left)
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        self._data.mocap_pos[self._mocap_right_id] = tcp_pos_right
        self._data.mocap_pos[self._mocap_left_id] = tcp_pos_left
        self._data.mocap_quat[self._mocap_right_id] = tcp_quat_right
        self._data.mocap_quat[self._mocap_left_id] = tcp_quat_left

        mujoco.mj_forward(self._model, self._data)

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            frictionloss = self._micro_friction0 * float(
                np.random.uniform(self._micro_friction_mul[0], self._micro_friction_mul[1])
            )
            self._model.dof_frictionloss[self._micro_dof_id] = frictionloss

            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._microwave_body_id] = self._microwave_mass0 * mass_mul
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._hot_dog_body_id] = self._hot_dog_mass0 * mass_mul

        # print(
        #     "microwave door: frictionloss="
        #     f"{self._model.dof_frictionloss[self._micro_dof_id]}"
        # )
        # print(
        #     "microwave mass="
        #     f"{self._model.body_mass[self._microwave_body_id]}, "
        #     "hot_dog mass="
        #     f"{self._model.body_mass[self._hot_dog_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()

        obs = self._compute_observation()
        return obs, {"succeed": False}

    def _geom_in_contact(self, target_geom_name: str, dist_threshold: float = 0.0) -> bool:
        """
        Return True if `target_geom_name` is in contact this step (dist <= dist_threshold).
        Must be called after a physics step (mujoco.mj_step / sim.step).
        """
        ncon = int(self.data.ncon)
        # debug
        # print("DEBUG: ncon =", ncon)

        for i in range(ncon):
            c = self.data.contact[i]

            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1))
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))

            # robust name lookup
            # try:
            #     g1 = self.model.geom_id2name(int(c.geom1))
            #     g2 = self.model.geom_id2name(int(c.geom2))
            # except Exception:
            #     # fallback for some bindings that return tuples
            #     if hasattr(c, "__len__") and len(c) > 0:
            #         g1 = self.model.geom_id2name(int(c[0].geom1))
            #         g2 = self.model.geom_id2name(int(c[0].geom2))
            #     else:
            #         g1 = g2 = None

            # debug prints (uncomment for interactive debugging)
            # print(f"Contact {i}: {g1} - {g2}, dist={getattr(c,'dist', None)}")

            dist = float(getattr(c, "dist", 0.0))
            if (g1 == target_geom_name or g2 == target_geom_name) and (dist <= dist_threshold):
                return True
        return False

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

        start_time = time.time()

        is_zero_action = False
        if action is None:
            is_zero_action = True
        else:
            try:
                if np.isscalar(action) and action == 0:
                    is_zero_action = True
                elif isinstance(action, np.ndarray) and action.size == 1 and action.item() == 0:
                    is_zero_action = True
            except Exception:
                is_zero_action = False

        # ----- parse action when not zero -----
        if not is_zero_action:
            # action expected to be dict: {"right": arr23, "left": arr23}
            right = np.asarray(action["right"])
            left = np.asarray(action["left"])
            # Parse franka delta
            x_r, y_r, z_r, w_r, qx_r, qy_r, qz_r = right[0:7]
            x_l, y_l, z_l, w_l, qx_l, qy_l, qz_l = left[0:7]

            # Parse current dexterous hand qpos targets.
            allegro_r = np.asarray(right[7:7 + self._n_hand_right], dtype=np.float64)
            allegro_l = np.asarray(left[7:7 + self._n_hand_left], dtype=np.float64)
            allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
            # print("Parsed allegro angles:", allegro_angles)
        else:
            # zero action -> do not change mocap nor allegro
            x_r = y_r = z_r = w_r = qx_r = qy_r = qz_r = 0.0
            x_l = y_l = z_l = w_l = qx_l = qy_l = qz_l = 0.0
            allegro_angles = None

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
        if not is_zero_action and not (np.allclose(tpos_r, 0.0) and np.allclose(tquat_r, 0.0)):
            self._data.mocap_pos[self._mocap_right_id] = tpos_r
            self._data.mocap_quat[self._mocap_right_id] = tquat_r
        else:
            # keep original right pose
            self._data.mocap_pos[self._mocap_right_id] = r_pos
            self._data.mocap_quat[self._mocap_right_id] = r_quat

        # ----- apply mocap for left  -----
        if not is_zero_action and not (np.allclose(tpos_l, 0.0) and np.allclose(tquat_l, 0.0)):
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
                pos_gains=self._arm_pos_gains,
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
                pos_gains=self._arm_pos_gains,
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

        obs = self._compute_observation()

        self.env_step += 1
        terminated = False
        if self.env_step >= self._max_episode_steps:
            terminated = True

        # ---- human rendering ----
        if self.render_mode == "human":
            try:
                self._viewer.render("human")
            except Exception:
                pass
                # print("[Warning] human render failed:", e)

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        success = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = terminated or success

        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": 0.0}

    def _compute_success(self):
        # ---- door must be closed ----
        micro_qpos = float(self._data.joint("microjoint").qpos)
        microwave_closed = abs(micro_qpos) < 1e-2

        # ---- hot_dog must be inside interior bbox ----
        inside = self.hot_dog_inside_microwave(margin=0.01)

        contact = self._geom_in_contact("start_button")

        # Debug helpers kept nearby for local inspection when needed.
        # print("microjoint qpos:", micro_qpos)
        # print("hot_dog inside:", inside)
        # print("start button pressed:", contact)

        return inside and microwave_closed and contact

    # ==========================
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            if cam_id >= 0:
                rendered_frames.append(self._viewer.render(render_mode="rgb_array", camera_id=cam_id))
            else:
                rendered_frames.append(None)
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

        # 按 hand spec 的 qpos 地址读取当前灵巧手状态，避免 Allegro 专属 sensor 名假设。
        allegro_right_qpos = self._data.qpos[self._allegro_dof_right_ids].astype(np.float32)
        allegro_left_qpos = self._data.qpos[self._allegro_dof_left_ids].astype(np.float32)
        allegro_qpos = np.concatenate([allegro_right_qpos, allegro_left_qpos])

        if self.image_obs:
            obs["images"] = {}
            (
                obs["images"]["random_camera" if self.randomize else "ego"],
                obs["images"]["ego_left"],
                obs["images"]["ego_right"],
                obs["images"]["wrist_left"],
                obs["images"]["wrist_right"],
            ) = self.render()

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "hot_dog_ori_pose": self._hot_dog_ori_pose,
            "microwave_ori_pose": self._microwave_ori_pose,
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


    def world_to_body(self, body_name, p_world):
        bid = self._model.body(body_name).id
        # body_pos = self._data.body_xpos[bid]               # world
        # body_mat = self._data.body_xmat[bid].reshape(3, 3) # world_R_body

        body_pos = self._data.body("microwave_object").xpos
        body_mat = self._data.body("microwave_object").xmat.reshape(3, 3)
        # p_local = R^T (p_world - body_pos)
        return body_mat.T @ (p_world - body_pos)

    def hot_dog_inside_microwave(self, margin=0.0):
        # ---- hot_dog world position ----
        p_world = self._data.sensor("hot_dog_pos").data

        # ---- transform to microwave local frame ----
        p_local = self.world_to_body("microwave_object", p_world)

        # ---- read interior box sites (local frame) ----
        p0 = self._model.site("int_p0").pos
        px = self._model.site("int_px").pos
        py = self._model.site("int_py").pos
        pz = self._model.site("int_pz").pos

        inside_x = (p0[0] - margin) <= p_local[0] <= (px[0] + margin)
        inside_y = (p0[1] - margin) <= p_local[1] <= (py[1] + margin)
        inside_z = (p0[2] - margin) <= p_local[2] <= (pz[2] + margin)

        return inside_x and inside_y and inside_z

if __name__ == "__main__":
    env = PandaBimanualMicrowaveCookGymEnv(render_mode="human")
    env.reset()
    for i in range(100):
        env.step(np.random.uniform(-1, 1, 4))
        env.render()
    env.close()
