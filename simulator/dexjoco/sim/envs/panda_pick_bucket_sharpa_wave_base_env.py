import random
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from ..controllers import opspace
from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from ..rendering import MujocoRenderer
from .arm_profiles import configure_arm
from .base_hand_specs import (
    ALLEGRO_RIGHT_ACTUATOR_NAMES,
    ALLEGRO_RIGHT_HOME,
    ALLEGRO_RIGHT_JOINT_NAMES,
    shadow_hand_e3m5_joint_targets_to_actuator_ctrl,
    get_pick_bucket_hand_spec,
)

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_bucket_pick_sharpa_wave.xml"
_CARTESIAN_BOUNDS = np.asarray([[-0.8, -0.8, -0.8], [0.8, 0.8, 0.8]])
_SAMPLING_BOUNDS = np.asarray(
    [[-0.20, -0.2], [-0.15, -0.25]]
)  # x:[-0.20,-0.15], y:[-0.2,-0.25]
_FOOD_SAMPLING_BOUNDS = np.array([[-0.35, 0.15], [-0.3, 0.20]])
_YAW_PERTURB_BOUNDS = np.array([-10, 10])

class PandaPickBucketGymEnv(MujocoGymEnv):
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
        hz=10,
        xml_path: Path | None = None,
        hand_joint_names: Tuple[str, ...] = ALLEGRO_RIGHT_JOINT_NAMES,
        hand_actuator_names: Tuple[str, ...] = ALLEGRO_RIGHT_ACTUATOR_NAMES,
        hand_home: np.ndarray = ALLEGRO_RIGHT_HOME,
        hand_type: str = "allegro_right",
        hand_command_mode: str = "actuator",
        max_episode_steps: int = 1200,
        arm_nullspace_stiffness: float = 0.5,
        arm_use_home_nullspace_target: bool = True,
        arm_max_ori_acceleration: float | None = None,
        arm_pos_gain_base: float = 400.0,
        arm_ori_gain_base: float = 200.0,
        arm_damping_ratio: float = 4.0,
    ):
        self.hz = int(hz)
        self._action_scale = action_scale
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics
        self._hand_type = str(hand_type).lower()
        self._hand_command_mode = str(hand_command_mode).lower()
        self._arm_nullspace_stiffness = float(arm_nullspace_stiffness)
        self._arm_use_home_nullspace_target = bool(arm_use_home_nullspace_target)
        self._arm_max_ori_acceleration = (
            None if arm_max_ori_acceleration is None else float(arm_max_ori_acceleration)
        )
        self._arm_pos_gain_base = float(arm_pos_gain_base)
        self._arm_ori_gain_base = float(arm_ori_gain_base)
        self._arm_damping_ratio = float(arm_damping_ratio)
        if not np.isfinite(self._arm_pos_gain_base) or self._arm_pos_gain_base <= 0.0:
            raise ValueError("arm_pos_gain_base must be finite and positive.")
        if not np.isfinite(self._arm_ori_gain_base) or self._arm_ori_gain_base <= 0.0:
            raise ValueError("arm_ori_gain_base must be finite and positive.")
        if not np.isfinite(self._arm_damping_ratio) or self._arm_damping_ratio <= 0.0:
            raise ValueError("arm_damping_ratio must be finite and positive.")
        if xml_path is None:
            default_spec = get_pick_bucket_hand_spec(hand_type)
            xml_path = default_spec.xml_path

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
        self._max_episode_steps = int(max_episode_steps)
        self.intervened = True

        self._arm_profile, self._arm_dof_ids, self._arm_ctrl_ids = configure_arm(self._model)
        self._arm_home = self._arm_profile.home_qpos.copy()

        self._site_id = self._model.site("attachment_site").id

        self._allegro_joint_names = list(hand_joint_names)
        self._hand_home = np.asarray(hand_home, dtype=np.float32)
        self._n_hand = int(len(self._allegro_joint_names))
        self._allegro_dof_ids = np.asarray(
            [
                int(np.asarray(self._model.joint(n).qposadr).reshape(-1)[0])
                for n in self._allegro_joint_names
            ],
            dtype=int,
        )
        self._allegro_vel_ids = np.asarray(
            [
                int(np.asarray(self._model.joint(n).dofadr).reshape(-1)[0])
                for n in self._allegro_joint_names
            ],
            dtype=int,
        )
        allegro_ids = []
        for name in hand_actuator_names:
            try:
                aid = self._model.actuator(name).id
            except Exception:
                try:
                    aid = mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                    )
                except Exception:
                    aid = -1
            allegro_ids.append(aid)
        self._allegro_ctrl_ids = np.asarray(allegro_ids, dtype=int)
        self._last_hand_follow_debug_t = 0.0

        # Observation space
        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "gripper_pose": spaces.Box(
                    -np.inf, np.inf, shape=(self._n_hand,), dtype=np.float32
                ),
                "bucket_ori_pose": spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype=np.float64
                ),
                "boxed_food_ori_pose": spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype=np.float64
                ),
                "table_delta_height": spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

        observation_space_dict = {"state": state_space}
        if self.image_obs:
            image_h = int(self._model.vis.global_.offheight)
            image_w = int(self._model.vis.global_.offwidth)
            observation_space_dict["images"] = spaces.Dict(
                {
                    "wrist": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "random_camera" if self.randomize else "front": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "ego_left": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                    "ego_right": spaces.Box(
                        0, 255, shape=(image_h, image_w, 3), dtype=np.uint8
                    ),
                }
            )

        self.observation_space = spaces.Dict(observation_space_dict)

        # Action space
        self.action_space = spaces.Box(
            low=np.full(7 + self._n_hand, -1.0, dtype=np.float32),
            high=np.full(7 + self._n_hand, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        self._front_camera_id = self._get_cam_id_by_name("front")
        self._ego_left_camera_id = self._get_cam_id_by_name("left")
        self._ego_right_camera_id = self._get_cam_id_by_name("right")
        self._wrist_camera_id = self._get_cam_id_by_name("handcam_rgb")

        missing = []
        if self._front_camera_id < 0:
            missing.append("front")
        if self._wrist_camera_id < 0:
            missing.append("handcam_rgb")
        if len(missing) > 0:
            raise RuntimeError(
                f"Required camera(s) not found in MuJoCo model: {missing}. "
                "Please ensure these cameras exist in your XML (names: 'front', 'handcam_rgb')."
            )
        self.camera_id = (
            self._front_camera_id,
            self._ego_left_camera_id,
            self._ego_right_camera_id,
            self._wrist_camera_id,
        )

        self._camera_params = np.load(_HERE / "replay_cameras.npy")
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

        self._table_body_z0 = float(self._model.body("table").pos[2])
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

        self._bucket_body_z0 = float(self._model.body("bucket").pos[2])
        self._boxed_food_body_z0 = float(self._model.body("boxed_food_0").pos[2])
        self._bucket_z = self._bucket_body_z0
        self.delta_h = 0.0
        self._bucket_bottom_site_ids = [
            self.model.site(f"bucket_ref_{i}").id for i in (0, 2, 4, 6)
        ]
        self._bucket_bottom_z0 = None

        # for dynamics randomization
        self._bucket_body_id = self._model.body("bucket").id
        self._boxed_food_body_id = self._model.body("boxed_food_0").id
        self._bucket_mass0 = float(self._model.body_mass[self._bucket_body_id])
        self._boxed_food_mass0 = float(self._model.body_mass[self._boxed_food_body_id])
        self._mass_mul = (0.75, 1.25)
        self._bucket_joint_id = self._model.joint("bucket_joint_0").id
        self._bucket_dof_id = int(self._model.jnt_dofadr[self._bucket_joint_id])
        self._bucket_friction_mul = (0.75, 1.25)

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
        self._viewer.render(render_mode="rgb_array", camera_id=self._wrist_camera_id)
        self._viewer.render(render_mode="rgb_array", camera_id=self._front_camera_id)
        if self._ego_left_camera_id >= 0:
            self._viewer.render(
                render_mode="rgb_array", camera_id=self._ego_left_camera_id
            )
        if self._ego_right_camera_id >= 0:
            self._viewer.render(
                render_mode="rgb_array", camera_id=self._ego_right_camera_id
            )

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

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # reset table height
        self.delta_h = np.float64(np.random.uniform(0.0, 0.05))

        # Move the whole table body (absolute)
        self._model.body_pos[self._table_body_id, 2] = (
            self._table_body_z0 + self.delta_h
        )

        # Adjust legs (absolute): extend so feet stay on floor
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = (
                self._table_leg_half_len0[gid] + self.delta_h
            )

        # Reset arm to home position.
        self._data.qpos[self._arm_dof_ids] = self._arm_home
        if self._hand_type != "shadow_hand_e3m5_right":
            self._data.qpos[self._allegro_dof_ids] = self._hand_home

        mujoco.mj_forward(self._model, self._data)
        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        self._data.mocap_pos[0] = tcp_pos
        self._data.mocap_quat[0] = tcp_quat

        # Sample a new bucket position.
        bucket_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._bucket_z = self._bucket_body_z0 + self.delta_h

        # --- Apply ±20° yaw perturbation to bucket ---
        bucket_orig = np.array(
            self._data.jnt("bucket_root").qpos[3:7], dtype=np.float64
        )  # (w,x,y,z)
        bucket_yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        bqw, bqz = np.cos(bucket_yaw / 2), np.sin(bucket_yaw / 2)
        bw1, bx1, by1, bz1 = bqw, 0, 0, bqz
        bw2, bx2, by2, bz2 = bucket_orig
        bucket_q_new = np.array(
            [
                bw1 * bw2 - bx1 * bx2 - by1 * by2 - bz1 * bz2,
                bw1 * bx2 + bx1 * bw2 + by1 * bz2 - bz1 * by2,
                bw1 * by2 - bx1 * bz2 + by1 * bw2 + bz1 * bx2,
                bw1 * bz2 + bx1 * by2 - by1 * bx2 + bz1 * bw2,
            ]
        )
        bucket_q_new /= np.linalg.norm(bucket_q_new)
        self.bucket_ori_pose = np.concatenate(
            [bucket_xy, [self._bucket_z], bucket_q_new]
        ).astype(np.float64)
        self._data.jnt("bucket_root").qpos = self.bucket_ori_pose

        # Sample a new boxfood position.
        box_food_xy = np.random.uniform(*_FOOD_SAMPLING_BOUNDS)

        # --- Apply ±20° yaw perturbation to original quaternion ---
        orig = np.array(
            self._data.jnt("boxed_food_0_freejoint").qpos[3:7], dtype=np.float64
        )  # (w,x,y,z)
        yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
        # quaternion multiplication q_new = q_delta * orig
        w1, x1, y1, z1 = qw, 0, 0, qz
        w2, x2, y2, z2 = orig
        q_new = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        q_new /= np.linalg.norm(q_new)

        self.box_food_ori_pose = np.concatenate(
            [box_food_xy, [self._boxed_food_body_z0 + self.delta_h], q_new]
        ).astype(np.float64)  # no yaw perturb for food
        self._data.jnt("boxed_food_0_freejoint").qpos = self.box_food_ori_pose

        if self.randomize:
            self.randomize_lighting()
            self.randomize_camera()
            self.randomize_desktop_texture()

        if self._randomize_dynamics:
            frictionloss = float(
                self._model.dof_frictionloss[self._bucket_dof_id]
            ) * float(
                np.random.uniform(
                    self._bucket_friction_mul[0], self._bucket_friction_mul[1]
                )
            )

            self._model.dof_frictionloss[self._bucket_dof_id] = frictionloss

            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._bucket_body_id] = self._bucket_mass0 * mass_mul
            mass_mul = float(np.random.uniform(self._mass_mul[0], self._mass_mul[1]))
            self._model.body_mass[self._boxed_food_body_id] = (
                self._boxed_food_mass0 * mass_mul
            )

        # print(
        #     "bucket handle: frictionloss="
        #     f"{self._model.dof_frictionloss[self._bucket_dof_id]}, "
        #     "stiffness="
        #     f"{self._model.jnt_stiffness[self._bucket_joint_id]}"
        # )
        # print(
        #     "bucket mass="
        #     f"{self._model.body_mass[self._bucket_body_id]}, "
        #     "boxed_food mass="
        #     f"{self._model.body_mass[self._boxed_food_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._bucket_bottom_z0 = self._data.site_xpos[
            self._bucket_bottom_site_ids, 2
        ].copy()
        self._prime_rgb_array_renderer()

        obs = self._compute_observation()
        return obs, {"succeed": False}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()

        if action is None or action.shape[0] < 7:
            raise ValueError("Action must have at least 7 elements (franka delta).")

        x, y, z, w, qx, qy, qz = (
            action[0],
            action[1],
            action[2],
            action[3],
            action[4],
            action[5],
            action[6],
        )

        if action.shape[0] >= 7 + self._n_hand:
            allegro_angles = np.asarray(action[7 : 7 + self._n_hand], dtype=np.float64)
        else:
            allegro_angles = np.zeros(self._n_hand, dtype=np.float64)

        joint_ids = [self._model.joint(n).id for n in self._allegro_joint_names]
        allegro_angles = np.clip(
            allegro_angles,
            self._model.jnt_range[joint_ids, 0],
            self._model.jnt_range[joint_ids, 1],
        )

        pos = self._data.mocap_pos[0].copy()
        quat = self._data.mocap_quat[0].copy()

        tpos = np.asarray([x, y, z])
        tquat = np.array([w, qx, qy, qz])

        if np.allclose(tpos, 0.0) and np.allclose(tquat, 0.0):
            self._data.mocap_pos[0] = pos
            self._data.mocap_quat[0] = quat
        else:
            self._data.mocap_pos[0] = tpos
            self._data.mocap_quat[0] = tquat

        for _ in range(self._n_substeps):
            joint_target = self._arm_home if self._arm_use_home_nullspace_target else None
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._site_id,
                dof_ids=self._arm_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=joint_target,
                gravity_comp=True,
                pos_gains=(
                    self._arm_pos_gain_base,
                    self._arm_pos_gain_base,
                    self._arm_pos_gain_base,
                ),
                ori_gains=(
                    self._arm_ori_gain_base,
                    self._arm_ori_gain_base,
                    self._arm_ori_gain_base,
                ),
                damping_ratio=self._arm_damping_ratio,
                nullspace_stiffness=self._arm_nullspace_stiffness,
                max_ori_acceleration=self._arm_max_ori_acceleration,
            )
            self._data.ctrl[self._arm_ctrl_ids] = tau

            try:
                if self._hand_command_mode == "qpos" and self._hand_type != "shadow_hand_e3m5_right":
                    self._data.qpos[self._allegro_dof_ids] = allegro_angles
                    self._data.qvel[self._allegro_vel_ids] = 0.0
                    ctrl_ids = self._allegro_ctrl_ids
                    valid_mask = ctrl_ids >= 0
                    if np.any(valid_mask):
                        self._data.ctrl[ctrl_ids[valid_mask].astype(int)] = allegro_angles[
                            valid_mask
                        ]
                else:
                    ctrl_ids = self._allegro_ctrl_ids
                    valid_mask = ctrl_ids >= 0

                    if np.any(valid_mask):
                        if self._hand_type == "shadow_hand_e3m5_right":
                            target_ctrl = shadow_hand_e3m5_joint_targets_to_actuator_ctrl(
                                allegro_angles,
                                ctrl_ids,
                                self._model.actuator_ctrlrange,
                                tuple(self._allegro_joint_names),
                            )
                        else:
                            target_ctrl = allegro_angles
                        self._data.ctrl[ctrl_ids[valid_mask].astype(int)] = target_ctrl[
                            valid_mask
                        ]

            except Exception:
                pass
                # print("[Warning] failed to write Allegro ctrl:", e)

            mujoco.mj_step(self._model, self._data)

        if "leap" in self._hand_type and self._n_hand >= 16:
            now = time.monotonic()
            if now - self._last_hand_follow_debug_t >= 2.0:
                actual = self._data.qpos[self._allegro_dof_ids]
                print(
                    "[pick_bucket_env] "
                    f"thumb_target={np.round(allegro_angles[12:16], 3)} "
                    f"thumb_actual={np.round(actual[12:16], 3)}"
                )
                self._last_hand_follow_debug_t = now

        # cone alpha logic (same as before)

        obs = self._compute_observation()

        self.env_step += 1
        terminated = False
        if self.env_step >= self._max_episode_steps:
            terminated = True

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)
        dt = time.time() - start_time
        # if self.intervened:
        time.sleep(max(0, (1.0 / self.hz) - dt))
        success = self._compute_success()
        rew = 1.0 if success else 0.0
        terminated = terminated or success

        return obs, rew, terminated, False, {"succeed": success, "grasp_penalty": 0.0}

    def _compute_success(self):
        box_pos = np.array(self._data.sensor("boxed_food_0_pos").data, dtype=float)
        site_ids = [self.model.site(f"bucket_ref_{i}").id for i in range(8)]
        corners = np.array(self._data.site_xpos[site_ids], dtype=float)  # (8,3)
        mins = corners.min(axis=0)
        maxs = corners.max(axis=0)
        inside = np.all(box_pos >= mins) and np.all(box_pos <= maxs)
        bottom_z = self._data.site_xpos[self._bucket_bottom_site_ids, 2]
        lifted = np.all(bottom_z - self._bucket_bottom_z0 >= 0.15)
        success = inside and lifted

        return success

    # ==========================
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    # Helper methods.
    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("franka/flange_pos").data
        tcp_quat = self._data.sensor("franka/flange_quat").data
        tcp_pose = np.concatenate([tcp_pos, tcp_quat])
        allegro_qpos = self._data.qpos[self._allegro_dof_ids].astype(np.float32)

        obs["state"] = {
            "tcp_pose": tcp_pose,
            "gripper_pose": allegro_qpos,
            "bucket_ori_pose": self.bucket_ori_pose,
            "boxed_food_ori_pose": self.box_food_ori_pose,
            "table_delta_height": np.asarray([self.delta_h], dtype=np.float64),
        }

        # if self.image_obs:
        obs["images"] = {}
        (
            obs["images"]["random_camera" if self.randomize else "front"],
            obs["images"]["ego_left"],
            obs["images"]["ego_right"],
            obs["images"]["wrist"],
        ) = self.render()

        return obs

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        pos = self._data.mocap_pos[0]
        quat = self._data.mocap_quat[0]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot_mat = R.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos
        return T
