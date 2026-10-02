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
_XML_PATH = _HERE / "xmls" / "arena_arm_hand_bimanual_photograph.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
_ALLEGRO_HOME = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float32)

_TABLE_HEIGHT_BOUNDS = np.asarray([0.0, 0.05], dtype=np.float64)
_LOGO_SAMPLING_BOUNDS_YZ = np.asarray([[-0.10, 1.22], [0.10, 1.38]], dtype=np.float64)
_CAMERA_SAMPLING_BOUNDS = np.asarray([[-0.30, 0.1], [-0.20, 0.2]], dtype=np.float64)
_TARGET_REGION_OFFSET = np.asarray([-0.40, 0.0, 0.0], dtype=np.float64)
_TARGET_REGION_RADIUS = 0.16
_TARGET_REGION_HALF_HEIGHT = 0.10
_TARGET_ANGLE_DEG = 10.0
_WORLD_POS_X = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
_PASSIVE_VIEWER_PATCHED = False
_PASSIVE_MODEL_TARGET_REGION = {}


def _register_passive_target_region(model, target_region_geom_id: int) -> None:
    _PASSIVE_MODEL_TARGET_REGION[id(model)] = int(target_region_geom_id)


def _patch_mujoco_passive_viewer_sync() -> None:
    global _PASSIVE_VIEWER_PATCHED
    if _PASSIVE_VIEWER_PATCHED:
        return
    try:
        viewer_mod = getattr(mujoco, "viewer", None)
        if viewer_mod is None:
            return
        launch_passive_orig = getattr(viewer_mod, "launch_passive", None)
        if launch_passive_orig is None:
            return
        if getattr(launch_passive_orig, "_photo_group5_patch", False):
            _PASSIVE_VIEWER_PATCHED = True
            return

        def launch_passive_patched(*args, **kwargs):
            ctx = launch_passive_orig(*args, **kwargs)
            model = args[0] if len(args) > 0 else kwargs.get("model", None)

            class _CtxWrapper:
                def __enter__(self):
                    v = ctx.__enter__()
                    orig_sync = getattr(v, "sync", None)
                    if callable(orig_sync):
                        def sync_patched(*s_args, **s_kwargs):
                            try:
                                gid = _PASSIVE_MODEL_TARGET_REGION.get(id(model), None)
                                if gid is not None:
                                    cam = getattr(v, "cam", None)
                                    is_free = False
                                    if cam is not None:
                                        try:
                                            is_free = int(cam.type) == int(mujoco.mjtCamera.mjCAMERA_FREE)
                                        except Exception:
                                            is_free = False
                                    model.geom_group[gid] = 0 if is_free else 5
                            except Exception:
                                pass
                            return orig_sync(*s_args, **s_kwargs)
                        v.sync = sync_patched
                    return v

                def __exit__(self, exc_type, exc, tb):
                    return ctx.__exit__(exc_type, exc, tb)

            return _CtxWrapper()

        launch_passive_patched._photo_group5_patch = True
        viewer_mod.launch_passive = launch_passive_patched
        _PASSIVE_VIEWER_PATCHED = True
    except Exception:
        pass


class PandaBimanualPhotographGymEnv(MujocoGymEnv):
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
        hz: int = 30,
        camera_screen_effect: bool = False,
        max_episode_steps: int = 1000,
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
        target_angle_deg: float = _TARGET_ANGLE_DEG,
        shutter_fingertip_geom_names: tuple[str, ...] | None = None,
        shutter_fingertip_body_names: tuple[str, ...] | None = None,
        shutter_touch_contact_geom_names: tuple[str, ...] | None = None,
        camera_object_scale: float = 1.0,
        camera_screen_update_interval: int = 1,
    ):
        self.hz = hz
        self._action_scale = action_scale
        self.randomize = randomize
        self._randomize_dynamics = randomize_dynamics
        self._camera_screen_effect = bool(camera_screen_effect)
        self._camera_screen_update_interval = max(1, int(camera_screen_update_interval))
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
        self._target_angle_deg = float(target_angle_deg)
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
        # 变量名保留 allegro 是为了少改原始逻辑；现在它表示“当前手型”。
        self._allegro_joint_right_names = list(right_hand_joint_names)
        self._allegro_joint_left_names = list(left_hand_joint_names)
        self._allegro_actuator_right_names = list(right_hand_actuator_names)
        self._allegro_actuator_left_names = list(left_hand_actuator_names)

        allegro_ids = []
        for name in allegro_actuator_right_names:
            allegro_ids.append(self._model.actuator(name).id)
        for name in allegro_actuator_left_names:
            allegro_ids.append(self._model.actuator(name).id)
        self._allegro_ctrl_ids = np.asarray(allegro_ids, dtype=int)
        self._allegro_ctrl_ids_right = self._allegro_ctrl_ids[: len(allegro_actuator_right_names)]
        self._allegro_ctrl_ids_left = self._allegro_ctrl_ids[len(allegro_actuator_right_names) :]

        self._allegro_dof_right_ids = np.asarray(
            [int(self._model.joint(n).qposadr) for n in self._allegro_joint_right_names],
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

        self._mocap_right_id = int(self._model.body("target_right").mocapid)
        self._mocap_left_id = int(self._model.body("target_left").mocapid)

        self._table_body_id = self._model.body("table").id
        self._table_z = float(self._model.body("table").pos[2])
        self._table_leg_geom_ids = [
            gid
            for gid in range(self._model.ngeom)
            if self._model.geom_bodyid[gid] == self._table_body_id
            and self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self._model.geom_size[gid, 1]) for gid in self._table_leg_geom_ids
        }

        self._logo_geom_id = int(self._model.geom("photograph_logo_disk").id)
        self._target_region_geom_id = int(self._model.geom("photograph_target_region_visual").id)
        self._camera_axis_z_geom_id = int(self._model.geom("camera_axis_z").id)
        self._view_site_id = int(self._model.site("view_direction_site").id)
        self._camera_body_id = int(self._model.body("link_19").id)
        self._camera_object_scale = float(camera_object_scale)
        if not np.isclose(self._camera_object_scale, 1.0):
            self._scale_camera_object_model(self._camera_body_id, self._camera_object_scale)
        self._shutter_geom_id = int(self._model.geom("camera_body_top_cap").id)
        self._camera_geom_ids = self._collect_object_geom_ids(self._camera_body_id)
        self._shutter_fingertip_geom_ids = self._collect_fingertip_geom_ids(
            geom_names=shutter_fingertip_geom_names,
            body_names=shutter_fingertip_body_names,
        )
        self._shutter_touch_contact_geom_ids = self._collect_geom_ids_by_name(shutter_touch_contact_geom_names)
        self._logo_local_pos0 = self._model.geom_pos[self._logo_geom_id].copy()
        self._logo_local_quat0 = self._model.geom_quat[self._logo_geom_id].copy()
        self._logo_size0 = self._model.geom_size[self._logo_geom_id].copy()
        self._target_region_local_pos0 = self._model.geom_pos[self._target_region_geom_id].copy()
        self._target_region_size0 = self._model.geom_size[self._target_region_geom_id].copy()
        self._camera_body_pos0 = self._model.body_pos[self._camera_body_id].copy()
        self._camera_body_quat0 = self._model.body_quat[self._camera_body_id].copy()
        self._camera_body_z0 = float(self._camera_body_pos0[2])
        self._camera_mass0 = float(self._model.body_mass[self._camera_body_id])
        self._camera_mass_mul = (0.75, 1.25)

        _register_passive_target_region(self._model, self._target_region_geom_id)
        _patch_mujoco_passive_viewer_sync()

        self._group5_geom_ids = []
        try:
            geom_groups = np.asarray(self._model.geom_group)
            self._group5_geom_ids = np.where(geom_groups == 5)[0].tolist()
        except Exception:
            self._group5_geom_ids = []

        self._viewer = MujocoRenderer(self.model, self.data)
        try:
            self._viewer.render(self.render_mode)
        except Exception:
            pass

        self._screen_tex_id = -1
        self._screen_cam_id = -1
        self._screen_tex_w = 0
        self._screen_tex_h = 0
        self._screen_tex_adr = 0
        self._screen_tex_nchan = 3
        self._screen_last_update_step = -1
        self._init_camera_screen_streaming()

        if not self._camera_screen_effect:
            self._hide_camera_screen_geom()

        # Photo capture / flash state.
        self._capture_flash_total = 5  # ~0.17 s of flash at hz=30
        self._captured_image = None
        self._capture_flash_frame = 0
        self._latest_shutter_pressed = False

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
                "camera_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "logo_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )

        observation_space_dict = {"state": state_space}
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

        self.action_space = spaces.Box(
            low=np.full(7 + max(self._n_hand_right, self._n_hand_left), -1.0, dtype=np.float32),
            high=np.full(7 + max(self._n_hand_right, self._n_hand_left), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_cam_id_by_name(self, name: str) -> int:
        try:
            return int(self._model.camera(name).id)
        except Exception:
            try:
                return int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name))
            except Exception:
                return -1

    def _collect_object_geom_ids(self, root_body_id: int) -> set:
        body_ids = self._collect_body_subtree_ids(root_body_id)
        return {gid for gid in range(self._model.ngeom) if int(self._model.geom_bodyid[gid]) in body_ids}

    def _collect_body_subtree_ids(self, root_body_id: int) -> set[int]:
        body_ids = {root_body_id}
        changed = True
        while changed:
            changed = False
            for bid in range(self._model.nbody):
                pid = int(self._model.body_parentid[bid])
                if pid in body_ids and bid not in body_ids:
                    body_ids.add(bid)
                    changed = True
        return body_ids

    def _scale_camera_object_model(self, root_body_id: int, scale: float) -> None:
        if not np.isfinite(scale) or scale <= 0.0:
            return

        body_ids = self._collect_body_subtree_ids(root_body_id)
        geom_ids = [
            gid for gid in range(self._model.ngeom)
            if int(self._model.geom_bodyid[gid]) in body_ids
        ]

        for bid in body_ids:
            if bid != root_body_id:
                self._model.body_pos[bid] *= scale
            self._model.body_mass[bid] *= scale ** 3
            self._model.body_inertia[bid] *= scale ** 5

        scaled_mesh_ids: set[int] = set()
        for gid in geom_ids:
            self._model.geom_pos[gid] *= scale
            self._model.geom_size[gid] *= scale

            if int(self._model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue

            mesh_id = int(self._model.geom_dataid[gid])
            if mesh_id < 0 or mesh_id in scaled_mesh_ids:
                continue
            vert_adr = int(self._model.mesh_vertadr[mesh_id])
            vert_num = int(self._model.mesh_vertnum[mesh_id])
            self._model.mesh_vert[vert_adr:vert_adr + vert_num] *= scale
            scaled_mesh_ids.add(mesh_id)

        for sid in range(self._model.nsite):
            if int(self._model.site_bodyid[sid]) in body_ids:
                self._model.site_pos[sid] *= scale
                self._model.site_size[sid] *= scale

        for cid in range(self._model.ncam):
            if int(self._model.cam_bodyid[cid]) in body_ids:
                self._model.cam_pos[cid] *= scale

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
        geom_names: tuple[str, ...] | None,
        body_names: tuple[str, ...] | None,
    ) -> set[int]:
        geom_ids = self._collect_geom_ids_by_name(geom_names)
        body_ids: set[int] = set()
        for body_name in body_names or ():
            try:
                body_ids.add(int(self._model.body(body_name).id))
            except Exception:
                pass
        if body_ids:
            for gid in range(self._model.ngeom):
                if int(self._model.geom_bodyid[gid]) in body_ids:
                    geom_ids.add(gid)
        return geom_ids

    def _parse_action(self, action):
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

        if is_zero_action:
            return is_zero_action, np.zeros(7), np.zeros(7), None

        right = np.asarray(action["right"], dtype=np.float64)
        left = np.asarray(action["left"], dtype=np.float64)
        pose_r = right[0:7]
        pose_l = left[0:7]
        allegro_r = np.asarray(right[7:7 + self._n_hand_right], dtype=np.float64)
        allegro_l = np.asarray(left[7:7 + self._n_hand_left], dtype=np.float64)
        allegro_angles = np.concatenate([allegro_r, allegro_l], axis=0)
        return is_zero_action, pose_r, pose_l, allegro_angles

    def _set_group_for_geom_ids(self, geom_ids, new_group: int):
        if len(geom_ids) == 0:
            return
        try:
            for gid in geom_ids:
                self._model.geom_group[gid] = new_group
        except Exception:
            try:
                tmp = np.asarray(self._model.geom_group).copy()
                for gid in geom_ids:
                    tmp[gid] = new_group
                self._model.geom_group[:] = tmp
            except Exception:
                pass

    def _temporarily_show_group5_in_gui(self):
        if not self._group5_geom_ids:
            return
        self._set_group_for_geom_ids(self._group5_geom_ids, 0)

    def _restore_group5(self):
        if not self._group5_geom_ids:
            return
        self._set_group_for_geom_ids(self._group5_geom_ids, 5)

    def _is_human_camera_free_pose(self) -> bool:
        free_type = int(mujoco.mjtCamera.mjCAMERA_FREE)

        viewer_candidates = []
        try:
            viewers = getattr(self._viewer, "_viewers", None)
            if isinstance(viewers, dict):
                human_viewer = viewers.get("human", None)
                if human_viewer is not None:
                    viewer_candidates.append(human_viewer)
        except Exception:
            pass

        viewer_candidates.extend(
            [
                getattr(self._viewer, "viewer", None),
                getattr(self._viewer, "_viewer", None),
                self._viewer,
            ]
        )

        for viewer_obj in viewer_candidates:
            if viewer_obj is None:
                continue
            cam = getattr(viewer_obj, "cam", None)
            if cam is None:
                cam = getattr(viewer_obj, "_cam", None)
            if cam is None:
                continue
            cam_type = getattr(cam, "type", None)
            if cam_type is None:
                continue
            try:
                if int(cam_type) == free_type:
                    return True
                fixedcamid = int(getattr(cam, "fixedcamid", -1))
                trackbodyid = int(getattr(cam, "trackbodyid", -1))
                # Free camera usually has no fixed camera id and no tracking body.
                return fixedcamid < 0 and trackbodyid < 0
            except Exception:
                continue

        return False

    def _set_human_geom_group_visibility(self, group_idx: int, visible: bool) -> bool:
        viewer_candidates = []
        try:
            viewers = getattr(self._viewer, "_viewers", None)
            if isinstance(viewers, dict):
                human_viewer = viewers.get("human", None)
                if human_viewer is not None:
                    viewer_candidates.append(human_viewer)
        except Exception:
            pass

        viewer_candidates.extend(
            [
                getattr(self._viewer, "viewer", None),
                getattr(self._viewer, "_viewer", None),
                self._viewer,
            ]
        )

        for viewer_obj in viewer_candidates:
            if viewer_obj is None:
                continue
            for opt_name in ("opt", "vopt", "_opt", "_vopt"):
                opt = getattr(viewer_obj, opt_name, None)
                if opt is None:
                    continue
                geomgroup = getattr(opt, "geomgroup", None)
                if geomgroup is None:
                    continue
                try:
                    geomgroup[group_idx] = 1 if visible else 0
                    return True
                except Exception:
                    continue
        return False

    def _set_offscreen_geom_group_visibility(self, group_idx: int, visible: bool) -> bool:
        try:
            viewers = getattr(self._viewer, "_viewers", None)
            if not isinstance(viewers, dict):
                return False
        except Exception:
            return False

        viewer_candidates = []
        for key, viewer_obj in viewers.items():
            if key == "human":
                continue
            if viewer_obj is not None:
                viewer_candidates.append(viewer_obj)

        for viewer_obj in viewer_candidates:
            for opt_name in ("opt", "vopt", "_opt", "_vopt"):
                opt = getattr(viewer_obj, opt_name, None)
                if opt is None:
                    continue
                geomgroup = getattr(opt, "geomgroup", None)
                if geomgroup is None:
                    continue
                try:
                    geomgroup[group_idx] = 1 if visible else 0
                except Exception:
                    continue

        return len(viewer_candidates) > 0

    def _init_camera_screen_streaming(self):
        """Look up texture and camera ids used to stream cam_pos_z onto the camera screen."""
        try:
            tex_id = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_TEXTURE, "camera_screen_tex")
            )
        except Exception:
            tex_id = -1
        try:
            cam_id = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_pos_z")
            )
        except Exception:
            cam_id = -1
        if tex_id < 0 or cam_id < 0:
            return
        self._screen_tex_id = tex_id
        self._screen_cam_id = cam_id
        self._screen_tex_w = int(self._model.tex_width[tex_id])
        self._screen_tex_h = int(self._model.tex_height[tex_id])
        self._screen_tex_adr = int(self._model.tex_adr[tex_id])
        self._screen_tex_nchan = int(self._model.tex_nchannel[tex_id])

    def _update_camera_screen_texture(self):
        """Drive the camera screen: live preview, capture flash, or frozen photo."""
        if self._screen_tex_id < 0 or not self._camera_screen_effect:
            return

        current_step = int(getattr(self, "env_step", 0))
        if self._screen_last_update_step == current_step:
            return

        capture_active = self._latest_shutter_pressed or (
            self._captured_image is not None
            and self._capture_flash_frame < self._capture_flash_total
        )
        if (
            not capture_active
            and self._camera_screen_update_interval > 1
            and current_step % self._camera_screen_update_interval != 0
        ):
            return

        if self._captured_image is None:
            try:
                self._set_offscreen_geom_group_visibility(5, False)
                img = self._viewer.render(
                    render_mode="rgb_array", camera_id=self._screen_cam_id
                )
            except Exception:
                return
            if img is None or img.ndim != 3 or img.shape[2] < self._screen_tex_nchan:
                return
            img = self._prepare_camera_screen_image(img)

            if self._latest_shutter_pressed:
                # First shutter press: cache this frame, start the flash.
                self._captured_image = img.copy()
                self._capture_flash_frame = 0
                final = self._compose_flash_over(img)
                self._capture_flash_frame += 1
            else:
                final = img
        else:
            if self._capture_flash_frame < self._capture_flash_total:
                final = self._compose_flash_over(self._captured_image)
                self._capture_flash_frame += 1
            else:
                final = self._captured_image

        self._write_screen_texture(final)
        self._screen_last_update_step = current_step

    def _fit_image_to_screen(self, img: np.ndarray) -> np.ndarray:
        """Center-crop to the screen aspect ratio and resample to texture size."""
        th, tw = self._screen_tex_h, self._screen_tex_w
        h_in, w_in = img.shape[:2]

        tex_aspect = tw / max(th, 1)
        img_aspect = w_in / max(h_in, 1)
        if abs(img_aspect - tex_aspect) > 1e-3:
            if img_aspect > tex_aspect:
                new_w = max(1, min(int(round(h_in * tex_aspect)), w_in))
                x0 = (w_in - new_w) // 2
                img = img[:, x0:x0 + new_w, :]
            else:
                new_h = max(1, min(int(round(w_in / tex_aspect)), h_in))
                y0 = (h_in - new_h) // 2
                img = img[y0:y0 + new_h, :, :]
            h_in, w_in = img.shape[:2]

        if h_in != th or w_in != tw:
            ys = np.linspace(0, h_in - 1, th).astype(np.int64)
            xs = np.linspace(0, w_in - 1, tw).astype(np.int64)
            img = img[np.ix_(ys, xs)]
        return img

    def _prepare_camera_screen_image(self, img: np.ndarray) -> np.ndarray:
        """Fit the rendered view and compensate for the screen geom's mirrored UVs."""
        img = self._fit_image_to_screen(img)
        return np.ascontiguousarray(img[:, ::-1, :])

    def _compose_flash_over(self, base_img: np.ndarray) -> np.ndarray:
        """Blend a white sheet over the captured image, alpha fading 1 -> 0."""
        total = max(self._capture_flash_total, 1)
        progress = min(1.0, self._capture_flash_frame / total)
        alpha = max(0.0, 1.0 - progress)
        if alpha <= 0.0:
            return base_img
        blended = alpha * 255.0 + (1.0 - alpha) * base_img.astype(np.float32)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _write_screen_texture(self, img: np.ndarray):
        nchan = self._screen_tex_nchan
        flat = np.ascontiguousarray(img[:, :, :nchan], dtype=np.uint8).reshape(-1)
        adr = self._screen_tex_adr
        self._model.tex_data[adr:adr + flat.size] = flat
        self._upload_screen_texture_to_all_viewers()

    def _hide_camera_screen_geom(self):
        """Make the camera_live_screen geom visually disappear (used when the
        camera-screen effect is disabled)."""
        try:
            gid = int(self._model.geom("camera_live_screen").id)
        except Exception:
            return
        try:
            self._model.geom_matid[gid] = -1
            self._model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._model.geom_size[gid] = np.array([1e-9, 1e-9, 1e-9], dtype=np.float64)
        except Exception:
            pass

    def _upload_screen_texture_to_all_viewers(self):
        if self._screen_tex_id < 0:
            return
        viewers = getattr(self._viewer, "_viewers", None)
        if not isinstance(viewers, dict):
            return
        for v in viewers.values():
            if v is None:
                continue
            con = getattr(v, "con", None)
            if con is None:
                continue
            try:
                mkc = getattr(v, "make_context_current", None)
                if callable(mkc):
                    mkc()
                mujoco.mjr_uploadTexture(self._model, con, self._screen_tex_id)
            except Exception:
                pass

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
        self._set_offscreen_geom_group_visibility(5, False)
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
        mujoco.mj_resetData(self._model, self._data)

        # Clear any prior capture so the screen returns to live preview.
        self._captured_image = None
        self._capture_flash_frame = 0
        self._latest_shutter_pressed = False
        self._screen_last_update_step = -1

        self.delta_h = np.float64(np.random.uniform(*_TABLE_HEIGHT_BOUNDS))
        table_pos = self._model.body("table").pos
        table_pos[2] = self.delta_h + self._table_z
        self._model.body("table").pos = table_pos
        for gid in self._table_leg_geom_ids:
            self._model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + self.delta_h

        logo_yz = np.random.uniform(_LOGO_SAMPLING_BOUNDS_YZ[0], _LOGO_SAMPLING_BOUNDS_YZ[1])
        logo_local_pos = self._logo_local_pos0.copy()
        logo_local_pos[1:] = logo_yz
        logpo_local_quat = self._logo_local_quat0.copy()
        self.logo_ori_pose = np.concatenate([logo_local_pos, logpo_local_quat]).astype(np.float64)
        self._model.geom_pos[self._logo_geom_id] = self.logo_ori_pose[0:3]
        self._model.geom_quat[self._logo_geom_id] = self.logo_ori_pose[3:7]
        self._model.geom_size[self._logo_geom_id] = self._logo_size0

        target_region_local_pos = logo_local_pos + _TARGET_REGION_OFFSET
        self._model.geom_pos[self._target_region_geom_id] = target_region_local_pos
        self._model.geom_size[self._target_region_geom_id, 0] = _TARGET_REGION_RADIUS
        self._model.geom_size[self._target_region_geom_id, 1] = _TARGET_REGION_HALF_HEIGHT
        self._model.geom_size[self._target_region_geom_id, 2] = self._target_region_size0[2]

        camera_xy = np.random.uniform(_CAMERA_SAMPLING_BOUNDS[0], _CAMERA_SAMPLING_BOUNDS[1])
        camera_body_pos = self._camera_body_pos0.copy()
        camera_body_pos[:2] = camera_xy
        camera_body_pos[2] = self._camera_body_z0 + self.delta_h + 0.002
        self.camera_ori_pose = np.concatenate([camera_body_pos, self._camera_body_quat0]).astype(np.float64)
        self._data.jnt("camera_root").qpos = self.camera_ori_pose


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
            mass_mul = float(np.random.uniform(self._camera_mass_mul[0], self._camera_mass_mul[1]))
            self._model.body_mass[self._camera_body_id] = self._camera_mass0 * mass_mul

        # print(
        #     "camera mass="
        #     f"{self._model.body_mass[self._camera_body_id]}"
        # )

        mujoco.mj_forward(self._model, self._data)

        self.env_step = 0
        self._prime_rgb_array_renderer()
        obs = self._compute_observation()
        return obs, {
            "succeed": False,
            "region_pass": False,
            "angle_pass": False,
            "shutter_pressed": False,
            "progress_num": 0,
            "view_angle_deg": float(180.0),
        }

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        is_zero_action, pose_r, pose_l, allegro_angles = self._parse_action(action)

        r_pos = self._data.mocap_pos[self._mocap_right_id].copy()
        l_pos = self._data.mocap_pos[self._mocap_left_id].copy()
        r_quat = self._data.mocap_quat[self._mocap_right_id].copy()
        l_quat = self._data.mocap_quat[self._mocap_left_id].copy()

        tpos_r = np.asarray(pose_r[0:3], dtype=np.float64)
        tquat_r = np.asarray([pose_r[3], pose_r[4], pose_r[5], pose_r[6]], dtype=np.float64)
        tpos_l = np.asarray(pose_l[0:3], dtype=np.float64)
        tquat_l = np.asarray([pose_l[3], pose_l[4], pose_l[5], pose_l[6]], dtype=np.float64)

        if not is_zero_action and not (np.allclose(tpos_r, 0.0) and np.allclose(tquat_r, 0.0)):
            self._data.mocap_pos[self._mocap_right_id] = tpos_r
            self._data.mocap_quat[self._mocap_right_id] = tquat_r
        else:
            self._data.mocap_pos[self._mocap_right_id] = r_pos
            self._data.mocap_quat[self._mocap_right_id] = r_quat

        if not is_zero_action and not (np.allclose(tpos_l, 0.0) and np.allclose(tquat_l, 0.0)):
            self._data.mocap_pos[self._mocap_left_id] = tpos_l
            self._data.mocap_quat[self._mocap_left_id] = tquat_l
        else:
            self._data.mocap_pos[self._mocap_left_id] = l_pos
            self._data.mocap_quat[self._mocap_left_id] = l_quat

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

        success, metrics = self._compute_success_metrics()
        self._latest_shutter_pressed = bool(metrics.get("shutter_pressed", False))
        obs = self._compute_observation(metrics=metrics)
        self.env_step += 1

        terminated = self.env_step >= self._max_episode_steps or success

        if self.render_mode == "human":
            try:
                # Refresh the camera screen texture for the human viewer in case
                # render() was not called this step (e.g. image_obs is False).
                self._update_camera_screen_texture()
                # Keep target-region geom (group 5) visible in MuJoCo human window only.
                self._set_human_geom_group_visibility(5, True)
                self._add_success_debug_overlay(metrics)
                self._viewer.render("human")
            except Exception:
                pass

        dt = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - dt))

        reward = 1.0 if success else 0.0
        info = {"succeed": bool(success)}
        info.update(metrics)
        return obs, reward, terminated, False, info

    def _compute_success_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        region_center = self._data.geom_xpos[self._target_region_geom_id].copy()
        region_rot = self._data.geom_xmat[self._target_region_geom_id].reshape(3, 3)
        view_pos = self._data.site_xpos[self._view_site_id].copy()
        delta_local = region_rot.T @ (view_pos - region_center)
        radial_dist = float(np.linalg.norm(delta_local[:2]))
        region_height_dist = float(abs(delta_local[2]))
        region_pass = radial_dist <= _TARGET_REGION_RADIUS and region_height_dist <= _TARGET_REGION_HALF_HEIGHT

        axis_z_world = self._data.geom_xmat[self._camera_axis_z_geom_id].reshape(3, 3)[:, 2].copy()
        axis_z_norm = float(np.linalg.norm(axis_z_world))
        if axis_z_norm < 1e-8:
            axis_z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis_z_world = axis_z_world / axis_z_norm

        cos_val = float(np.clip(np.dot(axis_z_world, _WORLD_POS_X), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        angle_pass = angle_deg <= self._target_angle_deg

        shutter_pressed = False
        shutter_contact_count = 0
        shutter_touch_count = 0
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 == self._shutter_geom_id:
                if g2 not in self._camera_geom_ids:
                    shutter_pressed = True
                    shutter_contact_count += 1
            elif g2 == self._shutter_geom_id:
                if g1 not in self._camera_geom_ids:
                    shutter_pressed = True
                    shutter_contact_count += 1

        if not shutter_pressed:
            shutter_touch_count = self._compute_shutter_touch_count()
            if shutter_touch_count > 0:
                shutter_pressed = True

        progress_num = 0
        if region_pass:
            progress_num = 1
            if angle_pass:
                progress_num = 2
                if shutter_pressed:
                    progress_num = 3

        success = region_pass and angle_pass and shutter_pressed
        return bool(success), {
            "region_pass": bool(region_pass),
            "angle_pass": bool(angle_pass),
            "shutter_pressed": bool(shutter_pressed),
            "progress_num": int(progress_num),
            "radial_dist": radial_dist,
            "region_height_dist": region_height_dist,
            "view_angle_deg": angle_deg,
            "shutter_contact_count": int(shutter_contact_count + shutter_touch_count),
            "shutter_direct_contact_count": int(shutter_contact_count),
            "shutter_touch_count": int(shutter_touch_count),
        }

    def _compute_success(self) -> bool:
        success, _ = self._compute_success_metrics()
        return bool(success)

    def _compute_shutter_touch_count(self) -> int:
        if not self._shutter_fingertip_geom_ids or not self._shutter_touch_contact_geom_ids:
            return 0

        shutter_center = self._data.geom_xpos[self._shutter_geom_id].copy()
        shutter_rot = self._data.geom_xmat[self._shutter_geom_id].reshape(3, 3)
        shutter_radius = float(self._model.geom_size[self._shutter_geom_id, 0])
        shutter_half_height = float(self._model.geom_size[self._shutter_geom_id, 1])
        radial_limit = shutter_radius * 1.45
        axial_limit = shutter_half_height + 0.018

        touch_count = 0
        ncon = int(self._data.ncon)
        for i in range(ncon):
            c = self._data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            if g1 in self._shutter_touch_contact_geom_ids and g2 in self._shutter_fingertip_geom_ids:
                contact_pos = np.asarray(c.pos, dtype=np.float64).copy()
            elif g2 in self._shutter_touch_contact_geom_ids and g1 in self._shutter_fingertip_geom_ids:
                contact_pos = np.asarray(c.pos, dtype=np.float64).copy()
            else:
                continue

            local = shutter_rot.T @ (contact_pos - shutter_center)
            radial_dist = float(np.linalg.norm(local[:2]))
            axial_dist = float(abs(local[2]))
            if radial_dist <= radial_limit and axial_dist <= axial_limit:
                touch_count += 1

        return touch_count

    def _add_success_debug_overlay(self, metrics: Dict[str, Any]) -> None:
        viewer = self._viewer._get_viewer(render_mode="human")
        viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "num",
            str(int(metrics.get("progress_num", 0))),
        )
        viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "photo",
            (
                f"region:{int(metrics.get('region_pass', False))} "
                f"angle:{int(metrics.get('angle_pass', False))} "
                f"shutter:{int(metrics.get('shutter_pressed', False))}"
            ),
        )
        viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "detail",
            (
                f"r={float(metrics.get('radial_dist', 0.0)):.3f} "
                f"h={float(metrics.get('region_height_dist', 0.0)):.3f} "
                f"deg={float(metrics.get('view_angle_deg', 0.0)):.1f} "
                f"cnt={int(metrics.get('shutter_contact_count', 0))}"
            ),
        )

    def render(self):
        # Refresh the camera screen texture with the latest cam_pos_z view.
        self._update_camera_screen_texture()
        # Hide target-region geom (group 5) in offscreen rgb_array cameras.
        self._set_offscreen_geom_group_visibility(5, False)
        rendered_frames = []
        for cam_id in self.camera_id:
            if cam_id >= 0:
                rendered_frames.append(self._viewer.render(render_mode="rgb_array", camera_id=cam_id))
            else:
                rendered_frames.append(None)
        return rendered_frames

    def _compute_observation(self, metrics: Dict[str, Any] = None) -> dict:
        if metrics is None:
            _, metrics = self._compute_success_metrics()

        obs = {"state": {}}

        tcp_pos_right = self._data.sensor("franka/flange_pos_right").data
        tcp_quat_right = self._data.sensor("franka/flange_quat_right").data
        tcp_pos_left = self._data.sensor("franka/flange_pos_left").data
        tcp_quat_left = self._data.sensor("franka/flange_quat_left").data
        tcp_pose = np.concatenate(
            [np.concatenate([tcp_pos_right, tcp_quat_right]), np.concatenate([tcp_pos_left, tcp_quat_left])]
        )

        # 从 qpos 地址读取当前手型的关节状态，避免 Allegro 专属 sensor 名假设。
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
            "camera_ori_pose": self.camera_ori_pose,
            "logo_ori_pose": self.logo_ori_pose,
            "table_delta_height": self.delta_h
        }
        return obs

    def get_end_effector_pose_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        pos_r = self._data.mocap_pos[self._mocap_right_id]
        quat_r = self._data.mocap_quat[self._mocap_right_id]
        quat_r = np.array([quat_r[1], quat_r[2], quat_r[3], quat_r[0]])
        rot_r = R.from_quat(quat_r).as_matrix()
        T_right = np.eye(4)
        T_right[:3, :3] = rot_r
        T_right[:3, 3] = pos_r

        pos_l = self._data.mocap_pos[self._mocap_left_id]
        quat_l = self._data.mocap_quat[self._mocap_left_id]
        quat_l = np.array([quat_l[1], quat_l[2], quat_l[3], quat_l[0]])
        rot_l = R.from_quat(quat_l).as_matrix()
        T_left = np.eye(4)
        T_left[:3, :3] = rot_l
        T_left[:3, 3] = pos_l
        return T_right, T_left


if __name__ == "__main__":
    env = PandaBimanualPhotographGymEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(200):
        action = {
            "right": np.zeros(23, dtype=np.float32),
            "left": np.zeros(23, dtype=np.float32),
        }
        obs, rew, done, trunc, info = env.step(action)
        if done or trunc:
            break
    env.close()
