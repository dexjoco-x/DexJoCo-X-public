from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np

'''
height       渲染图像高度，默认 640
width        渲染图像宽度，默认 640
camera_id    用哪个 MuJoCo 相机，-1 通常表示默认相机
mode         渲染模式，rgb_array 或 human
'''
@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 640
    width: int = 640
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"

# 这段代码定义了一个 MuJoCo 环境基类，给上层任务环境提供统一的 Gymnasium 接口
'''
1. 从 XML 加载 MuJoCo 模型
2. 创建 MuJoCo 仿真数据 data
3. 设置物理仿真步长 physics_dt
4. 设置控制周期 control_dt
5. 计算每个 env.step 里要跑多少个 MuJoCo 小步 _n_substeps
6. 提供 render / close / model / data 等基础接口
'''
class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    _LEAP_HAND_ROOT_BODIES = ("leap_palm", "leap_palm_right", "leap_palm_left")
    _LEAP_HAND_FRICTION = np.asarray((4.0, 0.18, 0.009), dtype=np.float64)
    _LEAP_FINGERTIP_FRICTION = np.asarray((6.0, 0.30, 0.015), dtype=np.float64)
    _LEAP_HAND_SOLREF = np.asarray((0.006, 1.2), dtype=np.float64)
    _LEAP_FINGERTIP_SOLREF = np.asarray((0.005, 1.2), dtype=np.float64)
    _LEAP_HAND_SOLIMP = np.asarray((0.92, 0.99, 0.001, 0.5, 2.0), dtype=np.float64)
    _LEAP_HAND_MARGIN = 0.001
    _LEAP_FINGERTIP_MARGIN = 0.002
    _LEAP_HAND_CONDIM = 4
    _LEAP_HAND_PRIORITY = 2
    _LEAP_FINGERTIP_PAD_MIN_RADIUS = 0.008

    _LEAP_TASK_OBJECT_ROOT_BODIES = (
        "bucket",
        "boxed_food_0",
    )
    _LEAP_TASK_OBJECT_FRICTION = np.asarray((2.2, 0.12, 0.008), dtype=np.float64)
    _LEAP_TASK_OBJECT_SOLREF = np.asarray((0.008, 1.2), dtype=np.float64)
    _LEAP_TASK_OBJECT_SOLIMP = np.asarray((0.92, 0.99, 0.001, 0.5, 2.0), dtype=np.float64)
    _LEAP_TASK_OBJECT_MARGIN = 0.001
    _LEAP_TASK_OBJECT_PRIORITY = 2
    _LEAP_ASSEMBLY_OBJECT_ROOT_BODIES = (
        "industreal_round_peg_8mm",
        "industreal_tray_insert_round_peg_8mm",
    )
    _LEAP_ASSEMBLY_TABLE_GEOMS = ("table_collision",)
    # The assembly peg and tray are easy to press into the tabletop with LEAP's
    # position-controlled fingers.  Give the support contacts a firm low-level
    # model, while damping the free bodies so the objects do not bounce at reset.
    _LEAP_ASSEMBLY_OBJECT_FRICTION = np.asarray(
        (2.2, 0.12, 0.010),
        dtype=np.float64,
    )
    _LEAP_ASSEMBLY_OBJECT_SOLREF = np.asarray((0.0035, 1.8), dtype=np.float64)
    _LEAP_ASSEMBLY_OBJECT_SOLIMP = np.asarray(
        (0.97, 0.995, 0.0001, 0.5, 2.0),
        dtype=np.float64,
    )
    _LEAP_ASSEMBLY_OBJECT_MARGIN = 0.0015
    _LEAP_ASSEMBLY_OBJECT_CONDIM = 6
    _LEAP_ASSEMBLY_OBJECT_PRIORITY = 2
    _LEAP_ASSEMBLY_TABLE_FRICTION = np.asarray(
        (2.5, 0.12, 0.010),
        dtype=np.float64,
    )
    _LEAP_ASSEMBLY_TABLE_SOLREF = np.asarray((0.0025, 1.9), dtype=np.float64)
    _LEAP_ASSEMBLY_TABLE_SOLIMP = np.asarray(
        (0.98, 0.998, 0.0001, 0.5, 2.0),
        dtype=np.float64,
    )
    _LEAP_ASSEMBLY_TABLE_MARGIN = 0.001
    _LEAP_ASSEMBLY_TABLE_CONDIM = 6
    _LEAP_ASSEMBLY_TABLE_PRIORITY = 2
    _LEAP_ASSEMBLY_FREEJOINT_DAMPING = np.asarray(
        (0.12, 0.12, 0.12, 0.04, 0.04, 0.04),
        dtype=np.float64,
    )
    # Mirror the Sharpa Wave assembly hand feel: high-friction, compliant
    # fingertip contacts, while keeping LEAP slightly stiffer to avoid visual
    # sink-in from its non-deforming collision shells.
    _LEAP_ASSEMBLY_HAND_FRICTION = np.asarray(
        (4.0, 0.15, 0.020),
        dtype=np.float64,
    )
    _LEAP_ASSEMBLY_FINGERTIP_FRICTION = np.asarray(
        (4.5, 0.18, 0.025), dtype=np.float64
    )
    _LEAP_ASSEMBLY_HAND_SOLREF = np.asarray((0.012, 1.1), dtype=np.float64)
    _LEAP_ASSEMBLY_FINGERTIP_SOLREF = np.asarray((0.010, 1.1), dtype=np.float64)
    _LEAP_ASSEMBLY_HAND_SOLIMP = np.asarray(
        (0.95, 0.995, 0.001, 0.5, 2.0), dtype=np.float64
    )
    _LEAP_ASSEMBLY_HAND_MARGIN = 0.0015
    _LEAP_ASSEMBLY_FINGERTIP_MARGIN = 0.003
    _LEAP_ASSEMBLY_HAND_PRIORITY = 3
    _LEAP_HANOI_DISK_ROOT_BODIES = (
        "hanoi_disk_large",
        "hanoi_disk_medium",
        "hanoi_disk_small",
    )
    # Keep Hanoi disks and posts exactly on the Inspire contact model.  The
    # scene's four-plate collision topology and explicit disk/post pairs are
    # shared between hands; only LEAP's hand-side collision shell differs.
    _LEAP_HANOI_HAND_FRICTION = np.asarray((5.0, 0.25, 0.012), dtype=np.float64)
    _LEAP_HANOI_FINGERTIP_FRICTION = np.asarray(
        (7.0, 0.35, 0.018), dtype=np.float64
    )
    _LEAP_HANOI_HAND_SOLREF = np.asarray((0.0035, 1.35), dtype=np.float64)
    _LEAP_HANOI_FINGERTIP_SOLREF = np.asarray((0.0025, 1.4), dtype=np.float64)
    _LEAP_HANOI_HAND_SOLIMP = np.asarray(
        (0.94, 0.995, 0.001, 0.5, 2.0), dtype=np.float64
    )
    _LEAP_HANOI_HAND_MARGIN = 0.002
    _LEAP_HANOI_FINGERTIP_MARGIN = 0.004
    _LEAP_HANOI_HAND_PRIORITY = 3
    _LEAP_HANOI_HAND_GUARD_BIT = 2
    _LEAP_HANOI_THUMB_PAD_BIT = 4
    _LEAP_HANOI_DISK_FREEJOINT_DAMPING = np.asarray(
        (0.04, 0.04, 0.04, 0.01, 0.01, 0.01),
        dtype=np.float64,
    )
    _LEAP_HANOI_THUMB_PAD_GEOMS = (
        "leap_thumb_fingertip_shell_right",
        "leap_thumb_fingertip_shell_left",
        "leap_thumb_fingertip_pad_capsule_right",
        "leap_thumb_fingertip_pad_capsule_left",
    )
    _LEAP_BUCKET_MIN_HALF_SIZES = {
        "handle1": (0.003, 0.008, 0.12),
        "handle2": (0.09, 0.008, 0.003),
        "handle3": (0.09, 0.008, 0.003),
        "bucket_l": (0.004, 0.015, 0.128),
        "bucket_r": (0.004, 0.015, 0.128),
        "bucket_left": (0.003, 0.09, 0.128),
        "bucket_right": (0.003, 0.09, 0.128),
        "bucket_bottom": (0.163, 0.004, 0.12),
    }
    _LEAP_BUCKET_HANDLE_DAMPING = 0.08
    _LEAP_BUCKET_HANDLE_ARMATURE = 0.03
    _LEAP_BUCKET_HANDLE_FRICTIONLOSS = 0.04
    _WUJI_HAND_FRICTION = np.asarray((3.0, 0.35, 0.02), dtype=np.float64)
    _WUJI_FINGERTIP_FRICTION = np.asarray((4.0, 1.0, 0.45), dtype=np.float64)
    _WUJI_HAND_SOLREF = np.asarray((0.003, 1.5), dtype=np.float64)
    _WUJI_FINGERTIP_SOLREF = np.asarray((0.0015, 2.0), dtype=np.float64)
    _WUJI_HAND_SOLIMP = np.asarray((0.96, 0.995, 0.0005, 0.5, 2.0), dtype=np.float64)
    _WUJI_FINGERTIP_SOLIMP = np.asarray((0.98, 0.995, 0.0001, 0.5, 2.0), dtype=np.float64)
    _WUJI_HAND_MARGIN = 0.0008
    _WUJI_FINGERTIP_MARGIN = 0.0018
    _WUJI_HAND_CONDIM = 4
    _WUJI_HAND_PRIORITY = 2
    _WUJI_FINGERTIP_PRIORITY = 3
    _WUJI_FINGERTIP_PAD_MIN_RADII = (
        ("thumb_fingertip_pad", 0.0115),
        ("index_fingertip_pad", 0.0095),
        ("middle_fingertip_pad", 0.0095),
        ("ring_fingertip_pad", 0.0095),
        ("pinky_fingertip_pad", 0.0095),
        ("thumb_index_web_pad", 0.0075),
    )
    _WUJI_GLASS_ROOT_BODIES = ("glass",)
    _WUJI_GLASS_FRICTION = np.asarray((1.8, 0.35, 0.12), dtype=np.float64)
    _WUJI_GLASS_SOLREF = np.asarray((0.0015, 2.0), dtype=np.float64)
    _WUJI_GLASS_SOLIMP = np.asarray((0.98, 0.995, 0.0001, 0.5, 2.0), dtype=np.float64)
    _WUJI_GLASS_MARGIN = 0.001
    _WUJI_GLASS_CONDIM = 4
    _WUJI_GLASS_PRIORITY = 2

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        self._xml_path = Path(xml_path).resolve()
        # 这行从 XML 文件加载 MuJoCo 模型
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._apply_builtin_contact_tuning()
        # 加载完成后，self._model 里包含：
            # body
            # joint
            # geom
            # actuator
            # sensor
            # camera
            # light
            # material
            # timestep
            # 质量、惯性、碰撞配置等

        # 设置离屏渲染分辨率
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height

        # self._data 是仿真运行时状态，会随着仿真的运行而变化
        # self._data.qpos
        # self._data.qvel
        # self._data.ctrl
        # self._data.time
        # self._data.contact
        self._data = mujoco.MjData(self._model)

        # 设置仿真运行的步长500hz
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        # 最大仿真时间，没有用到
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random

    def _apply_builtin_contact_tuning(self) -> None:
        self._apply_leap_hand_contact_tuning()
        self._apply_leap_task_object_contact_tuning()
        self._apply_wuji_hand_contact_tuning()
        self._apply_wuji_glass_contact_tuning()

    def _apply_leap_hand_contact_tuning(self) -> None:
        """Raise friction on the real LEAP hand collision geoms.

        Some LEAP XMLs keep unused collision defaults, while the actual hand
        links are direct mesh geoms. Tuning the compiled model avoids relying on
        XML defaults that may not be inherited by those geoms.
        """
        if not self._has_leap_hand():
            return

        self._activate_leap_fingertip_collision_pads()

        tuned_geom_ids: set[int] = set()
        for root_body_name in self._LEAP_HAND_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if geom_id in tuned_geom_ids:
                    continue
                body_id = int(self._model.geom_bodyid[geom_id])
                if body_id not in body_ids:
                    continue
                if not self._is_collision_geom(geom_id):
                    continue

                body_name = (
                    mujoco.mj_id2name(
                        self._model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        body_id,
                    )
                    or ""
                )
                is_fingertip = "fingertip" in body_name
                self._apply_contact_params(
                    geom_id,
                    friction=(
                        self._LEAP_FINGERTIP_FRICTION
                        if is_fingertip
                        else self._LEAP_HAND_FRICTION
                    ),
                    condim=self._LEAP_HAND_CONDIM,
                    solref=(
                        self._LEAP_FINGERTIP_SOLREF
                        if is_fingertip
                        else self._LEAP_HAND_SOLREF
                    ),
                    solimp=self._LEAP_HAND_SOLIMP,
                    margin=(
                        self._LEAP_FINGERTIP_MARGIN
                        if is_fingertip
                        else self._LEAP_HAND_MARGIN
                    ),
                    priority=self._LEAP_HAND_PRIORITY,
                )
                tuned_geom_ids.add(geom_id)

    def _activate_leap_fingertip_collision_pads(self) -> None:
        """Use the visible LEAP fingertip pads as collision shells.

        The LEAP XMLs contain 9 mm fingertip pad spheres that visually represent
        the contact surface, but mark them as visual-only. Leaving only the
        detailed mesh and tiny debug spheres active makes thin object edges look
        like they cut through the finger during teleop grasps.
        """
        for geom_id in range(self._model.ngeom):
            if int(self._model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_SPHERE:
                continue
            if (
                float(self._model.geom_size[geom_id, 0])
                < self._LEAP_FINGERTIP_PAD_MIN_RADIUS
            ):
                continue
            body_id = int(self._model.geom_bodyid[geom_id])
            body_name = (
                mujoco.mj_id2name(
                    self._model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )
                or ""
            )
            if "fingertip" not in body_name:
                continue
            self._model.geom_contype[geom_id] = 1
            self._model.geom_conaffinity[geom_id] = 1

    def _apply_leap_task_object_contact_tuning(self) -> None:
        if not self._has_leap_hand():
            return

        self._thicken_leap_bucket_collision()
        self._stabilize_leap_bucket_handle_joint()
        self._apply_leap_assembly_object_contact_tuning()
        self._apply_leap_assembly_grasp_contact_tuning()
        self._apply_leap_hanoi_grasp_contact_tuning()

        tuned_geom_ids: set[int] = set()
        for root_body_name in self._LEAP_TASK_OBJECT_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if geom_id in tuned_geom_ids:
                    continue
                if int(self._model.geom_bodyid[geom_id]) not in body_ids:
                    continue
                if not self._is_collision_geom(geom_id):
                    continue
                self._apply_contact_params(
                    geom_id,
                    friction=self._LEAP_TASK_OBJECT_FRICTION,
                    condim=self._LEAP_HAND_CONDIM,
                    solref=self._LEAP_TASK_OBJECT_SOLREF,
                    solimp=self._LEAP_TASK_OBJECT_SOLIMP,
                    margin=self._LEAP_TASK_OBJECT_MARGIN,
                    priority=self._LEAP_TASK_OBJECT_PRIORITY,
                )
                tuned_geom_ids.add(geom_id)

    def _has_leap_assembly_objects(self) -> bool:
        return any(
            mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            >= 0
            for root_body_name in self._LEAP_ASSEMBLY_OBJECT_ROOT_BODIES
        )

    def _apply_leap_assembly_grasp_contact_tuning(self) -> None:
        """Make LEAP-to-assembly contacts firm without sacrificing support.

        Contact parameters are selected from the geom with the highest
        priority. The table and assembly parts use priority 2 support
        contacts; hand geoms are priority 3 so grasps still get LEAP fingertip
        friction without making the tabletop soft.
        """
        if not self._has_leap_assembly_objects():
            return

        self._damp_leap_assembly_freejoints()

        tuned_geom_ids: set[int] = set()
        for root_body_name in self._LEAP_HAND_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if geom_id in tuned_geom_ids:
                    continue
                body_id = int(self._model.geom_bodyid[geom_id])
                if body_id not in body_ids or not self._is_collision_geom(geom_id):
                    continue
                is_fingertip = "fingertip" in self._body_name(body_id)
                self._apply_contact_params(
                    geom_id,
                    friction=(
                        self._LEAP_ASSEMBLY_FINGERTIP_FRICTION
                        if is_fingertip
                        else self._LEAP_ASSEMBLY_HAND_FRICTION
                    ),
                    condim=self._LEAP_HAND_CONDIM,
                    solref=(
                        self._LEAP_ASSEMBLY_FINGERTIP_SOLREF
                        if is_fingertip
                        else self._LEAP_ASSEMBLY_HAND_SOLREF
                    ),
                    solimp=self._LEAP_ASSEMBLY_HAND_SOLIMP,
                    margin=(
                        self._LEAP_ASSEMBLY_FINGERTIP_MARGIN
                        if is_fingertip
                        else self._LEAP_ASSEMBLY_HAND_MARGIN
                    ),
                    priority=self._LEAP_ASSEMBLY_HAND_PRIORITY,
                )
                tuned_geom_ids.add(geom_id)

    def _apply_leap_assembly_object_contact_tuning(self) -> None:
        if not self._has_leap_assembly_objects():
            return

        self._damp_leap_assembly_freejoints()
        self._apply_leap_assembly_table_contact_tuning()

        tuned_geom_ids: set[int] = set()
        for root_body_name in self._LEAP_ASSEMBLY_OBJECT_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if geom_id in tuned_geom_ids:
                    continue
                if int(self._model.geom_bodyid[geom_id]) not in body_ids:
                    continue
                if not self._is_collision_geom(geom_id):
                    continue
                self._apply_contact_params(
                    geom_id,
                    friction=self._LEAP_ASSEMBLY_OBJECT_FRICTION,
                    condim=self._LEAP_ASSEMBLY_OBJECT_CONDIM,
                    solref=self._LEAP_ASSEMBLY_OBJECT_SOLREF,
                    solimp=self._LEAP_ASSEMBLY_OBJECT_SOLIMP,
                    margin=self._LEAP_ASSEMBLY_OBJECT_MARGIN,
                    priority=self._LEAP_ASSEMBLY_OBJECT_PRIORITY,
                )
                tuned_geom_ids.add(geom_id)

    def _apply_leap_assembly_table_contact_tuning(self) -> None:
        for geom_name in self._LEAP_ASSEMBLY_TABLE_GEOMS:
            geom_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            if geom_id < 0:
                continue
            self._apply_contact_params(
                geom_id,
                friction=self._LEAP_ASSEMBLY_TABLE_FRICTION,
                condim=self._LEAP_ASSEMBLY_TABLE_CONDIM,
                solref=self._LEAP_ASSEMBLY_TABLE_SOLREF,
                solimp=self._LEAP_ASSEMBLY_TABLE_SOLIMP,
                margin=self._LEAP_ASSEMBLY_TABLE_MARGIN,
                priority=self._LEAP_ASSEMBLY_TABLE_PRIORITY,
            )

    def _damp_leap_assembly_freejoints(self) -> None:
        for body_name in self._LEAP_ASSEMBLY_OBJECT_ROOT_BODIES:
            joint_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{body_name}_joint",
            )
            if joint_id < 0:
                continue
            dof_adr = int(self._model.jnt_dofadr[joint_id])
            if dof_adr < 0:
                continue
            self._model.dof_damping[dof_adr : dof_adr + 6] = np.maximum(
                self._model.dof_damping[dof_adr : dof_adr + 6],
                self._LEAP_ASSEMBLY_FREEJOINT_DAMPING,
            )

    def _apply_leap_hanoi_grasp_contact_tuning(self) -> None:
        """Make the LEAP hand win against Inspire-equivalent Hanoi disks."""
        if not any(
            mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            >= 0
            for root_body_name in self._LEAP_HANOI_DISK_ROOT_BODIES
        ):
            return

        self._damp_leap_hanoi_disk_freejoints()

        tuned_geom_ids: set[int] = set()
        for root_body_name in self._LEAP_HAND_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if geom_id in tuned_geom_ids:
                    continue
                body_id = int(self._model.geom_bodyid[geom_id])
                if body_id not in body_ids or not self._is_collision_geom(geom_id):
                    continue
                is_fingertip = "fingertip" in self._body_name(body_id)
                self._apply_contact_params(
                    geom_id,
                    friction=(
                        self._LEAP_HANOI_FINGERTIP_FRICTION
                        if is_fingertip
                        else self._LEAP_HANOI_HAND_FRICTION
                    ),
                    condim=self._LEAP_HAND_CONDIM,
                    solref=(
                        self._LEAP_HANOI_FINGERTIP_SOLREF
                        if is_fingertip
                        else self._LEAP_HANOI_HAND_SOLREF
                    ),
                    solimp=self._LEAP_HANOI_HAND_SOLIMP,
                    margin=(
                        self._LEAP_HANOI_FINGERTIP_MARGIN
                        if is_fingertip
                        else self._LEAP_HANOI_HAND_MARGIN
                    ),
                    priority=self._LEAP_HANOI_HAND_PRIORITY,
                )
                # Preserve normal object contacts (bit 1), and add a private
                # bit for the ring-centre guards below.  Normal Hanoi parts
                # still collide through their contype/conaffinity bit 1.
                self._model.geom_contype[geom_id] |= self._LEAP_HANOI_HAND_GUARD_BIT
                tuned_geom_ids.add(geom_id)

        self._activate_leap_hanoi_thumb_pad_capsules()
        self._activate_leap_hanoi_inner_grasp_guards()

    def _damp_leap_hanoi_disk_freejoints(self) -> None:
        for disk_name in self._LEAP_HANOI_DISK_ROOT_BODIES:
            joint_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{disk_name}_joint",
            )
            if joint_id < 0:
                continue
            dof_adr = int(self._model.jnt_dofadr[joint_id])
            if dof_adr < 0:
                continue
            self._model.dof_damping[dof_adr : dof_adr + 6] = np.maximum(
                self._model.dof_damping[dof_adr : dof_adr + 6],
                self._LEAP_HANOI_DISK_FREEJOINT_DAMPING,
            )

    def _activate_leap_hanoi_thumb_pad_capsules(self) -> None:
        for geom_name in self._LEAP_HANOI_THUMB_PAD_GEOMS:
            geom_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            if geom_id < 0:
                continue
            self._model.geom_contype[geom_id] = self._LEAP_HANOI_THUMB_PAD_BIT
            self._model.geom_conaffinity[geom_id] = self._LEAP_HANOI_THUMB_PAD_BIT
            self._apply_contact_params(
                geom_id,
                friction=self._LEAP_HANOI_FINGERTIP_FRICTION,
                condim=self._LEAP_HAND_CONDIM,
                solref=self._LEAP_HANOI_FINGERTIP_SOLREF,
                solimp=self._LEAP_HANOI_HAND_SOLIMP,
                margin=self._LEAP_HANOI_FINGERTIP_MARGIN,
                priority=self._LEAP_HANOI_HAND_PRIORITY,
            )

        self._enable_hanoi_thumb_pad_contacts()

    def _enable_hanoi_thumb_pad_contacts(self) -> None:
        for disk_name in self._LEAP_HANOI_DISK_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                disk_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if int(self._model.geom_bodyid[geom_id]) not in body_ids:
                    continue
                geom_name = self._geom_name(geom_id)
                if (
                    "_collision_" not in geom_name
                    and not geom_name.endswith("_leap_hand_guard")
                ):
                    continue
                self._model.geom_conaffinity[geom_id] |= self._LEAP_HANOI_THUMB_PAD_BIT

    def _activate_leap_hanoi_inner_grasp_guards(self) -> None:
        """Block LEAP fingers from entering a Hanoi ring's centre hole.

        The invisible guards use a private collision bit shared only with the
        LEAP hand.  They therefore prevent visual hand-through-ring contact
        while the normal Hanoi post remains free to pass through the same hole.
        """
        for disk_name in self._LEAP_HANOI_DISK_ROOT_BODIES:
            guard_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{disk_name}_leap_hand_guard",
            )
            if guard_id < 0:
                continue
            self._model.geom_contype[guard_id] = self._LEAP_HANOI_HAND_GUARD_BIT
            self._model.geom_conaffinity[guard_id] = self._LEAP_HANOI_HAND_GUARD_BIT
            self._model.geom_friction[guard_id] = self._LEAP_HANOI_FINGERTIP_FRICTION
            self._model.geom_condim[guard_id] = self._LEAP_HAND_CONDIM
            self._model.geom_solref[guard_id] = self._LEAP_HANOI_FINGERTIP_SOLREF
            self._model.geom_solimp[guard_id] = self._LEAP_HANOI_HAND_SOLIMP
            self._model.geom_margin[guard_id] = self._LEAP_HANOI_FINGERTIP_MARGIN
            self._model.geom_priority[guard_id] = self._LEAP_HANOI_HAND_PRIORITY

    def _thicken_leap_bucket_collision(self) -> None:
        for geom_name, min_half_size in self._LEAP_BUCKET_MIN_HALF_SIZES.items():
            geom_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            if geom_id < 0:
                continue
            self._model.geom_size[geom_id, :3] = np.maximum(
                self._model.geom_size[geom_id, :3],
                np.asarray(min_half_size, dtype=np.float64),
            )

    def _stabilize_leap_bucket_handle_joint(self) -> None:
        joint_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "bucket_joint_0",
        )
        if joint_id < 0:
            return
        dof_id = int(self._model.jnt_dofadr[joint_id])
        self._model.dof_damping[dof_id] = max(
            float(self._model.dof_damping[dof_id]),
            self._LEAP_BUCKET_HANDLE_DAMPING,
        )
        self._model.dof_armature[dof_id] = max(
            float(self._model.dof_armature[dof_id]),
            self._LEAP_BUCKET_HANDLE_ARMATURE,
        )
        self._model.dof_frictionloss[dof_id] = max(
            float(self._model.dof_frictionloss[dof_id]),
            self._LEAP_BUCKET_HANDLE_FRICTIONLOSS,
        )

    def _apply_wuji_hand_contact_tuning(self) -> None:
        if not self._has_wuji_hand():
            return

        self._inflate_wuji_contact_pads()
        for geom_id in range(self._model.ngeom):
            if not self._is_wuji_fingertip_geom(geom_id):
                continue
            if not self._is_collision_geom(geom_id):
                continue

            self._apply_contact_params(
                geom_id,
                friction=self._WUJI_FINGERTIP_FRICTION,
                condim=self._WUJI_HAND_CONDIM,
                solref=self._WUJI_FINGERTIP_SOLREF,
                solimp=self._WUJI_FINGERTIP_SOLIMP,
                margin=self._WUJI_FINGERTIP_MARGIN,
                priority=self._WUJI_FINGERTIP_PRIORITY,
            )

    def _inflate_wuji_contact_pads(self) -> None:
        for geom_id in range(self._model.ngeom):
            geom_name = self._geom_name(geom_id)
            if not geom_name:
                continue
            for pattern, min_radius in self._WUJI_FINGERTIP_PAD_MIN_RADII:
                if pattern not in geom_name:
                    continue
                self._model.geom_size[geom_id, 0] = max(
                    float(self._model.geom_size[geom_id, 0]),
                    float(min_radius),
                )
                break

    def _apply_wuji_glass_contact_tuning(self) -> None:
        if not self._has_wuji_hand():
            return

        tuned_geom_ids: set[int] = set()
        for root_body_name in self._WUJI_GLASS_ROOT_BODIES:
            root_body_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name,
            )
            if root_body_id < 0:
                continue
            body_ids = self._collect_body_subtree_ids(root_body_id)
            for geom_id in range(self._model.ngeom):
                if geom_id in tuned_geom_ids:
                    continue
                if int(self._model.geom_bodyid[geom_id]) not in body_ids:
                    continue
                if not self._is_collision_geom(geom_id):
                    continue
                self._apply_contact_params(
                    geom_id,
                    friction=self._WUJI_GLASS_FRICTION,
                    condim=self._WUJI_GLASS_CONDIM,
                    solref=self._WUJI_GLASS_SOLREF,
                    solimp=self._WUJI_GLASS_SOLIMP,
                    margin=self._WUJI_GLASS_MARGIN,
                    priority=self._WUJI_GLASS_PRIORITY,
                )
                tuned_geom_ids.add(geom_id)

    def _apply_contact_params(
        self,
        geom_id: int,
        *,
        friction: np.ndarray,
        condim: int,
        solref: np.ndarray,
        solimp: np.ndarray,
        margin: float,
        priority: int,
    ) -> None:
        self._model.geom_friction[geom_id] = friction
        self._model.geom_condim[geom_id] = condim
        self._model.geom_solref[geom_id] = solref
        self._model.geom_solimp[geom_id] = solimp
        self._model.geom_margin[geom_id] = max(float(self._model.geom_margin[geom_id]), margin)
        self._model.geom_priority[geom_id] = max(int(self._model.geom_priority[geom_id]), priority)

    def _has_leap_hand(self) -> bool:
        for root_body_name in self._LEAP_HAND_ROOT_BODIES:
            if (
                mujoco.mj_name2id(
                    self._model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    root_body_name,
                )
                >= 0
            ):
                return True
        return False

    def _has_wuji_hand(self) -> bool:
        for body_id in range(self._model.nbody):
            if self._body_name(body_id).startswith("wuji_"):
                return True
        return False

    def _is_wuji_hand_geom(self, geom_id: int) -> bool:
        body_id = int(self._model.geom_bodyid[geom_id])
        return (
            self._geom_name(geom_id).startswith("wuji_")
            or self._body_name(body_id).startswith("wuji_")
            or self._geom_mesh_name(geom_id).startswith("wuji_")
        )

    def _is_wuji_fingertip_geom(self, geom_id: int) -> bool:
        names = (
            self._geom_name(geom_id),
            self._body_name(int(self._model.geom_bodyid[geom_id])),
            self._geom_mesh_name(geom_id),
        )
        return any(
            "fingertip_pad" in name
            or "thumb_index_web_pad" in name
            or name.startswith("wuji_fingertip_")
            for name in names
        )

    def _is_collision_geom(self, geom_id: int) -> bool:
        return (
            int(self._model.geom_contype[geom_id]) != 0
            or int(self._model.geom_conaffinity[geom_id]) != 0
        )

    def _body_name(self, body_id: int) -> str:
        return (
            mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
            )
            or ""
        )

    def _geom_name(self, geom_id: int) -> str:
        return (
            mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(geom_id),
            )
            or ""
        )

    def _geom_mesh_name(self, geom_id: int) -> str:
        mesh_id = int(self._model.geom_dataid[geom_id])
        if mesh_id < 0:
            return ""
        return (
            mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_id,
            )
            or ""
        )

    def _collect_body_subtree_ids(self, root_body_id: int) -> set[int]:
        body_ids = set()
        for body_id in range(self._model.nbody):
            cur = int(body_id)
            while cur > 0:
                if cur == root_body_id:
                    body_ids.add(body_id)
                    break
                cur = int(self._model.body_parentid[cur])
        return body_ids
