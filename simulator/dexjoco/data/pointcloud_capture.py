from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from dexjoco.data.depth_capture import collect_depth_frames, collect_segmentation_frames


@dataclass(frozen=True)
class PointCloudProfile:
    """Versioned RGB-D preprocessing contract shared by export and evaluation."""

    name: str
    camera_position_world: tuple[float, float, float] | None = None
    camera_target_world: tuple[float, float, float] | None = None
    camera_fovy_deg: float | None = None
    # Roll around the camera's own optical axis.  A 180-degree roll preserves
    # the viewpoint and depth while rotating RGB, point-cloud X/Y, and the
    # camera-frame wrist state together.
    camera_roll_deg: float = 0.0
    # Optional task-fixed policy frame.  The physical camera controls
    # visibility only; XYZ point clouds and wrist poses are expressed in this
    # frame so changing camera height/tilt does not change policy coordinates.
    # The origin is task-centred while target only defines the fixed
    # camera-like orientation (x right, y up, z back).
    policy_frame_origin_world: tuple[float, float, float] | None = None
    policy_frame_target_world: tuple[float, float, float] | None = None
    min_depth_m: float | None = None
    max_depth_m: float | None = None
    workspace_bounds_world: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] | None = None
    support_plane_site: str | None = None
    min_height_above_support_m: float = 0.0
    # Visible MuJoCo geoms removed by the aligned segmentation buffer before
    # point projection.  Existing profiles leave this empty and are unchanged.
    semantic_exclude_geom_names: tuple[str, ...] = ()
    # Visible geoms whose contribution is capped after crop/filter and before
    # fusion.  This differs from semantic exclusion: the surface remains
    # observable, but cannot dominate a fixed-size FPS cloud.  The fraction is
    # applied independently to each view so the dual-view contract is stable.
    semantic_limit_geom_names: tuple[str, ...] = ()
    semantic_limit_max_fraction: float | None = None
    sampling_method: str = "random"
    supported_camera_keys: tuple[str, ...] = ("front", "random_camera")


# The bounds are fixed per task, not per hand, so the same visual protocol can
# be used for cross-embodiment comparisons.  They retain the hand/wrist,
# hammer, nail, wood board, and hand/wrist while excluding the room, floor,
# table legs, and most of the robot base.  Final profiles may additionally use
# a calibrated support-plane height mask before FPS so a large table plane does
# not consume most of the fixed 10k-point observation.
_POINTCLOUD_PROFILES = {
    "legacy": PointCloudProfile(name="legacy"),
    # Frozen opposite-side camera retained only so existing v1 exports and
    # checkpoints remain reproducible.  New hammer datasets use ego_v2.
    "unidex_hammer_v1": PointCloudProfile(
        name="unidex_hammer_v1",
        camera_position_world=(0.60, 0.15, 1.48),
        camera_target_world=(-0.18, -0.18, 1.18),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=1.25,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        sampling_method="fps",
    ),
    # Fixed world camera on the robot-base side.  The elevated, slightly
    # lateral view approximates UniDex's external egocentric L515 setup while
    # avoiding eye-in-hand motion and forearm occlusion.  These world-frame
    # extrinsics are shared by every hammer_nail hand embodiment.
    "unidex_hammer_ego_v2": PointCloudProfile(
        name="unidex_hammer_ego_v2",
        camera_position_world=(-1.15, -0.35, 2.20),
        camera_target_world=(-0.15, -0.12, 1.08),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=2.25,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        sampling_method="fps",
    ),
    # Main cross-embodiment hammer protocol.  The fixed opposite-side elevated
    # camera avoids robot-arm foreground occlusion.  ``table_top`` is a
    # geometric calibration plane (not a semantic object mask): its actual
    # world height follows the task's per-episode 0--5 cm table randomization.
    # Points at/below that support plane are removed from dense RGB-D before
    # FPS, preserving the wood board and task objects above it.
    "unidex_hammer_opposite_high_center_v3": PointCloudProfile(
        name="unidex_hammer_opposite_high_center_v3",
        camera_position_world=(0.95, 0.05, 2.30),
        camera_target_world=(-0.20, -0.16, 1.06),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=2.50,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Deprecated experiment retained for checkpoint/data reproducibility.  Its
    # task-centred policy frame is not the camera-optical-frame convention used
    # by UniDex Real data, so it must not be selected by ``auto``.
    "unidex_hammer_topdown_taskframe_v4": PointCloudProfile(
        name="unidex_hammer_topdown_taskframe_v4",
        camera_position_world=(-0.05, -0.07, 2.65),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        policy_frame_origin_world=(-0.20, -0.16, 1.05),
        policy_frame_target_world=(-0.35, -0.25, -0.55),
        min_depth_m=0.20,
        max_depth_m=2.50,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Main native-30-Hz UniDex Real-style protocol.  The physical camera is on
    # the robot-base side of the task with a lateral shoulder-like offset.  It
    # is 0.85 m above the work centre with an approximately 30-degree oblique
    # top-down view, avoiding the severe forearm occlusion produced by looking
    # directly along the arm.  There is no policy-frame override: point-cloud
    # XYZ and wrist pose share the real camera optical centre.
    "unidex_hammer_topdown_real_v5": PointCloudProfile(
        name="unidex_hammer_topdown_real_v5",
        camera_position_world=(-0.35, 0.30, 1.90),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Candidate heights for manual RGB/PLY review.  They share the exact v5
    # preprocessing contract and differ only in physical camera height.
    "unidex_hammer_topdown_real_v5_075_preview": PointCloudProfile(
        name="unidex_hammer_topdown_real_v5_075_preview",
        camera_position_world=(-0.05, -0.07, 1.80),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_topdown_real_v5_095_preview": PointCloudProfile(
        name="unidex_hammer_topdown_real_v5_095_preview",
        camera_position_world=(-0.05, -0.07, 2.00),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Robot-base-side candidates for the final ego-like Real camera.  The arm
    # base is near (-0.8, 0.0), so all three sit behind the wrist on that side
    # and look along the arm towards the hammer/nail workspace.  Lateral
    # variants make the arm/hand occlusion trade-off directly inspectable.
    "unidex_hammer_arm_side_center_preview": PointCloudProfile(
        name="unidex_hammer_arm_side_center_preview",
        camera_position_world=(-0.48, -0.08, 1.90),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_arm_side_left_preview": PointCloudProfile(
        name="unidex_hammer_arm_side_left_preview",
        camera_position_world=(-0.48, 0.04, 1.90),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_arm_side_right_preview": PointCloudProfile(
        name="unidex_hammer_arm_side_right_preview",
        camera_position_world=(-0.48, -0.20, 1.90),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Arm-side, laterally offset variants.  These remain on the robot-base
    # half-plane (x < task-centre x) but avoid looking straight down the arm,
    # which hides the hammer/nail during the strike.
    "unidex_hammer_arm_side_oblique_left_preview": PointCloudProfile(
        name="unidex_hammer_arm_side_oblique_left_preview",
        camera_position_world=(-0.38, 0.22, 2.00),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_arm_side_oblique_wide_left_preview": PointCloudProfile(
        name="unidex_hammer_arm_side_oblique_wide_left_preview",
        camera_position_world=(-0.35, 0.38, 2.00),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_arm_side_oblique_balanced_preview": PointCloudProfile(
        name="unidex_hammer_arm_side_oblique_balanced_preview",
        camera_position_world=(-0.35, 0.30, 1.90),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Camera above the rear (robot-facing) part of the wood board rather than
    # laterally beside the arm.  Both candidates look near the nail/board
    # centre and preserve the physical camera optical frame used by Real data.
    "unidex_hammer_board_rear_center_preview": PointCloudProfile(
        name="unidex_hammer_board_rear_center_preview",
        camera_position_world=(-0.32, 0.02, 1.90),
        camera_target_world=(-0.08, 0.04, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_board_rear_near_vertical_preview": PointCloudProfile(
        name="unidex_hammer_board_rear_near_vertical_preview",
        camera_position_world=(-0.26, 0.02, 1.90),
        camera_target_world=(-0.08, 0.04, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_board_rear_workspace_preview": PointCloudProfile(
        name="unidex_hammer_board_rear_workspace_preview",
        camera_position_world=(-0.30, 0.02, 1.90),
        camera_target_world=(-0.15, -0.12, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Board-rear camera with no XY-plane yaw: camera and look-at target share
    # the same world Y, so the board/world X-Y axes stay aligned to the image
    # axes while the optical axis only pitches down along world X.
    "unidex_hammer_board_rear_axis_aligned_preview": PointCloudProfile(
        name="unidex_hammer_board_rear_axis_aligned_preview",
        camera_position_world=(-0.30, -0.10, 1.90),
        camera_target_world=(-0.15, -0.10, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Axis-aligned cameras pushed forward over the board.  Camera and target
    # move together towards negative world Y to reserve image space for the
    # hammer/hand on the right, while equal Y keeps zero XY-plane yaw.
    "unidex_hammer_board_forward_axis_aligned_preview": PointCloudProfile(
        name="unidex_hammer_board_forward_axis_aligned_preview",
        camera_position_world=(-0.20, -0.16, 1.90),
        camera_target_world=(-0.08, -0.16, 1.05),
        camera_fovy_deg=58.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_board_forward_wide_axis_aligned_preview": PointCloudProfile(
        name="unidex_hammer_board_forward_wide_axis_aligned_preview",
        camera_position_world=(-0.12, -0.18, 1.90),
        camera_target_world=(0.00, -0.18, 1.05),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Front-over-board look-back candidates.  The camera crosses to the
    # positive-X side of the workspace and looks back towards the board/arm.
    # Camera and target keep identical world Y, so this adds only an X-Z-plane
    # pitch and does not introduce the unwanted XY-plane yaw seen in earlier
    # oblique previews.  The two variants bracket a mild and stronger tilt.
    "unidex_hammer_board_front_lookback_mild_preview": PointCloudProfile(
        name="unidex_hammer_board_front_lookback_mild_preview",
        camera_position_world=(0.08, -0.18, 1.90),
        camera_target_world=(-0.12, -0.18, 1.05),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    "unidex_hammer_board_front_lookback_tilted_preview": PointCloudProfile(
        name="unidex_hammer_board_front_lookback_tilted_preview",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Same optical centre/direction as the tilted look-back candidate, but
    # rolled 180 degrees so the robot enters from the image bottom and the
    # board lies ahead/top in the ego-view semantic orientation.
    "unidex_hammer_board_front_lookback_ego_roll_preview": PointCloudProfile(
        name="unidex_hammer_board_front_lookback_ego_roll_preview",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Approved UniDex Real-style camera contract.  It keeps the calibrated
    # physical camera frame for point clouds and wrist poses, looks back from
    # the front-over-board side, and rolls the sensor 180 degrees so the robot
    # enters from the image bottom while the task workspace lies ahead.
    "unidex_hammer_ego_lookback_real_v6": PointCloudProfile(
        name="unidex_hammer_ego_lookback_real_v6",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Pick-bucket counterpart of the approved UniDex Real-style camera.  The
    # task spans both sides of the table in Y (boxed food near +0.18 m and the
    # bucket near -0.23 m), so this view is centred between the two objects and
    # uses a slightly wider crop than hammer-nail.  The support-plane filter
    # removes the large table surface while retaining the bucket, food box,
    # hand, and arm.  Point clouds and wrist/action poses remain expressed in
    # this fixed physical camera frame, matching the UniDex Real convention.
    "unidex_pick_bucket_ego_lookback_real_v1": PointCloudProfile(
        name="unidex_pick_bucket_ego_lookback_real_v1",
        camera_position_world=(0.22, -0.04, 1.95),
        camera_target_world=(-0.22, -0.02, 1.08),
        camera_fovy_deg=65.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.45,
        workspace_bounds_world=(
            (-0.70, -0.58, 0.88),
            (0.22, 0.45, 1.95),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
    ),
    # Fixed opposite-side observer used only as the secondary view of the
    # UniDex v10 fusion contract.  These extrinsics reproduce the task XML's
    # readable ``front`` camera while assigning them to ``ego_left`` so the
    # approved v6 primary camera can continue to own the observation key
    # ``front`` and the UniDex wrist/action reference frame.
    "unidex_hammer_front_observer_real_v10": PointCloudProfile(
        name="unidex_hammer_front_observer_real_v10",
        camera_position_world=(1.30, 0.0, 1.60),
        camera_target_world=(-0.20, 0.0, 0.90),
        camera_fovy_deg=45.0,
        min_depth_m=0.20,
        max_depth_m=2.25,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        sampling_method="fps",
        supported_camera_keys=("ego_left",),
    ),
    # Low-board-density control for OPFA+DP3.  Cameras, crop, support-plane
    # filtering and policy frame are byte-for-byte equivalent to v10.  Only
    # the sampling mix changes: visible wood-board points are retained but
    # capped at 10% per view, leaving at least 90% of the 2048-point fused
    # observation for the hand, arm, hammer, nail and other task geometry.
    "opfa_dp3_hammer_ego_low_board_v12": PointCloudProfile(
        name="opfa_dp3_hammer_ego_low_board_v12",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        semantic_limit_geom_names=("wood_collision", "wood_visual"),
        semantic_limit_max_fraction=0.10,
        sampling_method="fps",
    ),
    "opfa_dp3_hammer_front_low_board_v12": PointCloudProfile(
        name="opfa_dp3_hammer_front_low_board_v12",
        camera_position_world=(1.30, 0.0, 1.60),
        camera_target_world=(-0.20, 0.0, 0.90),
        camera_fovy_deg=45.0,
        min_depth_m=0.20,
        max_depth_m=2.25,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        semantic_limit_geom_names=("wood_collision", "wood_visual"),
        semantic_limit_max_fraction=0.10,
        sampling_method="fps",
        supported_camera_keys=("ego_left",),
    ),
    # Board-free control for the v10 top+front experiment.  The camera,
    # task crop, support-plane cut, sampling, and policy frame are unchanged;
    # only pixels belonging to the two wood-board geoms are rejected through
    # MuJoCo's RGB-D-aligned segmentation image.  A semantic mask is required
    # here: raising the height threshold would also remove the low hammer
    # handle, curled fingertips, and the inserted part of the nail.
    "unidex_hammer_ego_lookback_real_no_board_v11": PointCloudProfile(
        name="unidex_hammer_ego_lookback_real_no_board_v11",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        semantic_exclude_geom_names=("wood_collision", "wood_visual"),
        sampling_method="fps",
    ),
    "unidex_hammer_front_observer_real_no_board_v11": PointCloudProfile(
        name="unidex_hammer_front_observer_real_no_board_v11",
        camera_position_world=(1.30, 0.0, 1.60),
        camera_target_world=(-0.20, 0.0, 0.90),
        camera_fovy_deg=45.0,
        min_depth_m=0.20,
        max_depth_m=2.25,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        semantic_exclude_geom_names=("wood_collision", "wood_visual"),
        sampling_method="fps",
        supported_camera_keys=("ego_left",),
    ),
    # Experimental OPFA+DP3 contract derived from the approved v6 camera.
    # Unlike v6, the calibrated cut is taken above ``wood_top`` rather than
    # ``table_top``.  This removes the wood board's large horizontal surface
    # before FPS while retaining every visible task point above the board:
    # the hand, hammer, and exposed nail.  The 4 mm clearance rejects depth
    # quantisation around the board plane but remains below the exposed nail
    # head even at the 48 mm success threshold.  Keep this profile explicit
    # (not ``auto``) until the short DP3 ablation has passed evaluation.
    "opfa_dp3_hammer_object_only_v7": PointCloudProfile(
        name="opfa_dp3_hammer_object_only_v7",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="wood_top",
        min_height_above_support_m=0.004,
        sampling_method="fps",
    ),
    # Corrected OPFA+DP3 hammer contract.  It retains the approved v6 camera,
    # workspace crop, and table-plane rejection, but replaces the destructive
    # wood_top+4mm cut with an RGB-D-aligned MuJoCo geom mask.  Only the wood
    # board is removed among task surfaces; hammer, nail, hand, and arm points
    # survive regardless of their height above the board.
    "opfa_dp3_hammer_semantic_wood_v8": PointCloudProfile(
        name="opfa_dp3_hammer_semantic_wood_v8",
        camera_position_world=(0.18, -0.18, 1.90),
        camera_target_world=(-0.15, -0.18, 1.05),
        camera_fovy_deg=60.0,
        camera_roll_deg=180.0,
        min_depth_m=0.20,
        max_depth_m=1.15,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        semantic_exclude_geom_names=("wood_collision", "wood_visual"),
        sampling_method="fps",
    ),
    # Secondary view for the dual-view v9 ablation.  The camera is low and
    # lateral to the board, complementing v8's steep view when the hand and
    # hammer occlude the nail late in the trajectory.  It uses exactly the
    # same task-fixed crop and semantic exclusions as v8.  Each view is
    # filtered and sampled independently before v9 transforms this cloud into
    # the primary ``front`` camera frame.
    "opfa_dp3_hammer_semantic_wood_side_v9": PointCloudProfile(
        name="opfa_dp3_hammer_semantic_wood_side_v9",
        camera_position_world=(-0.20, 0.85, 1.50),
        camera_target_world=(-0.20, -0.16, 1.05),
        camera_fovy_deg=55.0,
        min_depth_m=0.20,
        max_depth_m=1.75,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        support_plane_site="table_top",
        min_height_above_support_m=0.005,
        semantic_exclude_geom_names=("wood_collision", "wood_visual"),
        sampling_method="fps",
        supported_camera_keys=("ego_left",),
    ),
    # Opposite-side elevated candidates used only for camera review.  They
    # deliberately share the final task crop and FPS contract with ego_v2;
    # none is selected by ``auto`` until the view is approved.
    "unidex_hammer_opposite_high_left_preview": PointCloudProfile(
        name="unidex_hammer_opposite_high_left_preview",
        camera_position_world=(0.85, 0.45, 2.25),
        camera_target_world=(-0.20, -0.16, 1.06),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=2.50,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        sampling_method="fps",
    ),
    "unidex_hammer_opposite_high_center_preview": PointCloudProfile(
        name="unidex_hammer_opposite_high_center_preview",
        camera_position_world=(0.95, 0.05, 2.30),
        camera_target_world=(-0.20, -0.16, 1.06),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=2.50,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        sampling_method="fps",
    ),
    "unidex_hammer_opposite_high_right_preview": PointCloudProfile(
        name="unidex_hammer_opposite_high_right_preview",
        camera_position_world=(0.85, -0.38, 2.25),
        camera_target_world=(-0.20, -0.16, 1.06),
        camera_fovy_deg=60.0,
        min_depth_m=0.20,
        max_depth_m=2.50,
        workspace_bounds_world=(
            (-0.65, -0.62, 0.88),
            (0.15, 0.30, 1.75),
        ),
        sampling_method="fps",
    ),
}


def _raw_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def resolve_pointcloud_profile(
    profile: str | PointCloudProfile | None,
    *,
    task: str | None = None,
) -> PointCloudProfile:
    """Resolve a named profile; ``auto`` selects the task-specific contract."""
    if isinstance(profile, PointCloudProfile):
        return profile
    name = "legacy" if profile is None else str(profile)
    if name == "auto":
        task_name = str(task or "")
        if task_name.startswith("hammer_nail"):
            name = "unidex_hammer_ego_lookback_real_v6"
        elif task_name.startswith("pick_bucket"):
            name = "unidex_pick_bucket_ego_lookback_real_v1"
        else:
            name = "legacy"
    try:
        return _POINTCLOUD_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_POINTCLOUD_PROFILES))
        raise ValueError(f"Unknown point-cloud profile {name!r}; expected one of: {choices}, auto") from exc


def pointcloud_profile_metadata(
    profile: str | PointCloudProfile | None,
    *,
    task: str | None = None,
) -> dict:
    result = asdict(resolve_pointcloud_profile(profile, task=task))
    result["workspace_bounds_world"] = (
        None
        if result["workspace_bounds_world"] is None
        else [list(bounds) for bounds in result["workspace_bounds_world"]]
    )
    # Preserve byte-for-byte-compatible metadata contracts for legacy
    # camera-frame profiles.  New task-frame profiles explicitly carry both
    # fields and are therefore versioned separately.
    if result["policy_frame_origin_world"] is None:
        result.pop("policy_frame_origin_world")
        result.pop("policy_frame_target_world")
    if float(result["camera_roll_deg"]) == 0.0:
        result.pop("camera_roll_deg")
    if not result["semantic_exclude_geom_names"]:
        result.pop("semantic_exclude_geom_names")
    if not result["semantic_limit_geom_names"]:
        result.pop("semantic_limit_geom_names")
    if result["semantic_limit_max_fraction"] is None:
        result.pop("semantic_limit_max_fraction")
    return result


def _camera_ids_for_keys(raw_env, image_keys):
    cam_ids = getattr(raw_env, "camera_id", None)
    if cam_ids is not None:
        cam_ids = tuple(cam_ids)
        if len(cam_ids) == len(image_keys):
            return dict(zip(image_keys, cam_ids))

    mapping = {
        "front": getattr(raw_env, "_front_camera_id", None),
        "random_camera": getattr(raw_env, "_front_camera_id", None),
        "ego_left": getattr(raw_env, "_left_camera_id", None),
        "ego_right": getattr(raw_env, "_right_camera_id", None),
        "wrist": getattr(raw_env, "_wrist_camera_id", None),
        "wrist_left": getattr(raw_env, "_wrist_left_camera_id", None),
        "wrist_right": getattr(raw_env, "_wrist_right_camera_id", None),
    }
    # Hammer-nail exposes RGB observations in this fixed order.  The explicit
    # fallback matters for subsets such as ["ego_left"], where the historical
    # length-based zip above cannot resolve the camera id.
    if cam_ids is not None and len(cam_ids) >= 4:
        ordered = {
            "front": cam_ids[0],
            "random_camera": cam_ids[0],
            "ego_left": cam_ids[1],
            "ego_right": cam_ids[2],
            "wrist": cam_ids[3],
        }
        for key, camera_id in ordered.items():
            if mapping.get(key) is None:
                mapping[key] = camera_id
    return {key: mapping[key] for key in image_keys if key in mapping and mapping[key] is not None}


def _look_at_quaternion_wxyz(
    position: np.ndarray,
    target: np.ndarray,
    *,
    camera_roll_deg: float = 0.0,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-8:
        raise ValueError("Camera position and target must be different")
    forward /= norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        raise ValueError("Camera viewing direction cannot be parallel to world up")
    right /= right_norm
    up = np.cross(right, forward)
    camera_to_world = np.column_stack([right, up, -forward])
    if float(camera_roll_deg) != 0.0:
        # Post-multiplication applies the roll in camera-local coordinates,
        # leaving the optical axis (local Z) unchanged.
        local_roll = Rotation.from_euler("z", float(camera_roll_deg), degrees=True).as_matrix()
        camera_to_world = camera_to_world @ local_roll
    return Rotation.from_matrix(camera_to_world).as_quat(scalar_first=True)


def configure_camera_for_pointcloud(
    env,
    image_key: str,
    profile: str | PointCloudProfile | None,
    *,
    task: str | None = None,
) -> PointCloudProfile:
    """Apply a profile's calibrated camera before ``env.reset()`` renders RGB."""
    resolved = resolve_pointcloud_profile(profile, task=task)
    if resolved.camera_position_world is None:
        return resolved
    if image_key not in resolved.supported_camera_keys:
        raise ValueError(
            f"Point-cloud profile {resolved.name!r} supports camera keys "
            f"{resolved.supported_camera_keys}, got {image_key!r}"
        )

    raw = _raw_env(env)
    camera_ids = _camera_ids_for_keys(raw, [image_key])
    if image_key not in camera_ids:
        raise KeyError(f"Cannot resolve MuJoCo camera id for observation key {image_key!r}")
    camera_id = int(camera_ids[image_key])
    position = np.asarray(resolved.camera_position_world, dtype=np.float64)
    target = np.asarray(resolved.camera_target_world, dtype=np.float64)
    raw._model.cam_pos[camera_id] = position
    raw._model.cam_quat[camera_id] = _look_at_quaternion_wxyz(
        position,
        target,
        camera_roll_deg=resolved.camera_roll_deg,
    )
    raw._model.cam_fovy[camera_id] = float(resolved.camera_fovy_deg)

    # Keep cam_xpos/cam_xmat synchronized for callers that query the camera
    # before the next simulator step.
    try:
        import mujoco

        mujoco.mj_forward(raw._model, raw._data)
    except (ImportError, AttributeError):
        pass
    return resolved


def camera_pose_world(env, image_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return MuJoCo camera position and camera-to-world rotation."""
    raw = _raw_env(env)
    camera_ids = _camera_ids_for_keys(raw, [image_key])
    if image_key not in camera_ids:
        raise KeyError(f"Cannot resolve MuJoCo camera id for observation key {image_key!r}")
    camera_id = int(camera_ids[image_key])
    position = np.asarray(raw._data.cam_xpos[camera_id], dtype=np.float32).copy()
    rotation = np.asarray(raw._data.cam_xmat[camera_id], dtype=np.float32).reshape(3, 3).copy()
    return position, rotation


def policy_frame_pose_world(
    env,
    image_key: str,
    profile: str | PointCloudProfile | None,
    *,
    task: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed policy-frame origin and frame-to-world rotation.

    Profiles without an explicit policy frame retain the historical behavior
    and use the physical camera frame.  This makes the new contract opt-in and
    keeps existing datasets/checkpoints reproducible.
    """
    resolved = resolve_pointcloud_profile(profile, task=task)
    if resolved.policy_frame_origin_world is None:
        return camera_pose_world(env, image_key)
    if resolved.policy_frame_target_world is None:
        raise ValueError(
            f"Point-cloud profile {resolved.name!r} defines a policy-frame origin "
            "without a target"
        )
    from scipy.spatial.transform import Rotation

    origin = np.asarray(resolved.policy_frame_origin_world, dtype=np.float32)
    target = np.asarray(resolved.policy_frame_target_world, dtype=np.float64)
    quat = _look_at_quaternion_wxyz(origin.astype(np.float64), target)
    rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix().astype(np.float32)
    return origin, rotation


def transform_pointcloud_to_policy_frame(
    pointcloud: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    policy_position_world: np.ndarray,
    policy_rotation_world: np.ndarray,
    *,
    coordinate_convention: str = "unidex",
) -> np.ndarray:
    """Re-express a colored camera point cloud in a fixed policy frame."""
    points = np.asarray(pointcloud, dtype=np.float32)
    if len(points) == 0:
        return points
    camera_xyz = points[:, :3]
    if coordinate_convention == "opencv":
        camera_xyz = camera_xyz * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    elif coordinate_convention != "unidex":
        raise ValueError(f"Unknown point-cloud coordinate convention: {coordinate_convention!r}")
    camera_position = np.asarray(camera_position_world, dtype=np.float32)
    camera_rotation = np.asarray(camera_rotation_world, dtype=np.float32).reshape(3, 3)
    policy_position = np.asarray(policy_position_world, dtype=np.float32)
    policy_rotation = np.asarray(policy_rotation_world, dtype=np.float32).reshape(3, 3)
    world_xyz = camera_xyz @ camera_rotation.T + camera_position
    policy_xyz = (world_xyz - policy_position) @ policy_rotation
    transformed = points.copy()
    transformed[:, :3] = policy_xyz
    if coordinate_convention == "opencv":
        transformed[:, :3] *= np.array([1.0, -1.0, -1.0], dtype=np.float32)
    return transformed


def depth_to_camera_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    fovy_deg: float,
    num_points: int | None = 10000,
    rng: np.random.Generator | None = None,
    coordinate_convention: str = "opencv",
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    pixel_keep_mask: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape HxWx3, got {rgb.shape}")
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"Depth shape {depth.shape} does not match RGB shape {rgb.shape[:2]}")

    finite = np.isfinite(depth) & (depth > 0)
    if pixel_keep_mask is not None:
        pixel_keep_mask = np.asarray(pixel_keep_mask, dtype=bool)
        if pixel_keep_mask.shape != depth.shape:
            raise ValueError(
                f"Pixel keep mask shape {pixel_keep_mask.shape} does not match depth {depth.shape}"
            )
        finite &= pixel_keep_mask
    if min_depth_m is not None:
        finite &= depth >= float(min_depth_m)
    if max_depth_m is not None:
        finite &= depth <= float(max_depth_m)
    if not np.any(finite):
        return np.zeros((0, 6), dtype=np.float32)

    height, width = depth.shape
    v, u = np.nonzero(finite)
    z = depth[v, u]
    fovy = math.radians(float(fovy_deg))
    fy = 0.5 * height / math.tan(0.5 * fovy)
    # MuJoCo pixels are square.  Horizontal focal length follows from fy,
    # including for non-square render targets.
    fx = fy
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    colors = rgb[v, u].astype(np.float32)
    if coordinate_convention == "unidex":
        y = -y
        z = -z
        colors = colors / 255.0
    elif coordinate_convention != "opencv":
        raise ValueError(f"Unknown point-cloud coordinate convention: {coordinate_convention!r}")
    points = np.column_stack([x, y, z, colors]).astype(np.float32)

    # Preserve the legacy API for callers that explicitly request random
    # image-wide sampling.  UniDex profiles defer sampling until after crop.
    if num_points is not None and num_points > 0 and len(points) > num_points:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(points), size=num_points, replace=False)
        points = points[idx]
    return points


def crop_pointcloud_in_world(
    pointcloud: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    workspace_bounds_world: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    | None,
    *,
    coordinate_convention: str = "unidex",
) -> np.ndarray:
    """Crop camera-frame points with a task-fixed world-frame AABB."""
    points = np.asarray(pointcloud, dtype=np.float32)
    if workspace_bounds_world is None or len(points) == 0:
        return points
    camera_xyz = points[:, :3]
    if coordinate_convention == "opencv":
        camera_xyz = camera_xyz * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    elif coordinate_convention != "unidex":
        raise ValueError(f"Unknown point-cloud coordinate convention: {coordinate_convention!r}")
    position = np.asarray(camera_position_world, dtype=np.float32)
    rotation = np.asarray(camera_rotation_world, dtype=np.float32).reshape(3, 3)
    world_xyz = camera_xyz @ rotation.T + position
    lower, upper = (np.asarray(bounds, dtype=np.float32) for bounds in workspace_bounds_world)
    keep = np.all(world_xyz >= lower, axis=1) & np.all(world_xyz <= upper, axis=1)
    return points[keep]


def filter_pointcloud_above_support_plane_in_world(
    pointcloud: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    support_plane_z_world: float,
    min_height_above_support_m: float = 0.0,
    *,
    coordinate_convention: str = "unidex",
) -> np.ndarray:
    """Remove a calibrated horizontal support plane before point sampling.

    The returned points remain in the original camera coordinate frame.  Only
    the temporary XYZ copy is transformed to world coordinates to apply the
    fixed visual contract.
    """
    points = np.asarray(pointcloud, dtype=np.float32)
    if len(points) == 0:
        return points
    camera_xyz = points[:, :3]
    if coordinate_convention == "opencv":
        camera_xyz = camera_xyz * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    elif coordinate_convention != "unidex":
        raise ValueError(f"Unknown point-cloud coordinate convention: {coordinate_convention!r}")
    position = np.asarray(camera_position_world, dtype=np.float32)
    rotation = np.asarray(camera_rotation_world, dtype=np.float32).reshape(3, 3)
    world_z = (camera_xyz @ rotation.T + position)[:, 2]
    threshold = float(support_plane_z_world) + float(min_height_above_support_m)
    return points[world_z >= threshold]


def support_plane_height_world(env, site_name: str) -> float:
    """Return a named calibration site's current world-frame Z coordinate."""
    raw = _raw_env(env)
    try:
        site_id = int(raw._model.site(site_name).id)
    except (AttributeError, KeyError) as exc:
        raise KeyError(f"Cannot resolve support-plane site {site_name!r}") from exc
    return float(raw._data.site_xpos[site_id, 2])


def sample_pointcloud(
    pointcloud: np.ndarray,
    num_points: int | None,
    *,
    method: str,
    rng: np.random.Generator | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Return an exact-size point cloud using random or official-style FPS."""
    points = np.asarray(pointcloud, dtype=np.float32)
    if num_points is None or num_points <= 0:
        return points
    if len(points) == 0:
        raise ValueError("Point-cloud preprocessing removed every RGB-D point")
    if rng is None:
        rng = np.random.default_rng(0)
    if len(points) < num_points:
        pad = points[rng.choice(len(points), num_points - len(points), replace=True)]
        points = np.concatenate([points, pad], axis=0)
    if len(points) == num_points:
        return points.astype(np.float32, copy=False)
    if method == "random":
        return points[rng.choice(len(points), num_points, replace=False)]
    if method != "fps":
        raise ValueError(f"Unknown point-cloud sampling method: {method!r}")

    return sample_pointcloud_batch(
        [points],
        num_points,
        method=method,
        rng=rng,
        device=device,
    )[0]


def sample_pointcloud_batch(
    pointclouds: list[np.ndarray] | tuple[np.ndarray, ...],
    num_points: int,
    *,
    method: str,
    rng: np.random.Generator | None = None,
    device: str | None = None,
) -> list[np.ndarray]:
    """Sample multiple variable-size clouds in one FPS kernel launch.

    This mirrors UniDex's official batched cache path and is materially faster
    for dense RGB-D export than launching a 10k-point FPS kernel per frame.
    """
    clouds = [np.asarray(pointcloud, dtype=np.float32) for pointcloud in pointclouds]
    if not clouds:
        return []
    if num_points <= 0:
        return clouds
    if rng is None:
        rng = np.random.default_rng(0)
    if any(len(points) == 0 for points in clouds):
        raise ValueError("Point-cloud preprocessing removed every RGB-D point")
    if method == "random":
        sampled = []
        for points in clouds:
            if len(points) < num_points:
                pad = points[rng.choice(len(points), num_points - len(points), replace=True)]
                points = np.concatenate([points, pad], axis=0)
            if len(points) > num_points:
                points = points[rng.choice(len(points), num_points, replace=False)]
            sampled.append(points.astype(np.float32, copy=False))
        return sampled
    if method != "fps":
        raise ValueError(f"Unknown point-cloud sampling method: {method!r}")

    try:
        import torch
        from pytorch3d.ops import sample_farthest_points
    except ImportError as exc:
        raise RuntimeError(
            "The UniDex point-cloud profile requires PyTorch3D for farthest-point sampling"
        ) from exc

    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    padded_clouds = []
    effective_lengths = []
    for points in clouds:
        if len(points) < num_points:
            pad = points[rng.choice(len(points), num_points - len(points), replace=True)]
            points = np.concatenate([points, pad], axis=0)
        padded_clouds.append(points)
        effective_lengths.append(len(points))
    max_points = max(effective_lengths)
    for index, points in enumerate(padded_clouds):
        if len(points) < max_points:
            pad = points[rng.choice(len(points), max_points - len(points), replace=True)]
            padded_clouds[index] = np.concatenate([points, pad], axis=0)
    tensor = torch.from_numpy(np.stack(padded_clouds)).to(device=device, dtype=torch.float32)
    lengths = torch.as_tensor(effective_lengths, dtype=torch.int64, device=device)
    with torch.inference_mode():
        _, indices = sample_farthest_points(
            tensor[..., :3],
            lengths=lengths,
            K=int(num_points),
        )
        sampled = torch.gather(
            tensor,
            1,
            indices[..., None].expand(-1, -1, tensor.shape[-1]),
        )
    array = sampled.cpu().numpy().astype(np.float32, copy=False)
    return [array[index] for index in range(len(clouds))]


def collect_pointcloud_frames(
    env,
    observation: dict,
    image_keys: list[str] | tuple[str, ...],
    num_points: int | None = 10000,
    rng: np.random.Generator | None = None,
    coordinate_convention: str = "opencv",
    profile: str | PointCloudProfile | None = None,
    task: str | None = None,
    sampling_device: str | None = None,
) -> dict[str, np.ndarray]:
    resolved = resolve_pointcloud_profile(profile, task=task)
    raw = _raw_env(env)
    depths = collect_depth_frames(env, image_keys)
    segmentations = (
        collect_segmentation_frames(env, image_keys)
        if (
            resolved.semantic_exclude_geom_names
            or resolved.semantic_limit_geom_names
        )
        else {}
    )
    if bool(resolved.semantic_limit_geom_names) != (
        resolved.semantic_limit_max_fraction is not None
    ):
        raise ValueError(
            "semantic_limit_geom_names and semantic_limit_max_fraction must "
            "be configured together"
        )
    if resolved.semantic_limit_max_fraction is not None and not (
        0.0 <= float(resolved.semantic_limit_max_fraction) <= 1.0
    ):
        raise ValueError(
            "semantic_limit_max_fraction must be within [0, 1], got "
            f"{resolved.semantic_limit_max_fraction}"
        )
    cam_id_by_key = _camera_ids_for_keys(raw, image_keys)
    out = {}
    for key in image_keys:
        if key not in depths or key not in observation or key not in cam_id_by_key:
            continue
        if resolved.name != "legacy" and key not in resolved.supported_camera_keys:
            raise ValueError(
                f"Point-cloud profile {resolved.name!r} does not support camera key {key!r}"
            )
        cam_id = int(cam_id_by_key[key])
        fovy = float(raw._model.cam_fovy[cam_id])
        pixel_keep_mask = None
        semantic_limit_mask = None
        if resolved.semantic_exclude_geom_names or resolved.semantic_limit_geom_names:
            import mujoco

            segmentation = segmentations.get(key)
            if segmentation is None or segmentation.shape != (*depths[key].shape, 2):
                raise RuntimeError(
                    f"Missing or misaligned segmentation for camera {key!r}: "
                    f"{None if segmentation is None else segmentation.shape}"
                )
            is_geom = segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
            excluded_geom_ids = []
            for geom_name in resolved.semantic_exclude_geom_names:
                try:
                    excluded_geom_ids.append(int(raw._model.geom(geom_name).id))
                except (AttributeError, KeyError) as exc:
                    raise KeyError(
                        f"Cannot resolve semantic exclusion geom {geom_name!r}"
                    ) from exc
            if excluded_geom_ids:
                is_excluded = np.isin(segmentation[..., 1], excluded_geom_ids)
                pixel_keep_mask = ~(is_geom & is_excluded)
            limited_geom_ids = []
            for geom_name in resolved.semantic_limit_geom_names:
                try:
                    limited_geom_ids.append(int(raw._model.geom(geom_name).id))
                except (AttributeError, KeyError) as exc:
                    raise KeyError(
                        f"Cannot resolve semantic limit geom {geom_name!r}"
                    ) from exc
            if limited_geom_ids:
                semantic_limit_mask = is_geom & np.isin(
                    segmentation[..., 1], limited_geom_ids
                )

        camera_position = camera_rotation = None
        if resolved.name != "legacy":
            camera_position, camera_rotation = camera_pose_world(env, key)

        def _project_and_filter(mask: np.ndarray | None) -> np.ndarray:
            points = depth_to_camera_pointcloud(
                observation[key],
                depths[key],
                fovy_deg=fovy,
                num_points=None if resolved.name != "legacy" else num_points,
                rng=rng,
                coordinate_convention=coordinate_convention,
                min_depth_m=resolved.min_depth_m,
                max_depth_m=resolved.max_depth_m,
                pixel_keep_mask=mask,
            )
            if resolved.name == "legacy":
                return points
            points = crop_pointcloud_in_world(
                points,
                camera_position,
                camera_rotation,
                resolved.workspace_bounds_world,
                coordinate_convention=coordinate_convention,
            )
            if resolved.support_plane_site is not None:
                points = filter_pointcloud_above_support_plane_in_world(
                    points,
                    camera_position,
                    camera_rotation,
                    support_plane_height_world(env, resolved.support_plane_site),
                    resolved.min_height_above_support_m,
                    coordinate_convention=coordinate_convention,
                )
            if resolved.policy_frame_origin_world is not None:
                policy_position, policy_rotation = policy_frame_pose_world(
                    env,
                    key,
                    resolved,
                    task=task,
                )
                points = transform_pointcloud_to_policy_frame(
                    points,
                    camera_position,
                    camera_rotation,
                    policy_position,
                    policy_rotation,
                    coordinate_convention=coordinate_convention,
                )
            return points

        if semantic_limit_mask is None or resolved.name == "legacy":
            points = _project_and_filter(pixel_keep_mask)
            if resolved.name != "legacy":
                points = sample_pointcloud(
                    points,
                    num_points,
                    method=resolved.sampling_method,
                    rng=rng,
                    device=sampling_device,
                )
        else:
            keep = (
                np.ones_like(semantic_limit_mask, dtype=bool)
                if pixel_keep_mask is None
                else pixel_keep_mask
            )
            limited_points = _project_and_filter(keep & semantic_limit_mask)
            other_points = _project_and_filter(keep & ~semantic_limit_mask)
            if num_points is None or int(num_points) <= 0:
                points = np.concatenate([other_points, limited_points], axis=0)
            else:
                if len(other_points) == 0:
                    raise ValueError(
                        f"Semantic quota profile {resolved.name!r} retained no "
                        "non-limited task points"
                    )
                limited_count = (
                    min(
                        int(round(int(num_points) * float(
                            resolved.semantic_limit_max_fraction
                        ))),
                        int(num_points),
                    )
                    if len(limited_points)
                    else 0
                )
                other_count = int(num_points) - limited_count
                sampled_other = sample_pointcloud(
                    other_points,
                    other_count,
                    method=resolved.sampling_method,
                    rng=rng,
                    device=sampling_device,
                )
                if limited_count:
                    sampled_limited = sample_pointcloud(
                        limited_points,
                        limited_count,
                        method=resolved.sampling_method,
                        rng=rng,
                        device=sampling_device,
                    )
                    points = np.concatenate(
                        [sampled_other, sampled_limited], axis=0
                    )
                    points = points[rng.permutation(len(points))]
                else:
                    points = sampled_other
        out[key] = points
    return out


DUAL_VIEW_V9_PRIMARY_PROFILE = "opfa_dp3_hammer_semantic_wood_v8"
DUAL_VIEW_V9_SECONDARY_PROFILE = "opfa_dp3_hammer_semantic_wood_side_v9"

HAMMER_TOP_FRONT_FUSION_V10 = "unidex_hammer_top_front_fused_v10"
DUAL_VIEW_V10_PRIMARY_PROFILE = "unidex_hammer_ego_lookback_real_v6"
DUAL_VIEW_V10_SECONDARY_PROFILE = "unidex_hammer_front_observer_real_v10"
DUAL_VIEW_V10_PRIMARY_CAMERA = "front"
DUAL_VIEW_V10_SECONDARY_CAMERA = "ego_left"
DUAL_VIEW_V10_PRIMARY_FRACTION = 0.60

HAMMER_TOP_FRONT_NO_BOARD_FUSION_V11 = (
    "unidex_hammer_top_front_fused_no_board_v11"
)
DUAL_VIEW_V11_PRIMARY_PROFILE = "unidex_hammer_ego_lookback_real_no_board_v11"
DUAL_VIEW_V11_SECONDARY_PROFILE = (
    "unidex_hammer_front_observer_real_no_board_v11"
)

HAMMER_TOP_FRONT_LOW_BOARD_FUSION_V12 = (
    "opfa_dp3_hammer_top_front_fused_low_board_v12"
)
DUAL_VIEW_V12_PRIMARY_PROFILE = "opfa_dp3_hammer_ego_low_board_v12"
DUAL_VIEW_V12_SECONDARY_PROFILE = "opfa_dp3_hammer_front_low_board_v12"


def configure_hammer_dual_view_v9(env) -> tuple[PointCloudProfile, PointCloudProfile]:
    """Configure the fixed primary and lateral cameras used by dual-view v9.

    This is intentionally an opt-in helper so no historical single-view
    dataset or checkpoint changes behavior.
    """
    primary = configure_camera_for_pointcloud(
        env,
        "front",
        DUAL_VIEW_V9_PRIMARY_PROFILE,
        task="hammer_nail_inspire",
    )
    secondary = configure_camera_for_pointcloud(
        env,
        "ego_left",
        DUAL_VIEW_V9_SECONDARY_PROFILE,
        task="hammer_nail_inspire",
    )
    return primary, secondary


def collect_hammer_dual_view_v9(
    env,
    observation: dict,
    *,
    num_points: int = 1024,
    primary_points: int | None = None,
    rng: np.random.Generator | None = None,
    sampling_device: str | None = None,
) -> dict[str, np.ndarray]:
    """Build an exact-size two-camera cloud in the primary camera frame.

    Each RGB-D view is cropped, semantically filtered, and FPS sampled on its
    own.  The secondary cloud is then transformed using MuJoCo's fixed camera
    extrinsics.  Raw camera-local clouds must never be concatenated directly.
    """
    if int(num_points) <= 1:
        raise ValueError("dual-view fusion requires at least two points")
    if primary_points is None:
        primary_points = int(num_points) // 2
    secondary_points = int(num_points) - int(primary_points)
    if primary_points <= 0 or secondary_points <= 0:
        raise ValueError(
            f"Both views need points, got primary={primary_points}, "
            f"secondary={secondary_points}"
        )
    if rng is None:
        rng = np.random.default_rng(0)

    primary = collect_pointcloud_frames(
        env,
        observation,
        ["front"],
        num_points=int(primary_points),
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V9_PRIMARY_PROFILE,
        task="hammer_nail_inspire",
        sampling_device=sampling_device,
    ).get("front")
    secondary = collect_pointcloud_frames(
        env,
        observation,
        ["ego_left"],
        num_points=secondary_points,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V9_SECONDARY_PROFILE,
        task="hammer_nail_inspire",
        sampling_device=sampling_device,
    ).get("ego_left")
    expected_primary = (int(primary_points), 6)
    expected_secondary = (secondary_points, 6)
    if primary is None or primary.shape != expected_primary:
        raise RuntimeError(
            f"Unexpected dual-view primary cloud: "
            f"{None if primary is None else primary.shape}; expected {expected_primary}"
        )
    if secondary is None or secondary.shape != expected_secondary:
        raise RuntimeError(
            f"Unexpected dual-view secondary cloud: "
            f"{None if secondary is None else secondary.shape}; expected {expected_secondary}"
        )

    primary_position, primary_rotation = camera_pose_world(env, "front")
    secondary_position, secondary_rotation = camera_pose_world(env, "ego_left")
    secondary_in_primary = transform_pointcloud_to_policy_frame(
        secondary,
        secondary_position,
        secondary_rotation,
        primary_position,
        primary_rotation,
        coordinate_convention="unidex",
    )
    fused = np.concatenate([primary, secondary_in_primary], axis=0).astype(
        np.float32, copy=False
    )
    if fused.shape != (int(num_points), 6) or not np.all(np.isfinite(fused)):
        raise RuntimeError(
            f"Invalid fused dual-view cloud: shape={fused.shape}, "
            f"finite={bool(np.all(np.isfinite(fused)))}"
        )
    return {
        "front": primary,
        "ego_left_in_front": secondary_in_primary,
        "fused": fused,
    }


def hammer_dual_view_v9_metadata(
    *, num_points: int = 1024, primary_points: int | None = None
) -> dict:
    """Serializable visual-input contract for manifests and eval reports."""
    if primary_points is None:
        primary_points = int(num_points) // 2
    return {
        "name": "opfa_dp3_hammer_dual_view_v9",
        "primary_camera_key": "front",
        "secondary_camera_key": "ego_left",
        "primary_points": int(primary_points),
        "secondary_points": int(num_points) - int(primary_points),
        "total_points": int(num_points),
        "fusion_frame": "primary_camera_unidex",
        "fusion_transform": "fixed_mujoco_camera_extrinsics",
        "sampling_order": "semantic_filter_crop_fps_per_view_then_transform_concat",
        "primary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V9_PRIMARY_PROFILE, task="hammer_nail_inspire"
        ),
        "secondary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V9_SECONDARY_PROFILE, task="hammer_nail_inspire"
        ),
    }


def configure_hammer_top_front_fusion_v10(
    env,
    *,
    task: str = "hammer_nail_inspire",
) -> tuple[PointCloudProfile, PointCloudProfile]:
    """Configure the fixed v6 primary and opposite-side front cameras.

    The primary camera remains the sole policy/wrist reference.  The second
    camera changes visibility only and must be transformed into the primary
    frame before the two clouds are concatenated.
    """
    primary = configure_camera_for_pointcloud(
        env,
        DUAL_VIEW_V10_PRIMARY_CAMERA,
        DUAL_VIEW_V10_PRIMARY_PROFILE,
        task=task,
    )
    secondary = configure_camera_for_pointcloud(
        env,
        DUAL_VIEW_V10_SECONDARY_CAMERA,
        DUAL_VIEW_V10_SECONDARY_PROFILE,
        task=task,
    )
    return primary, secondary


def _dual_view_v10_point_counts(
    num_points: int,
    primary_points: int | None,
) -> tuple[int, int]:
    total = int(num_points)
    if total <= 1:
        raise ValueError("top+front fusion requires at least two points")
    if primary_points is None:
        primary_points = int(round(total * DUAL_VIEW_V10_PRIMARY_FRACTION))
    primary = int(primary_points)
    secondary = total - primary
    if primary <= 0 or secondary <= 0:
        raise ValueError(
            f"Both fusion views need points, got primary={primary}, secondary={secondary}"
        )
    return primary, secondary


def collect_hammer_top_front_fusion_v10(
    env,
    observation: dict,
    *,
    num_points: int = 10000,
    primary_points: int | None = None,
    rng: np.random.Generator | None = None,
    sampling_device: str | None = None,
    task: str = "hammer_nail_inspire",
) -> dict[str, np.ndarray]:
    """Return v6 top, transformed front, and fused colored point clouds.

    Both views are independently cropped and FPS sampled.  The secondary
    camera cloud is re-expressed in the v6 primary camera frame using the live
    fixed MuJoCo extrinsics; raw camera-local XYZ arrays are never concatenated.
    """
    primary_count, secondary_count = _dual_view_v10_point_counts(
        num_points,
        primary_points,
    )
    if rng is None:
        rng = np.random.default_rng(0)

    primary = collect_pointcloud_frames(
        env,
        observation,
        [DUAL_VIEW_V10_PRIMARY_CAMERA],
        num_points=primary_count,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V10_PRIMARY_PROFILE,
        task=task,
        sampling_device=sampling_device,
    ).get(DUAL_VIEW_V10_PRIMARY_CAMERA)
    secondary = collect_pointcloud_frames(
        env,
        observation,
        [DUAL_VIEW_V10_SECONDARY_CAMERA],
        num_points=secondary_count,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V10_SECONDARY_PROFILE,
        task=task,
        sampling_device=sampling_device,
    ).get(DUAL_VIEW_V10_SECONDARY_CAMERA)
    expected_primary = (primary_count, 6)
    expected_secondary = (secondary_count, 6)
    if primary is None or primary.shape != expected_primary:
        raise RuntimeError(
            "Unexpected top+front primary cloud: "
            f"{None if primary is None else primary.shape}; expected {expected_primary}"
        )
    if secondary is None or secondary.shape != expected_secondary:
        raise RuntimeError(
            "Unexpected top+front secondary cloud: "
            f"{None if secondary is None else secondary.shape}; expected {expected_secondary}"
        )

    primary_position, primary_rotation = camera_pose_world(
        env,
        DUAL_VIEW_V10_PRIMARY_CAMERA,
    )
    secondary_position, secondary_rotation = camera_pose_world(
        env,
        DUAL_VIEW_V10_SECONDARY_CAMERA,
    )
    secondary_in_primary = transform_pointcloud_to_policy_frame(
        secondary,
        secondary_position,
        secondary_rotation,
        primary_position,
        primary_rotation,
        coordinate_convention="unidex",
    )
    fused = np.concatenate([primary, secondary_in_primary], axis=0).astype(
        np.float32,
        copy=False,
    )
    expected_fused = (int(num_points), 6)
    if fused.shape != expected_fused or not np.all(np.isfinite(fused)):
        raise RuntimeError(
            f"Invalid top+front fused cloud: shape={fused.shape}, "
            f"expected={expected_fused}, finite={bool(np.all(np.isfinite(fused)))}"
        )
    return {
        "primary": primary,
        "secondary_in_primary": secondary_in_primary,
        "fused": fused,
    }


def hammer_top_front_fusion_v10_metadata(
    *,
    num_points: int = 10000,
    primary_points: int | None = None,
) -> dict:
    """Return the complete serializable UniDex v10 fusion contract."""
    primary_count, secondary_count = _dual_view_v10_point_counts(
        num_points,
        primary_points,
    )
    return {
        "name": HAMMER_TOP_FRONT_FUSION_V10,
        "primary_camera_key": DUAL_VIEW_V10_PRIMARY_CAMERA,
        "secondary_camera_key": DUAL_VIEW_V10_SECONDARY_CAMERA,
        "primary_points": primary_count,
        "secondary_points": secondary_count,
        "total_points": int(num_points),
        "fusion_frame": "primary_v6_camera_unidex",
        "fusion_transform": "fixed_mujoco_camera_extrinsics",
        "sampling_order": "crop_filter_fps_per_view_then_transform_concat",
        "state_action_reference": "primary_v6_camera_only",
        "primary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V10_PRIMARY_PROFILE,
            task="hammer_nail_inspire",
        ),
        "secondary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V10_SECONDARY_PROFILE,
            task="hammer_nail_inspire",
        ),
    }


def configure_hammer_top_front_low_board_fusion_v12(
    env,
    *,
    task: str = "hammer_nail_inspire",
) -> tuple[PointCloudProfile, PointCloudProfile]:
    """Configure v10 cameras with a per-view 10% wood-board point cap."""
    primary = configure_camera_for_pointcloud(
        env,
        DUAL_VIEW_V10_PRIMARY_CAMERA,
        DUAL_VIEW_V12_PRIMARY_PROFILE,
        task=task,
    )
    secondary = configure_camera_for_pointcloud(
        env,
        DUAL_VIEW_V10_SECONDARY_CAMERA,
        DUAL_VIEW_V12_SECONDARY_PROFILE,
        task=task,
    )
    return primary, secondary


def collect_hammer_top_front_low_board_fusion_v12(
    env,
    observation: dict,
    *,
    num_points: int = 2048,
    primary_points: int | None = None,
    rng: np.random.Generator | None = None,
    sampling_device: str | None = None,
    task: str = "hammer_nail_inspire",
) -> dict[str, np.ndarray]:
    """Return an exact-size low-board-density cloud in the v10 policy frame."""
    primary_count, secondary_count = _dual_view_v10_point_counts(
        num_points,
        primary_points,
    )
    if rng is None:
        rng = np.random.default_rng(0)
    primary = collect_pointcloud_frames(
        env,
        observation,
        [DUAL_VIEW_V10_PRIMARY_CAMERA],
        num_points=primary_count,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V12_PRIMARY_PROFILE,
        task=task,
        sampling_device=sampling_device,
    ).get(DUAL_VIEW_V10_PRIMARY_CAMERA)
    secondary = collect_pointcloud_frames(
        env,
        observation,
        [DUAL_VIEW_V10_SECONDARY_CAMERA],
        num_points=secondary_count,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V12_SECONDARY_PROFILE,
        task=task,
        sampling_device=sampling_device,
    ).get(DUAL_VIEW_V10_SECONDARY_CAMERA)
    expected_primary = (primary_count, 6)
    expected_secondary = (secondary_count, 6)
    if primary is None or primary.shape != expected_primary:
        raise RuntimeError(
            "Unexpected v12 primary cloud: "
            f"{None if primary is None else primary.shape}; expected {expected_primary}"
        )
    if secondary is None or secondary.shape != expected_secondary:
        raise RuntimeError(
            "Unexpected v12 secondary cloud: "
            f"{None if secondary is None else secondary.shape}; expected {expected_secondary}"
        )
    primary_position, primary_rotation = camera_pose_world(
        env, DUAL_VIEW_V10_PRIMARY_CAMERA
    )
    secondary_position, secondary_rotation = camera_pose_world(
        env, DUAL_VIEW_V10_SECONDARY_CAMERA
    )
    secondary_in_primary = transform_pointcloud_to_policy_frame(
        secondary,
        secondary_position,
        secondary_rotation,
        primary_position,
        primary_rotation,
        coordinate_convention="unidex",
    )
    fused = np.concatenate([primary, secondary_in_primary], axis=0).astype(
        np.float32, copy=False
    )
    expected_fused = (int(num_points), 6)
    if fused.shape != expected_fused or not np.all(np.isfinite(fused)):
        raise RuntimeError(
            f"Invalid v12 fused cloud: shape={fused.shape}, "
            f"expected={expected_fused}, finite={bool(np.all(np.isfinite(fused)))}"
        )
    return {
        "primary": primary,
        "secondary_in_primary": secondary_in_primary,
        "fused": fused,
    }


def hammer_top_front_low_board_fusion_v12_metadata(
    *,
    num_points: int = 2048,
    primary_points: int | None = None,
) -> dict:
    """Return the serializable v12 camera, quota and fusion contract."""
    primary_count, secondary_count = _dual_view_v10_point_counts(
        num_points, primary_points
    )
    primary_board_cap = int(round(primary_count * 0.10))
    secondary_board_cap = int(round(secondary_count * 0.10))
    return {
        "name": HAMMER_TOP_FRONT_LOW_BOARD_FUSION_V12,
        "primary_camera_key": DUAL_VIEW_V10_PRIMARY_CAMERA,
        "secondary_camera_key": DUAL_VIEW_V10_SECONDARY_CAMERA,
        "primary_points": primary_count,
        "secondary_points": secondary_count,
        "total_points": int(num_points),
        "wood_board_point_cap": primary_board_cap + secondary_board_cap,
        "wood_board_fraction_cap": (
            primary_board_cap + secondary_board_cap
        ) / float(num_points),
        "wood_board_quota_scope": "independent_per_view",
        "non_board_point_floor": int(num_points)
        - primary_board_cap
        - secondary_board_cap,
        "fusion_frame": "primary_v6_camera_unidex",
        "fusion_transform": "fixed_mujoco_camera_extrinsics",
        "sampling_order": (
            "semantic_split_crop_filter_fps_by_quota_per_view_then_transform_concat"
        ),
        "state_action_reference": "primary_v6_camera_only",
        "primary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V12_PRIMARY_PROFILE,
            task="hammer_nail_inspire",
        ),
        "secondary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V12_SECONDARY_PROFILE,
            task="hammer_nail_inspire",
        ),
    }


def configure_hammer_top_front_no_board_fusion_v11(
    env,
    *,
    task: str = "hammer_nail_inspire",
) -> tuple[PointCloudProfile, PointCloudProfile]:
    """Configure the v10 cameras with semantic wood-board rejection."""
    primary = configure_camera_for_pointcloud(
        env,
        DUAL_VIEW_V10_PRIMARY_CAMERA,
        DUAL_VIEW_V11_PRIMARY_PROFILE,
        task=task,
    )
    secondary = configure_camera_for_pointcloud(
        env,
        DUAL_VIEW_V10_SECONDARY_CAMERA,
        DUAL_VIEW_V11_SECONDARY_PROFILE,
        task=task,
    )
    return primary, secondary


def collect_hammer_top_front_no_board_fusion_v11(
    env,
    observation: dict,
    *,
    num_points: int = 10000,
    primary_points: int | None = None,
    rng: np.random.Generator | None = None,
    sampling_device: str | None = None,
    task: str = "hammer_nail_inspire",
) -> dict[str, np.ndarray]:
    """Return the board-free 6k top + 4k transformed-front observation."""
    primary_count, secondary_count = _dual_view_v10_point_counts(
        num_points,
        primary_points,
    )
    if rng is None:
        rng = np.random.default_rng(0)

    primary = collect_pointcloud_frames(
        env,
        observation,
        [DUAL_VIEW_V10_PRIMARY_CAMERA],
        num_points=primary_count,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V11_PRIMARY_PROFILE,
        task=task,
        sampling_device=sampling_device,
    ).get(DUAL_VIEW_V10_PRIMARY_CAMERA)
    secondary = collect_pointcloud_frames(
        env,
        observation,
        [DUAL_VIEW_V10_SECONDARY_CAMERA],
        num_points=secondary_count,
        rng=rng,
        coordinate_convention="unidex",
        profile=DUAL_VIEW_V11_SECONDARY_PROFILE,
        task=task,
        sampling_device=sampling_device,
    ).get(DUAL_VIEW_V10_SECONDARY_CAMERA)
    expected_primary = (primary_count, 6)
    expected_secondary = (secondary_count, 6)
    if primary is None or primary.shape != expected_primary:
        raise RuntimeError(
            "Unexpected no-board primary cloud: "
            f"{None if primary is None else primary.shape}; expected {expected_primary}"
        )
    if secondary is None or secondary.shape != expected_secondary:
        raise RuntimeError(
            "Unexpected no-board secondary cloud: "
            f"{None if secondary is None else secondary.shape}; expected {expected_secondary}"
        )

    primary_position, primary_rotation = camera_pose_world(
        env,
        DUAL_VIEW_V10_PRIMARY_CAMERA,
    )
    secondary_position, secondary_rotation = camera_pose_world(
        env,
        DUAL_VIEW_V10_SECONDARY_CAMERA,
    )
    secondary_in_primary = transform_pointcloud_to_policy_frame(
        secondary,
        secondary_position,
        secondary_rotation,
        primary_position,
        primary_rotation,
        coordinate_convention="unidex",
    )
    fused = np.concatenate([primary, secondary_in_primary], axis=0).astype(
        np.float32,
        copy=False,
    )
    expected_fused = (int(num_points), 6)
    if fused.shape != expected_fused or not np.all(np.isfinite(fused)):
        raise RuntimeError(
            f"Invalid no-board fused cloud: shape={fused.shape}, "
            f"expected={expected_fused}, finite={bool(np.all(np.isfinite(fused)))}"
        )
    return {
        "primary": primary,
        "secondary_in_primary": secondary_in_primary,
        "fused": fused,
    }


def hammer_top_front_no_board_fusion_v11_metadata(
    *,
    num_points: int = 10000,
    primary_points: int | None = None,
) -> dict:
    """Return the complete board-free comparison contract."""
    primary_count, secondary_count = _dual_view_v10_point_counts(
        num_points,
        primary_points,
    )
    return {
        "name": HAMMER_TOP_FRONT_NO_BOARD_FUSION_V11,
        "primary_camera_key": DUAL_VIEW_V10_PRIMARY_CAMERA,
        "secondary_camera_key": DUAL_VIEW_V10_SECONDARY_CAMERA,
        "primary_points": primary_count,
        "secondary_points": secondary_count,
        "total_points": int(num_points),
        "fusion_frame": "primary_v6_camera_unidex",
        "fusion_transform": "fixed_mujoco_camera_extrinsics",
        "sampling_order": (
            "semantic_wood_filter_crop_filter_fps_per_view_then_transform_concat"
        ),
        "state_action_reference": "primary_v6_camera_only",
        "semantic_exclusion": {
            "object": "wood_board",
            "geom_names": ["wood_collision", "wood_visual"],
            "method": "aligned_mujoco_segmentation_before_projection_and_fps",
        },
        "primary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V11_PRIMARY_PROFILE,
            task="hammer_nail_inspire",
        ),
        "secondary_profile": pointcloud_profile_metadata(
            DUAL_VIEW_V11_SECONDARY_PROFILE,
            task="hammer_nail_inspire",
        ),
    }
